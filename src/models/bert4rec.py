import torch
from torch import nn as nn
import pytorch_lightning as pl
from .embedding import SimpleEmbedding, PositionalEmbedding  # 导入嵌入层
from .new_transformer import TransformerBlock  # 导入Transformer块
from .utils import SAGP, wasserstein_distance_matmul  # 导入辅助函数

class BERT(pl.LightningModule):
    def __init__(self,
        max_len: int = None,       # 序列最大长度
        num_items: int = None,      # 物品总数
        n_layer: int = None,        # Transformer层数
        n_head: int = None,         # 注意力头数
        num_users: int = None,      # 用户总数
        n_b: int = None,            # 行为类型数量
        d_model: int = None,        # 模型维度
        dropout: float = .0,        # Dropout概率
    ):
        super().__init__()
        # 初始化模型参数
        self.d_model = d_model
        self.num_items = num_items
        self.num_users = num_users
        self.max_len = max_len
        self.n_b = n_b
        self.activation = nn.ELU()  # ELU激活函数
        # SAGP（自适应的高斯模式）相关参数
        self.Wub = nn.Linear(d_model, d_model)  # 用户行为转换矩阵
        self.WPub = nn.Linear(d_model, d_model)  # 位置行为转换矩阵
        ''' 动态表示编码 '''
        # 实体分布嵌入层（均值和协方差）
        self.item_embedding_m = SimpleEmbedding(vocab_size=num_items+2, embed_size=d_model, dropout=dropout)  # 物品均值嵌入
        self.item_embedding_c = SimpleEmbedding(vocab_size=num_items+2, embed_size=d_model, dropout=dropout)  # 物品协方差嵌入
        self.b_embedding_m = SimpleEmbedding(vocab_size=n_b+1, embed_size=d_model, dropout=dropout)  # 行为均值嵌入
        self.b_embedding_c = SimpleEmbedding(vocab_size=n_b+1, embed_size=d_model, dropout=dropout)  # 行为协方差嵌入
        self.u_embedding_m = SimpleEmbedding(vocab_size=num_users+1, embed_size=d_model, dropout=dropout)  # 用户均值嵌入
        self.u_embedding_c = SimpleEmbedding(vocab_size=num_users+1, embed_size=d_model, dropout=dropout)  # 用户协方差嵌入
        self.p_embedding_m = PositionalEmbedding(max_len=max_len, d_model=d_model)  # 位置均值嵌入
        self.p_embedding_c = PositionalEmbedding(max_len=max_len, d_model=d_model)  # 位置协方差嵌入
        # 行为关系分布嵌入层
        self.bb_embedding_m = SimpleEmbedding(vocab_size=n_b*n_b+1, embed_size=d_model, dropout=dropout)  # 行为关系均值嵌入
        self.bb_embedding_c = SimpleEmbedding(vocab_size=n_b*n_b+1, embed_size=d_model, dropout=dropout)  # 行为关系协方差嵌入
        # 多层Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_model * 4, n_b, dropout)  # Transformer块
            for _ in range(n_layer)  # 创建n_layer层
        ])

    def forward(self, x, b_seq, u_id):
        """前向传播"""
        # 创建掩码（非填充位置）
        mask = (x > 0)
        # 获取物品、行为和用户的嵌入表示（均值和协方差）
        x_m = self.item_embedding_m(x)  # 物品均值嵌入
        x_c = self.item_embedding_c(x) + 1  # 物品协方差嵌入（+1确保正值）
        b_m = self.b_embedding_m(b_seq)  # 行为均值嵌入
        b_c = self.b_embedding_c(b_seq) + 1  # 行为协方差嵌入（+1确保正值）
        u_m = self.u_embedding_m(u_id).squeeze()  # 用户均值嵌入（压缩维度）
        u_c = self.u_embedding_c(u_id).squeeze() + 1  # 用户协方差嵌入（+1确保正值）
        p_m = self.p_embedding_m(x)  # 位置均值嵌入
        p_c = self.p_embedding_c(x) + 1  # 位置协方差嵌入（+1确保正值）
        batch_size = u_id.size(0)  # 获取批次大小
        ''' 个性化模式学习 '''
        # 创建所有行为类型的索引（0到n_b）
        behavior_indices = torch.LongTensor([[0,1,2,3,4]] * u_id.size(0)).to(self.device)
        b_mi = self.b_embedding_m(behavior_indices).squeeze()  # 所有行为均值
        b_ci = self.b_embedding_c(behavior_indices).squeeze()  # 所有行为协方差
        b_ci = self.activation(b_ci) + 1  # 应用激活函数并确保正值
        # 计算个性化行为模式分布（SAGP操作）
        P_user_behavior_m, P_user_behavior_c = SAGP(
            u_m.unsqueeze(1),  # 用户均值（增加维度）
            self.Wub(b_mi),     # 转换后的行为均值
            u_c.unsqueeze(1),   # 用户协方差（增加维度）
            b_ci                # 行为协方差
        )
        ''' 行为协作影响因子 '''
        # 计算行为间的Wasserstein距离作为影响权重
        Weight_user_behavior = -wasserstein_distance_matmul(
            P_user_behavior_m, P_user_behavior_c,  # 源分布
            P_user_behavior_m, P_user_behavior_c   # 目标分布
        )
        # 初始化行为关系表示
        bb_m = torch.zeros(u_id.size(0), self.n_b+1, self.n_b+1, self.d_model).cuda()  # 均值矩阵
        bb_c = torch.zeros(u_id.size(0), self.n_b+1, self.n_b+1, self.d_model).cuda()  # 协方差矩阵
        # 填充行为关系矩阵
        for i in range(self.n_b):
            for j in range(self.n_b):
                # 计算行为i到j的关系表示（加权组合）
                rel_idx = i * self.n_b + j + 1  # 关系索引
                # 均值部分
                bb_m[:, i+1, j+1, :] = torch.matmul(
                    Weight_user_behavior[:, i+1, j+1].unsqueeze(1),  # 权重
                    self.bb_embedding_m(torch.LongTensor([rel_idx]).cuda())  # 关系嵌入
                )
                # 协方差部分
                bb_c[:, i+1, j+1, :] = torch.matmul(
                    Weight_user_behavior[:, i+1, j+1].unsqueeze(1),  # 权重
                    self.bb_embedding_c(torch.LongTensor([rel_idx]).cuda())  # 关系嵌入
                )
        bb_c = self.activation(bb_c) + 1  # 应用激活函数并确保正值
        # 通过多个Transformer块处理序列
        W_probs = None  # 初始化注意力权重
        for transformer in self.transformer_blocks:
            # 前向传播通过Transformer块
            x_m, x_c, W_probs = transformer.forward(
                x_m, x_c,     # 物品表示
                b_seq,        # 行为序列
                b_m, b_c,     # 行为表示
                bb_m, bb_c,   # 行为关系表示
                p_m, p_c,     # 位置表示
                mask          # 序列掩码
            )
        ''' 模式感知的下一个物品预测 '''
        # 获取当前序列位置对应的行为表示
        user_behavior_m = P_user_behavior_m[torch.arange(batch_size)[:, None], b_seq[torch.arange(batch_size)], :]
        user_behavior_c = P_user_behavior_c[torch.arange(batch_size)[:, None], b_seq[torch.arange(batch_size)], :]
        # 应用SAGP操作融合物品表示和行为模式
        x_m, x_c = SAGP(
            x_m,  # 物品均值
            self.WPub(user_behavior_m),  # 转换后的行为均值
            x_c,   # 物品协方差
            user_behavior_c  # 行为协方差
        )
        # 返回最终表示和注意力权重
        return x_m, x_c, W_probs