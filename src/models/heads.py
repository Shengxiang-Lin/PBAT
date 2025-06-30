import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import wasserstein_distance, wasserstein_distance_matmul  # 导入Wasserstein距离计算函数

# 用于BERT4Rec的Wasserstein预测头
class WassersteinPredictionHead(nn.Module):
    def __init__(self, d_model, num_items, token_embeddings_m, token_embeddings_c):
        """
        :param d_model: 模型维度
        :param num_items: 物品总数
        :param token_embeddings_m: 物品均值嵌入层
        :param token_embeddings_c: 物品协方差嵌入层
        """
        super().__init__()
        self.token_embeddings_m = token_embeddings_m  # 物品均值嵌入
        self.token_embeddings_c = token_embeddings_c  # 物品协方差嵌入
        self.vocab_size = num_items + 1  # 词汇表大小（物品数+1）
        # 输出转换层
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),  # 线性层
            nn.ELU(),                    # ELU激活函数
        )
        self.activation = nn.ELU()        # 额外的ELU激活函数
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))  # 偏置项

    def forward(self, x_m, x_c, b_seq, candidates=None):
        """
        :param x_m: 输入的均值表示 (B x H 或 M x H)
        :param x_c: 输入的协方差表示 (B x H 或 M x H)
        :param b_seq: 行为序列（本函数中未使用，但保留接口）
        :param candidates: 候选物品集，如果为None则计算所有物品
        """
        # 通过输出层转换输入
        x_m = self.out(x_m)  # 转换均值表示
        x_c = self.out(x_c)  # 转换协方差表示
        if candidates is not None:  # 如果提供了候选物品集
            # 获取候选物品的嵌入表示
            emb1 = self.token_embeddings_m(candidates)  # 候选物品均值嵌入 (B x C x H)
            # 处理协方差嵌入并确保正值
            emb2 = self.activation(self.token_embeddings_c(candidates)) + 1  # (B x C x H)
            # 计算输入与候选物品之间的Wasserstein距离
            # x_m.unsqueeze(1) 形状变为 B x 1 x H
            # x_c.unsqueeze(1) 形状变为 B x 1 x H
            # emb1 形状为 B x C x H
            # emb2 形状为 B x C x H
            logits = wasserstein_distance_matmul(
                x_m.unsqueeze(1), 
                x_c.unsqueeze(1), 
                emb1, 
                emb2
            ).squeeze()  # 结果形状 B x C，然后压缩为 B x C
        else:  # 如果没有提供候选物品集，计算所有物品
            # 获取所有物品的嵌入表示
            emb1 = self.token_embeddings_m.weight[:self.vocab_size]  # 所有物品均值嵌入 (V x H)
            # 处理协方差嵌入并确保正值
            emb2 = self.activation(self.token_embeddings_c.weight[:self.vocab_size]) + 1  # (V x H)
            # 计算输入与所有物品之间的Wasserstein距离
            # x_m.unsqueeze(1) 形状变为 M x 1 x H
            # x_c.unsqueeze(1) 形状变为 M x 1 x H
            # emb1 形状为 V x H -> 1 x V x H
            # emb2 形状为 V x H -> 1 x V x H
            logits = wasserstein_distance_matmul(
                x_m.unsqueeze(1), 
                x_c.unsqueeze(1), 
                emb1, 
                emb2
            ).squeeze()  # 结果形状 M x V
        return logits  # 返回预测分数