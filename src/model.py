import torch
import pytorch_lightning as pl
from .models import WassersteinPredictionHead  # 导入Wasserstein预测头
from .models.bert4rec import BERT  # 导入BERT模型
from .utils import recalls_and_ndcgs_for_ks  # 导入评估指标计算工具

class RecModel(pl.LightningModule):
    def __init__(self,
            backbone: BERT,  # 主干网络，BERT模型实例
        ):
        super().__init__()
        self.backbone = backbone  # 主干网络
        self.n_b = backbone.n_b  # 行为类型数量（从主干网络获取）
        self.max_len = backbone.max_len  # 序列最大长度（从主干网络获取）
        # 初始化Wasserstein预测头
        self.head = WassersteinPredictionHead(
            backbone.d_model,  # 特征维度
            backbone.num_items,  # 物品总数
            self.backbone.item_embedding_m.token,  # 物品embedding矩阵（均值）
            self.backbone.item_embedding_c.token  # 物品embedding矩阵（协方差）
        )
        # 交叉熵损失函数，忽略索引0（填充值）
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self, input_ids, b_seq, u_id):
        """前向传播：通过主干网络"""
        return self.backbone(input_ids, b_seq, u_id)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        input_ids = batch['input_ids']  # 获取输入序列
        b_seq = batch['behaviors']  # 获取行为序列
        user_id = batch['user_id']  # 获取用户ID
        # 通过主干网络获取输出（均值、协方差和注意力权重）
        outputsm, outputsc, W_pro = self(input_ids, b_seq, user_id)
        # 重塑输出：合并批次和时间维度 (B*T x H)
        outputsm = outputsm.view(-1, outputsm.size(-1))
        outputsc = outputsc.view(-1, outputsc.size(-1))
        labels = batch['labels']  # 获取标签
        labels = labels.view(-1)  # 展平标签 (B*T)
        # 创建有效标签掩码（忽略填充值0）
        valid = labels > 0
        valid_index = valid.nonzero().squeeze()  # 获取有效索引
        # 提取有效位置的输出和标签
        valid_outputsm = outputsm[valid_index]
        valid_outputsc = outputsc[valid_index]
        valid_b_seq = b_seq.view(-1)[valid_index]  # 有效行为序列
        valid_labels = labels[valid_index]
        # 通过预测头计算有效位置的logits
        valid_logits = self.head(valid_outputsm, valid_outputsc, valid_b_seq)
        # 计算损失
        loss = self.loss(valid_logits, valid_labels)
        loss = loss.unsqueeze(0)  # 确保损失是张量
        return {'loss': loss}  # 返回损失字典

        
    def training_epoch_end(self, training_step_outputs):
        """训练周期结束：汇总平均损失并记录"""
        # 聚合所有批次的损失
        loss = torch.cat([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)  # 记录到日志

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        input_ids = batch['input_ids']  # 获取输入序列
        b_seq = batch['behaviors']  # 获取行为序列
        user_id = batch['user_id']  # 获取用户ID
        # 通过主干网络获取输出
        outputsm, outputsc, W_pro = self(input_ids, b_seq, user_id)
        # 提取序列最后一个位置的特征（用于预测）
        last_outputsm = outputsm[:, -1, :]  # 形状 (B x H)
        last_outputsc = outputsc[:, -1, :]   # 形状 (B x H)
        last_b_seq = b_seq[:, -1]  # 最后位置的行为类型
        # 获取候选物品集 (B x C)
        candidates = batch['candidates'].squeeze() 
        # 计算候选物品的logits
        logits = self.head(last_outputsm, last_outputsc, last_b_seq, candidates)
        labels = batch['labels'].squeeze()  # 获取真实标签
        # 计算多个K值下的召回率和NDCG
        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])
        return metrics  # 返回指标字典
    
    def validation_epoch_end(self, validation_step_outputs):
        """验证周期结束：汇总平均指标并记录"""
        keys = validation_step_outputs[0].keys()  # 获取指标名称
        print("Validation Metrics:")
        for k in keys:
            # 聚合所有批次的该指标
            tmp = [o[k] for o in validation_step_outputs]
            print(f"{k}:   {torch.Tensor(tmp).mean()}")
            # 计算指标均值并记录到日志
            self.log(f'Val:{k}', torch.Tensor(tmp).mean())