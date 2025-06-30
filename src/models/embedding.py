from torch import nn as nn
import torch

class LayerNorm(nn.Module):
    """层归一化模块"""
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 缩放参数
        self.bias = nn.Parameter(torch.zeros(hidden_size))   # 偏移参数
        self.variance_epsilon = eps  # 防止除零的小常数

    def forward(self, x):
        """前向传播"""
        u = x.mean(-1, keepdim=True)  # 计算最后一个维度的均值
        s = (x - u).pow(2).mean(-1, keepdim=True)  # 计算方差
        # 归一化处理
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        # 应用缩放和偏移
        return self.weight * x + self.bias

class PositionalEmbedding(nn.Module):
    """位置嵌入模块"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)  # 位置嵌入层
        self.apply(self._init_weights)  # 初始化权重

    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)  # 获取批次大小
        # 扩展位置嵌入以匹配批次大小
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def _init_weights(self, module):
        """权重初始化方法"""
        if isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果有填充索引，将其权重置零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SimpleEmbedding(nn.Module):
    """简单嵌入模块（包含归一化和激活）"""
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: 词汇表大小
        :param embed_size: 嵌入维度
        :param dropout: dropout概率
        """
        super().__init__()
        # 嵌入层（使用padding_idx=0处理填充）
        self.token = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)  # dropout层
        self.LayerNorm = LayerNorm(embed_size, eps=1e-12)  # 层归一化
        self.embed_size = embed_size  # 保存嵌入维度
        self.activation = nn.ELU()  # ELU激活函数
        self.apply(self._init_weights)  # 初始化权重

    def forward(self, sequence):
        """前向传播"""
        x = self.token(sequence)  # 获取嵌入表示
        x = self.LayerNorm(x)    # 应用层归一化
        x = self.dropout(x)       # 应用dropout
        x = self.activation(x)    # 应用激活函数
        return x
    
    def _init_weights(self, module):
        """权重初始化方法"""
        if isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果有填充索引，将其权重置零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()