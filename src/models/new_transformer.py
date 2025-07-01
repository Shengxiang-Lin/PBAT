from torch import nn as nn
import torch.nn.functional as F
import torch
import math
from .utils import BSFFL, TriSAGP ,SAGP
from .utils import wasserstein_distance, wasserstein_distance_matmul
from .embedding import LayerNorm

class MultiHeadedAttention(nn.Module):
    '''Fused Behavior-Aware Attention'''
    def __init__(self, h, n_b,  d_model, dropout=0.1):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 确保 d_model 能被 h 整除
        assert d_model % h == 0
        # 计算每个头的维度
        self.d_k = d_model // h
        # 多头注意力的头数
        self.h = h
        # 行为的数量
        self.n_b = n_b
        # 定义输入 x 的均值线性层参数
        self.linear_layers_xm = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        # 定义输入 x 的协方差线性层参数
        self.linear_layers_xc = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        # 定义行为 b 的均值线性层参数
        self.linear_layers_bm = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        # 定义行为 b 的协方差线性层参数
        self.linear_layers_bc = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        # 对输入 x 的均值线性层参数进行初始化，使用正态分布
        self.linear_layers_xm.data.normal_(mean=0.0, std=0.02)
        # 对输入 x 的协方差线性层参数进行初始化，使用正态分布
        self.linear_layers_xc.data.normal_(mean=0.0, std=0.02)
        # 对行为 b 的均值线性层参数进行初始化，使用正态分布
        self.linear_layers_bm.data.normal_(mean=0.0, std=0.02)
        # 对行为 b 的协方差线性层参数进行初始化，使用正态分布
        self.linear_layers_bc.data.normal_(mean=0.0, std=0.02)
        # 定义 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 定义激活函数 ELU
        self.activation = nn.ELU()
        # 定义均值的全连接层
        self.mean_dense = nn.Linear(d_model, d_model)
        # 定义协方差的全连接层
        self.cov_dense = nn.Linear(d_model, d_model)
        # 定义输出的 Dropout 层
        self.out_dropout = nn.Dropout(dropout)
        # 定义 LayerNorm 层，用于归一化
        self.LayerNorm = LayerNorm(d_model, eps=1e-12)
        # 定义查询的第一个线性层
        self.Wq1 = nn.Linear(self.d_k, self.d_k)
        # 定义查询的第二个线性层
        self.Wq2 = nn.Linear(self.d_k, self.d_k)
        # 定义键的第一个线性层
        self.Wk1 = nn.Linear(self.d_k, self.d_k)
        # 定义键的第二个线性层
        self.Wk2 = nn.Linear(self.d_k, self.d_k)

    def transpose_for_scores(self, x):
        # 计算新的形状，添加多头维度
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        # 调整张量的形状
        x = x.view(*new_x_shape)
        return x

    def forward(self, x_m, x_c , b_seq, b_m ,b_c, bb_m, bb_c, p_m, p_c, mask=None):
        # 获取批次大小和序列长度
        batch_size, seq_len = x_m.size(0), x_m.size(1)
        # 初始化查询、键和值，这里 x_m 同时作为查询、键和值
        queryx1, keyx1, valuex1 = x_m, x_m, x_m
        # 初始化查询、键和值，这里 x_c 同时作为查询、键和值
        queryx2, keyx2, valuex2 = x_c, x_c, x_c
        # 初始化查询、键和值，这里 b_m 同时作为查询、键和值
        queryb1, keyb1, valueb1 = b_m, b_m, b_m
        # 初始化查询、键和值，这里 b_c 同时作为查询、键和值
        queryb2, keyb2, valueb2 = b_c, b_c, b_c
        # 通过 einsum 函数和线性层参数计算行为特定的查询、键和值（x 的均值部分）
        queryx1, keyx1, valuex1= [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_xm[l])
                        for l, x in zip(range(3), (queryx1, keyx1, valuex1))]
        # 通过 einsum 函数和线性层参数计算行为特定的查询、键和值（x 的协方差部分）
        queryx2, keyx2, valuex2= [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_xc[l])
                        for l, x in zip(range(3), (queryx2, keyx2, valuex2))]
        # 通过 einsum 函数和线性层参数计算行为特定的查询、键和值（b 的均值部分）
        queryb1, keyb1, valueb1= [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_bm[l])
                        for l, x in zip(range(3), (queryb1, keyb1, valueb1))]
        # 通过 einsum 函数和线性层参数计算行为特定的查询、键和值（b 的协方差部分）
        queryb2, keyb2, valueb2= [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_bc[l])
                        for l, x in zip(range(3), (queryb2, keyb2, valueb2))]
        # 将 x 和 b 的均值查询相加
        query1 = queryx1 + queryb1
        # 将 x 和 b 的均值键相加
        key1 = keyx1 + keyb1
        # 将 x 和 b 的均值值相加
        value1 = valuex1 + valueb1
        # 对 x 和 b 的协方差查询应用激活函数并加 1
        query2 = self.activation(queryx2 + queryb2) + 1
        # 对 x 和 b 的协方差键应用激活函数并加 1
        key2 = self.activation(keyx2 + keyb2) + 1
        # 对 x 和 b 的协方差值应用激活函数并加 1
        value2 = self.activation(valuex2 + valueb2) + 1
        # 位置增强的行为感知融合：对 bb_m 进行维度转换和重排
        bb_m1 = self.transpose_for_scores(bb_m).permute(0, 3, 1, 2, 4).contiguous() # batch * h * b * b * k 
        # 位置增强的行为感知融合：对 bb_c 进行维度转换和重排
        bb_c1 = self.transpose_for_scores(bb_c).permute(0, 3, 1, 2, 4).contiguous()
        # 位置增强的行为感知融合：对 p_m 进行维度转换和重排
        p_m1 = self.transpose_for_scores(p_m).permute(0, 2, 1, 3).contiguous()   # batch * h * n * k
        # 位置增强的行为感知融合：对 p_c 进行维度转换和重排
        p_c1 = self.transpose_for_scores(p_c).permute(0, 2, 1, 3).contiguous()
        # 初始化批次化的 bb_m 张量
        bb_m1_batch = torch.zeros(batch_size,self.h,seq_len,seq_len,self.d_k).cuda()
        # 初始化批次化的 bb_c 张量
        bb_c1_batch = torch.zeros(batch_size,self.h,seq_len,seq_len,self.d_k).cuda()
        # 根据行为序列 b_seq 对 bb_m 进行索引和重排
        bb_m1_batch = bb_m1[torch.arange(batch_size)[:,None,None],:,b_seq[torch.arange(batch_size)][:,None],b_seq[torch.arange(batch_size)].unsqueeze(2),:].permute(0, 3, 2, 1, 4).contiguous()
        # 根据行为序列 b_seq 对 bb_c 进行索引和重排
        bb_c1_batch = bb_c1[torch.arange(batch_size)[:,None,None],:,b_seq[torch.arange(batch_size)][:,None],b_seq[torch.arange(batch_size)].unsqueeze(2),:].permute(0, 3, 2, 1, 4).contiguous()
        # 将查询扩展维度，并通过 TriSAGP 函数进行融合，得到融合后的查询均值和协方差
        # query B * H * N * K -> B * H * N * j * K    (j:position)
        # fusionQ = B * H * N * N * K
        fusion_Q_m ,fusion_Q_c = TriSAGP(query1.unsqueeze(3), 
                                        self.Wq1(bb_m1_batch),
                                        self.Wq2(p_m1).unsqueeze(3), 
                                        query2.unsqueeze(3), 
                                        bb_c1_batch,
                                        p_c1.unsqueeze(3) )
        # 将键扩展维度，并通过 TriSAGP 函数进行融合，得到融合后的键均值和协方差
        # key B * H * N * K -> B * H * i * N * K   (i:position)
        # fusionK = B * H * N * N * K   
        fusion_K_m ,fusion_K_c = TriSAGP(key1.unsqueeze(2), 
                                        self.Wk1(bb_m1_batch),
                                        self.Wk2(p_m1).unsqueeze(2), 
                                        key2.unsqueeze(2), 
                                        bb_c1_batch,
                                        p_c1.unsqueeze(2) )
        # 计算 Wasserstein 距离注意力分数
        Wass_scores = -wasserstein_distance_matmul( fusion_Q_m.unsqueeze(4), fusion_Q_c.unsqueeze(4),
                                                    fusion_K_m.unsqueeze(4), fusion_K_c.unsqueeze(4)).squeeze()
        # 进行注意力聚合，对分数进行缩放
        Wass_scores = Wass_scores / math.sqrt(self.d_k)
        # 处理填充和 softmax 操作
        if mask is not None:
            # 确保掩码的维度为 2
            assert len(mask.shape) == 2
            # 扩展掩码的维度
            mask = (mask[:,:,None] & mask[:,None,:]).unsqueeze(1)
            # 如果分数的数据类型为 float16，使用 -65500 填充掩码位置
            if Wass_scores.dtype == torch.float16:
                Wass_scores = Wass_scores.masked_fill(mask == 0, -65500)
            # 否则使用 -1e30 填充掩码位置
            else:
                Wass_scores = Wass_scores.masked_fill(mask == 0, -1e30)
        # 对分数应用 softmax 函数并进行 Dropout 操作
        Wass_probs = self.dropout(nn.functional.softmax(Wass_scores, dim=-1))
        # 计算均值上下文层
        mean_context_layer = torch.matmul(Wass_probs, value1)
        # 计算协方差上下文层
        cov_context_layer = torch.matmul(Wass_probs ** 2, value2)
        # 对均值上下文层进行维度重排
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        # 对协方差上下文层进行维度重排
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()
        # 计算新的上下文层形状
        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.h * self.d_k,)
        # 调整均值上下文层的形状
        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        # 调整协方差上下文层的形状
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)
        # 通过全连接层处理均值上下文层
        mean_hidden_states = self.mean_dense(mean_context_layer)
        # 对均值隐藏状态应用 Dropout 操作
        mean_hidden_states = self.out_dropout(mean_hidden_states)
        # 对均值隐藏状态进行 LayerNorm 归一化
        mean_hidden_states = self.LayerNorm(mean_hidden_states + x_m)
        # 通过全连接层处理协方差上下文层
        cov_hidden_states = self.cov_dense(cov_context_layer)
        # 对协方差隐藏状态应用 Dropout 操作
        cov_hidden_states = self.out_dropout(cov_hidden_states)
        # 对协方差隐藏状态进行 LayerNorm 归一化
        cov_hidden_states = self.LayerNorm(cov_hidden_states + x_c)
        # 返回均值隐藏状态、协方差隐藏状态和注意力概率
        return mean_hidden_states, cov_hidden_states, Wass_probs

class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, n_b, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        :param n_b: number of behaviors
        """
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 初始化多头注意力层
        self.attention = MultiHeadedAttention(h=attn_heads, n_b=n_b, d_model=hidden, dropout=dropout)
        # 初始化行为特定的前馈网络层
        self.feed_forward = BSFFL(d_model=hidden, d_ff=feed_forward_hidden, n_b=n_b, dropout=dropout)
        # 定义 Dropout 层
        self.dropout = nn.Dropout(p=dropout)
        # 定义激活函数 ELU
        self.activation_func = nn.ELU()

    def forward(self, x_m, x_c, b_seq , b_m, b_c, bb_m, bb_c, p_m, p_c, mask):
        # 通过多头注意力层进行计算
        x_m, x_c, W_probs = self.attention(x_m, x_c , b_seq, b_m ,b_c, bb_m, bb_c, p_m, p_c,mask=mask)
        # 通过前馈网络层处理均值输入
        x_m = self.feed_forward(x_m,b_seq)
        # 通过前馈网络层处理协方差输入，并应用激活函数加 1
        x_c = self.activation_func(self.feed_forward(x_c,b_seq)) + 1
        # 返回处理后的均值、协方差和注意力概率
        return x_m, x_c, W_probs