import torch

def recall(scores, labels, k):
    """计算召回率@k"""
    scores = scores.cpu()  # 将分数张量移动到CPU
    labels = labels.cpu()  # 将标签张量移动到CPU
    rank = (-scores).argsort(dim=1)  # 对分数降序排序，获取排名索引
    cut = rank[:, :k]  # 获取前k个预测
    hit = labels.gather(1, cut)  # 收集这些位置的真实标签
    # 计算召回率：命中数/总相关项数，然后取批次平均
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()

def ndcg(scores, labels, k):
    """计算NDCG@k"""
    scores = scores.cpu()  # 将分数张量移动到CPU
    labels = labels.cpu()  # 将标签张量移动到CPU
    rank = (-scores).argsort(dim=1)  # 对分数降序排序，获取排名索引
    cut = rank[:, :k]  # 获取前k个预测
    hits = labels.gather(1, cut)  # 收集这些位置的真实标签
    position = torch.arange(2, 2+k)  # 创建位置向量 [2,3,...,k+1]
    weights = 1 / torch.log2(position.float())  # 计算折扣权重 (1/log2(位置))
    dcg = (hits.float() * weights).sum(1)  # 计算折现累计收益(DCG)
    # 计算理想DCG(IDCG)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg  # 计算NDCG
    return ndcg.mean()  # 返回批次平均NDCG

def hr_at_k(scores, labels, k):
    """计算HR@k"""
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut).sum(1)
    hr = (hits > 0).float().mean().item()
    return hr

def recalls_and_ndcgs_for_ks(scores, labels, ks):
    """计算多个k值的召回率和NDCG指标"""
    metrics = {}  # 存储结果的字典
    scores = scores.cpu()  # 将分数张量移动到CPU
    labels = labels.cpu()  # 将标签张量移动到CPU
    answer_count = labels.sum(1)  # 每个样本的相关项总数
    answer_count_float = answer_count.float()  # 转换为浮点数
    labels_float = labels.float()  # 标签转换为浮点数
    rank = (-scores).argsort(dim=1)  # 对分数降序排序，获取排名索引
    cut = rank  # 初始化cut为完整排名
    # 按k值从大到小遍历（优化计算效率）
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]  # 截取前k个预测
        hits = labels_float.gather(1, cut)  # 收集这些位置的真实标签
        # 计算召回率@k
        metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()
        # 计算NDCG@k
        position = torch.arange(2, 2+k)  # 位置向量
        weights = 1 / torch.log2(position.float())  # 折扣权重
        dcg = (hits * weights).sum(1)  # 折现累计收益
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])  # 理想DCG
        ndcg = (dcg / idcg).mean()  # 计算NDCG并取平均
        metrics['NDCG@%d' % k] = ndcg
        metrics['HR@%d' % k] = hr_at_k(scores, labels, k)

    return metrics  # 返回指标字典

def split_at_index(dim, index, t):
    """在指定维度dim的索引index处分割张量"""
    pre_slices = (slice(None),) * dim  # 创建前置切片元组
    l = (*pre_slices, slice(None, index))  # 左半部分切片
    r = (*pre_slices, slice(index, None))  # 右半部分切片
    return t[l], t[r]  # 返回分割后的两个张量