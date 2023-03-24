import numpy as np


def precision_at_k(r, k):
    """
    计算查准率P@k
    :param r: 单个查询返回的排序相关列表 [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    :param k: 计算前几个查准率
    :return: P@k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        r = 0.0
        # raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """
    计算AP
    :param r: 单个查询返回的排序相关列表 [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    :return: AP
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """
    计算mAP
    :param rs: 各个查询排序后的相关列表[[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]或是[[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    :return:返回mAP
    """
    return np.mean([average_precision(r) for r in rs])


def nmrr(r, ng, k):
    """
    计算归一化修正后的检索秩
    :param r:代表单个查询是否相关的列表[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    :param ng:在图像库中与查询qi实际相关的个数
    :param k: k=min(4*ng,2M) m=max{Ng(q1),Ng(q2)，....,Ng(qn)}
    :return:归一化修正后的检索秩
    """
    sum = 0
    rank = [index + 1 for index, value in enumerate(r) if value]
    # print(rank)
    for i in range(0,ng):
        if i < len(rank):
            if rank[i] <= k:
                sum = sum +  rank[i]
            else:
                sum = sum + (k + 1)
        else:
            sum = sum + (k + 1)
    avr = sum/ng
    mrr = avr - 0.5 * (1 + ng)
    nmrr = mrr / (k + 0.5 - 0.5 * ng)
    return avr, mrr, nmrr

def anmrr(nmrr):
    """
    计算平均归一化修正后的检索秩
    :param nmrr: 包含nq个查询q的nmrr列表
    :return: anmrr
    """
    anmrr =  np.mean(nmrr)

    return  anmrr


if __name__ == "__main__":
    # [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    # r = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
    r = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    # avr_at_k(r,10)
    avr, mrr, nmrr = nmrr(r, 10, 2 * 10)
    print(avr)
    print(mrr)
    print(nmrr)