import torch as tc
import torch.nn as nn


def dot_attention(Q, K, V, return_attention=False):
    '''点积注意力'''
    # 词向量长度
    d = tc.tensor(K.shape[-1])
    # 在(b,s,w)的(s,w)部分转置
    K_T = K.transpose(-1, -2).contiguous()
    # 注意力分布
    A_ = Q@K_T/tc.sqrt(d)
    # 归一化
    A = tc.softmax(A_, dim=-1)
    # 查询结果
    Y = A@V

    if return_attention:
        # 顺便返回注意力分布
        return Y, A
    else:
        return Y


def init_normal(net, seed=0):
    '''根据正态分布，初始化模型的所有权重'''
    for param in net.parameters():
        _ = tc.random.manual_seed(seed)
        _ = nn.init.normal_(param.data, mean=0, std=1)
