from typing import Literal

import torch.nn as nn


def conv2dBN(c_in: int,
             c_out: int,
             kernel_size: int = 3,
             stride: int = 1,
             padding: int = 0,
             groups: int = 1,
             bias: bool = False,
             use_batchNorm: bool = True,
             activation: Literal['relu', 'relu6', 'leakyRelu', 'none'] = 'relu') -> nn.Module:
    '''卷积+块标准化+激活函数'''

    # 卷积层
    layers = nn.Sequential(
        nn.Conv2d(c_in,
                  c_out,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=bias))

    # 批标准化
    if use_batchNorm:
        layers.append(nn.BatchNorm2d(c_out))

    # 激活函数
    if activation == 'relu':
        # 激活函数
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'relu6':
        layers.append(nn.ReLU6(inplace=True))
    elif activation == 'leakyRelu':
        layers.append(nn.LeakyReLU(0.1, inplace=True))

    return layers
