import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


def sampleForRes(c_in, c_out, stride=1):
    '''
    专用于残差路径的采样块
    用于调整输入X的通道数
    '''
    return nn.Sequential(
        # 1*1卷积
        nn.Conv2d(c_in, c_out,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(c_out)
    )


class DoubleConvRes(nn.Module):
    '''基本的残差块'''

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out,
                               kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(c_out)

        self.conv2 = nn.Conv2d(c_out, c_out,
                               kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(c_out)

        self.use_downsample = False
        if not downsample is None:
            # 下采样通道的层
            self.downsample = downsample
            self.use_downsample = True

    def forward(self, X):
        Y = self.bn1(self.conv1(X))
        Y = F.relu(Y)
        Y = self.bn2(self.conv2(Y))

        # 残差连接
        if self.use_downsample:
            identity = self.downsample(X)
        else:
            identity = X

        return F.relu(Y+identity)


class Bottleneck(nn.Module):
    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super().__init__()
        # 取c_hid的四倍
        c_hid = int(c_out/4)

        self.conv1 = nn.Conv2d(
            c_in, c_hid,
            kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_hid)

        self.conv2 = nn.Conv2d(
            c_hid, c_hid,
            kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(c_hid)

        self.conv3 = nn.Conv2d(
            c_hid, c_out,
            kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)

        self.use_downsample = False
        if not downsample is None:
            # 下采样通道的层
            self.downsample = downsample
            self.use_downsample = True

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        # 残差连接
        if self.use_downsample:
            identity = self.downsample(X)
        else:
            identity = X
        return F.relu(Y+identity)


class ResNet(nn.Module):
    '''残差网络'''

    def __init__(self, c_in, c_out,
                 c_base=64,
                 c_linear=512,
                 num_layers=[2, 2, 2, 2]):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c_base = c_base
        # 四个res块的深度
        self.num_layers = num_layers
        self.c_linear = c_linear

        # 生成网络
        self.build()

    def build(self):
        # 最开始的输入层
        self.inputBlock = nn.Sequential(
            nn.Conv2d(
                self.c_in, self.c_base,
                kernel_size=7, stride=2,
                padding=3, bias=False),
            nn.BatchNorm2d(self.c_base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.mainBlock = nn.Sequential(
            self._block(self.c_base, self.c_base,
                        stride_first=1, num_layer=self.num_layers[0]),
            self._block(self.c_base, self.c_base*2,
                        stride_first=2, num_layer=self.num_layers[1]),
            self._block(self.c_base*2, self.c_base*4,
                        stride_first=2, num_layer=self.num_layers[2]),
            self._block(self.c_base*4, self.c_base*8,
                        stride_first=2, num_layer=self.num_layers[3]),
        )

        # 用于分类的输出块
        self.outputBlock = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(self.c_base*8, self.c_out)
        )

    def _block(self, c_in, c, stride_first=1, num_layer=1):
        '''
        ResNet的主要组成块
        参数:
            c_in  第一个卷积层的通道数
            stride  第一个卷积层的步长
        '''
        layers = nn.Sequential()
        conv_first = models.resnet.BasicBlock(
            inplanes=c_in, planes=c, stride=stride_first,
            # 调整通道数
            downsample=sampleForRes(
                c_in, c, stride_first) if c_in != c else None
        )
        layers.append(conv_first)

        for _ in range(1, num_layer):
            # 从第二个块开始，输入输出通道数不再改变
            layers.append(models.resnet.BasicBlock(
                inplanes=c, planes=c, stride=1,
                downsample=None))
        return layers

    def forward(self, X):
        Y = self.inputBlock(X)
        Y = self.mainBlock(Y)
        return self.outputBlock(Y)


class ResNetWithBottleNeck(nn.Module):
    '''残差块'''

    def __init__(self, c_in, c_out,
                 c_base=64,
                 num_layers=[3, 4, 6, 3],
                 c_linear=512):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c_base = c_base
        # 四个res块的深度
        self.num_layers = num_layers
        # 全连接层的特征数
        self.c_linear = c_linear

        # 生成网络
        self.build()

    def build(self):
        # 最开始的输入层
        self.inputBlock = nn.Sequential(
            nn.Conv2d(
                self.c_in, self.c_base,
                kernel_size=7, stride=2,
                padding=3, bias=False),
            nn.BatchNorm2d(self.c_base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.mainBlock = nn.Sequential(
            self._block(self.c_base, self.c_base*4,
                        stride_first=1, num_layer=self.num_layers[0]),
            self._block(self.c_base*4, self.c_base*8,
                        stride_first=2, num_layer=self.num_layers[1]),
            self._block(self.c_base*8, self.c_base*16,
                        stride_first=2, num_layer=self.num_layers[2]),
            self._block(self.c_base*16, self.c_base*32,
                        stride_first=2, num_layer=self.num_layers[3]),
        )

        # 用于分类的输出块
        self.outputBlock = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(self.c_linear, self.c_out)
        )

    def _block(self, c_in, c, stride_first=1, num_layer=1):
        '''
        ResNet的主要组成块
        参数:
            c_in  第一个卷积层的通道数
            stride  第一个卷积层的步长
        '''
        layers = nn.Sequential()

        layers.append(models.resnet.Bottleneck(
            inplanes=c_in, planes=int(c/4), stride=stride_first,
            # 调整通道数
            downsample=sampleForRes(
                c_in, c, stride_first) if c_in != c else None
        ))

        for _ in range(1, num_layer):
            # 从第二个块开始，输入输出通道数不再改变
            layers.append(models.resnet.Bottleneck(
                inplanes=c, planes=int(c/4), stride=1,
                downsample=None))
        return layers

    def forward(self, X):
        Y = self.inputBlock(X)
        Y = self.mainBlock(Y)
        return self.outputBlock(Y)
