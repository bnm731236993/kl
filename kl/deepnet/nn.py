import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, c_in, c_out, c_base=64, fc_hid=4096, input_height=224,
                 dropout_rate=0.1):
        super().__init__()
        # 输入通道数
        self.c_in = c_in
        # 输出特征维度（分类数）
        self.c_out = c_out
        # 卷积层的基准层数
        self.c_base = c_base
        # 全连接层的隐藏层
        self.fc_hid = fc_hid
        # 输入单个样本的高度
        self.input_height = input_height
        self.dropout_rate = dropout_rate

        # 创建网络
        self.build()

    @property
    def device(self):
        '''手动创建设备属性'''
        # 设备名取自某个权重
        return next(self.parameters()).device

    def build(self):
        # 1+1+2+2+2=8个卷积层
        self.convAll = nn.Sequential(
            self.vgg_block(self.c_in, self.c_base, num_conv=1),
            self.vgg_block(self.c_base, self.c_base*2, num_conv=1),
            self.vgg_block(self.c_base*2, self.c_base*4, num_conv=2),
            self.vgg_block(self.c_base*4, self.c_base*8, num_conv=2),
            self.vgg_block(self.c_base*8, self.c_base*8, num_conv=2),
        )

        # 卷积层的最终输出高度
        last_conv_output_height = self.input_height/(2**5)
        last_conv_output_height = int(last_conv_output_height)
        # 平铺后（保留批维度）的长度
        fc_in = (last_conv_output_height**2)*(self.c_base*8)
        fc_in = int(fc_in)

        # 三个全连接层
        self.threeLinear = nn.Sequential(
            # 平铺层,保留批维度
            nn.Flatten(start_dim=1),
            nn.Linear(fc_in, self.fc_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hid, self.fc_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hid, self.c_out)
        )

    def vgg_block(self, c_in, c_out, num_conv=1):
        '''
        创建VGG块
        参数:
            num_conv  卷积层的数量
        '''
        seq = nn.Sequential()
        for _ in range(num_conv):
            # 卷积
            _ = seq.append(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1))
            # 激活函数
            _ = seq.append(nn.ReLU())
            # 隐卷积层的输入通道数
            c_in = c_out
        # 最大池化
        _ = seq.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return seq

    def forward(self, X):
        Y = self.convAll(X)
        return self.threeLinear(Y)
