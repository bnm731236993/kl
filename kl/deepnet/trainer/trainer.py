import os
from pathlib import Path

from typing import Union, Optional

import json
import time

import torch as tc

from .early_stop import EarlyStop
from .checkpoint_manager import CheckpointManager
from .summary_manager import SummaryManager


class Trainer:
    '''训练器'''

    def __init__(self,
                 net,
                 loss_fn=None,
                 optimizer=None,
                 metric=None,
                 data_train_loader=None,
                 data_test_loader=None,
                 num_epoch: int = 1,
                 verbose: int = 1,
                 roor_dir: Union[str, Path] = './trainer_output/',
                 createFolderByDate: bool = True,
                 enable_checkpointManager: bool = True,
                 checkpoint_limit: int = 3,
                 device: str = 'cpu',
                 enable_tensorboard: bool = False,
                 earlyStop_config: Optional[dict] = None):
        # 网络
        self.net = net
        # 损失函数
        self.loss_fn = loss_fn
        # 优化器
        self.optimizer = optimizer
        # 精度计算器
        self.metric = metric

        # 批数据迭代器
        self.data_train_loader = data_train_loader
        self.data_test_loader = data_test_loader

        # 所使用的设备
        # 模型应该提前移动到指定位置
        # 该属性只用于数据转移
        self.device = device if isinstance(device, tc.device) \
            else tc.device(device)

        # 当前轮次
        self.epoch = 1
        # 总轮次
        self.num_epoch = num_epoch
        # 批大小
        self.batch_size = self.data_train_loader.batch_size
        # 每轮的批数量
        self.num_batch_train = len(self.data_train_loader)
        self.num_batch_test = len(self.data_test_loader)

        # 根目录
        self.roor_dir = Path(roor_dir)
        # 是否在根目录下创建子文件夹
        # 由当前时间命名
        if createFolderByDate:
            current_time = time.strftime('%Y-%m-%d+%H-%M-%S', time.localtime())
            self.roor_dir = self.roor_dir/Path(current_time)
        # 创建文件夹
        self.roor_dir.mkdir(parents=True, exist_ok=True)

        # 打印信息的等级
        # 0  不记录任何信息
        # 1  只记录批信息
        # 2  记录轮信息、批信息
        self.verbose = verbose

        # ==============================================
        # 信息管理器
        self.summaryManager = SummaryManager(
            root_dir=self.roor_dir/Path('summary'),
            enable_tensorboard=enable_tensorboard)

        # ==============================================
        # 权重管理器
        self.enable_checkpointManager = enable_checkpointManager
        if self.enable_checkpointManager:
            # 权重管理器
            self.checkpointManager = CheckpointManager(
                modObj={
                    'net': self.net,
                    'loss_fn': self.loss_fn,
                    'optimizer': self.optimizer
                },
                max_count=checkpoint_limit,
                root_dir=self.roor_dir/Path('checkpoint'))

        # ==============================================
        # 早停功能
        self.enable_earlyStop = False
        if isinstance(earlyStop_config, dict):
            self.enable_earlyStop = True
            # 早停对象
            self.earlyStop = EarlyStop(
                verbose=self.verbose,
                **earlyStop_config)
            # 指定早停要监视的对象
            self.earlyStop_aim = earlyStop_config.get('object', 'loss')

    def predict(self, data: tc.tensor):
        '''
        主要的计算函数
        应该重新定义！！！！
        '''
        # 设备
        X = data[0].to(self.device)
        Y_true = data[1].to(self.device)

        # 预测
        Y = self.net(X)
        return Y, Y_true

    def _fit_batch(self, is_training: bool = True):
        if is_training:
            # 训练模式
            self.net.train()
            # 数据加载器
            data_loder = self.data_train_loader
        else:
            # 测试模式
            self.net.eval()
            data_loder = self.data_test_loader

        # 精度计算器重置
        _ = self.metric.reset()

        # 遍历数据集
        for batch, data in enumerate(data_loder):
            self.batch = batch+1
            # 计时
            time_batch_begin = time.time()

            if is_training:
                # 梯度归零
                self.optimizer.zero_grad()
                # 预测
                Y, Y_true = self.predict(data)
                # 损失
                loss = self.loss_fn(Y, Y_true)
                # 计算梯度
                _ = loss.backward()
                # 更新权重
                self.optimizer.step()
            else:
                with tc.no_grad():
                    # 预测
                    Y, Y_true = self.predict(data)

            with tc.no_grad():
                # 更新精度
                _ = self.metric.update(Y, Y_true)
                acc = self.metric.compute()

            # 计时
            time_batch_delta = time.time() - time_batch_begin

            # 打印
            self._print(
                text='\r{} Epoch:{}-Batch:{}/{}, {}Accuracy:{:.4f}, Time:{:.6f}'.format(
                    'Train' if is_training else 'Test',
                    self.epoch,
                    self.batch,
                    self.num_batch_train if is_training else self.num_batch_test,
                    'Loss:{:.4f}, '.format(loss.item()) if is_training else '',
                    acc.item(),
                    time_batch_delta,
                ),
                end='', verbose=2)
        # 补充换行符
        self._print('', end='\n', verbose=2)

        # 本轮训练结果报表
        epoch_info = {
            'epoch': self.epoch,
            'mode': 'train' if is_training else 'test',
            'loss': loss.item() if is_training else None,
            'accuracy': acc.item()
        }

        # 记录轮信息
        self.summaryManager.append(
            input=epoch_info)

        if self.enable_earlyStop and not is_training:
            # 更新早停记录器
            self.earlyStop.update(epoch_info[self.earlyStop_aim])

    def _print(self, text, verbose=0, *args, **kwargs):
        '''
        根据verbose等级打印信息
        参数:
            verbose  允许打印的最低等级
                用self.verbose与之比较
        '''
        if self.verbose >= verbose:
            print(text, *args, **kwargs)

    def save(self):
        # 保存权重
        if self.enable_checkpointManager:
            # 自动调用权重保存
            self.checkpointManager.save()
            # 保存记录
            self.summaryManager.save()

    def fit(self):
        # 遍历批
        for epoch in range(self.epoch, self.num_epoch+1):
            # 开始训练
            self.epoch = epoch
            self._print(
                text=f'Train Epoch:{self.epoch}/{self.num_epoch} Begin',
                verbose=1)
            # 批训练
            self._fit_batch(is_training=True)

            # =======================================================
            # 开始测试
            self._print(
                text=f'Test Epoch:{self.epoch} Begin',
                verbose=1)
            # 批测试
            self._fit_batch(is_training=False)

            # =======================================================
            # 保存权重
            self.save()

            # =======================================================
            # 检查早停
            if self.enable_earlyStop and self.earlyStop.status:
                # 触发早停
                self._print('==========Early Stop==========',
                            verbose=1)
                break

        self._print('\n========Finish========\n',
                    verbose=1)
