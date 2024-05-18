from typing import Union, Optional

from .util import number2string


class EarlyStop:
    '''早停控制器'''
    # 最佳分数
    best_score = -10e5
    # 计数器
    counter = 0
    # 是否触发早停
    status = False

    def __init__(self,
                 patience: int = 3,
                 gap: Union[int, float] = 0,
                 verbose: int = 1,
                 *args,
                 **kwargs):
        # 耐心
        self.patience = patience
        # 最小优化幅度(比例)
        self.gap = gap
        self.verbose = verbose

    def reset(self, reset_score: bool = False) -> None:
        '''重置'''
        self.counter = 0
        self.status = False

        if reset_score:
            # 重置最佳记录
            self.best_score = -10e5

    def update(self, score: Union[int, float]) -> bool:
        if score > self.best_score*(1+self.gap):
            # 发现更加的分数
            # 更新
            self.best_score = score
            # 重置计数器
            self.counter = 0
            if self.verbose >= 2:
                print(
                    'Better score found.'.format(
                        number2string(self.best_score)),
                    end='\n')
            # 结束
            return

        # 没有发现更佳的分数
        self.counter += 1
        if self.verbose >= 2:
            print(
                'No update, counter is {}.'.format(
                    number2string(self.counter)),
                end='\n')

        # 检查耐心
        self._judge()

    def _judge(self):
        '''判断是否达到耐心上限'''
        if self.counter >= self.patience:
            # 在更新一次最佳分数后
            # 如果在接下来三次内(包括第三次)没有更加的分数
            if self.verbose >= 1:
                print(
                    f'No better score found in {self.patience} patience, tigger early stop.')
            # 触发早停
            self.status = True
