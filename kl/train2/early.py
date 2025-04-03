from typing import Union, Optional


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
                 gap: Union[int, float] = 0):
        # 耐心数
        self.patience = patience
        # 最小优化幅度(比例)
        self.gap = gap

    def reset(self, reset_score: bool = False) -> None:
        '''重置'''
        self.counter = 0
        self.status = False

        if reset_score:
            # 重置最佳记录
            self.best_score = -10e5

    def update(self, score: Union[int, float]) -> bool:
        # 如果发现更好的分数
        if score > self.best_score*(1+self.gap):
            # 更新
            self.best_score = score
            # 重置计数器
            self.counter = 0
            # 结束
            return False

        # 没有发现更佳的分数
        self.counter += 1
        # 检查耐心
        self._judge()

        return self.status

    def _judge(self):
        '''判断是否达到耐心上限'''

        # 在更新一次最佳分数后，如果在接下来三次内(包括第三次)没有更加的分数
        if self.counter >= self.patience:
            # 触发早停
            self.status = True


if __name__ == '__main__':
    import time
    import random

    es = EarlyStop(patience=3, gap=0)
    for k in range(100):
        k2 = random.random()
        if es.update(k2):
            print('触发早停')
            break
        print(es.best_score, es.counter)
        time.sleep(0.1)
