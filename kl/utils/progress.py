from .system import is_ipynb

if is_ipynb():
    # 如果是Jupyter环境
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ProgressBar:
    """ 进度条类 """

    def __init__(self,
                 total: int,
                 title: str):
        # 初始化进度条
        self.bar = tqdm(initial=0,
                        total=total,
                        desc=title)

    def update(self, n: int):
        _ = self.bar.update(n)

    def __del__(self):
        self.bar.close()


if __name__ == '__main__':
    import time

    bar1 = ProgressBar(10, '测试A')
    bar2 = ProgressBar(50, '测试B')
    for k in range(10):
        # 更新进度条
        _ = bar1.update(1)
        _ = bar2.update(2)

        time.sleep(0.1)
