import time
import threading

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from .app import is_ipynb


class MyThread:
    """多线程执行工具"""

    # 线程锁
    lock = threading.Lock()

    # 线程列表
    threads = []

    # 成功列表
    succeeds = []
    faileds = []

    # 收集结果
    results = []

    # 计数
    num_succeed = 0
    num_failed = 0

    # 每次执行后暂停
    use_time_wait = False

    def __init__(self, func, args, use_bar=False, time_wait=0, pool_sema=8):
        """初始化"""

        # 目标函数
        self.func = func

        # 参数列表
        self.args = args
        # 总长度
        self.num_all = len(args)

        # 是否使用进度条
        self.use_bar = use_bar

        # 等待时间
        self.time_wait = time_wait
        if self.time_wait > 0:
            self.use_time_wait = True

        # 限制最大线程数量
        self.pool_sema = threading.BoundedSemaphore(pool_sema)

    def run(self):
        """开始执行"""

        if self.use_bar:
            # 显示进度条
            # 进度条一创建就会立即显示，应该放在run方法内
            self.__init_bar()

        # 添加线程
        for arg in self.args:
            thread = threading.Thread(target=self.__process, args=(arg,))
            # 线程开始执行
            # 必须在创建下一个Thread对象之前start
            # 否则“Thread对象不会随着类销毁而消失，影响下一次类创建”
            thread.start()

            self.threads.append(thread)

        for thread in self.threads:
            # 把所有线程加入等待序列
            # 应该把一批线程同时加入
            # 不能start后立马join，将会导致该线程处理完才创建新线程！！！
            thread.join()

        if self.use_bar:
            # 进度条最后再加1，否则最后缺少一位
            self.update_bar()

    def __process(self, arg):
        # 限制线程数
        self.pool_sema.acquire()

        # 执行成功标志
        succeed = True

        try:
            # 执行函数
            result = self.func(arg)

            if self.use_time_wait:
                # 等待时间
                time.sleep(self.time_wait)
        except:
            succeed = False

        # 锁定变量
        self.lock.acquire()
        if succeed:
            # 如果执行成功
            self.num_succeed += 1
            self.succeeds.append(arg)
            if not result is None:
                self.results.append({"arg": arg, "result": result})
        else:
            self.num_failed += 1
            self.faileds.append(arg)

        if self.use_bar:
            # 更新进度条
            self.update_bar()
        else:
            # 打印相关信息
            print('Succeed: %d\tFailed: %d' %
                  (self.num_succeed, self.num_failed))

        # 释放变量
        self.lock.release()

        # 释放线程
        self.pool_sema.release()

    def __init_bar(self):
        """创建进度条"""

        # 区分ipynb和py环境
        if is_ipynb():
            self.bar = tqdm_notebook(total=self.num_all, initial=0)
        else:
            self.bar = tqdm(total=self.num_all)

    def update_bar(self):
        """更新进度条"""

        self.bar.set_postfix(
            ordered_dict={"成功": self.num_succeed, "失败": self.num_failed}
        )
        self.bar.update(1)
