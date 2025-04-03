from typing import Union, Optional
import os
from pathlib import Path

import torch as tc


class CheckpointManager:
    '''权重文件管理器'''

    def __init__(self,
                 modObj: dict = None,
                 root_dir: Union[str, Path] = './checkpoint/',
                 max_count: int = 3):
        # “网络名-网络”的字典
        # 可以后续修改
        self.modObj = modObj
        # 存放路径
        self.root_dir = Path(root_dir)
        # 创建根文件夹
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # 最大文件数
        self.max_count = max_count

        # 最新的记录点序号
        self.stack = []
        # 检查已有的权重
        self.check_existing_pth()

    def get_last_idx(self):
        '''得到已有的最新的序号'''
        if len(self.stack) > 0:
            return self.stack[-1]

        # 如果没有过去的权重文件
        return 0

    def check_existing_pth(self):
        '''检查已有的权重文件'''
        # 转化为数字
        files = [int(fp.stem) for fp in self.root_dir.glob('*.pth')]
        # 记录权重序号
        # 从小到大排序
        self.stack = sorted(files)

    def save(self):
        '''保存当前的权重'''
        # 新权重的序号
        new_idx = self.get_last_idx()+1
        pth_path = self.root_dir / Path(str(new_idx)+'.pth')

        # 保存权重
        tc.save(
            obj={mod_name: mod.state_dict()
                 for mod_name, mod in self.modObj.items()},
            f=pth_path)

        if len(self.stack) >= self.max_count:
            # 如果已经达到数量限制
            # 弹出最旧的
            oldest_idx = self.stack.pop(0)
            oldest_pth_path = self.root_dir / Path(str(oldest_idx)+'.pth')
            # 删除该文件
            os.remove(oldest_pth_path)

        # 加入新的权重序号
        self.stack.append(new_idx)

    def load(self, pth_path: Union[str, Path, None] = None):
        '''加载权重'''
        if pth_path is None:
            # 加载最新的权重
            last_idx = self.get_last_idx()
            pth_path = self.root_dir / Path(str(last_idx)+'.pth')
        # 打开权重文件
        pth = tc.load(pth_path)
        for mod_name, mod in self.modObj.items():
            # 网络加载对应权重
            _ = mod.load_state_dict(pth[mod_name])
