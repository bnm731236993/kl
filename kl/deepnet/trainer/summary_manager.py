from typing import Union, Optional
from pathlib import Path
import json

from .util import number2string


class SummaryManager:
    '''信息管理器'''
    # 保存的数据
    storage = []

    def __init__(self,
                 root_dir: Union[str, Path, None],
                 # 要记录的数值
                 columns: list = ['mode', 'epoch', 'batch',
                                  'loss', 'accuracy', 'time'],
                 enable_tensorboard: bool = False,
                 # 要上传到Tensorboard的数值
                 board_columns: list = ['loss', 'accuracy'],
                 ):
        # 根路径
        # 使用Path对象
        self.root_dir = Path(root_dir)
        # 创建文件夹
        self.root_dir.mkdir(parents=True,
                            exist_ok=True)

        # 需要记录的列名
        self.columns = columns

        # 使用Tensorboard图表
        self.enable_tensorboard = enable_tensorboard
        if self.enable_tensorboard:
            # 需要写入图表的列名
            self.board_columns = board_columns

            from torch.utils.tensorboard import SummaryWriter

            # 写入器
            self.boardWriter = SummaryWriter(
                log_dir=self.root_dir
            )

    def append(self, input: dict):
        '''添加单个数据'''
        # 储存数据
        self._storage_add(input)

        # 追加到Tensorboard
        if self.enable_tensorboard:
            self._board_add(input)

    def _storage_add(self, input: dict):
        '''追加单个数据'''
        input2 = {}
        # 只留需要的键值对
        for key in self.columns:
            value = input.get(key, None)
            if value is None:
                # 如果不存在于表单中
                continue
            input2[key] = value

        self.storage.append(input2)

    def _board_add(self, input: dict):
        '''写入Tensorboard'''
        # 横坐标
        # 注意epoch从1开始
        for key in self.board_columns:
            # 纵坐标
            y = input.get(key, None)
            if y is None:
                # 如果没有该数据
                continue

            # 图表名
            tag = '{}/{}'.format(input['mode'], key.title())
            # 记入表内
            self.boardWriter.add_scalar(
                tag=tag,
                scalar_value=y,
                global_step=input['epoch'])

    def __len__(self) -> int:
        return len(self.storage)

    def get_text(self, idx: int) -> str:
        '''
        获取指定条目的字符串形式
        用于打印
        '''
        text_list = []
        # 信息的字典
        obj = self.storage[idx]
        for column in self.columns:
            value = obj.get(column, None)
            if value is None:
                # 如果没有该记录
                continue

            if isinstance(value, int):
                # 整数
                # 首字母大写
                text = '{}:{}'.format(column.title(),
                                      number2string(value, dtype='int'))
            elif isinstance(value, float):
                # 浮点数
                text = '{}:{}'.format(column.title(),
                                      number2string(value, dtype='float'))
            text_list.append(text)
        return ' '.join(text_list)

    def save(self):
        '''保存为JSON'''
        path = self.root_dir/Path('summary.json')
        with open(path, mode='w', encoding='utf-8') as fp:
            json.dump(self.storage,
                      fp=fp,
                      indent=4,
                      ensure_ascii=False)
