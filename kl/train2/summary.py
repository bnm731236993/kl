from typing import Union, Dict, Any
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Summary:
    def __init__(self,
                 title: Union[str, Path],
                 output: Union[str, Path]):
        """ 日志记录器
        args:
            title: 日志标题，会在output下创建子文件夹，用于在界面中区分不同实验
            output: 日志输出路径
        """
        # 创建日志文件夹
        Path(output).mkdir(exist_ok=True,  parents=True)
        # 日志记录器
        self.writer = SummaryWriter(log_dir=Path(output)/Path(title))

    def add(self,
            name: str,
            step: int,
            value: Any):
        """ 添加单个记录
        args:
            tag: 图表名称标签
            step: 步数
            value: 记录值
        """
        # 添加记录
        self.writer.add_scalar(tag=name,
                               scalar_value=value,
                               global_step=step)
        # 刷新
        self.writer.flush()

    def adds(self,
             name_value: Dict[str, Any],
             step: int):
        """ 添加多个记录 """
        for name, value in name_value.items():
            self.add(name=name,
                     value=value,
                     step=step)

    def __del__(self):
        """ 析构函数 """
        # 刷新
        self.writer.flush()
        # 关闭写入流
        self.writer.close()
