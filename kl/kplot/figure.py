from typing import Optional, Union, Literal

import numpy as np
import matplotlib.pyplot as plt

from 。 import util


class KFigure:
    def __init__(self,
                 # 行列数
                 grid: tuple[int, int] = (1, 1),
                 # 图表大小
                 size: Optional[Union[tuple, list, np.ndarray]] = None,
                 is_inch: bool = False,
                 dpi: int = 300) -> None:
        if grid[0] < 1 or grid[1] < 1:
            raise Exception('The number of grid is less than 1')
        self.grid = grid

        if dpi < 1:
            raise Exception('The value of dpi is less than 1')
        self.dpi = dpi

        # 创建图表对象
        self.figure, _ = plt.subplots(nrows=self.grid[0],
                                      ncols=self.grid[1],
                                      dpi=self.dpi)

        if size is not None:
            # 缩放
            # 使用位置参数可能有问题
            self.resize(size=size, is_inch=is_inch)

        # 调整样式
        # self.set_padding()
        # self.set_space()

    @property
    def axes(self) -> list:
        '''所有子表的列表（平铺）'''
        return self.figure.axes

    def ax(self,
           coord: tuple[int, int] = (0, 0)
           ) -> plt.Axes:
        '''获取指定子表'''
        if coord[0] < 0 or coord[1] < 0 \
                or self.grid[0] <= coord[0] or self.grid[1] <= coord[1]:
            raise Exception(
                f'Coord {coord} is out of the range of grid {self.grid}')

        index = int(self.grid[1]*coord[0]+coord[1])
        return self.axes[index]

    def resize(self,
               size: tuple,
               scale: int = 1,
               is_inch: bool = False) -> None:
        '''
        修改图表的形状
        参数:
            scale  在size基础上进一步缩放
            is_inch  是否传入的是英寸
        '''
        if is_inch:
            # 如果传入的尺寸是英寸
            size_inch = size
        else:
            # 图片尺寸换算
            size_inch = util.calc_figsize(size, self.dpi)*scale
        # 在此处修改dpi会有问题
        self.figure.set_figheight(size_inch[0])
        self.figure.set_figwidth(size_inch[1])

    def set_padding(self,
                    padding: tuple[float, float, float, float] = (
                        0.05, 0.05, 0.05, 0.05),
                    disable: bool = False) -> None:
        '''
        设定内边距
        参数:
            padding  比例  左上右下
            disable  直接清除padding，也就是为0
        '''
        if disable:
            padding = (0, 0, 0, 0)
        _ = self.figure.subplots_adjust(left=padding[0],
                                        bottom=padding[3],
                                        top=1-padding[1],
                                        right=1-padding[2])

    def set_space(self,
                  h: float = 0.1,
                  w: float = 0.1) -> None:
        '''设定子图的间距'''
        _ = self.figure.subplots_adjust(hspace=h,
                                        wspace=w)

    def display(self) -> None:
        '''绘制Figure'''
        util.display_figure(self.figure)

    def __del__(self):
        # 删除图表对象
        plt.close()

    def imshow(self,
               mat: np.ndarray,
               coord: tuple = (0, 0)) -> None:
        '''（快捷）绘制图片到指定子表'''
        util.imshow_to_ax(mat=mat, ax=self.ax(coord=coord))
