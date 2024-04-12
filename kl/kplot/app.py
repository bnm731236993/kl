from typing import Optional

import numpy as np

import matplotlib.pyplot as plt

# 加载当前目录下的PY文件
from . import util


def imshow_on_axis(mat: np.ndarray,
                   axis: plt.Axes) -> None:
    '''在网格处绘制图片'''
    if np.issubdtype(mat.dtype, np.floating):
        if mat.min() < 0 or mat.max() > 1:
            raise Exception('The range of float data is [0,1]')
    elif np.issubdtype(mat.dtype, np.integer):
        if mat.min() < 0 or mat.max() > 255:
            raise Exception('The range of integer data is [0,255]')

    # 关闭网格
    util.disable_grid_and_axis(axis)
    # 绘图
    _ = axis.imshow(mat)


def imshow(mat: np.ndarray,
           size: Optional[tuple[int, int]] = None,
           scale: int = 1,
           dpi: int = 300,
           channel_first: bool = False,
           display_figure: bool = True,
           return_figure: bool = False):
    '''
    绘制图片
    返回Pillow格式

    参数:
        size  图像大小  [高，宽]
        scale  缩放
        channel_first  通道维度是否为第一维度
        return_figure  是否返回Figure对象
    '''
    if scale <= 0:
        # 如果缩放倍率错误
        raise Exception('Scale must be greater than zero')
    if channel_first:
        # 调整维度顺序
        mat = util.channel_revise(mat)

    # 创建图表对象
    figure, axis = plt.subplots(dpi=dpi)

    # 如果没有传入图片形状
    if size is None:
        # 此处维度已经被调整
        size = mat.shape[:2]

    # 设定图片大小
    util.figure_resize(figure=figure,
                       size=size,
                       scale=scale,
                       dpi=dpi,
                       is_inch=False)

    # 清除边缘
    _ = figure.subplots_adjust(left=0, right=1,
                               top=1, bottom=0)
    # 绘图
    imshow_on_axis(mat, axis)

    if display_figure:
        # 显示图片
        util.show_figure(figure)

    if return_figure:
        # 如果需要返回Figure
        return figure
    else:
        # 否则删除figure对象
        plt.close()


def imshows(mats: np.ndarray,
            grid: tuple = (1, 1),
            size: tuple[int, int] = (500, 500),
            padding: tuple[float, float] = (0.05, 0.95),
            space: tuple[float, float] = (0.1, 0.1),
            dpi: int = 300,
            channel_first: bool = False,
            display_figure: bool = True,
            return_figure: bool = False):
    '''
    在网格上绘制多副图

    参数:
        grid  网格排布
        space  图片hw间距
        display_figure  是否自动显示图像
    '''
    if len(mats) != grid[0]*grid[1]:
        raise Exception('Input length is not equal to the number of grids')

    # 创建图表对象
    figure, axis_grid = plt.subplots(nrows=grid[0],
                                     ncols=grid[1],
                                     dpi=dpi)
    # 设定图表大小
    util.figure_resize(figure=figure,
                       size=size,
                       scale=1,
                       dpi=dpi,
                       is_inch=False)

    # 设置最小边缘
    _ = figure.subplots_adjust(left=padding[0],
                               right=padding[1],
                               bottom=padding[0],
                               top=padding[1],
                               hspace=space[0],
                               wspace=space[1])

    if grid[0] == 1:
        # 如果只有一行
        for c in range(grid[1]):
            axis = axis_grid[c]
            mat = mats[c]
            if channel_first:
                # 调整维度顺序
                mat = util.channel_revise(mat)
            imshow_on_axis(mat, axis)
    elif grid[1] == 1:
        # 如果只有一列
        for c in range(grid[0]):
            axis = axis_grid[c]
            mat = mats[c]
            if channel_first:
                # 调整维度顺序
                mat = util.channel_revise(mat)
            imshow_on_axis(mat, axis)
    else:
        # 如果有多行
        for r in range(grid[0]):
            for c in range(grid[1]):
                axis = axis_grid[r][c]
                # 1D索引
                idx = r*grid[1]+c
                mat = mats[idx]
                if channel_first:
                    # 调整维度顺序
                    mat = util.channel_revise(mat)
                # 绘图
                imshow_on_axis(mat, axis)

    if display_figure:
        # 显示图片
        util.show_figure(figure)

    if return_figure:
        # 如果需要返回Figure
        return figure
    else:
        # 否则删除Figure
        plt.close()


def imshows_with_titles(mats,
                        # 标题
                        titles=None,
                        font_size=16,
                        font_family='Times New Roman',
                        font_color='black',
                        grid=None,
                        size=(500, 500),
                        padding=(0.05, 0.95),
                        space=(0.1, 0.1),
                        dpi=300,
                        title_shift=0.1,
                        return_figure=False):
    '''
    在网格上绘制多副图
    附带标题

    参数:
        title_shift  标题在y轴上的偏移
            以上为正值
    '''
    # 网格默认为横
    if grid is None:
        grid = (1, len(mats))

    # 图像大小换算
    figsize_inch = util.calc_figsize(size, dpi)
    figure, axis_grid = plt.subplots(
        nrows=grid[0],
        ncols=grid[1],
        figsize=figsize_inch, dpi=dpi)

    # 字体大小换算
    font_size_inch = font_size/dpi*100
    fontdict = {
        'fontsize': font_size_inch,
        'family': font_family,
        'fontweight': 100,
        'color': font_color
    }

    # 设置最小边缘
    _ = figure.subplots_adjust(left=padding[0],
                               right=padding[1],
                               bottom=padding[0],
                               top=padding[1],
                               hspace=space[0],
                               wspace=space[1])

    if grid[0] == 1:
        # 如果只有一行
        for c in range(grid[1]):
            axis = axis_grid[c]
            imshow_on_axis(mats[c], axis)
            _ = axis.set_title(label=titles[c],
                               loc='center',
                               y=-0.1,
                               pad=0,
                               fontdict=fontdict)
    else:
        # 如果有多行
        for r in range(grid[0]):
            for c in range(grid[1]):
                axis = axis_grid[r][c]
                # 1D索引
                idx = r*grid[1]+c
                # 绘图
                imshow_on_axis(mats[idx], axis)
                _ = axis.set_title(label=titles[idx],
                                   loc='center',
                                   y=title_shift,
                                   pad=0,
                                   fontdict=fontdict)

    # 显示图片
    util.show_figure(figure)

    if return_figure:
        # 如果需要返回Figure
        return figure
    # 否则删除Figure
    plt.close()
