import matplotlib.pyplot as plt

# 加载当前目录下的PY文件
from . import util


def imshow_on_axis(mat, axis):
    '''在网格处绘制图片'''
    # 关闭网格
    _ = axis.grid(False)
    # 关闭坐标轴
    _ = axis.set_axis_off()
    # 绘图
    _ = axis.imshow(mat)


def imshow(mat, size=None, scale=1, dpi=300, return_figure=False):
    '''
    绘制图片
    返回Pillow格式

    参数:
        size  图像大小
            留空则使用原图大小
        scale  缩放
        return_figure  是否返回Figure对象
    '''
    if scale <= 0:
        raise Exception('ERROR: scale<=0')

    if size is not None:
        # 图片尺寸换算
        figsize_inch = util.calc_figsize(size, dpi)*scale
    else:
        figsize_inch = util.calc_figsize(mat.shape[:2], dpi)*scale

    figure, ax = plt.subplots(figsize=figsize_inch, dpi=dpi)
    # 清除边缘
    _ = figure.subplots_adjust(left=0, right=1,
                               top=1, bottom=0)
    # 绘图
    imshow_on_axis(mat, ax)
    # 显示图片
    util.show_figure(figure)

    if return_figure:
        # 如果需要返回Figure
        return figure
    # 否则删除Figure
    plt.close()


def imshows(mats,
            grid=None,
            size=(500, 500),
            padding=(0.05, 0.95),
            space=(0.1, 0.1),
            dpi=300,
            return_figure=False):
    '''
    在网格上绘制多副图

    参数:
        grid  网格排布
        space  图片hw间距
    '''
    if grid is None:
        grid = (1, len(mats))

    figsize_inch = util.calc_figsize(size, dpi)
    figure, axis_grid = plt.subplots(
        nrows=grid[0],
        ncols=grid[1],
        figsize=figsize_inch, dpi=dpi)
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
    else:
        # 如果有多行
        for r in range(grid[0]):
            for c in range(grid[1]):
                axis = axis_grid[r][c]
                # 1D索引
                idx = r*grid[1]+c
                # 绘图
                imshow_on_axis(mats[idx], axis)

    # 显示图片
    util.show_figure(figure)

    if return_figure:
        # 如果需要返回Figure
        return figure
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
