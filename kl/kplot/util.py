import io
# 大写避免混淆
from IPython import display as Display

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def figure_resize(figure: plt.Figure,
                  size: tuple,
                  scale: int = 1,
                  dpi: int = 300,
                  is_inch: bool = False) -> None:
    '''
    修改图表的形状
    参数:
        is_inch  是否传入的是英寸
    '''
    if is_inch:
        # 如果传入的尺寸是英寸
        size_inch = size
    else:
        # 图片尺寸换算
        size_inch = calc_figsize(size, dpi)*scale
    # 在此处修改dpi会有问题
    # figure.set_dpi(dpi)
    figure.set_figheight(size_inch[0])
    figure.set_figwidth(size_inch[1])


def disable_grid_and_axis(ax: plt.Axes) -> None:
    '''关闭轴的网格和坐标轴'''
    # 关闭网格
    _ = ax.grid(False)
    # 关闭坐标轴
    _ = ax.set_axis_off()


def imshow_to_ax(mat: np.ndarray,
                 ax: plt.Axes) -> None:
    '''在网格处绘制图片（不即时显示）'''
    if np.issubdtype(mat.dtype, np.floating):
        if mat.min() < 0 or mat.max() > 1:
            raise Exception(
                '''Input's dtype is float, but value is't within the range of [0,1]''')
    elif np.issubdtype(mat.dtype, np.integer):
        if mat.min() < 0 or mat.max() > 255:
            raise Exception(
                '''Input's dtype is integer, but value is't within the range of [0,255]''')

    # 绘图
    _ = ax.imshow(mat)


def disable_auto_display() -> None:
    '''关闭Matplotlib对图片的自动显示'''
    # 判断“交互性”
    if plt.isinteractive():
        # 关闭Matplotlib的自动显示
        _ = plt.ioff()


def figure_to_PIL(figure: plt.Figure,
                  format: str = 'png'):
    '''Matplotlib转Pillow'''
    # 字节流
    img_buffer = io.BytesIO()
    # 将图片输出到字节流
    _ = figure.savefig(img_buffer, format=format)
    # 从字节流读取
    image = Image.open(img_buffer)
    return image


def display_figure(figure: plt.Figure) -> None:
    '''绘制Figure'''
    # 转换为Pillow图片
    fig_img = figure_to_PIL(figure, format='png')
    # Pillow无法正常显示RGBA图片
    # 用display显示
    Display.display_png(fig_img)


def calc_figsize(size: tuple[int, int],
                 dpi: int = 300) -> tuple:
    '''
    将像素形状换算为适用于Matplotlib的形状
    '''
    if size[0] <= 0 or size[1] <= 0:
        raise Exception('height or width <= 0')
    return size[0]/dpi, size[1]/dpi


def channel_revise(mat):
    '''
    对于第一维度为通道维度的情况
    调整到第三维度
    '''
    if len(mat.shape) < 3:
        # 如果图片没有三个维度
        raise Exception(f"Image's dimension is {len(mat.shape)}")

    # 第一维度为通道维度
    return np.transpose(mat, axes=(1, 2, 0))
