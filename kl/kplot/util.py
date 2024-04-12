import io
from IPython import display

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


def disable_grid_and_axis(axis: plt.Axes) -> None:
    '''关闭轴的网格和坐标轴'''
    # 关闭网格
    _ = axis.grid(False)
    # 关闭坐标轴
    _ = axis.set_axis_off()


def disable_auto_display() -> None:
    '''关闭Matplotlib对图片的自动显示'''
    # 判断“交互性”
    if plt.isinteractive():
        # 关闭Matplotlib的自动显示
        _ = plt.ioff()


def figure_to_PIL(figure, format='png'):
    '''Matplotlib转Pillow'''
    # 字节流
    img_buffer = io.BytesIO()
    # 将图片输出到字节流
    _ = figure.savefig(img_buffer, format=format)
    # 从字节流读取
    img = Image.open(img_buffer)
    return img


def show_figure(figure):
    '''绘制Figure'''
    # 转换为Pillow图片
    fig_img = figure_to_PIL(figure)
    # Pillow无法正常显示RGBA图片
    # 直接用display显示
    display.display_png(fig_img)


def calc_figsize(size_px, dpi=300):
    '''
    将像素形状换算为适用于Matplotlib的形状
    '''
    if not isinstance(size_px, np.ndarray):
        # 转化为Num数据
        size_px = np.array(size_px)
    # 输出适用于matplotlib的形状
    size_inch = size_px.astype(np.float32)/dpi
    return size_inch


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
