import io
from IPython import display

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def disable_auto_display():
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
