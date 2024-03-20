import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image


def 绘图(
    x,
    y,
    kind="line",
    color=None,
    xlabel="x",
    ylabel="y",
    xlim=None,
    ylim=None,
    figsize=(3, 3),
    style="whitegrid",
):
    """
    xlim/ylim  应该传入“[0,1]”
    """

    fig, ax = plt.subplots(figsize=figsize)
    with sns.axes_style(style):
        数据表 = pd.DataFrame({"x": x, "y": y})
        if kind == "scatter":
            sns.scatterplot(data=数据表, x="x", y="y", color=color, ax=ax)
        else:
            sns.lineplot(data=数据表, x="x", y="y", color=color, ax=ax)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set(xlabel=xlabel, ylabel=ylabel)


def 复合绘图(
    xs,
    ys,
    kinds=["line"],
    xlabel="x",
    ylabel="y",
    figsize=(5, 5),
    style="whitegrid",
    color_palette=None,
):
    """
    xs, ys, kinds, colors  应该是长度相同的列表
    color_palette  调色板
    """

    # 生成的色板是一个列表
    # 每个颜色都是RGB(0-1浮点)元组
    colors = sns.color_palette(palette=color_palette, n_colors=len(ys))

    fig, ax = plt.subplots(figsize=figsize)
    with sns.axes_style(style):
        for x, y, kind, color in zip(xs, ys, kinds, colors):
            data = pd.DataFrame({"x": x, "y": y})
            if kind == "scatter":
                # 必须指定颜色，否则每次绘图的颜色都一样
                sns.scatterplot(data=data, x="x", y="y", color=color, ax=ax)
            elif kind == "line":
                sns.lineplot(data=data, x="x", y="y", color=color, ax=ax)

    ax.set(xlabel=xlabel, ylabel=ylabel)


def 热力图(X, figsize=(6, 5), style="whitegrid"):
    fig, ax = plt.subplots(figsize=figsize)
    with sns.axes_style(style):
        sns.heatmap(
            X,
            vmin=0,
            vmax=1,
            # annot=True,
            # fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            cmap="YlGnBu",
            cbar=True,
            square=True,
            ax=ax,
        )


def 显示图片(M):
    """显示图片"""

    if np.max(M) <= 1:
        # 将0-1放大到0-255
        M *= 255

    if not np.issubsctype(M, np.integer):
        # 数据类型更改为整型
        M = M.astype(np.uint8)

    if M.ndim == 3:
        # 三通道图片
        img = Image.fromarray(M, mode="RGB")
    else:
        # 单通道图片
        img = Image.fromarray(M, mode="L")
    return img
