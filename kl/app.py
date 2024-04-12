from typing import Literal

import sys


def is_ipynb() -> bool:
    """判断是否是ipynb环境"""

    return 'ipykernel' in sys.modules


def display_all_output(
        ast_node_interactivity: Literal['all', 'last_expr', 'none'] = 'all') -> None:
    """
    在ipynb中显示每行的输出
        all  显示所有表达式大返回值
        last_expr  只有最后一个表达式
        none  不显示
    """

    if is_ipynb():
        from IPython.core.interactiveshell import InteractiveShell

        InteractiveShell.ast_node_interactivity = ast_node_interactivity


def sample(iterObject, n=1):
    '''
    从可迭代对象中获取N个项
    参数:
        n  项的个数
    返回:
        项的列表
    '''
    result = []
    # 不用enumerate
    idx = 0
    for obj in iterObject:
        result.append(obj)
        idx += 1
        if idx >= n:
            # 结束
            break
    return result
