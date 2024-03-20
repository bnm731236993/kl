import sys


def isIpynb():
    """判断是否是ipynb环境"""

    return 'ipykernel' in sys.modules


def display_all_output(ast_node_interactivity='all'):
    """
    在ipynb中显示每行的输出
        all  显示所有表达式大返回值
        last_expr  只有最后一个表达式
        none  不显示
    """

    if isIpynb():
        from IPython.core.interactiveshell import InteractiveShell

        InteractiveShell.ast_node_interactivity = ast_node_interactivity
