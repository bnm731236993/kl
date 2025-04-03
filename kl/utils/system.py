import sys


def is_ipynb() -> bool:
    """判断是否是ipynb环境"""

    return 'ipykernel' in sys.modules
