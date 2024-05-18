from typing import Union, Optional


def number2string(
        value: Union[int, float],
        precision: int = 4,
        dtype: str = 'int') -> str:
    '''将数值转化为字符串'''
    if dtype == 'int' or isinstance(value, int):
        # 整数
        return f'{value}'
    elif dtype == 'float' or isinstance(value, float):
        # 浮点数
        template = '{:.%df}' % precision
        return template.format(value)
