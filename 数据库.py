import sqlite3


class MySQlite:
    def __init__(self, datebase):
        """初始化"""
        self.connect = sqlite3.connect(datebase)

    def __del__(self):
        """析构"""
        self.connect.close()

    def execute(self, order):
        """执行SQl语句"""

        try:
            with self.connect:
                cur = self.connect.execute(order)
            return cur
        except:
            return False

    def insertMany(self, table_name, cols, rows):
        """
        插入多组数据
        参数:
            cols  列名的列表
            rows  多行数据
        """

        order = "insert into %s (%s) values (?, ?)" % (table_name, ",".join(cols))

        try:
            with self.connect:
                self.connect.executemany(order, rows)
            return True
        except:
            return False
