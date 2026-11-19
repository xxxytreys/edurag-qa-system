# 该脚本用于：mysql 基本操作
import os
import pandas as pd
import sys
import pymysql  #用于mysql数据库链接

# todo 1.路径配置E:\pythonproject\NLP_project\mysql_system
# 1获取当前文件根目录
current_dir=os.path.dirname(os.path.abspath(__file__))
# print(f'当前目录：{current_dir}')
# 2
module_dir=os.path.dirname(current_dir)
# print(module_dir)
# 3
project_name=os.path.dirname(module_dir)
# print(project_name)
# 4将E:\pythonproject\NLP_project\mysql_system插入系统路径 保证跨目录导入模块成功
sys.path.insert(0,project_name)

# 5
from base import Config,logger
# 从base文件导入config类和logger记录器

# todo 2.定义mysql客户端类 封装mysql链接 表操作 数据操作 连接关闭等
class MySQLClient:
    # todo 2.1初始化方法 建链接 创建游标 链接异常
    def __init__(self):
    # 1将全局logger属性为类属性，可以logger.类名调用
        self.logger=logger
        try:
            # 2建立链接
            self.connection=pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE,
                charset='utf8mb4'
            )
            # 3创建游标对象
            self.cursor=self.connection.cursor()
            self.logger.info('MYSQL连接成功')
            # 21: 51:43, 189 - EduRAG - INFO - MYSQL连接成功
        except pymysql.MySQLError as e:
            self.logger.error(f'数据库链接失败: {e}')
            raise #抛出异常不屏蔽错误

    # todo 2.2创建表
    def create_table(self):
        # 1定义创建表sql语句创建jpkb存储学科知识
        create_table_query = """
            CREATE TABLE IF NOT EXISTS jpkb(
                id INT AUTO_INCREMENT PRIMARY KEY,
                subject_name VARCHAR(20),
                question VARCHAR(1000),
                answer VARCHAR(1000)
            )
        """

        try:
            # 2执行sql语句 创建表
            self.cursor.execute(create_table_query)
            # 3提交事务
            self.connection.commit()  #当提交那一刻上面的语句都成功就成功，有一个失败当没有执行
            # 4记录日志
            self.logger.info('创建表jpkb成功')
        except pymysql.MySQLError as e:
            self.logger.error(f'创建表失败{e}')
            raise

    # todo 2.3从csv文件批量插入数据到表
    def insert_data(self,csv_path):
        # 1
        try:
            # 1读取csv
            data=pd.read_csv(csv_path)
            print(data.head())
            # 2循环遍历csv执行插入操作
            for _,row in data.iterrows():
                # 2.1定义sql语句 使用%s占位符，避免sql注入攻击问题,insert into table_name(列名)
                insert_query='insert into jpkb(subject_name,question,answer) values(%s,%s,%s)'
                # 2.2执行sql
                self.cursor.execute(insert_query,(row['学科名称'],row['问题'],row['答案']))
            # 3提交事务
            self.connection.commit()
            # 4输出信息日志
            self.logger.info(f'插入数据成功')

        except pymysql.MySQLError as e:
            self.logger.error(f'插入数据错误：{e}')
            # 5事务回滚，把数据库恢复到事务开始前状态
            self.connection.rollback()
            raise

    # todo 2.4从jpkb中获取所有问题
    def fetch_question(self):
        try:
            # 1执行sql
            self.cursor.execute('select question from jpkb')
            # 2获取查询结果
            results=self.cursor.fetchall()

            self.logger.info('查询所有问题成功')
            # 4返回
            # print(results)
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f'查询问题错误{e}')
            return []

    # todo 2.5从表中获取对应的回答
    def fetch_answer(self,question):
        try:
            # 1执行sql 问题得一样不能模糊
            self.cursor.execute("select answer from jpkb where question=%s",(question,))
            # 2获取查询结果
            result=self.cursor.fetchone()  #获取单行数据
            # 3输出查询结果
            print(result[0])
            self.logger.info(f'问题：{question}的答案已查出')
            # 4处理结果，有匹配数据则返回答案（元组第一个元素）无则返回NONE
            return result[0] if result else None

        except pymysql.MySQLError as e:
            self.logger.error(f'未找到答案：{e}')
            return None

    # todo 2.6关闭
    def close_db(self):
        # 关闭数据库连接
        try:
            self.cursor.close()
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            self.logger.error(f"关闭连接失败: {e}")

if __name__ == '__main__':
    # 1实例化mysqlclient类
    mysql_client=MySQLClient()
    # mysql_client.create_table()
    # csv_path='../data/JP学科知识问答.csv'  #数据在此文件的父类文件夹同级的数据集data下
    # mysql_client.insert_data(csv_path)
    # results=mysql_client.fetch_question()
    # print(results)
    # question='getattr作用'
    # result=mysql_client.fetch_answer(question)
    # mysql_client.close_db()
    mysql_client.fetch_question()