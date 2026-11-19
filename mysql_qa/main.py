# 该脚本用于mysql入门程序

from db.mysql_client import MySQLClient
from cache.redis_client import RedisClient
from retrieval.BM25_search import BM25Search
from base import logger
import time

# todo 1.定义问答系统类 整合mysql redis BM25搜索功能，提供统一的查询接口
class MySQLQASystem:
    def __init__(self):
        # 1.日志记录器
        self.logger=logger
        # 2mysql客户端
        self.mysql_client=MySQLClient()
        # 3redis客户端
        self.redis_client=RedisClient()
        # 4BM25搜索模型
        self.bm25_search=BM25Search(self.redis_client,self.mysql_client)

    # 处理用户的查询，1调用BM25搜索 2记录日志耗时 3返回答案
    def query(self,query):
        """
        处理用户查询 通过BM25搜索从mysql获取答案，返回结果并记录日志
        :param query:用户输入的查询文本
        :return:匹配的答案，没找到返回提示
        """
        # 1记录查询开始时间
        start_time=time.time()
        # 2.记录日志，标记当前正在处理的查询内容
        self.logger.info(f'处理查询：{query}')
        # 3调用BM25模型，获取答案
        # _表示 忽略是否需要进一步处理，这里我们只获取答案
        result=self.bm25_search.search(query,threshold=0.85)

        if isinstance(result, (list, tuple)) and len(result) >= 2:
            answer, similarity = result[0], result[1]
        else:
            answer = result
            similarity = None

        # 4分支处理搜索结果，找到答案就记录，未找到就返回默认提示
        if answer:
            self.logger.info(f'找到答案：{answer}')
        else:
            self.logger.info('未找到答案,需要调用rag系统')
            answer='未找到匹配答案'

        # 5计算处理时间
        process_time=time.time() - start_time
        self.logger.info(f'处理时间：{process_time:.3f}秒')

        return answer


def main():
    mysql_qa=MySQLQASystem()
    try:
        print('欢迎使用Mysql-qa系统')
        print('输入查询进行回答，输入exit退出')

        while True:
            query = input('\n请输入查询：').strip()
            if query.lower() == 'exit':
                logger.info('退出mysql系统')
                print('再见')
                break  # 退出循环

            # 将具体的查询逻辑放在 try 里面，但 try 不要包着整个 while
            try:
                answer = mysql_qa.query(query)
                print(f'答案是：{answer}')
            except Exception as e:
                logger.error(f'处理查询时出现错误：{e}')
                print(f"处理查询时发生错误: {e}，请重试。")
                # 这里不要 break，程序会继续回到 while True 继续让用户输入
                continue
    finally:
        mysql_qa.mysql_client.close_db()


if __name__=='__main__':
    main()



