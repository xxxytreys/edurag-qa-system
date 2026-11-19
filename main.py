from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.BM25_search import BM25Search

from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem

from base.logger import logger
from base.config import config
from openai import OpenAI

import time


# 集成问答系统的核心类
# 该类的功能：集成问答系统的核心功能，把各种工具拼起来
# 先查mysql现成答案，靠谱就用，不靠谱就用RAG重新生成，啥也没有就没找到
class IntegratedQASystem:
    def __init__(self):
        self.logger=logger
        self.config=config
        self.mysql_client=MySQLClient()
        self.redis_client=RedisClient()
        self.bm25_search=BM25Search(self.redis_client,self.mysql_client)

        try:
            self.client=OpenAI(api_key=self.config.DASHSCOPE_API_KEY,base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            self.logger.error(f'OpenAI 客户端初始化失败：{e}')
            raise

        self.vectorstore=VectorStore()
        self.ragsystem=RAGSystem(self.vectorstore,self.call_dashscope)


    # 调用LLM的方法
    def call_dashcope(self,prompt):
        try:
            # 创建聊天请求
            completion=self.client.chat.completions.create(
                model=self.config.LLM_MODEL,            #使用配置中的语言模型
                messages=[                              #系统提示和用户输入
                    {"role":"system","content":"你是一个有用的助手。"},
                    {"role":"user","content":prompt}
                ]
            )
            return completion.choices[0].message.content if completion.choices else '错误：无效的LLM'
        except Exception as e:
            self.logger.error(f'LLM调用失败：{e}')
            return f'错误：LLM调用事边-{e}'

    # 处理查询核心方法
    # ****************************************讲出来******************************
    def query(self,query,source_filter):
        # 1处理query 2处理source_filter 3写系统的逻辑
        """
        函数功能：处理用户问题按步骤，先查mysql+BM25  不行再RAG   返回答案
        :param query: 问题
        :param source_filter:数据科目
        :return:
        """
        # 1.记录开始时间
        start_time=time.time()
        self.logger.info(f'开始处理问题：{query},来源过滤：{source_filter}')

        # 2.调用BM25查mysql答案，如果两个答案，答案本身是否要走RAG
        answer , need_rag =self.bm25_search.search(query,threshold=0.85)
        # need_rag=False,说明redis有数据，或者mysql中查到了数据且阈值在0.85以上，不用rag处理;

        # 3根据BM25处理结果判断
        if answer:
            # 拿到了答案，返回
            self.logger.info(f'BM25找到了答案：{answer[:50]}')   #只显示前50条
            # 算一下用了多久
            processing_time=time.time()-start_time
            self.logger.info(f'处理时间：{processing_time}')
            return answer
        elif need_rag:
            # 答案不靠谱，重新RAG生成，need_rag=1 mysql查到但是阈值小于0.85，需要生成
            self.logger.info(f'BM25答案不靠谱，RAG处理')
            answer=self.ragsystem.generate_answer(query,source_filter)
            print(answer)

            processing_time=time.time()-start_time
            self.logger.info(f'处理时间：{processing_time}')
        else:
            # 告诉用户没找到，MYSQL找不到或者查询异常
            print(f'问题有误，请重新输入')


def main():
    """
    函数功能：系统入口，用户输入问题，来源过滤
    :return:
    """

    qa_system=IntegratedQASystem()
    qa_system.logger.info(f'系统初始化完成！！！@@@@可以开始问问题@@@@@')

    try:
        print(f'\n--------集成问答系统-----------')
        print(f'支持的来源：{qa_system.config.VALID_SOURCES}')
        print("用法：输入问题按回车查看答案")

        while True:
            query=input('\n请输入问题').strip()
            if query=='exit':
                qa_system.logger.info('用户输入exit 退出系统')
                print('拜拜！')
                break
            source_filter=input(f'请输入来源过滤：(可选)，支持：{"./".join(qa_system.config.VALID_SOURCES)},直接回车表示不限制')
            if source_filter:
                if source_filter not in qa_system.config.VALID_SOURCES:
                    qa_system.logger.warning(f'用户输入无效的科目{source_filter}')
                    source_filter=None
                else:
                    qa_system.logger.info(f'用户选择:{source_filter}')

            answer=qa_system.query(query,source_filter)
            print(f'\n答案：{answer}')
    except Exception as e:
        qa_system.logger.error(f'系统出问题了。######需要重试######')
        print(f'处理问题出错：{e},请打电话摇人：{qa_system.config.CUSTOMER_SERVICE_PHONE}')
    finally:
        qa_system.mysql_client.close_db()
        qa_system.logger('mysql连接关闭，系统退出')


if __name__=='__main__':
    main()






