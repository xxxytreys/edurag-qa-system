
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.BM25_search import BM25Search

from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem

from base.logger import logger
from base.config import config
from openai import OpenAI

import time
import uuid  #会话ID
import pymysql #python操作mysql的库，用于捕获异常，增删改查

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

        # 优化1
        # self.init_conversation_table()
        # 初始化对话历史表，在mysql中创建存储对话的表结构,在Mysql直接写好了

    def call_dashcope(self,prompt):
        try:
            # 创建聊天请求
            completion=self.client.chat.completions.create(
                model=self.config.LLM_MODEL,            #使用配置中的语言模型
                messages=[                              #系统提示和用户输入
                    {"role":"system","content":"你是一个有用的助手。"},
                    {"role":"user","content":prompt}
                ],
                timeout=30,
                stream=True
            )

            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content=chunk.choices[0].delta.content
                    yield content
        except Exception as e:
            self.logger.error(f'LLM调用失败：{e}')
            return f'错误：LLM调用事边-{e}'

    def fetch_recent_histor(self,session_id):
        try:
            with self.mysql_client.cursor() as cursor:
                sql = (
                    "SELECT id, session_id, question, answer, timestamp "
                    "FROM conversation "
                    "WHERE session_id = %s "
                    "ORDER BY timestamp DESC "
                    "LIMIT %s"
                )
                cursor.execute(sql, (session_id, 5))
                rows = cursor.fetchall()
            # 反转为时间正序，方便按“旧→新”拼接历史
            return list(reversed(rows))

        except Exception as e:
            self.logger.error(f'查MYSQL失败：{e}')
            return f'错误-{e}'

# yield 用法：
# def func()
#     for i in range(10):
#         yield i
# 相当于 list=[]   list.append(i)   return list
# print(func())


    def updata_session_history(self,session_id:str,question:str,answer:str):
        """
        更新会话历史的mysql 保留最近的5轮对话
        :param session_id:会话的标识，一个用户一个标识，区分人
        :param question:问题
        :param answer:答案
        :return:
        """
        try:
            self.mysql_client.cursor.execute(
                """
                INSERT INTO conversation(session_id,question,answer,timestamp)
                VALUES (%s,%s,%s,NOW())
                """,(session_id,question,answer)
            )
            history=self.fetch_recent_histor(session_id)
            # 删除超过五轮的记录
            self.mysql_client.cursor.execute("""
                DELETE 
                FROM conversation          
                WHERE session_id=%s
                    AND id NOT IN (SELECT id
                            FROM  ( SELECT id
                                    FROM conversation
                                    WHERE session_id=%s
                                    ORDER BY timestamp DESC
                                        LIMIT %s) AS sub)
            """,(session_id,session_id,5))
            # 根据时间排序，拿到最近五条，只要最近5条的ID，不在这五条范围里的其他会话ID，全删掉
            self.mysql_client.connection.commit()
            self.logger.info(f'会话{session_id}历史更新成功')
            return history

        except Exception as e:
            self.logger.error(f'错误：{e}')
            self.mysql_client.connection.rollback()
            raise


    def clear_session_history(self,session_id:str):
        """
        函数功能：清除指定会话历史
        :param session_id:会话唯一标识
        :return:True 清除成功 False 清除失败
        """
        try:
            self.mysql_client.cursor.execute(
                """
                DELETE
                FROM conversation
                WHERE session_id=%s
                """,(session_id,)
            )
            self.mysql_client.connection.commit()
            self.logger.info(f'会话:{session_id}历史清除成功')
            return True
        except Exception as e:
            self.logger.error(f'异常：{e}')
            return False

# 要改ragsystem
    def query(self,query,session_id=None,source_filter=None):
        """
        函数功能：整合答案
        :param query: 问题
        :param session_id:会话ID
        :param source_filter: 学科
        :return:
        """
        # 记录开始时间和日志
        start_time=time.time()
        self.logger.info(f'处理查询:{query}')

        # 获取历史对话 没有session返回空列表
        history=self.fetch_recent_histor(session_id) if session_id else []

        # 执行BM25，获取答案和是否需要rag
        answer,need_rag=self.bm25_search.search(query,threshold=0.85)

        if answer:  #做返回的之前的一些列记录日志 处理时间 更新历史对话 返回列表操作
            # 最好的情况下，阈值大于0.85，直接返回不走rag
            self.logger.info(f'答案：{answer}')
            if session_id:
                self.updata_session_history(session_id,query,answer)
            processing_time=time.time()-start_time
            self.logger.info(f'处理时间：{processing_time}')
            return answer

        elif need_rag:
            # 没查到或者阈值低，走rag,大模型生成，从RAG获取流式输出，做返回的一系列操作：更新对话历史，记录日志 处理时间
            self.logger.info('走rag')
            collected_answer=''
            for token in self.ragsystem.generate_answer(query,source_filter=source_filter,history=history):
                collected_answer+=token
                yield token,False

            if session_id:
                self.updata_session_history(session_id,query,collected_answer)

            process_time=time.time()-start_time

            self.logger.info(f'处理时间：{process_time},')

            yield "",True  #返回空字符串，标记流结束

        else:
            process_time=time.time()-start_time

            self.logger.info(f'未找到答案，处理时间：{process_time},')

            yield "未找到答案",True  #返回空字符串，标记流结束


def main():
    # 1. 定义主函数，提供命令行交互界面 -> 初始化问答系统.
    qa_system = IntegratedQASystem()  # 初始化问答系统
    # 2. 生成唯一会话 ID
    session_id = str(uuid.uuid4())
    # 3. 打印欢迎信息
    print("\n欢迎使用集成问答系统！")
    # 打印会话 ID
    print(f"会话ID: {session_id}")
    # 打印支持的学科类别
    print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
    # 提示用户输入查询或退出
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        while True:
            # 4. 获取用户输入的查询
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 如果用户输入 exit，记录退出日志
                logger.info("退出系统")
                # 打印退出信息
                print("再见, 感谢您的使用, 期待下次再会！")
                # 退出循环
                break
            # 5. 获取用户输入的学科过滤 -> 允许用户指定学科类别, 无效则忽略.
            # 5.1 接收用户录入的学科类别
            source_filter = input(f"请输入学科类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            # 5.2 判断用户录入的 学科类别 是否有效.
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                # 如果学科过滤无效，记录警告日志
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                # 设置为空，忽略过滤
                source_filter = None
            # 6. 打印答案提示
            print("\n答案: ", end="", flush=True)
            # 初始化累积答案的字符串
            answer = ""
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    # 仅当 token 非空时打印
                    print(token, end="", flush=True)
                    # 累积答案
                    answer += token
                if is_complete:
                    # 如果是完整答案或流结束，换行并退出循环
                    print()
                    break
            # 7. 打印对话历史
            history = qa_system.fetch_recent_histor(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        qa_system.mysql_client.close_db()

if __name__=='__main__':
    main()



