# 该脚本用于BM25搜索，阈值0.85+

from rank_bm25 import BM25Okapi
import numpy as np
from mysql_qa.utils.preprocess import  preprocess_text
from base import logger
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import  RedisClient

# todo 1.定义BM25搜索类 封装数据加载 模型初始化 相似度计算 答案检索
class BM25Search:
    def __init__(self,redis_client,mysql_client):
        # 1日志实例
        self.logger=logger
        # 2redis客户端实例
        self.redis_client=redis_client
        # 3mysql客户端实例
        self.mysql_client=mysql_client
        # 4初始化BM25模型
        self.bm25=None
        # 5初始化分词后问题列表
        self.questions=None
        # 6初始化原始问题列表
        self.original_questions=None
        # 7调用数据加载方法
        self._load_data()  #_的函数表示;__init__ 这种前后两个的，是程序内置；__func()这种前面两个的，是私有；这种前面一个_的，是内置函数，默认都是只在类内用

    # def _load_data(self):
    #     original_key = "qa_original_question1"
    #     tokenized_key = "qa_tokenized_question1"
    #
    #     # 从redis读取
    #     original_questions = self.redis_client.get_data(original_key)
    #     tokenized_questions = self.redis_client.get_data(tokenized_key)
    #
    #
    #     # 缓存为空则从mysql加载
    #     if not original_questions or not tokenized_questions:
    #         original_questions = self.mysql_client.fetch_question()  #从mysql找所有问题，放入问题Key中
    #         print('redis是空的，进入if')
    #         if not original_questions:
    #             self.logger.warning('未从mysql中查询到任何问题数据')
    #             return
    #
    #     # 获取分词
    #     tokenized_questions = [preprocess_text(q[0]) for q in original_questions]
    #
    #     # 打印没问题之后存入
    #     print(original_questions[:10])   #(('用上下文管理器实现函数运行时间的计算?',), ("有字典{'a':1,'b':2,'c':3},现在有一个需求\n装饰器练习题",), ('两个人开发项目,我push到github上后\n别人直接pull可以看到我的代码吗\n为什么我要pull request 他才能看到我提交的代码',), ('关联子查询的执行顺序是什么',), ('定义类字典对象(继承于dict类)，如何修改读写字典元素方法',), ('Ubuntu上Navicat for MySQL 显示乱码',), ('为什么isalnum函数会通过中文',), ('pycharm资源管理区域文件的显示更新',), ('如何在 Ubuntu 中快速创建Pycharm桌面快捷方式？',), ('VMware安装VMware Tools时显示灰色如何解决',))
    #     print(tokenized_questions[:10])
    #
    #     # 存入redis
    #     # self.redis_client.save_data(original_key, [q[0] for q in original_questions])
    #     # self.redis_client.save_data(tokenized_key, tokenized_questions)
    #
    #     # 初始化BM25
    #     self.original_questions = original_questions
    #     self.questions = tokenized_questions
    #     self.bm25 = BM25Okapi(self.questions)
    #     self.logger.info('BM25模型初始化成功')

    def _load_data(self):
        original_key = "qa_original_question2"
        tokenized_key = "qa_tokenized_question2"

        original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        if not original_questions or not tokenized_questions:
            original_questions = self.mysql_client.fetch_question()
            print('redis是空的，进入if')
            if not original_questions:
                self.logger.warning('未从mysql中查询到任何问题数据')
                return

            # 关键：把元组转成字符串列表
            original_questions = [q[0] if isinstance(q, tuple) else q for q in original_questions]

        # 分词
        tokenized_questions = [preprocess_text(q) for q in original_questions]

        # print("问题数:", len(original_questions))
        # print("前3条:", original_questions[:3])
        # print("前3条分词:", tokenized_questions[:3])

        # 存入redis
        # self.redis_client.save_data(original_key, original_questions)
        # self.redis_client.save_data(tokenized_key, tokenized_questions)

        self.original_questions = original_questions
        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info('BM25模型初始化成功')

    def _softmax(self,scores):
        # 1对输入分数进行softmax归一化 输出总和为1的概率分布
        # 1.1计算指数分数 对每个分数减去最大值，防止数据大导致溢出,分数减去最大值后取指数级，所以最大等于1，其他小于1正数
        exp_scored=np.exp(scores-np.max(scores))
        # 1.2返回归一化结果，每一项算占总数的百分比
        return exp_scored/np.sum(exp_scored)


    def search(self,query,threshold):
        """
        根据输入查询检索 最相似的问题并返回答案 先判断是否问题成立
        :param query:用户查询文本
        :param threshold:相似度阈值
        :return:匹配成功返回answer Fasle（redis命中 不用新查询 用户问的直接在redis有） 失败返回None True（redis没命中 要新查询，就是用户问的和redis问题库里面不一样）
        """
        # 1判断问题是否成立
        if not query or not isinstance(query,str):
            return None,False

        # 2执行查询
        cached_answer=self.redis_client.get_answer(query)
        # 走到这有两种情况，一种缓存命中，一种没命中，需要if判断
        if cached_answer:
            # clean_answer = str(cached_answer).strip()
            # 关键点在于这个答案的末尾，包含了一个换行符 \n。在Python中，当你把包含 \n的字符串放到类似这样的格式化操作（比如使用了冒号:）中时，Python解释器会尝试把冒号后面的 \n当作格式说明符来解析，但 \n并不是合法的格式化参数，于是就会抛出Invalidformatspecifier。
            self.logger.info(f'在redis中找到了完全一致的问题，答案：{query}')
            return cached_answer ,False  #说明命中

        # 3走到这，说明缓存没命中，需要从数据库查询
        try:
            # 1对查询文本进行预处理 分词 和问题库预处理逻辑一致
            print(f'未从redis找到{query}问题的答案，进入mysql找')
            query_tokens=preprocess_text(query)
            # print(query_tokens)
            # 2计算该问题和数据库所有问题的BM25相似度
            scores=self.bm25.get_scores(query_tokens)
            # print(scores)
            # 3对数据进行归一化 转化为概率分布
            softmax_score=self._softmax(scores)
            # print(softmax_score)
            # 4找到最高相似度对应的索引和分数
            best_idx=np.argmax(softmax_score)
            best_score=np.max(softmax_score)
            print(best_score,best_idx)
            print("命中问题:", repr(self.original_questions[best_idx]))
            print("用户查询:", repr(query))

            # 5不是直接从数据库找到对应数据，要不概率大于0.85从数据库找 要不概率小于0.85走rag
            if best_score>=threshold:
                # 5.1从数据库查找答案
                original_question=self.original_questions[best_idx]
                # 5.2从数据库查找问题答案
                answer=self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 5.3将查询答案插入redis中
                    self.redis_client.save_data(query,answer)
                    # 5.4记录日志搜索成功及相似度
                    self.logger.info(f'查询{query}成功，-》{original_question},相似度：{best_score}')
                    return answer,False   #true代表要进一步做处理 false代表不用做进一步处理，从mysql找到了

            # 6若未超过阈值，则记录info和日志
            self.logger.info(f'未找到满足条件的可靠答案，相似度：{best_score}')
            return None,True
        except Exception as e:
            self.logger.error(f'数据库查询异常：{e}')
            return None,True  #true代表要进一步做处理


if __name__=='__main__':
    # 1实例化
    redis_client=RedisClient()
    mysql_client=MySQLClient()

    # 2创建BM25搜索实例
    bm25_search=BM25Search(redis_client,mysql_client)

    # 3测试搜索功能
    result=bm25_search.search(query='网络调试助手无法收到 ubuntu发送的包',threshold=0.85)
    print(f"结果：{result}")
