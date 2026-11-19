# 该脚本用于RAG系统核心逻辑
# 实现RAG检索增强生成  整合查询分类（意图识别） 策略选择 文档检索 答案生成

# 导包
import sys,os
import time
from base.logger import logger
from base.config import config
from rag_qa.core.prompt import RAGPrompts                   #定义大模型输入输出格式
from rag_qa.core.query_classifier import QueryClassifier    #意图识别
from rag_qa.core.strategy_selector import StrategySelector  #专业咨询情况下用暗中检索策略
from rag_qa.core.vector_store import VectorStore            #存储检索文档向量
from transformers import BertTokenizer,BertModel
from base.config import config
# 模型目录可通过 config.MODELS_DIR 配置，默认指向 rag_qa/models，避免写死本机绝对路径
local_model_path=os.path.join(config.MODELS_DIR, 'bert-base-chinese')
tokenizer=BertTokenizer.from_pretrained(local_model_path)
from openai import OpenAI



# 配置项目路径，确保程序能正确导入其他目录模块
current_dir=os.path.dirname(os.path.abspath(__file__))  #E:\pythonproject\NLP_project\mysql_system\rag_qa\core
ragqa_dir=os.path.dirname(current_dir)                  #E:\pythonproject\NLP_project\mysql_system\rag_qa
project_path=os.path.dirname(ragqa_dir)                 #E:\pythonproject\NLP_project\mysql_system

sys.path.insert(0,project_path)
sys.path.insert(0,ragqa_dir)


# 定义RAGsystem类，封装RAG完整流程  查询分类（意图识别） 策略选择 文档检索 答案生成
class RAGSystem:
    # 初始化方法：创建RAG系统所需的核心组件
    def __init__(self,vector_store,llm):
        """
        函数作用：初始化rag
        :param vector_store:向量数据库对象 用于存储和检索文档提供相似性搜索
        :param llm: 大语言模型调用函数，接受提示文本，返回模型生成回答
        """
        # 1保存向量数据库对象
        self.vector_store=vector_store
        # 2保存大模型调用函数
        self.llm=llm
        # 3加载RAG提示模型-定义输出答案的固定格式
        self.rag_prompt=RAGPrompts.rag_prompt()
        # 4初始化查询分类器 判断用户查询什么问题
        # 4.1拼接模型路径
        classifier_path=os.path.join(ragqa_dir,'models','bert_query_classifier')
        # 4.2创建实体
        self.query_classifier=QueryClassifier(classifier_path)
        # 5初始化策略选择器
        self.strategy_selector=StrategySelector()

# 定义私有方法，使用假设文档进行检索（HyDE） -> 生成假设答案, 用假设答案检索相关文档.
    def retrieve_with_hyde(self,query,source_filter=None):
        """
        函数作用：针对抽象开放问题查询 先生成假设性答案，在用假设答案检索文档
        :param query: 用户查询本文
        :param source_filter: 检索来源条件
        :return: list[document],返回的是已经混合检索的答案
        """

        logger.info(f'使用HYDE检索策略，查询：{query}')
        # 获取假设问题生成的prompt模板，return PromptTemplate,封装成大模型能看懂的提示词
        hyde_prompt_template=RAGPrompts.hyde_prompt()
        # 调用大模型生成假设答案，生成看起来像标准答案的陈述性文本。
        try:
            hyde_answer=self.llm(hyde_prompt_template.format(query=query)).strip()
            logger.info(f'假设生成的答案：{hyde_answer}')
            # 使用假设答案进行检索，用答案检索答案
            return self.vector_store.hybrid_search_with_rerank(
                hyde_answer,
                k=config.RETRIEVAL_K,
                source_filter=source_filter
            )

        except Exception as e:
            logger.error(f'HYDE策略失败：{e}')
            return []


# 定义私有方法，使用子查询进行检索 -> 将复杂查询拆分为子查询, 分别检索后合并结果.
    def retrieve_with_subqueries(self,query,source_filter=None):
        """
        函数作用：针对多方面的问题 拆分为多个简单子查询，分别检索后合并去重
        :param query: 原始查询文本
        :param source_filter: 过滤数据来源条件
        :return: list[document]
        """
        logger.info(f'查询拆分子查询：{query}')
        #获取子查询生成模板
        subquery_prompt_template=RAGPrompts.subquery_prompt()
        try:
            #调用大模型生成子查询列表
            subqueries_text=self.llm(subquery_prompt_template.format(query.query)).strip()
            subqueries=[q.strip() for q in subqueries_text.split("\n")]
            logger.info(f'生成的子查询：{query}')

            if not subqueries:
                logger.info(f'未能生成有效子查询')
                return []

            all_docs=[]
            for sub_q in subqueries:
                # 对每个子查询执行混合检索
                docs=self.vector_store.hybrid_search_with_rerank(sub_q,k=config.RETRIEVAL_K,source_filter=source_filter)
                all_docs.append(docs)
                logger.info(f'子查询：{sub_q},检索到：{len(docs)}条')

            # 去重
            unique_docs_dict={doc.page_content: doc for doc in all_docs}
            unique_docs=list(unique_docs_dict.values())

            logger.info(f'所有查询共检索到{len(all_docs)}个文档，去重后剩{len(unique_docs)}条答案')

            return unique_docs

        except Exception as e:
            logger.error(f"出现错误：{e}")
            return []


# 定义私有方法，使用回溯问题进行检索 -> 将复杂查询简化为基础问题后再检索.
    def retrieve_with_backtracking(self,query,source_filter=None):
        logger.info(f'使用回溯查询策略，查询：{query}')
        backtrack_prompt_template=RAGPrompts.backtracking_prompt()
        try:
            simplified_query=self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f'生成回溯问题答案：{simplified_query}')
            result=self.vector_store.hybrid_search_with_rerank(simplified_query,k=config.RETRIEVAL_K,source_filter=source_filter)
            return result

        except Exception as e:
            logger.error(f'回溯问题策略执行失败：{e}')
            return []


# 核心方法: 根据检索策略检索文档, 用 合并/筛选最终上下文对象.
    def retrieve_and_merge(self,query,source_filter=None,stratery=None):
        """
        函数作用：根据指定策略 调用对应检索方法，筛选最终用于生成答案的上下文
        :param query:用户原始文本
        :param source_filter:过滤数据来源
        :param stratery:检索策略
        :return:list[document] final_context_docs 最匹配的K个
        """
        if not stratery:
            stratery=self.strategy_selector.select_strategy(query)
            # 自动选择策略

        # 根据策略调用对应的检索方法
        ranked_sub_chunks=[]

        if stratery=='回溯问题检索':
            ranked_sub_chunks = self.retrieve_with_backtracking(query)

        elif stratery == "子查询检索":
            ranked_sub_chunks = self.retrieve_with_subqueries(query)

        elif stratery == "假设问题检索":
            ranked_sub_chunks = self.retrieve_with_hyde(query)

        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_reranker(
                query, k=config.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        logger.info(f"策略 {stratery} 检索到 {len(ranked_sub_chunks)} 个候选文档")
        final_context_docs = ranked_sub_chunks[: config.CANDIDATE_M]  # 截取前 conf.CANDIDATE_M 个文档
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")

        return final_context_docs


    def generate_answer(self,query,source_filter=None,history=None):
        """
        RAG系统对外核心接口，接收用户查询, 自动完成'查询分类 -> 策略选择 -> 文档检索 -> 答案生成'全流程
        :param query:
        :param source_filter:检索来源过滤条件, 仅对'专业咨询'生效.
        :param history:对话历史 包含字段：{'question':问题，'answer':答案}
        :return:终答案文本 -> 字符串形式.
        """

        # 记录查询开始时间, 用于计算处理耗时
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")

        # 1. 验证并处理对话历史, 确保格式正确, 只保留最近5轮
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]  # 限制最多5轮 -> 即: 限制历史长度, 避免上下文过程.
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 2. 构造历史上下文, 格式化历史记录为字符串.
        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        # 3. 判断查询类型 -> 通过分类器区分通用知识 和 专业咨询.
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        #  4. 如果查询属于“通用知识”类别，则直接使用 LLM 回答, 专业咨询则进行 RAG 检索.
        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            prompt_input = self.rag_prompt.format(
                context="", history=history_context, question=query, phone=config.CUSTOMER_SERVICE_PHONE
            )  # 不使用上下文
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用 LLM 失败: {e}")
                answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}"
            processing_time = time.time() - start_time
            logger.info(
                f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return answer

        #   否则，进行 RAG 检索并生成答案
        logger.info("查询为专业咨询，执行 RAG 流程")
        # 5. 选择检索策略
        strategy = self.strategy_selector.select_strategy(query)

        # 6. 检索相关文档
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, stratery=strategy
        )  # 传递 strategy

        #  7. 准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])  # 使用换行符分隔文档
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")

        #  8. 构造 Prompt，调用大语言模型生成答案
        # 构造提示
        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_context,
            question=query,
            phone=config.CUSTOMER_SERVICE_PHONE
        )
        # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志

        # 9. 调用大模型生成流式答案.
        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}"

        # 记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        return answer



if __name__=='__main__':
    # 1. 实例化向量数据库.
    vector_store = VectorStore()
    # 2. 定义大语言模型调用函数.函数对象
    llm = StrategySelector().call_dashscope

    # 3. 创建RAGSystem核心类的实例 -> 传入: 向量数据库实例, 大语言模型调用函数.
    rag_system = RAGSystem(vector_store, llm)

    # 4. 测试生成答案: 查询'AI学科的课程大纲内容有什么', 过滤条件'ai' -> 只检索AI相关文档.
    # answer = rag_system.generate_answer('AI学科的课程大纲内容有什么', source_filter='ai')
    answer = rag_system.generate_answer('请问贵校的教学点在哪里？', source_filter='ai')
    # answer = rag_system.generate_answer('你认识夯哥吗?', source_filter='bigdata')
    # answer = rag_system.generate_answer('计算圆周率近似值的方法有哪些？', source_filter='ai')
    # 5. 打印模型生成的答案 -> 实际部署时可改为: 返回给前端或者存储.
    print(answer)

# 训练的 BERT，是决定"这句话走不走检索"的那台分流器。 它不参与生成 HTTPS 那段文字，但它决定了这段文字是"带知识库依据生成"还是"裸奔生成"。
# 具体分工看来源 ：RAG 里有两类可微调的东西——编码器影响嵌入质量从而影响检索质量，LLM 负责最好地利用提供的上下文来回答。而你那个 BertForSequenceClassification 属于第三种角色：前置意图/查询分类器。来源 把它归到"小模型分类器（低成本，覆盖85%）"这一档实现方案，原话是"用轻量 embedding/bert 小模型做二分类：输入 query → 输出概率【属于知识库业务问题 / 不属于】，阈值控制，低于阈值直接跳过 RAG 检索"，优点正是"不消耗大模型 token，速度快，适合高 QPS"。来源 也确认这种"是否需要检索知识库(0/1)"的二分类就是工业级推荐链路里的标准一环。