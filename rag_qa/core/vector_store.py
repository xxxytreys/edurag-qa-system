# 该脚本用于向量检索和存储
# 核心流程：文档向量生成（BGEM3）——向量入库——混合检索（稠密+稀疏）——结果去重——重排序——返回文档
# 常用的检索策略：
#     1稠密
#     2稀疏


# 导入BGEM3 用于生成文档和查询向量表示
import os
# 强制要求所有加载本地模型的操作只读硬盘，绝对不联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from milvus_model.hybrid import BGEM3EmbeddingFunction
import torch.cuda  #GPU版本
# 导入数据库操作，客户端，数据类型，检索请求，排序器
from pymilvus import MilvusClient,DataType,AnnSearchRequest,WeightedRanker
# 导入document用于创建文档对象 统一文档数据格式
from langchain_core.documents import Document
# 导入crossencoder 用于重排序和NLI判断
from sentence_transformers import CrossEncoder
import hashlib  #用于生成唯一ID的哈希值
from rag_qa.core.document_processor import *
import sys
from FlagEmbedding import FlagReranker



# todo 1.路径配置模块 定位当前脚本，RAG核心目录，项目根目录，确保自定义模块可以导入
current_dir=os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
module_dir=os.path.dirname(current_dir)
# print(module_dir)
project_path=os.path.dirname(module_dir)
# print(project_path)
sys.path.insert(0,project_path)
from base.logger import logger
from base.config import Config


# todo 2.初始化全局配置对象
conf=Config()



# todo 3.定义vector store类，这个类封装数据库的核心操作，集合管理，文档入库，混合检索
class VectorStore:
    # todo 3.1
    def __init__(
            self,
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME
    ):
        # 初始化milvus核心设置
        self.collection_name =collection_name
        self.host=host
        self.port=port
        self.database=database

        # 初始化日志
        self.logger=logger

        # 3选择设备，优先GPU,再CPU
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'使用设备：{self.device}')

        # 4初始化BGE reranker模型优化搜索结果和相关性排序，重排
        reranker_path=os.path.join(module_dir,'models','bge-reranker-large')
        print(f'reranker_path:{reranker_path}')
        self.reranker=CrossEncoder(reranker_path,device=self.device)
        # self.reranker = FlagReranker(reranker_path,
        #                         use_fp16=True)

        # 5初始化BGEM3模型，用于生成文档和查询向量表示
        m3_path=os.path.join(module_dir,'models','bge-m3')
        print(f'm3_path:{m3_path}')
        self.embedding_function=BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            use_fp16=(self.device=='cuda'),  #在用GPU时，启用半精度计算，CPU禁用
            device=self.device
        )


        # 6获取BGE-M3稠密向量的维度，固定维度1024
        self.dense_dim=self.embedding_function.dim['dense']
        # print(f'稠密向量维度：{self.dense_dim}')

        # 7初始化milvus客户端，建立数据库链接
        self.client=MilvusClient(uri=f'http://{self.host}:{self.port}',db_name=self.database)

        # 8调用私有方法创建新集合
        self._create_or_load_collection()




    # todo 3.2调用私有方法，创建新集合
    def _create_or_load_collection(self):
        # 1. 检查指定集合是否已存在(避免重复创建)
        if not self.client.has_collection(self.collection_name):
            # 走这里, 说明集合不存在, 需要创建集合.
            # 2. 创建集合 Schema，禁用自动 ID(手动用文档哈希作为主键)，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 3.创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}        # 构建索引时,丢弃20%的低权重值(减少存储,噪声, 精度影响较小)
            )

            # 4. 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 5.记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")

        # 6.如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 7. 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)



    # todo 3.3定义函数 将文本转化成向量并插入milvus
    def add_documents(self,documents):
        # 1提取所有文档内容
        texts=[doc.page_content for doc in documents]  #格式：['文档1内容‘，’文档2内容‘]

        # 2使用BGE-M3模型生成文档向量
        # 输入：文档列表text 输出：字典，含稠密和稀疏向量
        embeddings=self.embedding_function(texts)

        # 3初始化空列表 将数据插入data
        data=[]

        # 4遍历每个文档 组装插入数据
        for i,doc in enumerate(documents):
            # 生成唯一ID，文档哈希值，hexdigest()将哈希结果转成16进制适合作为ID
            text_hash=hashlib.md5(doc.page_content.encode('UTF-8')).hexdigest()
            # 处理稀疏向量 返回的是矩阵，需要转成milvus支持的
            sparse_vector={}  #初始化是字典
            # 获取第i个文档的稀疏矩阵
            row=embeddings['sparse'][[i]]
            # print(f'稀疏向量行:{row},shape:{row.shape}')
            # (0,5) 0.029   --(行号, 列号) 权重    shape:(1,250002)250002 是模型词表（Vocabulary）的大小，你可以把它当成模型认识的一个包含 25 万个“词/字/子词（Token）”的超级大字典
            #  (0, 203593)	0.23054850101470947,shape:(1, 250002)

            # 获取稀疏向量的非零索引和对应权重
            indics=row.indices
            # print(f'非零索引列表：{indics}')
            values=row.data  #获取稀疏向量的非零值的权重别表
            # [4      5      6     12     15     16     37     43    106    247
            #  264    339    470    527    571    573    575    594    844    963
            #  1441   1635   2729   3272   3344   3742   4696   5001   7081   7200
            #  7272   7702   8107   8218   8316   8335   9449  11003  11394  12833
            #  13293  15635  17072  17866  19186  19541  20486  20798  21001  21245
            #  22566  22693  23556  23708  23906  24450  25953  27616  28748  29606
            #  30926  33542  35073  37150  38046  38274  40513  41953  50317  54553
            #  71277  75092  79178  81492  83658  97567 103942 118484 119613 121318
            #  125389 136229 146563 158204 171126 171815 182426 189061 203593]

            # 组装稀疏向量字典
            for idx,value in zip(indics,values):
                sparse_vector[idx]=value
                print(f'稀疏向量字典：{sparse_vector},长度：{len(sparse_vector)}')

                # 扩展：查看稠密向量的维度
                # print(f'稠密向量:{embeddings["dense"][i]},稠密向量维度：{len(embeddings["dense"][i])}')

            # 4.3组装带条数据
            data.append({
                "id": text_hash,  # 唯一ID(md5哈希)
                "text": doc.page_content,  # 文档内容
                "dense_vector": embeddings["dense"][i],  # 稠密向量(BGE-M3生成)
                "sparse_vector": sparse_vector,  # 稀疏向量(组装后的字典形式)
                "parent_id": doc.metadata["parent_id"],  # 父文档ID(从元数据获取, 例如: doc_0_parent_0)
                "parent_content": doc.metadata["parent_content"],  # 父文档内容(从元数据获取, 用于: 上下文补充)
                "source": doc.metadata.get("source", "unknown"),  # 学科类别(从元数据获取, 例如: ai)
                "timestamp": doc.metadata.get("timestamp", "unknown")  # 时间戳(从元数据获取)
            })

        # 5插入milvus
        if data:
            # 使用upsert或者insert
            self.client.insert(collection_name=self.collection_name,data=data)
            # 打印日志
            self.logger.info(f'已插入数据{len(data)}条')





    # todo 3.4定义函数，混合检索（稀疏+稠密）+重排序，返回精准答案
    #小块检索、大块生成,返回父文档
    def hybrid_search_with_reranker(self,query,k=conf.RETRIEVAL_K,source_filter=None):
        """
               该函数用于执行: 混合检索(稠密 + 稀疏向量) + 结果重排序, 返回精准父文档
               :param query: 用户查询文本, 例如: 'AI学科的课程内容是什么'
               :param k: 混合检索返回的TopK子块数量
               :param source_filter: 学科过滤条件, 例如: ai, 仅检索该学科的文档, 默认None不过滤.
               :return: 重排序后的Top-M个父文档列表(M是从conf中获取的)
               """
        # 1.使用 BGE-M3 嵌入函数生成查询的嵌入,把文本变成列表喂给模型生成向量
        query_embeddings = self.embedding_function([query])
        # 2.获取查询的稠密向量，[0]————list[float]，， 返回的字典里，dense_vecs 是一个 numpy 二维数组[[]]
        dense_query_vector = query_embeddings["dense"][0]

        # 3.初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # 3.1 获取查询稀疏向量的第 0 行数据 -> 仅1个查询, 故取第0行.
        row = query_embeddings["sparse"][[0]]
        # 3.2 获取稀疏向量的非零值索引
        indices = row.indices  # 索引列表
        # 3.3 获取稀疏向量的非零值
        values = row.data  # 权重列表
        # 3.4 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        # 4构建检索过滤表达式 如果sourcefilter有值等于它,学科
        filter_expr=f'source == "{source_filter}"' if source_filter else None
        # print(f'filter_expr:{filter_expr}')
        # filter_expr: source == "ai"

        # 5构建稠密向量检索请求
        dense_request=AnnSearchRequest(
            data=[dense_query_vector],
            anns_field='dense_vector',  #检索向量字段
            param={'metric_type':'IP',"params":{'nprobe':10}},
            limit=k,            #返回topk结果
            expr=filter_expr    #应用过滤表达式
        )
        # 6构建稀疏向量检索请求
        sparse_request=AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field='sparse_vector',  #检索向量字段
            param={'metric_type':'IP',"params":{}},
            limit=k,            #返回topk结果
            expr=filter_expr    #应用过滤表达式
        )

        # 7创建加权排序器，融合稠密稀疏，按加权计算结果
        ranker=WeightedRanker(1.0,0.8)

        # 8执行混合检索
        results=self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request,sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=['text','parent_id','parent_content','source','timestamp']
        )[0]
        # print(f'result:{results},类型：{type(results)},数据量：{len(results)}')
        #[{'id': '837980a4d3bc702a70cefab6ea8e663a', 'distance': 0.4412883222103119, 'entity':{'text','parent_id','parent_content','source','timestamp'}}],类型：<class 'list'>,数据量：3

        # 9 将结果转化成langchain document对象,从上面字典的形式变成[document(metadata={}),   ]对象
        sub_chunks=[self._doc_from_hit(hit['entity']) for hit in results]
        # print(f'子块列表：{sub_chunks},数据量：{len(sub_chunks)}')
        #[document1,2,3],数据量：3    Document(metadata=...., page_content= )

        # 10从子块中提取去重的父文档 避免父块多个子块重复返回
        parent_docs=self._get_unique_parent_docs(sub_chunks)

        # 11重排 父文档数量小于2跳过直接返回，数量=1直接返回
        if len(parent_docs)<2:
            return parent_docs[:conf.CANDIDATE_M]

        # 12父文档数量大于2执行重排，提升相关性，走到这一步就不是1了
        if parent_docs:
            # 构建查询-文档 配对列表 ，重排序模型需要改格式输入 每个配对为【query,doc_content】
            pairs=[[query,doc.page_content] for doc in parent_docs]
            # 计算相关得分
            scores=self.reranker.predict(pairs)
            print(f'相关得分：{scores}')
            # [0.9687488 0.9687488]
            # 按得分降序排序，将得分和父文档配对，表示降序sorted（数据，reverse=True）
            # zip 把两个长度相同的列表按下标一一对应，打包成元组。  sorted 对元组列表排序时，默认比较每个元组的第一个元素
            #sort的结果是[(0.87, 文档B), (0.55, 文档C), (0.31, 文档A)]
            #[doc for _, doc in ...] —— 解包并丢弃分数  _ 接住第一个位置的值（也就是分数），用下划线开头表示"这个值我不要，故意丢掉"；  [文档B, 文档C, 文档A]   # 只剩文档，且已按相关性从高到低排好
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), key=lambda x: x[0], reverse=True)]
        else:
            ranked_parent_docs=[]

        return ranked_parent_docs[:conf.CANDIDATE_M]



    # todo 3.5定义函数，从子块列表中提取去重的父文档
    def _get_unique_parent_docs(self,sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents=set()
        # 初始化列表用于存储唯一父文档
        unique_docs=[]
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容
            parent_content=chunk.metadata.get("parent_content",chunk.page_content)
            # 检查父块内容是否非空且非重复
            if parent_content and parent_content not in parent_contents:
                # 创建文件对象
                unique_docs.append(Document(page_content=parent_content,metadata=chunk.metadata))
                # 加入集合
                parent_contents.add(parent_content)
        # 返回去重后父文档列表
        return unique_docs




    # todo 3.6定义函数，将milvus转化为langchain document对象
    def _doc_from_hit(self,hit):
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id":hit.get("parent_id"),
                "parent_content":hit.get("parent_content"),
                "source":hit.get("source"),
                "timestamp":hit.get("timestamp")
            }
        )





if __name__=='__main__':
    vectorstore=VectorStore()

    # 2插入数据  2-4向量入库
    # dirtory_path='../data/ai_data'
    # # print(f'embedding_function.dim{vectorstore.embedding_function.dim}')
    # # 打印维度,embedding_function.dim{'dense': 1024, 'colbert_vecs': 1024, 'sparse': 250002}
    # # 密集向量：抓整体语义，避免关键词不同但是意思相近 的文本被漏掉
    # # COLBERT 抓词级上下文 避免同词异意误匹配的情况
    # # 稀疏向量抓住关键词权重 确保特定术语精准匹配  真正有语义贡献的非零项只有 43 个
    #
    # # 3获取所有文档
    # documents=process_documents(dirtory_path)
    # # 4加载到数据库
    # vectorstore.add_documents(documents)

    # 5测试混合检索
    query='LLM是什么'
    result=vectorstore.hybrid_search_with_reranker(query,source_filter='ai')
    print(result)