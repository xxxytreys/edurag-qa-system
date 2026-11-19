# 用于：通过RAGAS框架对检索和生成部分评估
# 生成部分：忠诚度 答案相关性
# 检索部分：上下文精确度 上下文召回率

import sys
import types
import json
import os
import pandas as pd

# 伪造一个模块，防止 ragas 强行导入已删除的 vertexai 报错
fake_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
fake_vertexai.ChatVertexAI = object  # 给个占位符就行
sys.modules["langchain_community.chat_models.vertexai"] = fake_vertexai


from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings,ChatOpenAI


# 密钥只从环境变量读取，源码中不保留真实 Key
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")



# 加载数据集
with open('rag_evaluate_data.json','r',encoding='utf-8') as f:
    data=json.load(f)
# print(data[0],len(data))  #<class 'list'> 30


# 转换数据格式 将原始数据转成RAGAS所需的数据
eval_data={
    'question':[item['question'] for item in data],   #[问题1，问题2]
    'answer':[item['answer'] for item in data],         #[答案1，答案2]
    'contexts':[item['context'] for item in data],          #[]
    'ground_truth':[item['ground_truth'] for item in data]  #[]
}
# print(len(eval_data),type(eval_data))   #4 <class 'dict'>


# 转化为dataset数据集
dataset=Dataset.from_dict(eval_data)
# print(len(dataset),dataset[0])  #30
# {'question': '人工智能就业课的课程版本是什么？', 'answer': '人工智能就业课的课程版本是V6.0。', 'contexts': ['人工智能学科全新升级——人工智能开发V6.0课程。'], 'ground_truth': 'V6.0'}




# 配置评估模型 选择LLM和嵌入模型
# 版本1 使用openai 用gpt-4模型
# llm=ChatOpenAI(model='gpt-4',openai_api_key="")
# embedding=OpenAIEmbeddings(open_ai_key="")

# 版本2 使用ollama 无需token
# 1初始化ollama聊天模型
# llm=ChatOllama(model='qwen2.5:7b',base_url='http://loaclhost:11434')
#
# # 2初始化嵌入模型
# embeddings=OllamaEmbeddings(model='qwen2.5:7b',base_url='http://loaclhost:11434')
#
# # 计算各项评估指标
# result= evaluate(
#     dataset=dataset,
#     metrics=[
#         faithfulness,
#         answer_relevancy,
#         context_precision,
#         context_recall
#     ],
#     llm=llm,
#     embeddings=embeddings
# )

# 版本3：千问大模型
llm = ChatOpenAI(
    model="qwen-plus",              # 打分建议用能力强的，如 qwen-max
    openai_api_key=QWEN_API_KEY,
    openai_api_base=QWEN_BASE_URL,  # 关键：指向千问兼容模式地址（等价于 base_url=）
    temperature=0,                  # 评估任务要稳定，温度设 0
)

# 2) 嵌入模型（answer_relevancy 算语义相似度要用），千问用 text-embedding-v3
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    openai_api_key=QWEN_API_KEY,
    openai_api_base=QWEN_BASE_URL,
)

# 计算各项评估指标
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
)
# 评估结果
print(f'RAGAS评估结果：{result}')






