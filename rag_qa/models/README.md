# rag_qa/models

本目录存放**本地模型权重**，总计约 6.3 GB，不随仓库分发。目录结构必须与代码中的路径一致：

```
rag_qa/models/
├── bert-base-chinese/                           # BERT 底座，约 1.2 GB
├── bert_query_classifier/                       # 微调二分类模型，约 390 MB
├── bge-m3/                                      # 嵌入模型（稠密 + 稀疏），约 2.2 GB
├── bge-reranker-large/                          # CrossEncoder 重排模型，约 2.1 GB
└── nlp_bert_document-segmentation_chinese-base/ # 中文语义切分模型，约 390 MB
```

获取方式（任选其一）：

- HuggingFace / ModelScope 搜索同名模型下载
- 用 `huggingface-cli download <repo> --local-dir rag_qa/models/<name>`
- 公司/课程提供的离线包直接拷贝

> `bert_query_classifier` 若不存在，代码会自动回退到 `bert-base-chinese` 底座，功能可用但分类效果未调优；
> 也可执行 `python rag_qa/core/query_classifier.py` 用 `rag_qa/classify_data/model_generic_5000.json` 重新训练。
