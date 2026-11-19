# mysql_qa/data

本目录存放**标准问答库数据**，用于初始化 MySQL `jpkb` 表，也是 BM25 召回的语料来源。

仓库中不包含业务数据文件。运行前请自行准备：

```
mysql_qa/data/
└── JP学科知识问答.csv     # 至少包含 question、answer 两列（有 source 列更佳）
```

也可以替换成你自己的问答 CSV，注意列名与 `mysql_qa/db/mysql_client.py` 中的 `insert_data()` 保持一致。
