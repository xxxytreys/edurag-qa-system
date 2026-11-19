# rag_qa/data

本目录存放**知识库原始文档**，向量库入库时会从这里读取。

仓库中不包含这些文件（含第三方课程资料版权，且体积较大）。运行前请自行放入，目录约定如下：

```
rag_qa/data/
└── ai_data/            # 按学科分类，目录名即 source_filter 取值
    ├── *.pdf
    ├── *.docx
    ├── *.pptx
    └── *.md / *.txt
```

支持的格式：PDF、Word（.docx）、PPT（.pptx）、图片（OCR）、纯文本 / Markdown。

入库方式见项目根目录 README 的「构建向量库」一节。
