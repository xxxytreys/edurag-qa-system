# 该脚本用于处理文本
# 该脚本用于: 处理输入文档, 分块并准备向量存储.

# 导包
# import sys
# import io
import os       # 文件操作
# if sys.platform == "win32":
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
#     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
#     os.system('chcp 65001')

import sys
import ctypes

# 强制设置控制台输出和输入编码为 UTF-8
ctypes.windll.kernel32.SetConsoleOutputCP(65001)
ctypes.windll.kernel32.SetConsoleCP(65001)

# 同时包装标准输出流
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


from langchain_community.document_loaders import TextLoader     # 加载纯文本文件(.txt)用的工具
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader    # 加载(.md)文件的工具, 保留标题,列表等结构化信息
from langchain_text_splitters import MarkdownTextSplitter       # 针对 .md文档的文本分割器.
from datetime import datetime               # 用于记录文档加载/处理时间.
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter        # 适配中文教育文本层级结构
# from rag_qa.edu_text_spliter import AliTextSplitter                     # 阿里中文优化(教育场景专用文本分割器)
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader  # 教育场景专用文档加载器(支持OCR, 处理扫描版课件, 图片文字等.)
from base.config import Config      # 配置类
from base.logger import logger      # 日志类

# todo 1. 定义配置类 -> 获取配置参数.
# 1. 创建配置实例.
conf = Config()
# 2. 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

# todo 2. 核心加载函数模块 -> 从指定目录递归加载所有支持类型文件, 添加原数据并返回文档列表.
# def load_documents_from_directory(directory_path):
#     """
#     从指定目录(含子目录)加载所有支持类型的文件, 为每个文档添加原数据
#     :param directory_path: 目标文件夹的绝对路径.
#     :return: 加载完成的文档列表, 每个元素为 LangChain Document对象, 含: page_content 和 metadata
#     """
#     # 1. 初始化空列表 -> 用于存储所有加载成功的文档.
#     documents = []
#     # 2. 获取支持的文件扩展名结合
#     supported_extensions = document_loaders.keys()
#     # 3. 从目录名提取 '学科类别'原数据, 例如: 'ai_data' -> 提取'ai'作为source
#     source = os.path.basename(directory_path).replace('_data', '')
#     # 4. 递归遍历目标目录及其所有子目录,  注意: os.walk()返回 当前目录路径, 子目录列表, 文件列表.
#     for root, _, files in os.walk(directory_path):
#         # 5. 遍历当前目录下的每个文件, 逐一处理.
#         for file in files:
#             # 5.1 构造文件的绝对路径  root + 文件名, 避免相对路径问题.
#             file_path = os.path.join(root, file)
#             # print(f'file_path: {file_path}')        # 例如: ../data/ai_data\LLM基础知识.pdf
#
#             # 5.2 提取文件扩展名并转为小写 -> 统一格式.
#             file_extension = os.path.splitext(file)[1].lower()
#             # print(f'file_extension: {file_extension}')
#
#             # 5.3 若文件类型在支持列表中, 执行加载逻辑.
#             if file_extension in supported_extensions:
#                 try:
#                     # 5.3.1 根据文件扩展名获取对应的加载器 -> 创建加载器实例.
#                     loader_class = document_loaders[file_extension]
#                     # 5.3.2 实例化加载器.
#                     if file_extension == '.txt':
#                         loader = loader_class(file_path, encoding='utf-8')  # txt文件强制使用utf-8编码
#                     else:
#                         loader = loader_class(file_path)
#                     # 5.3.3 调用加载器的load()方法, 加载文件内容 -> 返回文档列表, 单个文件可能拆分成多个Document.
#                     loaded_docs = loader.load()
#                     # print(f'加载文件: {loaded_docs}')
#                     # 5.3.4 给每个加载的文档添加元数据 -> 便于后续RAG检索时过滤/溯源.
#                     for doc in loaded_docs:
#                         # 为文档添加学科类别元数据, 例如: ai
#                         doc.metadata["source"] = source
#                         # 为文档添加文件路径元数据
#                         doc.metadata["file_path"] = file_path
#                         # 为文档添加当前时间戳元数据, 精确到秒
#                         doc.metadata["timestamp"] = datetime.now().isoformat()
#
#                     # 5.3.5 将添加元数据后的文档列表添加到总列表中.
#                     documents.extend(loaded_docs)
#                     # 5.3.6 记录INFO日志
#                     logger.info(f'成功加载文件: {file_path}')
#                 except Exception as e:
#                     # 5.3.7 捕获异常 -> 输出异常信息, 继续处理下一个文件.
#                     logger.error(f'加载文件 {file_path} 失败: {str(e)}')
#             else:
#                 # 5.4 若文件不再支持列表, 记录警告日志.
#                 logger.warning(f'不支持的文件类型: {file_path}')
#
#     # 6. 返回加载完成的文档列表.
#     return documents

def load_documents_from_directory(directory_path):
    """
    从指定目录（含子目录）加载所有支持类型的文件 为每个文件添加原数据
    :param directory_path: 目标文件夹的绝对路径
    :return: 加载完成的文档列表 每个元素为langchain document对象 含page_content和metadata
    """
    # 1初始化空列表，用于存储所有加载成功的文档
    documents=[]
    # 2获取支持的文件扩展集合
    supported_extensions=document_loaders.keys()
    # print(supported_extensions)  #dict_keys(['.txt', '.pdf', '.docx', '.ppt', '.pptx', '.jpg', '.png', '.md'])
    # 3从目录名提取学科类别
    source=os.path.basename(directory_path).replace('_data','')
    # print(source)  #ai
    # 4递归遍历目标目录以及所有子目录 注意os.walk()返回当前目录路径 子目录列表 文件列表
    for root,_,files in os.walk(directory_path):
        # print(root,files)
        # ../ data / ai_data  ['LLM基础知识.pdf', '人工智能就业课课程大纲.docx', '人工智能就业课课程大纲.md']
        # 5遍历当下目录下的每个文件
        for file in files:
            # 5.1构建文件绝对路径 root+文件名
            file_path=os.path.join(root,file)
            # print(file_path)
            # ../ data / ai_data\LLM基础知识.pdf
            # ../ data / ai_data\人工智能就业课课程大纲.docx
            # ../ data / ai_data\人工智能就业课课程大纲.md
            # 5.2获取文件的扩展名转为小写
            file_extension=os.path.splitext(file)[1].lower()
            # print(file_extension)
            # .pdf
            # .docx
            # .md
            # 5.3若文件类型再支持列表中，执行加载
            if file_extension in supported_extensions:
                try:
                    # 5.3.1根据文件扩展名获取对应的加载器
                    loader_class = document_loaders[file_extension]
                    # 5.3.2实例化加载器
                    if file_extension == '.txt':
                        loader = loader_class(file_path, encoding='utf-8')  # TextLoader
                    else:
                        loader = loader_class(file_path)
                    # 5.3.3调用加载器的load()方法 加载文件内容 返回文档列表 单个文件可能拆分多个document
                    loaded_docs = loader.load()  # 加载文件内容

                    # 5.4给每个文件添加元数据——便于RAG检索过滤溯源
                    for doc in loaded_docs:
                        if isinstance(doc.page_content, str):
                            doc.page_content = doc.page_content.replace("\x00", "").strip()

                        if not doc.page_content:
                            continue

                        doc.metadata["source"] = source  # 为文档添加元数据-来源
                        doc.metadata["file_path"] = file_path
                        doc.metadata["timestamp"] = datetime.now().isoformat()  # 为文档添加元数据-时间戳

                    # 5.5将元数据加载到总列表
                    documents.extend(loaded_docs)

                    # 5.6日志记录
                    logger.info(f'数据文档 {file} 已加载')

                except Exception as e:
                    logger.warn(f'文件类型支持，但是出错{e}')
            else:
                logger.warning(f'{file_path}文件类型不支持')


        # print(f'加载文件个数：{len(loaded_docs)}')

    # 6返回文档列表
    return documents



# todo 3.文档处理核心函数 -> 加载文档后进行分层切分(父块 + 子块): 返回带元数据的子块列表.
# def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
#                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
#                      chunk_overlap=conf.CHUNK_OVERLAP):
#     """
#     加载文档并进行分层切分: 先切成大粒度父块, 再将父块切成小粒度子块, 子块关联父块元数据.
#     :param directory_path:  文档目录路径
#     :param parent_chunk_size: 父块切分长度(大粒度, 例如: 1200)
#     :param child_chunk_size: 子块切分长度(小粒度, 例如: 300)
#     :param chunk_overlap: 子块重叠长度, 确保上下文关联, 例如: 50
#     :return:
#         child_chunks: 切分后的子块列表(每个子块含父块关联信息)
#     """
#
#     # 1. 调用加载函数 -> 从指定目录加载所有文档
#     documents = load_documents_from_directory(directory_path)
#     # 记录加载的文档总数日志
#     logger.info(f"加载的文档数量: {len(documents)}")
#
#     # 2. 初始化父块和子块分词器（通用）
#     parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
#     child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
#     # 初始化 Markdown 专用分词器
#     markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
#     markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
#
#     # 3. 初始化子块列表, 存储最终切分结果.
#     child_chunks = []
#
#     # 4. 遍历每个原始文档 -> 带索引i, 用于生成唯一ID
#     for i, doc in enumerate(documents):
#         # 4.1 获取文件扩展名 -> 用于判断是否是MarkDown文件, 选择对应的切分器.
#         file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
#         is_markdown = (file_extension == '.md')     # 标记是否是MarkDown文件
#
#         # 4.2 根据文件类型选择切分器.
#         parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter       # 父块切分器
#         child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter          # 子块切分器
#         logger.info(f'处理文档: {doc.metadata["file_path"]}, 使用切分器: {"Markdown" if is_markdown else "ChineseRecursive"}')
#
#         # 4.3 将文档切分成父块 -> 大粒度, 保留较多上下文.
#         # parent_docs的格式: [父块1Document对象, 父块2Document对象, ...]
#         parent_docs = parent_splitter_to_use.split_documents([doc])     # 传入列表格式, split_documents()函数要的是可迭代类型.
#
#         # 5. 遍历每个父块 -> 父块切分成子块 -> 小粒度, 并于精准匹配.
#         for j, parent_doc in enumerate(parent_docs):
#             # 5.1 生成父块唯一id, 格式: doc_文档索引_parent_父块索引
#             parent_id = f"doc_{i}_parent_{j}"
#             # 5.2 给父块添加元数据 -> 唯一的ID 和 自身内容, 供子块关联.
#             parent_doc.metadata["parent_id"] = parent_id
#             parent_doc.metadata["content"] = parent_doc.page_content        # 存储父块完整内容.
#             # 如下输出格式为: 父块: ../data/ai_data\LLM基础知识.pdf, 索引: 9, 唯一ID: doc_0_parent_9
#             # print(f'父块: {parent_doc.metadata["file_path"]}, 索引: {j}, 唯一ID: {parent_id}\n\n')
#
#             # 5.3 将父块切分成子块 -> 小粒度, 并于精准匹配.
#             sub_chunks = child_splitter_to_use.split_documents([parent_doc])
#
#             # 6. 遍历每个子块 -> 添加父块关联信息.
#             for k, sub_chunk in enumerate(sub_chunks):
#                 # 6.1 添加子块关联父块信息 -> 父块ID, 父块内容, 子块内容.
#                 sub_chunk.metadata["parent_id"] = parent_id                         # 关联父块ID
#                 sub_chunk.metadata["parent_content"] = parent_doc.page_content      # 关联父块的完整内容.
#
#                 # 6.2 生成子块唯一的ID, 格式: 父块ID_child_子块索引.
#                 sub_chunk.metadata["id"] = f'{parent_id}_child_{k}'
#                 # 6.3 将子块添加到结果列表.
#                 child_chunks.append(sub_chunk)
#
#     # 7. 记录日志.
#     logger.info(f"切分完成的子块总数: {len(child_chunks)}")
#     # 8. 返回子块列表.
#     return child_chunks

def process_documents(directory_path,parent_chunk_size=conf.PARENT_CHUNK_SIZE,child_chunk_size=conf.CHILD_CHUNK_SIZE,chunk_overlap=conf.CHUNK_OVERLAP):
    """
    加载文件并分层切分 先切成大粒度 再将父类切成小粒度，子块，子块关联父块
    :param directory_path:文档路径
    :param parent_chunk_size:父块大小
    :param child_chunk_size:子块大小
    :param chunk_overlap:子块重叠长度
    :return:child_chunks:切分后的子块列表
    """
    #1  从指定目录加载所有文档
    documents=load_documents_from_directory(directory_path)  #上一个函数的返回是文档的内容，文档列表，返回文件内容
    # 记录日志
    logger.info(f'加载的文档数量：{len(documents)}')

    # 2 初始化父块和子块的分词器
    parent_splitter=ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)
    child_splitter=ChineseRecursiveTextSplitter(chunk_size=child_chunk_size,chunk_overlap=chunk_overlap)

    # 初始化md专用分词器
    markdown_parent_splitter=MarkdownTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)
    markdown_child_splitter=MarkdownTextSplitter(chunk_size=child_chunk_size,chunk_overlap=chunk_overlap)

    # 3初始化子块列表，存储最终切分结果
    child_chunks=[]

    # 4遍历每个原始文档 带索引i 生成一个唯一的id
    for i,doc in enumerate(documents):
        # 4.1获取文件扩展名。用于判断是否md文件，选择不同切分器
        file_extension=os.path.splitext(doc.metadata.get("file_path",""))[1].lower()   #按照.切分，1表示拿后面的后缀，转小写
        is_markdown=(file_extension =='.md')

        # 4.2根据文件类型选择切分器
        parent_splitter_to_use=markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use=markdown_child_splitter if is_markdown else child_splitter
        logger.info(f'处理文本：{doc.metadata["file_path"]},使用切分器：{MarkdownTextSplitter if is_markdown else ChineseRecursiveTextSplitter}')

        # 4.3将文档切分一层，成父块
        parent_docs=parent_splitter_to_use.split_documents([doc])
        # print(parent_docs)  #每个父块格式：原数据+一块的数据文本，[父块document对象1，父块document对象2，]

        # 5遍历每个父块
        for j,parent_doc in enumerate(parent_docs):
        # 5.1生成唯一的文档id格式：doc_文档索引_parent_父类索引
            parent_id=f'doc_{i}_parent_{j}'
            # 5.2给父块添加元数据，包含唯一ID 自身内容 供子块关联
            parent_doc.metadata['parent_id']=parent_id
            parent_doc.metadata['content']=parent_doc.page_content  #存储父块完整内容
            # print(f'父类：{parent_doc.metadata["file_path"]},索引：{j},唯一ID：{parent_id}\n\n')

            # 5.3切分父块
            sub_chunks=child_splitter_to_use.split_documents([parent_doc])

            # 6遍历
            for k,sub_chunk in enumerate(sub_chunks):
                # 6.1给子块添加父块关联信息
                sub_chunk.metadata['parent_id']=parent_id
                sub_chunk.metadata['parent_content']=parent_doc.page_content
                # 6.2生成子块唯一ID
                sub_chunk.metadata['id']=f'{parent_id}_child_{k}'
                child_chunks.append(sub_chunk)

    # 7记录日志
    logger.info(f'切分子块总数：{len(child_chunks)}')
    # 8返回子块列表
    return child_chunks

# todo n. 测试代码
if __name__ == '__main__':
    # 1. 定义测试目录路径 -> 目标文件夹, 含待加载的各类文件.
    # directory_path = '../data/ai_data'
    # # # 2. 调用核心加载函数 -> 加载目录下所有的支持类型文件.
    # documents = load_documents_from_directory(directory_path)
    # # 3. 打印结果 -> 生产环境下, 需要注释.
    # print(f'加载完成的文档总数: {len(documents)}')  #加载完成的文档总数: 3
    # print(f'前1个文档详情: {documents[0] if documents else "未加载到任何文档"}')

    # # 4. 调用文档处理函数 -> 加载指定目录并切分文档.
    chunks = process_documents('../data/ai_data', conf.PARENT_CHUNK_SIZE, conf.CHILD_CHUNK_SIZE, conf.CHUNK_OVERLAP)
    print(f'切分完成的子块总数: {len(chunks)}')
    print(f'前1个子块详情: {chunks[0] if chunks else "未切分到任何子块"}')
