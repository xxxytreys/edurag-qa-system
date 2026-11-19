

# 该脚本是: RAG_QA系统的主入口.

import os
import sys
from base.config import config as conf
from base.logger import logger
from rag_qa.core.document_processor import process_documents  # 导入处理文档的函数
from rag_qa.core.vector_store import VectorStore              # 向量存储和检索
from rag_qa.core.rag_system import RAGSystem                  # RAG_QA系统的核心代码
from openai import OpenAI                                     # 使用 OpenAI 接口


# todo 1.定义主函数main -> 系统入口逻辑, 根据模式执行数据处理或交互式查询.
def main(query_mode=True, directory_path="data"):
    """
    函数功能: 系统主函数, 控制两种运行模式.
    :param query_mode:
        数据处理模式: query_model=False  解析指定目录的文档, 分割为文档块并存入向量数据.
        交互式查询模式: query_model=True 接收用户问题, 调用RAG系统生成答案并展示.
    :param directory_path: 数据处理模式下的文档根目录, 默认是: ./data, 需包含各个学科的子目录, 例如: ai_data, java_data
    :return: 无
    """
    # 1 初始化LLM客户端 创建调用大模型的客户端生成答案
    try:
        client=OpenAI(api_key=conf.DASHSCOPE_API_KEY,base_url=conf.DASHSCOPE_BASE_URL)
    except Exception as e:
        logger.error(f'初始化OPENAI客户端失败：{e}')
        if query_mode:  # 查询模式下必须要有 LLM
            print("错误：无法初始化语言模型客户端，无法进入查询模式。")
            return
        client=None

    # 2. 定义LLM调用函数(仅在需要时定义和使用) -> 封装大语言模型调用逻辑, 供RAG系统使用.
    # 把组装好的Prompt交给大模型，拿回文本结果
    def call_dashscope(prompt):
        if not client:
            logger.error('LLM客户端初始化未成功')
            return f"错误: LLM客户端不可用"

        try:
            completion=client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role":"system","content":"你是一个有用的助手"},
                    {"role":"user","content":"你是一个有用的助手"},
                ]
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                logger.error(f'LLM API 调用返回无效响应或空消息')
                return "错误：LLM返回无效响应"

        except Exception as e:
            logger.error(f"LLM API (call_dashscope) 调用失败: {e}")
            return f"错误: 调用LLM失败 - {e}"


    # 3. 初始化 VectorStore向量数据库 -> 连接Milvus服务, 用于存储和检索向量文档.
    try:
        vector_store=VectorStore(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME
        )

    except Exception as e:
        logger.error(f'初始化 VectorStore 失败 (请检查 Milvus 连接配置): {e}')
        return


    # 4. 根据模式执行不同操作 -> 数据处理模式: 解析文档并添加到向量数据库.
    if not query_mode:
        # 数据处理模式
        logger.info("进入数据处理模式")
        total_chunks_added=0
        for source_dir in conf.VALID_SOURCES:
            dir_path=os.path.join(directory_path,f'{source_dir}_data')           # directory_path默认是: ./data,拼接data/ai_data
            if os.path.exists(dir_path):
                logger.info(f'开始处理目录：{dir_path}')
                try:
                    chunks=process_documents(
                        dir_path,
                        conf.PARENT_CHUNK_SIZE,
                        conf.CHILD_CHUNK_SIZE,
                        conf.CHUNK_OVERLAP
                    )

                    if chunks:
                        vector_store.add_documents(chunks)
                        total_chunks_added+=len(chunks)
                        logger.info(f"成功处理目录 {dir_path}，添加了 {len(chunks)} 个文档块")

                    else:
                        logger.info(f"目录 {dir_path} 未发现有效文档或处理结果为空")


                except Exception as e:
                    logger.error(f'处理目录 {dir_path} 时出错: {e}')

            else:
                logger.warning(f"目录 {dir_path} 不存在，跳过处理")

        logger.info(f"数据处理完成，共添加了 {total_chunks_added} 个文档块到向量存储")

    else:
        # --- 交互式查询模式 ---
        if not client:
            print("错误：查询模式需要语言模型客户端，但初始化失败。")
            return

        logger.info("进入交互式查询模式...")

        try:
            pass
        except Exception as e:
            logger.error(f"初始化 RAGSystem 失败: {e}")
            return

        valid_sources = conf.VALID_SOURCES
        print("\n欢迎使用 EduRAG 交互式查询系统！")
        print(f"支持的学科类别：{valid_sources}")
        print("输入您的问题，或输入 'exit' 退出。")

        while True:
            query=input("\n请输入您的问题")

            if query=='exit':
                logger.info(f'退出查询模式')
                print('再见')
                break

            source_filter_input=input(f'请输入学科（{"/".join(valid_sources)}）:').strip()
            source_filter=None

            # 判断学科是否是可以处理的学科，是，就赋值给最终返回值
            if source_filter_input:
                if source_filter_input in valid_sources:
                    source_filter=source_filter_input
                    logger.info(f"用户选择了学科过滤: {source_filter}")

                else:
                    logger.warning(f"无效的学科类别 '{source_filter_input}'，将不过滤")
                    print(f"提示：输入的学科 '{source_filter_input}' 无效，将不过滤。")

            # 返回答案
            try:
                print("正在生成答案，请稍候...")
                ragsystem=RAGSystem(vector_store,client)
                answer=ragsystem.generate_answer(query,source_filter=source_filter)

                print("-" * 30)
                print(f"问题: {query}")
                print(f"回答: {answer}")
                print("-" * 30)


            except Exception as e:
                logger.error(f"处理查询 '{query}' 时失败: {str(e)}")



# todo 2. 程序的主入口: 解析命令参数, 决定运行模式 -> 处理处理 / 查询模式
if __name__ == "__main__":
    import argparse

    # 1. 创建参数解析器, 为解析器设置'描述信息', 当用户在终端执行.
    parser = argparse.ArgumentParser(description="EduRAG System Main Entry Point")
    # 2. 添加数据处理模式参数.    --data-processing -> 意思是: 进入数据处理模式.
    parser.add_argument('--data-processing', action='store_true',
                        help='Run in data processing mode instead of query mode.')
    # 3. 添加数据目录参数. --data-dir -> 含义: 指定数据目录.
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to the data directory.')
    # 4. 解析命令行参数.
    args = parser.parse_args()
    # 5. 根据参数调用main函数 -> query_mode为False表示数据处理模式, 否则为查询模式.
    main(query_mode=(not args.data_processing), directory_path=args.data_dir)
