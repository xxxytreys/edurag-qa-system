# 该脚本用于文本预处理

import jieba
from base import logger

# todo 1.定义文本预处理函数
def preprocess_text(text):
    # logger.info(f'开始处理文本{text}')
    try:
        return jieba.lcut(text.lower())

    except AttributeError as e:
        logger.error(f'文本预处理{text}失败:{e}')
        return []

# def preprocess_text(text: str) -> list[str]:
#     if not isinstance(text, str):
#         text = str(text)
#
#     text = text.replace('\r\n', '\n')
#     text = text.replace('\n', '')
#     text = text.replace('\r', '')
#     text = text.replace('\t', ' ')
#     text = text.strip()
#
#     try:
#         return jieba.lcut(text.lower())
#
#     except AttributeError as e:
#         logger.error(f'文本预处理{text}失败:{e}')
#         return []

if __name__=='__main__':
    print(preprocess_text("我是蜘蛛侠"))