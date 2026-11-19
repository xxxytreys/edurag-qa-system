# 该模型用于：基于BERT模型实现查询分类器，将用户查询分为通用知识和专业咨询
# 核心流程：加载预训练BERT模型——数据预处理——模型训练——模型评估——预测查询类别
# 使用场景：在RAG系统，可用于判断是否需要检索专业知识

# 导包
import json
import os
import torch
import sys
from base.logger import logger
import numpy as np
from transformers import BertTokenizer,BertForSequenceClassification    #BERT分词器和序列分类模型
from transformers import Trainer,TrainingArguments                      #trainer:负责实际执行训练过程按照后者的参数  TrainingArguments：参数传递
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

# 导入路径
current_dir=os.path.dirname(os.path.abspath(__file__))
rag_qa_dir=os.path.dirname(current_dir)
project_path=os.path.dirname(rag_qa_dir)
sys.path.insert(0,project_path)

# 定义QueryClassifier类,封装BERT查询分类的完整流程，包括模型加载 训练 评估 预测
# class QueryClassifier():
#     # 初始化方法，配置模型路径，加载分词器.下面的函数都在为train_model做准备，train_model保存到的模型是预训练模型，之后init拿来用，要是之后有预训练模型就init，没有就预训练走下面的函数
#     def __init__(self,model_path='../models/bert_query_classifier'):
#         # 加载bert
#         self.pre_trained_model_path=f'{rag_qa_dir}/models/bert_query_classifier'
#         # 模型训练后保存路径
#         self.model_path=model_path
#         # 加载BERT分词器,将文本转为机器能读的数据
#         self.tokenizer=BertTokenizer.from_pretrained(self.pre_trained_model_path)
#         # 初始化模型
#         self.model=None
#         # 确认设备
#         self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # 记录日志
#         logger.info(f'使用设备：{self.device}')
#         # 定义标签，区分专业知识还是正常问题
#         self.label_map={"通用知识":0,"专业咨询":1}
#         # 加载模型
#         self.load_model()

class QueryClassifier():
    def __init__(self, model_path='../models/bert_query_classifier'):
        """
        :param model_path: 微调模型的保存路径。如果不传，则使用默认路径。
        """
        # 1. 直接使用传入的 model_path，不再自己拼 pre_trained_model_path
        self.model_path = model_path

        # 2. 如果传进来的 path 不存在，说明还没有微调模型
        #    回退到预训练底座（注意这里要和 RAGSystem 里拼的路径字符串完全一致）
        if not os.path.exists(self.model_path):
            # 假设 RAGSystem 和 QueryClassifier 都在 core 包下，向上回退一级拿到 ragqa_dir
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rag_qa_dir = os.path.dirname(current_dir)

            # 拼接预训练底座路径，注意拼写要和 RAGSystem 里的一模一样（比如 clasifier 或 classifier）
            self.model_path = os.path.join(rag_qa_dir, 'models', 'bert_query_classifier')
            logger.info(f"微调模型路径 {model_path} 不存在，回退到预训练底座: {self.model_path}")

        # 3. 加载分词器（使用统一后的 self.model_path）
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

        # 后续逻辑保持不变...
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'使用设备：{self.device}')
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        self.load_model()



    def load_model(self):
        """
               函数功能: 加载模型, 优先从self.model_path加载, 不存在则初始化基于 bert-base-chinese的新模型
               :return:
               """
        # 1. 检查模型路径是否存在 -> 即: 是否有已训练的模型.
        if os.path.exists(self.model_path):
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path,_fast_init=False)
            # 将模型移到指定设备
            self.model.to(self.device)

            # _____新增
            weight_init = self.model.classifier.weight.data.std().item()
            bias_init = self.model.classifier.bias.data.std().item()

            logger.info(f"分类头权重标准差: {weight_init:.5f}, 偏置标准差: {bias_init:.5f}")
            if abs(weight_init - 0.0227) < 0.001:
                logger.warning("⚠️ 分类头权重标准差约为 0.0227，判定为【底层随机初始化】，未加载到微调权重！")
            else:
                logger.info("✅ 分类头权重状态异常，判定为【已加载微调模型】")
            # ————————优化结束

            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 2. 若模型不存在, 初始化新模型
            # 参1: 模型路径,  参2: 二分类任务.
            self.model = BertForSequenceClassification.from_pretrained("../models/bert-base-chinese", num_labels=2)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")


    # 保存模型 和分词器
    def save_model(self):
        self.model.save_pretrained(self.model_path)   #save_pretrained保存函数已写好，存到model_path
        self.tokenizer.save_pretrained(self.model_path)


    # 数据预处理方法，将文本和标签转化为BERT模型所需的输入
    def preprocess_data(self,texts,labels):
        """
        函数功能: 对文本进行分词, 阶段, 填充, 将标签转换为: 数字.
        :param texts:
        :param labels:
        :return:
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,        # 超过最大长度时截断
            padding=True,           # 不足最大长度时填充.
            max_length=128,         # 最大长度
            return_tensors="pt"     # 返回张量
        )
        # 2. 标签转换: 将文本标签转换为数字
        return encodings, [self.label_map[label] for label in labels]


    # 创建数据集方法：自定义dataset类，封装：编码和标签
    def create_dataset(self,encodings, labels):
        # 定义内部类 继承torch.utils.data.Dataset
        class Dataset(torch.utils.data.Dataset):
            # 初始化方法: 接收编码和标签, 创建内部Dataset类.
            def __init__(self,encodings, labels):
                super().__init__()
                self.encodings = encodings
                self.labels = labels

            # 根据索引返回单条数据(编码 + 标签)
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            # 返回数据集长度.
            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)


    # 训练模型方法，加载数据集 预处理 配置参数
    def train_model(self,data_file="training_dataset_hybrid_5000.json"):
        """
        函数功能: 训练BERT分类模型, 区分查询分类为'通用知识' 和 '专业咨询'
        :param data_file: 数据集文件路径.
        :return: 无
        """
        # 1. 检查数据集文件是否存在.
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        # 2. 加载数据集, 从Json文件中读取查询文本和对应标签.
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]

        # 3. 提取文本和标签, 分离查询文本和对应的分类标签.
        texts = [item["query"] for item in data]  # 格式: [查询文本1, 查询文本2, 查询文本3, ...]
        labels = [item["label"] for item in data]  # 格式: [标签1, 标签2, 标签3, ...]

        # 4. 划分训练集和测试集. 80%用于训练, 20%用于验证. 固定随机种子确保结果可复现.
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # 5. 数据预处理: 将文本和标签转换为BERT模型所需的输入格式 -> 调用自定义的preprocess_data() 预处理数据的方法.
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        # 6. 创建数据集对象, 将编码 和 标签封装为PyTorch可识别的 Dataset对象.
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        # 7. 配置训练参数: 定义模型训练的超参数.
        training_args = TrainingArguments(
            # 设置模型和检查点保存的目录路径
            output_dir="./bert_results",
            # 设置训练的总轮数为3轮
            num_train_epochs=3,
            # 设置每个设备（GPU/CPU）上的训练批次大小为8
            per_device_train_batch_size=8,
            # 设置每个设备（GPU/CPU）上的评估批次大小为8
            per_device_eval_batch_size=8,
            # 设置学习率预热步数为20步，训练初期学习率从0逐渐增加到设定值, 逐步提升学习率, 避免初始震荡.
            warmup_steps=20,
            # 设置权重衰减系数为0.01，用于防止过拟合
            weight_decay=0.01,
            # 设置日志文件保存的目录路径
            # logging_dir="./bert_logs",
            # 设置每10个训练步骤记录一次日志
            logging_steps=10,
            # 设置评估策略为每个epoch结束后进行评估
            # evaluation_strategy="epoch",
            eval_strategy="epoch",
            # 设置模型保存策略为每个epoch结束后保存
            save_strategy="epoch",
            # 设置训练结束后加载最佳模型而非最后一个模型
            load_best_model_at_end=True,
            # 设置最多保存1个检查点文件，超出时自动删除旧的
            save_total_limit=1,
            # 设置用于判断最佳模型的指标为评估损失
            metric_for_best_model="eval_loss",
            # 禁用FP16混合精度训练，使用FP32精度 简化配置, 需要GPU支持
            fp16=False,
        )

        # 8. 初始化 Trainer: 封装模型, 训练参数, 数据集, 评估指标计算方法.
        trainer = Trainer(
            # 传入要训练的模型实例
            model=self.model,
            # 传入上面定义的训练参数配置
            args=training_args,
            # 传入训练数据集
            train_dataset=train_dataset,
            # 传入验证数据集，用于训练过程中评估模型性能
            eval_dataset=val_dataset,
            # 传入计算评估指标的函数，用于在验证集上计算准确率等指标
            compute_metrics=self.compute_metrics  # 自定义评估指标(准确率)
        )
        # 9. 训练模型
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        self.save_model()

        # 10. 评估模型
        self.evaluate_model(val_texts, val_labels)


    # 评估指标方法，验证集的准确率
    def compute_metrics(self, eval_pred):
        """
        函数功能: 计算分类任务的评估指标(准确率)
        :param eval_pred:  包含模型输出logits 和 真实标签的元素.
        :return: 准确率字典.
        """
        # 1. 模型输出的 logits 和 真实标签.
        logits, labels = eval_pred
        # 2. 取logits最大值对应的索引作为预测结果.
        predictions = np.argmax(logits, axis=-1)
        # 3. 计算准确率: 预测正确的样本占比
        accuracy = (predictions == labels).mean()
        # 4. 返回准确率字典.
        return {"accuracy": accuracy}


    # 评估模型方法 生成分类报告和混淆矩阵，全面分析模型性能
    def evaluate_model(self, texts, labels):
        """
        函数功能: 在给定文本和标签上评估模型, 输出分类报告和混淆矩阵.
        :param texts:  待评估的文本列表
        :param labels: 文本对应的真实标签(数字形式)
        """
        # 1. 对文本进行编码, 仅处理文本, 因为标签已经转换为数字了.
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # 2. 创建评估数据集.
        dataset = self.create_dataset(encodings, labels)

        # 3. 使用Trainer进行预测.
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)

        # 4. 解析预测结果: 取logits最大值对应的索引作为预测结果.
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels  # 直接使用数字标签

        # 5. 记录评估结果到日志.
        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"]  # 指定标签名称, 使报告更易读.
        ))

        # 混淆矩阵: 展示预测标签 和 真实标签的匹配情况.
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))


# 预测类别方法 -> 对单个查询文本进行分类预测.
    def predict_category(self, query):
        """
        函数功能: 对单个查询文本进行类别预测 -> 通用知识 或者 专业咨询.
        :param query: 待分类的查询文本
        :return: 类别名称(文本标签)
        """
        # 1. 检查模型是否加载.
        if self.model is None:
            # 模型未加载
            logger.error("模型未训练或加载!")
            return '模型未训练通用知识'

        # 2. 对查询文本进行编码.
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # 3. 将编码数据迁移到指定设备.
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # 4. 模型推理 -> 不计算梯度, 提高效率.
        with torch.no_grad():
            # 4.1 获取模型输出 -> 包含logits
            outputs = self.model(**encoding)
            # 4.2 获取预测结果 -> 获取logits最大值对应的索引作为预测结果.
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # 5. 返回预测结果标签
        return '专业咨询' if prediction == 1 else '通用知识'


if __name__=="__main__":
    # 1实例化查询分类器
    query_classify=QueryClassifier()
    # 2训练模型
    # data_file='../classify_data/model_generic_5000.json'
    # query_classify.train_model(data_file)

    # 3 示例预测: 对查询'AI的课程大纲是什么'进行分类
    result = query_classify.predict_category('AI课程的项目是独立完成还是小组合作？')
    print(result)