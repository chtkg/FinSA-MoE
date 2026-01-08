# 训练双专家GLM模型的训练器类
# 涉及技术：参数高效微调、分布式训练、早停机制
# 数据流：数据流：CSV → DataFrame → Dataset → Tokenized → Batched

import os
import torch
import pandas as pd
import numpy as np
import importlib.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,           # 自动加载分词器
    TrainingArguments,       # 训练参数配置
    Trainer,                # 训练器基类
    DataCollatorWithPadding, # 动态填充数据
    EarlyStoppingCallback,    # 早停回调
    default_data_collator
)
from datasets import Dataset, DatasetDict
import yaml
from .model import DualExpertGLMClassifier
# from .data_augmentation import ChineseFinancialAugmentation
from .loss_functions import AdaptiveBalancedLoss
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from .metrics import compute_classification_metrics



class GLMFinancialTrainer:
    """GLM-Z1-9B金融情感分析训练器"""

    def __init__(
            self,
            config_path: str = r"/home/pc/WorkSpace/liutao/FinSA-MoE/config/training_config.yaml",
            dataset_type: str = "forum"  # "news" or "forum"
    ):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dataset_type = dataset_type

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True,
            padding_side="left"    # GLM模型特性：左侧填充,与GPT类模型一致
        )

        # 初始化模型
        self.model = DualExpertGLMClassifier(
            base_model_path=self.config['model']['name'],
            num_labels=self.config['model']['num_labels'],
            lora_config=self.config['lora']
        )
        # 依据当前dataset_type（"news"/"forum"）激活并设置可训练的LoRA适配器
        if hasattr(self.model, "prepare_for_training"):
            self.model.prepare_for_training(self.dataset_type)

        # 启用梯度检查点,用时间换空间，重新计算中间激活值而不是存储，减少显存占用约30-50%
        # self.model.base_model.gradient_checkpointing_enable()

        # 数据增强器
        # self.augmenter = ChineseFinancialAugmentation()

    def load_data(self, data_dir: str):
        """加载数据"""
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        valid_df = pd.read_csv(f"{data_dir}/valid.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")

        return train_df, valid_df, test_df

    def prepare_dataset(self, train_df, valid_df, test_df):
        """准备数据集"""
        # 标签映射
        label_map = {'利空': 0, '中性': 1, '利好': 2}
        # if self.dataset_type == "news":
        #     label_map = {'利空': 0, '中性': 1, '利好': 2}
        # else:
        #     label_map = {'消极': 0, '中性': 1, '积极': 2}

        # 数据增强
        # if self.config['augmentation']['enable']:
        #     target_dist = self.config['augmentation'][f'{self.dataset_type}_target_dist']
        #     texts, labels = self.augmenter.targeted_augmentation(
        #         train_df['text'].tolist(),
        #         train_df['label'].tolist(),
        #         target_dist
        #     )
        #     train_df = pd.DataFrame({'text': texts, 'label': labels})

        # 批量处理文本，转换为模型输入格式
        def preprocess_function(examples):
            if self.dataset_type == "forum":
                prompts = [
                    f"分析以下股吧评论的情感倾向：{text}\n情感分类："
                    for text in examples['text']
                ]
            else:
                prompts = [
                    f"分析以下金融新闻的市场情感：{text}\n情感分类："
                    for text in examples['text']
                ]

            model_inputs = self.tokenizer(
                prompts,
                max_length=self.config['training']['max_length'],
                truncation=True,
                # padding='max_length'
            )

            model_inputs['labels'] = [
                label_map[label] for label in examples['label']
            ]

            return model_inputs

        # 转换为Dataset格式
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        valid_dataset = Dataset.from_pandas(valid_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        # 应用预处理，并在 map 之后直接移除原始列（如 text/label/__index_level_0__ 等）
        train_dataset = train_dataset.map(
            preprocess_function, batched=True, remove_columns=train_dataset.column_names
        )
        valid_dataset = valid_dataset.map(
            preprocess_function, batched=True, remove_columns=valid_dataset.column_names
        )
        test_dataset = test_dataset.map(
            preprocess_function, batched=True, remove_columns=test_dataset.column_names
        )

        return DatasetDict({
            'train': train_dataset,
            'validation': valid_dataset,
            'test': test_dataset
        })

    def compute_metrics(self, eval_pred):
        # 计算评估指标
        predictions, labels = eval_pred
        return compute_classification_metrics(predictions, labels)

    def train(self, data_dir: str, output_dir: str):

        seed = int(self.config["training"].get("seed", 42))
        set_global_seed(seed)
        print(f"[JointMoETrainer] Global seed = {seed}")
        """执行训练"""
        # 加载数据
        train_df, valid_df, test_df = self.load_data(data_dir)

        # 准备数据集
        datasets = self.prepare_dataset(train_df, valid_df, test_df)

        # 统计训练集各类别样本数
        label_map = {'利空': 0, '中性': 1, '利好': 2}
        # if self.dataset_type == "news":
        #     label_map = {'利空': 0, '中性': 1, '利好': 2}
        # else:
        #     label_map = {'消极': 0, '中性': 1, '积极': 2}
        train_labels = [label_map[label] for label in train_df['label']]
        samples_per_class = [train_labels.count(i) for i in range(self.config['model']['num_labels'])]
        # 初始化并绑定自适应平衡损失函数
        self.model.loss_fn = AdaptiveBalancedLoss(samples_per_class)

        # 计算一个稳定的 DS 配置绝对路径（相对 trainer.py 文件位置，而不是当前工作目录）
        DS_CONFIG_PATH = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "deepspeed_config.json"))
        HAS_DEEPSPEED = (importlib.util.find_spec("deepspeed") is not None) and os.path.exists(DS_CONFIG_PATH)

        # === 在创建 TrainingArguments 之前，加上这些兜底转换 ===
        def _as_float(x, name):
            try:
                return float(x)
            except Exception as e:
                raise TypeError(
                    f"{name} 应为浮点数，当前值={x} (type={type(x)}). 请检查 config/training_config.yaml") from e

        def _as_int(x, name):
            # 兼容 "500" / 500.0 这种情况
            try:
                return int(float(x))
            except Exception as e:
                raise TypeError(
                    f"{name} 应为整数，当前值={x} (type={type(x)}). 请检查 config/training_config.yaml") from e

        lr = _as_float(self.config['training']['learning_rate'], 'training.learning_rate')
        warmup_steps = _as_int(self.config['training']['warmup_steps'], 'training.warmup_steps')
        eval_steps = _as_int(self.config['training']['eval_steps'], 'training.eval_steps')
        save_steps = _as_int(self.config['training']['save_steps'], 'training.save_steps')
        logging_steps = _as_int(self.config['training']['logging_steps'], 'training.logging_steps')
        per_device_bs = _as_int(self.config['training']['per_device_batch_size'], 'training.per_device_batch_size')
        grad_accum = _as_int(self.config['training']['gradient_accumulation_steps'],
                             'training.gradient_accumulation_steps')

        print(f"[DEBUG] learning_rate={lr} ({type(lr)}) | warmup_steps={warmup_steps} | eval_steps={eval_steps}")

        use_fp16 = bool(torch.cuda.is_available())
        training_args_kwargs = dict(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config['training']['num_epochs'],  # 如果这里也可能被改成字符串，可以同样 _as_int
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,  # ← 使用转换后的 float
            warmup_steps=warmup_steps,  # ← 使用转换后的 int
            # fp16=use_fp16,
            bf16=True,
            gradient_checkpointing=False,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,  # ← 使用转换后的 int
            save_strategy="steps",
            save_steps=save_steps,  # ← 使用转换后的 int
            save_total_limit=self.config['training']['save_total_limit'],
            eval_strategy="steps",
            eval_steps=eval_steps,  # ← 使用转换后的 int
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            push_to_hub=False,
        )
        if HAS_DEEPSPEED:
            training_args_kwargs["deepspeed"] = DS_CONFIG_PATH
            
        seed = int(self.config["training"].get("seed", 42))
        training_args_kwargs["seed"] = seed
        training_args_kwargs["data_seed"] = seed
        
        training_args = TrainingArguments(**training_args_kwargs)

        # 自定义Trainer
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(
                    **inputs,
                    labels=labels,
                    expert_type=dataset_type  # 训练单个专家
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                return (loss, outputs) if return_outputs else loss

        dataset_type = self.dataset_type  # 为lambda函数捕获变量

        # 数据collator
        # data_collator = default_data_collator
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        # 在初始化 Trainer 之前（或在 prepare_for_training 末尾）打印一下可训练参数数量，确保 > 0
        trainable, total = 0, 0
        for n, p in self.model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"[DEBUG] trainable params = {trainable:,} / {total:,}")
        assert trainable > 0, "没有任何参数处于可训练状态，请检查 LoRA adapter 是否正确激活并设为可训练。"

        # 初始化Trainer
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[
                # 早停机制监控验证集性能，连续N个评估步骤无改善时停止训练，防止过拟合
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['training']['early_stopping_patience']
                ),
                TrainStatsCallback(output_dir, tag=self.dataset_type),
            ]
        )

        # 训练
        train_result = trainer.train()

        # 保存模型
        trainer.save_model()

        # 评估测试集
        test_results = trainer.evaluate(datasets['test'])

        print(f"Training completed for {self.dataset_type} expert")
        print(f"Test results: {test_results}")

        return train_result, test_results