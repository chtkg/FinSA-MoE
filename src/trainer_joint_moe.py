import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
import importlib.util
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import yaml

from .model import DualExpertGLMClassifier
from .loss_functions import AdaptiveBalancedLoss
from .metrics import compute_classification_metrics
from src.utils_seed import set_global_seed
from src.train_stats_callback import TrainStatsCallback



class JointMoETrainer:
    """
    第二阶段：联合训练 MoE 路由 + 两个 LoRA 专家
    默认把 news / forum 两个数据集拼在一起训练。
    """

    def __init__(
        self,
        config_path: str,
        init_checkpoint: str = None,  # 用来初始化 MoE 模型的 checkpoint（可以给 news_expert）
    ):
        # 1. 加载配置
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 2. 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=True,
            padding_side="left",
        )

        # 3. 初始化 MoE 模型
        if init_checkpoint is not None:
            print(f"[JointMoETrainer] 初始化模型自 checkpoint: {init_checkpoint}")
            self.model = DualExpertGLMClassifier.from_checkpoint(
                ckpt_dir=init_checkpoint,
                base_model_path=self.config["model"]["name"],
                num_labels=self.config["model"]["num_labels"],
                lora_config=self.config["lora"],
            )
        else:
            print("[JointMoETrainer] 初始化模型自 base_model（不加载单专家权重）")
            self.model = DualExpertGLMClassifier(
                base_model_path=self.config["model"]["name"],
                num_labels=self.config["model"]["num_labels"],
                lora_config=self.config["lora"],
            )

    # -----------------------
    # 数据加载 / 预处理
    # -----------------------
    def load_joint_data(self, news_data_dir: str, forum_data_dir: str):
        """
        读取 news / forum 各自的 train/valid/test.csv，
        并添加 domain 列后按行拼接。
        """

        def _load_split(split: str):
            news_path = os.path.join(news_data_dir, f"{split}.csv")
            forum_path = os.path.join(forum_data_dir, f"{split}.csv")
            news_df = pd.read_csv(news_path)
            news_df["domain"] = "news"
            forum_df = pd.read_csv(forum_path)
            forum_df["domain"] = "forum"
            df = pd.concat([news_df, forum_df], ignore_index=True)
            return df

        train_df = _load_split("train")
        valid_df = _load_split("valid")
        test_df = _load_split("test")
        return train_df, valid_df, test_df

    def prepare_joint_dataset(self, train_df, valid_df, test_df):
        """
        把 text / label / domain -> tokenizer 输入 + labels + expert_labels
        expert_labels: 0=news, 1=forum
        """
        label_map = {"利空": 0, "中性": 1, "利好": 2}
        domain_map = {"news": 0, "forum": 1}

        def preprocess_function(examples):
            prompts = []
            expert_labels = []

            for text, dom in zip(examples["text"], examples["domain"]):
                if dom == "forum":
                    prompts.append(f"分析以下股吧评论的情感倾向：{text}\n情感分类：")
                    expert_labels.append(domain_map["forum"])
                else:
                    prompts.append(f"分析以下金融新闻的市场情感：{text}\n情感分类：")
                    expert_labels.append(domain_map["news"])

            model_inputs = self.tokenizer(
                prompts,
                max_length=self.config["training"]["max_length"],
                truncation=True,
            )

            # 文本标签 → [0,1,2]
            model_inputs["labels"] = [label_map[label] for label in examples["label"]]
            # 领域标签 → 路由监督标签
            model_inputs["expert_labels"] = expert_labels
            return model_inputs

        # pandas -> Dataset
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        valid_dataset = Dataset.from_pandas(valid_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        # map + 删除原始列
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        valid_dataset = valid_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=valid_dataset.column_names,
        )
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
        )

        return DatasetDict(
            {
                "train": train_dataset,
                "validation": valid_dataset,
                "test": test_dataset,
            }
        )

    # metric
    def compute_metrics(self, eval_pred):
        """
        标准写法：
        - predictions: logits (或包含 logits 的 tuple)
        - labels: 真实标签 id
        返回值交给 src/metrics.py 里的 compute_classification_metrics 计算 Accuracy/F1 等
        """
        import numpy as np

        predictions, labels = eval_pred

        # 1. 取出真正的 logits（有些情况 predictions 可能是 tuple/list）
        logits = predictions
        if isinstance(logits, (tuple, list)):
            for item in logits:
                if hasattr(item, "shape"):
                    logits = item
                    break

        # 2. logits -> 预测类别 id（一维）
        logits = np.array(logits)
        preds = logits.argmax(axis=-1)

        # 3. labels 转成一维 numpy 数组
        labels = np.array(labels)
        if labels.ndim > 1:
            # (N,1) 或 one-hot 情况，都压成 (N,)
            if labels.shape[-1] > 1:
                labels = labels.argmax(axis=-1)
            else:
                labels = labels.reshape(-1)

        # 4. 长度不一致就裁成一样长（极端防御）
        if preds.shape[0] != labels.shape[0]:
            n = min(preds.shape[0], labels.shape[0])
            preds = preds[:n]
            labels = labels[:n]

        # 5. 调用原来的指标函数（参数顺序：preds, labels）
        return compute_classification_metrics(preds, labels)



    # 参数冻结/解冻工具
    def _freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _unfreeze_router_and_classifier(self):
        # 路由器
        for p in self.model.router.parameters():
            p.requires_grad_(True)
        # MoE 分类头
        if hasattr(self.model, "classifier"):
            for p in self.model.classifier.parameters():
                p.requires_grad_(True)

    def _unfreeze_lora_router_classifier(self):
        # 先全部冻结
        self._freeze_all()
        # 再打开 LoRA + 路由器 + 分类头
        for name, p in self.model.named_parameters():
            # 简单粗暴：名字里带 "lora" 的都放开
            if "lora" in name:
                p.requires_grad_(True)

        self._unfreeze_router_and_classifier()


    # TrainingArguments 构建
    def _build_training_args(
        self,
        output_dir: str,
        max_steps: int = None,
        num_train_epochs: float = None,
    ) -> TrainingArguments:
        DS_CONFIG_PATH = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "config", "deepspeed_config.json"
            )
        )
        HAS_DEEPSPEED = (importlib.util.find_spec("deepspeed") is not None) and os.path.exists(
            DS_CONFIG_PATH
        )

        def _as_float(x, name):
            try:
                return float(x)
            except Exception as e:
                raise TypeError(
                    f"{name} 应为浮点数，当前值={x} (type={type(x)})"
                ) from e

        def _as_int(x, name):
            try:
                return int(float(x))
            except Exception as e:
                raise TypeError(
                    f"{name} 应为整数，当前值={x} (type={type(x)})"
                ) from e

        cfg = self.config["training"]
        lr = _as_float(cfg["learning_rate"], "training.learning_rate")
        warmup_steps = _as_int(cfg["warmup_steps"], "training.warmup_steps")
        eval_steps = _as_int(cfg["eval_steps"], "training.eval_steps")
        save_steps = _as_int(cfg["save_steps"], "training.save_steps")
        logging_steps = _as_int(cfg["logging_steps"], "training.logging_steps")
        per_device_bs = _as_int(
            cfg["per_device_batch_size"], "training.per_device_batch_size"
        )
        grad_accum = _as_int(
            cfg["gradient_accumulation_steps"], "training.gradient_accumulation_steps"
        )

        if num_train_epochs is None:
            num_train_epochs = cfg["num_epochs"]

        use_fp16 = bool(torch.cuda.is_available())
        kwargs = dict(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            warmup_steps=warmup_steps,
            # fp16=use_fp16,
            bf16=True,
            gradient_checkpointing=False,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=cfg["save_total_limit"],
            eval_strategy="steps",
            eval_steps=eval_steps,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            push_to_hub=False,
        )
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        if HAS_DEEPSPEED:
            kwargs["deepspeed"] = DS_CONFIG_PATH

        seed = int(self.config["training"].get("seed", 42))
        kwargs["seed"] = seed
        kwargs["data_seed"] = seed


        return TrainingArguments(**kwargs)
        
    # 对外主入口：联合训练
    def train_joint(
        self,
        news_data_dir: str,
        forum_data_dir: str,
        output_dir: str,
    ):
        seed = int(self.config["training"].get("seed", 42))
        set_global_seed(seed)
        print(f"[JointMoETrainer] Global seed = {seed}")
        # 1. 加载并拼接数据
        train_df, valid_df, test_df = self.load_joint_data(news_data_dir, forum_data_dir)

        # 2. 用训练集统计各类别样本数，初始化自适应损失
        label_map = {"利空": 0, "中性": 1, "利好": 2}
        train_labels = [label_map[label] for label in train_df["label"]]
        num_labels = self.config["model"]["num_labels"]
        samples_per_class = [train_labels.count(i) for i in range(num_labels)]
        self.model.loss_fn = AdaptiveBalancedLoss(samples_per_class)

        # 3. 构建 DatasetDict
        datasets = self.prepare_joint_dataset(train_df, valid_df, test_df)

        # 4. Router 预热阶段：只训练路由器 + 分类头
        router_warmup_steps = int(
            self.config.get("moe", {}).get("router_warmup_steps", 500)
        )
        print(f"[JointMoETrainer] Router warmup steps = {router_warmup_steps}")

        self._freeze_all()
        self._unfreeze_router_and_classifier()

        warmup_args = self._build_training_args(
            output_dir=os.path.join(output_dir, "router_warmup"),
            max_steps=router_warmup_steps,
            num_train_epochs=100,  # 有 max_steps 限制，具体epoch数不重要
        )

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)

        class MoETrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                expert_labels = inputs.pop("expert_labels")
                outputs = model(
                    **inputs,
                    labels=labels,
                    expert_labels=expert_labels,
                )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                if return_outputs:
                    # 只把 logits 暴露给 Trainer，避免 routing_weights 等里出现 None
                    if isinstance(outputs, dict):
                        return loss, {"logits": outputs["logits"]}
                    else:
                        # 理论上不会走到这里，但防一下
                        return loss, outputs
                else:
                    return loss

        warmup_trainer = MoETrainer(
            model=self.model,
            args=warmup_args,
            data_collator=data_collator,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.config["training"]["early_stopping_patience"]),
                TrainStatsCallback(output_dir=output_dir, tag="router_warmup"),
            ],
        )

        print("[JointMoETrainer] >>> 开始 Router 预热阶段训练（仅路由器 + 分类头）")
        warmup_trainer.train()
        print("[JointMoETrainer] >>> Router 预热结束")

        # 5. 联合训练阶段：解冻所有 LoRA + 继续训练路由器与分类头
        self._unfreeze_lora_router_classifier()

        # 打印一下 trainable 参数数
        trainable, total = 0, 0
        for n, p in self.model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(
            f"[JointMoETrainer] trainable params after unfreeze = {trainable:,} / {total:,}"
        )
        assert trainable > 0, "联合训练阶段没有任何可训练参数，请检查 LoRA/router/classifier 的 requires_grad 设置"

        joint_args = self._build_training_args(output_dir=output_dir)

        joint_trainer = MoETrainer(
            model=self.model,
            args=joint_args,
            data_collator=data_collator,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.config["training"]["early_stopping_patience"]),
                TrainStatsCallback(output_dir=output_dir, tag="joint"),
            ],
        )

        print("[JointMoETrainer] >>> 开始联合训练阶段（LoRA + Router + 分类头）")
        train_result = joint_trainer.train()
        joint_trainer.save_model()
        print("[JointMoETrainer] >>> 联合训练完成")

        # 6. 在 test 集上评估
        test_results = joint_trainer.evaluate(datasets["test"])
        print("[JointMoETrainer] Test results:", test_results)

        return train_result, test_results
