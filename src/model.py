# 双专家模型，模型采用了专家混合（Mixture of Experts, MoE）架构，结合了LoRA（Low-Rank Adaptation）技术和量化优化

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from .router import EfficientRouter
from peft import prepare_model_for_kbit_training
from safetensors.torch import load_file
import os


class DualExpertGLMClassifier(nn.Module):
    """双专家GLM金融情感分类器"""

    def __init__(
            self,
            base_model_path: str,
            num_labels: int = 3,
            lora_config: dict = None
    ):
        super().__init__()

        # 4-bit量化配置，将模型权重从32位浮点数压缩到4位，大幅减少内存占用（约8倍压缩），同时保持较好的性能
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 使用4位量化
            bnb_4bit_use_double_quant=True,  # 双重量化，进一步压缩
            bnb_4bit_quant_type="nf4",  # 使用NormalFloat4量化类型
            bnb_4bit_compute_dtype=torch.bfloat16  # 计算时使用bfloat16
        )

        # 加载基础模型
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=num_labels,
            dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="cuda",     # 训练单专家时如果报错，改回auto
            trust_remote_code=True
        )
        # 加载基础模型后，立刻做k-bit训练准备
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        # for name, module in self.base_model.named_modules():
        #     print(name)

        # LoRA配置
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
                "bias": "none",
                "task_type": "SEQ_CLS"
            }

        peft_config = LoraConfig(**lora_config)

        # 创建新闻专家，每个专家共享基础GLM模型，但有独立的LoRA参数，能够学习不同文本类型的特征。
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.base_model.add_adapter(adapter_name="news", peft_config=peft_config)

        # 创建股吧专家
        self.base_model.add_adapter(adapter_name="forum", peft_config=peft_config)

        # 显式激活并“标记可训练”要训练的适配器（以news为例）
        if hasattr(self.base_model, "set_adapter"):
            self.base_model.set_adapter("forum")  # 激活 news

        # 训练forum专家时，将news改为forum
        if hasattr(self.base_model, "train_adapter"):
            self.base_model.train_adapter("forum")

        # 打印一下，便于确认真的有可训练参数
        if hasattr(self.base_model, "print_trainable_parameters"):
            self.base_model.print_trainable_parameters()

        # 路由器
        hidden_size = self.base_model.config.hidden_size
        # 动态选择专家，输入文本的特征，自动判断应该更多依赖哪个专家的输出
        self.router = EfficientRouter(
            input_dim=hidden_size,
            num_experts=2,
            hidden_dim=64
        )

        # 分类头
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        # 权重损失
        self.route_loss_weight = 1.5    # λ_r[routerloss_weight_1:0, routerloss_weight_2:0.5, routerloss_weight_3:1.0,routerloss_weight_4:1.5(FinSA_MoE)，routerloss_weight_5:2.0]
        self.balance_loss_weight = 0.05  # λ_b=[balanceloss_weight_1:0, balanceloss_weight_2:0.005, balanceloss_weight_3:0.01, balanceloss_weight_4:0,05]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        expert_type: str = None,          # 单专家模式："news" 或 "forum"
        expert_labels=None,               # MoE 监督路由标签：0=news, 1=forum
        **kwargs,                         # 兜底，防止 Trainer 传入多余字段
    ):
        """
        统一的前向接口，兼容 Trainer / 推理：
        - 单专家模式：传 expert_type="news"/"forum"
        - MoE 模式：不传 expert_type，而是传 expert_labels（用于第二阶段联合训练）
        """
        
        # 1）单专家模式
        if expert_type is not None:
            # 选择对应适配器
            self.base_model.set_adapter(expert_type)

            # 直接用带分类头的 base_model 做前向
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,   # 不用底层自带的 CE，统一在外面算
                **kwargs,
            )

            logits = base_out.logits if hasattr(base_out, "logits") else base_out["logits"]

            loss = None
            if labels is not None:
                if hasattr(self, "loss_fn"):
                    loss = self.loss_fn(logits, labels)
                else:
                    loss = nn.CrossEntropyLoss()(logits, labels)

            return {
                "loss": loss,
                "logits": logits,
            }


        # 2）MoE 模式（第二阶段联合训练 / 自动路由推理）
        # 2.1 用 backbone 的最后一层 hidden state 做路由输入
        #     注意这里只做前向，不回传梯度到 backbone（梯度主要给 LoRA）
        with torch.no_grad():
            base_outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
        hidden_states = base_outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        # 2.2 路由器：得到 routing_weights 与 router_logits
        # routing_weights: (batch, num_experts)
        # src/model.py (MoE 分支)
        routing_weights, router_logits = self.router(
            hidden_states, return_router_logits=True, force_soft=kwargs.get("force_soft_routing", False)
        )


        # 2.3 两个专家分别前向（可以回传梯度到各自 LoRA）
        # 新闻专家
        self.base_model.set_adapter("news")
        news_outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        news_hidden = news_outputs.hidden_states[-1]

        # 股吧专家
        self.base_model.set_adapter("forum")
        forum_outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        forum_hidden = forum_outputs.hidden_states[-1]

        # 2.4 按路由权重融合专家输出（稀疏/致密都可以，一般 eval 下是 one-hot）
        # expert_stack: (batch, num_experts, seq_len, hidden_dim)
        expert_stack = torch.stack([news_hidden, forum_hidden], dim=1)
        # routing_weights: (batch, num_experts) -> (batch, num_experts, 1, 1)
        routing_weights_expanded = routing_weights.unsqueeze(-1).unsqueeze(-1)
        # 加权求和 -> (batch, seq_len, hidden_dim)
        weighted_hidden = (expert_stack * routing_weights_expanded).sum(dim=1)

        # 2.5 池化成句向量
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            weighted_sum = (weighted_hidden * mask_expanded).sum(dim=1)
            pooled_output = weighted_sum / mask_expanded.sum(dim=1).clamp(min=1e-6)
        else:
            pooled_output = weighted_hidden.mean(dim=1)

        pooled_output = self.dropout(pooled_output)

        # 2.6 统一分类头
        logits = self.classifier(pooled_output)

        # 2.7 损失：L = L_class + λ_r L_route + λ_b L_balance
        loss = loss_cls = loss_route = loss_balance = None
        if labels is not None:
            # 分类损失
            if hasattr(self, "loss_fn"):
                loss_cls = self.loss_fn(logits, labels)
            else:
                loss_cls = nn.CrossEntropyLoss()(logits, labels)

            if self.training:
                # 路由监督损失（有 expert_labels 才算，没有就置 0）
                if expert_labels is not None:
                    # Dataset 里 expert_labels 是 list[int]，这里转成 tensor
                    if not torch.is_tensor(expert_labels):
                        expert_labels = torch.as_tensor(
                            expert_labels, device=logits.device, dtype=torch.long
                        )
                    loss_route = self.router.compute_route_loss(
                        router_logits, expert_labels
                    )
                else:
                    loss_route = torch.zeros(1, device=logits.device)

                # 负载均衡损失
                loss_balance = self.router.compute_balance_loss(routing_weights)

                loss = (
                    loss_cls
                    + self.route_loss_weight * loss_route
                    + self.balance_loss_weight * loss_balance
                )
            else:
                # eval/推理阶段一般只看分类损失
                loss = loss_cls

        return {
            "loss": loss,
            "logits": logits,
            "routing_weights": routing_weights,
            "router_logits": router_logits,
            "loss_cls": loss_cls,
            "loss_route": loss_route,
            "loss_balance": loss_balance,
        }


        
    # --- 兼容 Transformers/Trainer 的梯度检查点相关调用 ---
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # 与HF一致：开启GC时关闭use_cache以节省显存/避免冲突
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "use_cache"):
            try:
                self.base_model.config.use_cache = False
            except Exception:
                pass
        # 透传给底层（PeftModel 或 PreTrainedModel）
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            try:
                return self.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except TypeError:
                # 兼容旧签名
                return self.base_model.gradient_checkpointing_enable()
        # 若底层也没有，静默返回
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()
        return None

    def enable_input_require_grads(self):
        # 一些Trainer版本在GC时会调用，用于让输入嵌入参与反向
        if hasattr(self.base_model, "enable_input_require_grads"):
            return self.base_model.enable_input_require_grads()
        # 退化实现：尝试对输入嵌入层强制require_grad
        try:
            emb = self.get_input_embeddings()
            if emb is not None:
                emb.weight.requires_grad_(True)
        except Exception:
            pass
        return None

    def get_input_embeddings(self):
        if hasattr(self.base_model, "get_input_embeddings"):
            return self.base_model.get_input_embeddings()
        # 兼容某些模型命名
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "get_input_embeddings"):
            return self.base_model.model.get_input_embeddings()
        return None

    def prepare_for_training(self, adapter_name: str):
        # 激活并标记该适配器为可训练
        if hasattr(self.base_model, "set_adapter"):
            self.base_model.set_adapter(adapter_name)
        if hasattr(self.base_model, "train_adapter"):
            self.base_model.train_adapter(adapter_name)
        # 可选：打印可训练参数
        if hasattr(self.base_model, "print_trainable_parameters"):
            self.base_model.print_trainable_parameters()

    @classmethod
    def from_pretrained(cls, adapter_dir, base_model_path, num_labels=3, lora_config=None, adapter_name="forum"):
        inst = cls(base_model_path=base_model_path, num_labels=num_labels, lora_config=lora_config)
        inst.base_model.load_adapter(adapter_dir, adapter_name=adapter_name)
        inst.base_model.set_adapter(adapter_name)
        return inst

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, base_model_path: str, num_labels: int = 3, lora_config: dict = None):
        # 先按训练时的方式构建模型骨架（会加载基础模型 & 注入LoRA结构 & 两个adapter壳）
        inst = cls(base_model_path=base_model_path, num_labels=num_labels, lora_config=lora_config)
        # 直接把 Trainer.save_model() 产出的 state_dict 灌回去
        ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
        state = load_file(ckpt_path)  # safetensors
        missing, unexpected = inst.load_state_dict(state, strict=False)
        print("[load_state_dict] missing:", missing)
        print("[load_state_dict] unexpected:", unexpected)
        return inst
