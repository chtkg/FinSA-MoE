# 推理引擎

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import time
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .model import DualExpertGLMClassifier


class ProductionInferenceEngine:
    """生产环境推理引擎"""

    def __init__(
            self,
            model_path: str,
            dataset_type: str = "news",  # "news" / "forum"，在单专家模式下仍然使用
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            batch_size: int = 8,
            use_router: bool = True      # True 时使用 MoE 稀疏路由，在单专家推理时一定要使用False
    ):
        self.device = device
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.use_router = use_router

        self.model = DualExpertGLMClassifier.from_checkpoint(
            ckpt_dir=model_path,
            base_model_path="/home/pc/WorkSpace/liutao/FinSA-MoE/models/GLM-Z1-9B",
            num_labels=3
        )

        self.model.to(device)
        self.model.eval()

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/pc/WorkSpace/liutao/FinSA-MoE/models/GLM-Z1-9B",
            trust_remote_code=True
        )

        # 标签映射
        self.id2label = {0: "利空", 1: "中性", 2: "利好"}
        # if self.dataset_type == "news":
        #     self.id2label = {0: "利空", 1: "中性", 2: "利好"}
        # elif self.dataset_type == "forum":
        #     self.id2label = {0: "消极", 1: "中性", 2: "积极"}
        # else:
        #     raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
    @torch.no_grad()
    def predict_batch(
        self,
        texts,
        return_all_scores: bool = False,
        return_expert_info: bool = False,
    ):
        results = []
        for t in texts:
            res = self.predict_single(
                t,
                return_all_scores=return_all_scores,
                return_expert_info=return_expert_info,
            )
            results.append(res)
        return results

    
    @torch.no_grad()
    def predict_single(
            self,
            text: str,
            return_all_scores: bool = False,
            return_expert_info: bool = True
    ) -> Dict:
        """单条文本预测"""
        start_time = time.time()

        # 1. 构造提示词
        if self.use_router:
            # MoE 模式
            prompt = f"分析以下金融文本的市场情感：{text}\n情感分类："
        else:
            # 单专家模式
            if self.dataset_type == "news":
                prompt = f"分析以下金融新闻的市场情感：{text}\n情感分类："
            else:
                prompt = f"分析以下股吧评论的情感倾向：{text}\n情感分类："

        # 2. 编码输入
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        ).to(self.device)

        # 3. 前向推理
        if self.use_router:
            # MoE 模式：不指定 expert_type，由路由器决定激活哪个专家
            outputs = self.model(**inputs)
        else:
            # 单专家模式：直接指定新闻/股评专家
            outputs = self.model(**inputs, expert_type=self.dataset_type)

        logits = outputs["logits"]

        # 4. 专家信息
        expert_info = {}
        if return_expert_info and ("routing_weights" in outputs):
            # --- MoE 模式：routing_weights 为 (batch, num_experts) ---
            routing_weights = outputs["routing_weights"]  # 推理时是一行 one-hot（稀疏路由）
            expert_weights = routing_weights.cpu().numpy()[0]
            # 0 → 新闻专家，1 → 股吧专家
            selected_idx = int(expert_weights.argmax())
            selected_expert = "新闻" if selected_idx == 0 else "股吧"
            expert_confidence = float(expert_weights[selected_idx])
            expert_info = {
                "expert": selected_expert,
                "expert_confidence": expert_confidence,
                "expert_weights": {
                    "新闻": float(expert_weights[0]),
                    "股吧": float(expert_weights[1]),
                },
            }
        elif return_expert_info:
            # --- 单专家模式：没有 routing_weights，就根据 dataset_type 构造 one-hot ---
            if self.dataset_type == "news":
                expert_info = {
                    "expert": "新闻",
                    "expert_confidence": 1.0,
                    "expert_weights": {"新闻": 1.0, "股吧": 0.0},
                }
            else:
                expert_info = {
                    "expert": "股吧",
                    "expert_confidence": 1.0,
                    "expert_weights": {"新闻": 0.0, "股吧": 1.0},
                }

        # 5. 分类概率
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))
        predicted_label = self.id2label[predicted_class]
        confidence = float(probs[predicted_class])

        # 6. 组装最终结果
        inference_time = time.time() - start_time
        result = {
            "text": text,
            "sentiment": predicted_label,
            "confidence": confidence,
            "inference_time_ms": inference_time * 1000,
        }

        if return_all_scores:
            result["scores"] = {
                self.id2label[i]: float(probs[i]) for i in range(len(probs))
            }

        result.update(expert_info)
        return result


    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """批量预测"""
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = [self.predict_single(text) for text in batch_texts]
            results.extend(batch_results)
        return results

    @torch.no_grad()
    def predict_proba(self, texts, return_expert_info=False, force_soft_routing=False):
        """
        texts: List[str]
        return: np.ndarray, shape=(N, 3)
        """
        probs_list = []
        expert_infos = []

        for text in texts:
            if self.use_router:
                prompt = f"分析以下金融文本的市场情感：{text}\n情感分类："
            else:
                if self.dataset_type == "news":
                    prompt = f"分析以下金融新闻的市场情感：{text}\n情感分类："
                else:
                    prompt = f"分析以下股吧评论的情感倾向：{text}\n情感分类："

            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            if self.use_router:
                outputs = self.model(**inputs, force_soft_routing=force_soft_routing)
            else:
                outputs = self.model(**inputs, expert_type=self.dataset_type)

            logits = outputs["logits"]
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            probs_list.append(probs)

            if return_expert_info and ("routing_weights" in outputs):
                rw = outputs["routing_weights"].detach().cpu().numpy()[0]
                expert_infos.append(rw)

        probs_arr = np.vstack(probs_list)
        if return_expert_info:
            return probs_arr, expert_infos
        return probs_arr

