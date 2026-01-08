
"""
运行代码：python evaluate.py --model_path /home/pc/WorkSpace/liutao/FinSA-MoE/models/seed/moe_joint_seed123 --dataset_type news --data_path /home/pc/WorkSpace/liutao/FinSA-MoE/data/processed/mix/test.csv
"""

import argparse
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import ProductionInferenceEngine
from src.metrics import compute_classification_metrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def evaluate(model_path: str, dataset_type: str, data_path: str):
    # 读取测试集
    df = pd.read_csv(data_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 初始化推理引擎
    # 在评估测试集性能时，use_router非常重要，单专家测试要使用False，不然性能会很差
    engine = ProductionInferenceEngine(model_path=model_path, dataset_type=dataset_type,device="cuda",use_router=False)

    # 预测
    preds = []
    for text in texts:
        result = engine.predict_single(text, return_all_scores=False, return_expert_info=False)
        preds.append(result["sentiment"])

    # 标签映射（利空 / 中性 / 利好）
    label_map = {"利空": 0, "中性": 1, "利好": 2}

    y_true = [label_map[l] for l in labels]
    y_pred = [label_map[l] for l in preds]

    # 计算评估指标（Accuracy / Precision / Recall / F1 / F1_weighted）
    metrics = compute_classification_metrics(np.array(y_pred), np.array(y_true))

    print("======== Evaluation Results ========")
    for k, v in metrics.items():
        print(f"{k:15}: {v:.4f}")

    # 详细分类报告：利空 / 中性 / 利好
    print("\nClassification Report:")
    target_names = list(label_map.keys())
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # ====== 混淆矩阵 ======
    # 行：真实标签；列：预测标签
    cm = confusion_matrix(y_true, y_pred, labels=list(label_map.values()))

    print("\nConfusion Matrix (rows = true labels, cols = predicted labels):")
    cm_df = pd.DataFrame(
        cm,
        index=[f"真实_{name}" for name in target_names],
        columns=[f"预测_{name}" for name in target_names]
    )
    print(cm_df)

    # 进一步给出每个类别的 TP / FP / FN / TN 统计
    print("\nPer-class stats derived from confusion matrix:")
    total = cm.sum()
    for idx, name in enumerate(target_names):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp        # 真实为该类，但被预测错
        fp = cm[:, idx].sum() - tp        # 预测为该类，但真实不是
        tn = total - tp - fn - fp         # 其它都算 TN
        print(
            f"{name}: "
            f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--dataset_type", type=str, choices=["news", "forum"], required=True, help="Dataset type")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test.csv file")
    args = parser.parse_args()

    evaluate(args.model_path, args.dataset_type, args.data_path)
