# src/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_classification_metrics(predictions, labels):
    """
    统一的分类任务指标计算
    输入:
        predictions: logits 或类别预测 (numpy array)
        labels: 真实标签 (numpy array)
    返回:
        指标字典
    """
    # 如果传入的是logits，先取argmax
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }
