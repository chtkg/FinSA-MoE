
"""
运行示例：
python evaluate_sample_correct_and_wrong.py --model_path /home/pc/WorkSpace/liutao/FinSA-MoE/models/moe_joint/moe --dataset_type forum --data_path /home/pc/WorkSpace/liutao/FinSA-MoE/data/processed/forum/test.csv --output_dir /home/pc/WorkSpace/liutao/FinSA-MoE/data/analysis --sample_size 100 --seed 42
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import ProductionInferenceEngine
from src.metrics import compute_classification_metrics


LABEL_MAP = {"利空": 0, "中性": 1, "利好": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def evaluate_and_sample(
    model_path: str,
    dataset_type: str,
    data_path: str,
    output_dir: str,
    sample_size: int = 100,
    seed: int = 42,
    device: str = "cuda",
    use_router: bool = True,    #使用单专家一定要用False
):
    os.makedirs(output_dir, exist_ok=True)

    # ===== 1. 读取测试集 =====
    df = pd.read_csv(data_path)

    required_cols = {"id", "text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {missing}")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    # ===== 2. 初始化推理引擎 =====
    engine = ProductionInferenceEngine(
        model_path=model_path,
        dataset_type=dataset_type,
        device=device,
        use_router=use_router,
    )

    # ===== 3. 执行预测 =====
    preds = []
    for text in texts:
        out = engine.predict_single(
            text, return_all_scores=False, return_expert_info=False
        )
        preds.append(out["sentiment"])

    # ===== 4. 标签映射 =====
    y_true = np.array([LABEL_MAP[l] for l in labels])
    y_pred = np.array([LABEL_MAP[p] for p in preds])

    # ===== 5. 评估指标 =====
    metrics = compute_classification_metrics(y_pred, y_true)
    print("\n======== Evaluation Results ========")
    for k, v in metrics.items():
        print(f"{k:15}: {v:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=list(LABEL_MAP.keys()), digits=4
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_MAP.values()))
    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            cm,
            index=[f"真实_{k}" for k in LABEL_MAP.keys()],
            columns=[f"预测_{k}" for k in LABEL_MAP.keys()],
        )
    )

    # ===== 6. 构造结果 DataFrame =====
    result_df = df.copy()
    result_df["pred"] = [INV_LABEL_MAP[i] for i in y_pred]
    result_df["is_correct"] = y_true == y_pred

    # ===== 7. 抽样 =====
    rng = np.random.RandomState(seed)

    correct_df = result_df[result_df["is_correct"]]
    wrong_df = result_df[~result_df["is_correct"]]

    print(f"\nCorrect samples: {len(correct_df)}")
    print(f"Wrong samples  : {len(wrong_df)}")

    correct_sample = (
        correct_df.sample(
            n=min(sample_size, len(correct_df)), random_state=rng
        )
        if len(correct_df) > 0
        else correct_df
    )

    wrong_sample = (
        wrong_df.sample(
            n=min(sample_size, len(wrong_df)), random_state=rng
        )
        if len(wrong_df) > 0
        else wrong_df
    )

    # ===== 8. 保存 CSV =====
    keep_cols = ["id", "text", "label", "pred"]

    correct_path = os.path.join(output_dir, "forum_correct_samples_100.csv")
    wrong_path = os.path.join(output_dir, "forum_wrong_samples_100.csv")

    correct_sample[keep_cols].to_csv(
        correct_path, index=False, encoding="utf-8-sig"
    )
    wrong_sample[keep_cols].to_csv(
        wrong_path, index=False, encoding="utf-8-sig"
    )

    print("\nSaved files:")
    print(f"  ✔ Correct samples: {correct_path}")
    print(f"  ✘ Wrong samples  : {wrong_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model and sample correct / wrong predictions"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, choices=["news", "forum"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_router", action="store_true")
    parser.set_defaults(use_router=True)

    args = parser.parse_args()

    evaluate_and_sample(
        model_path=args.model_path,
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        device=args.device,
        use_router=args.use_router,
    )
