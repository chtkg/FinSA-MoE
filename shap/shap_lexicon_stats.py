# 运行示例：
# python shap_lexicon_stats.py --model_path /home/pc/WorkSpace/liutao/FinSA-MoE/models/moe_joint/moe --dataset_type news --data_path /home/pc/WorkSpace/liutao/FinSA-MoE/data/shap/correct_samples_200.csv --lexicon_path /home/pc/WorkSpace/liutao/FinSA-MoE/shap/negative_lexicon.txt --out_csv negative_lexicon_shap_stats.csv --true_label 利空 --target_class 利空 --max_evals 2000 --use_router


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import shap
from collections import defaultdict
from src.inference import ProductionInferenceEngine

LABEL_ORDER = ["利空", "中性", "利好"]  # 确保与 predict_proba 输出顺序一致


def load_lexicon(path: str):
    """
    一行一个词；允许空行和#注释
    """
    lex = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            lex.append(w)
    # 去重但保持顺序
    seen = set()
    uniq = []
    for w in lex:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq


def normalize_token(tok: str) -> str:
    """
    针对常见 tokenizer 的 token 清洗：
    - 去掉BPE/WordPiece标记（##）
    - 去掉 sentencepiece 的下划线前缀（▁）
    - strip 空白
    """
    if tok is None:
        return ""
    t = str(tok)
    t = t.replace("##", "")
    t = t.replace("▁", "")
    return t.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="news")
    parser.add_argument("--data_path", type=str, required=True, help="CSV: id,text,label")
    parser.add_argument("--lexicon_path", type=str, required=True, help="利好词表 txt（一行一个词）")
    parser.add_argument("--out_csv", type=str, required=True, help="输出统计表 CSV")
    parser.add_argument("--out_token_csv", type=str, default="", help="可选：输出 token 级明细 CSV（不需要可不填）")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_router", action="store_true")
    parser.set_defaults(use_router=True)

    parser.add_argument("--max_evals", type=int, default=2000, help="SHAP 采样预算，越大越慢但更稳")
    parser.add_argument("--batch_size", type=int, default=16, help="predict_proba 批大小（如引擎支持）")

    # 只分析利好文本
    parser.add_argument("--true_label", type=str, default="利空", help="只对该标签文本做分析")
    parser.add_argument("--target_class", type=str, default="利空", choices=LABEL_ORDER, help="解释哪个类别的 SHAP")
    parser.add_argument("--min_freq", type=int, default=1, help="统计表中最小出现次数过滤")
    args = parser.parse_args()

    # 1) 初始化推理引擎
    engine = ProductionInferenceEngine(
        model_path=args.model_path,
        dataset_type=args.dataset_type,
        device=args.device,
        use_router=args.use_router,
    )

    # 2) 读取数据
    df = pd.read_csv(args.data_path)
    required_cols = {"id", "text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}, 需要: {required_cols}")

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()

    # 只保留 label=利好 的样本
    df_pos = df[df["label"] == args.true_label].copy()
    if df_pos.empty:
        raise ValueError(f"没有 label=={args.true_label} 的样本可分析。")

    texts = df_pos["text"].tolist()

    # 3) 读取词表
    lexicon = load_lexicon(args.lexicon_path)
    lexicon_set = set(lexicon)

    # 4) masker
    masker = shap.maskers.Text(engine.tokenizer)

    # 5) 预测函数（返回每类概率）
    def f(x):
        proba = engine.predict_proba(
            x,
            force_soft_routing=True,
            # batch_size=args.batch_size,  # 若 engine 实现支持可打开
        )
        return proba

    # 6) 构建 explainer
    explainer = shap.Explainer(
        f,
        masker,
        output_names=LABEL_ORDER,
    )

    # 7) 计算 SHAP
    shap_values = explainer(texts, max_evals=args.max_evals)

    # 8) 只取目标类别（默认“利好”）
    target_idx = LABEL_ORDER.index(args.target_class)

    # 9) 统计：词表 token 的贡献
    # word_stats[word] = [val1, val2, ...]
    word_stats = defaultdict(list)

    token_rows = []

    # shap_values.data[i] 是 token 列表；shap_values.values[i] shape: (num_tokens, num_classes)
    for i in range(len(texts)):
        tokens = shap_values.data[i]
        values = shap_values.values[i][:, target_idx]

        for tok, val in zip(tokens, values):
            t = normalize_token(tok)
            if not t:
                continue

            # 仅统计“词表中出现的 token”
            if t in lexicon_set:
                word_stats[t].append(float(val))

            if args.out_token_csv:
                token_rows.append({
                    "id": df_pos.iloc[i]["id"],
                    "label": df_pos.iloc[i]["label"],
                    "token": t,
                    "shap": float(val),
                    "in_lexicon": int(t in lexicon_set),
                })

    # 10) 输出统计表
    rows = []
    for w, vals in word_stats.items():
        freq = len(vals)
        if freq < args.min_freq:
            continue
        rows.append({
            "word": w,
            "frequency": freq,
            "mean_shap": float(np.mean(vals)),
            "mean_abs_shap": float(np.mean(np.abs(vals))),
            "sum_shap": float(np.sum(vals)),
            "sum_abs_shap": float(np.sum(np.abs(vals))),
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        result_df = pd.DataFrame(columns=["word", "frequency", "mean_shap", "mean_abs_shap", "sum_shap", "sum_abs_shap"])
    else:
        result_df = result_df.sort_values("mean_abs_shap", ascending=False)

    # ===== 计算占比  =====
    if not result_df.empty:
        total = result_df["sum_abs_shap"].sum()
        if total > 0:
            result_df["ratio"] = result_df["sum_abs_shap"] / total
        else:
            result_df["ratio"] = 0.0

    # 加一点元信息
    meta = {
        "model_path": args.model_path,
        "dataset_type": args.dataset_type,
        "data_path": args.data_path,
        "num_positive_samples": len(texts),
        "target_class": args.target_class,
        "max_evals": args.max_evals,
        "use_router": args.use_router,
        "lexicon_size": len(lexicon),
        "match_mode": "token_exact",
    }
    
    print("META:", meta)

    result_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved lexicon SHAP stats to: {args.out_csv}")
    print(f"Explained positive samples: {len(texts)}")
    print(f"Matched lexicon tokens: {len(result_df)}")

    if args.out_token_csv:
        token_df = pd.DataFrame(token_rows)
        token_df.to_csv(args.out_token_csv, index=False, encoding="utf-8-sig")
        print(f"Saved token-level details to: {args.out_token_csv}")


if __name__ == "__main__":
    main()
