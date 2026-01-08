# FinSA-MoE/scripts/merge_expert_lora.py
"""
把 news_expert 和 forum_expert 的 LoRA 权重合并到一个 ckpt 目录，
用于第二阶段 MoE 联合训练的初始化模型。

思路：
- 不加载模型本体，只操作 safetensors 里的 state_dict
- 以 news_expert 的 state_dict 为 base
- 用 forum_expert 里「forum adapter 的 LoRA 参数」覆盖 base
"""

import os
import shutil
import argparse
from safetensors.torch import load_file, save_file


def _collect_adapter_keys(state_dict, adapter_name: str):
    """
    过滤出属于某个 adapter 的 LoRA 参数 key 列表

    这里通过以下规则粗略判断：
      - key 中包含 adapter_name（"news" 或 "forum"）
      - 且包含 "lora_" 或 "lora_embedding"
    """
    keys = []
    for k in state_dict.keys():
        if adapter_name in k and ("lora_" in k or "lora_embedding" in k):
            keys.append(k)
    return keys


def merge_expert_adapters(news_ckpt_dir: str, forum_ckpt_dir: str, output_dir: str):
    # 1. 路径检查
    news_path = os.path.join(news_ckpt_dir, "model.safetensors")
    forum_path = os.path.join(forum_ckpt_dir, "model.safetensors")

    if not os.path.exists(news_path):
        raise FileNotFoundError(f"没有找到 news_expert ckpt: {news_path}")
    if not os.path.exists(forum_path):
        raise FileNotFoundError(f"没有找到 forum_expert ckpt: {forum_path}")

    if os.path.exists(output_dir):
        raise ValueError(
            f"输出目录 {output_dir} 已存在，请先删除或改名后再运行本脚本。"
        )

    # 2. 以 news_expert 目录为模板拷贝一份到 output_dir
    print(f"[merge] 复制 {news_ckpt_dir} -> {output_dir}")
    shutil.copytree(news_ckpt_dir, output_dir)

    # 3. 加载两个 state_dict（纯权重，不加载模型）
    print(f"[merge] 加载 news_expert 权重: {news_path}")
    news_state = load_file(news_path)
    print(f"[merge] 加载 forum_expert 权重: {forum_path}")
    forum_state = load_file(forum_path)

    print(f"[merge] news_expert 参数数:  {len(news_state)}")
    print(f"[merge] forum_expert 参数数: {len(forum_state)}")

    # 4. 找出各自的 LoRA key
    news_lora_news_keys = _collect_adapter_keys(news_state, "news")
    forum_lora_forum_keys = _collect_adapter_keys(forum_state, "forum")

    print(f"[merge] news_expert 中属于 'news' adapter 的 LoRA 参数数: {len(news_lora_news_keys)}")
    print(f"[merge] forum_expert 中属于 'forum' adapter 的 LoRA 参数数: {len(forum_lora_forum_keys)}")

    if len(news_lora_news_keys) == 0:
        print("[WARN] 在 news_expert ckpt 中没有找到任何包含 'news' 的 LoRA 参数，请检查一阶段新闻专家是否训练成功。")
    if len(forum_lora_forum_keys) == 0:
        print("[WARN] 在 forum_expert ckpt 中没有找到任何包含 'forum' 的 LoRA 参数，请检查一阶段股评专家是否训练成功。")

    # 5. 以 news_expert 为 base
    merged_state = dict(news_state)

    # 6. 用 forum_expert 中的 forum LoRA 参数覆盖 merged_state
    override_count = 0
    for k in forum_lora_forum_keys:
        if k in forum_state:
            merged_state[k] = forum_state[k]
            override_count += 1

    print(f"[merge] 覆盖 forum adapter LoRA 参数数: {override_count}")

    # 7. 保存合并后的 safetensors 到 output_dir
    out_path = os.path.join(output_dir, "model.safetensors")
    print(f"[merge] 保存合并后的权重到: {out_path}")
    save_file(merged_state, out_path)

    print("\n[merge] 完成合并！")
    print("[merge] 现在可以在第二阶段联合训练里把 init_checkpoint 设置为这个输出目录。")


def main():
    parser = argparse.ArgumentParser(description="合并 news_expert 和 forum_expert 的 LoRA 权重")
    parser.add_argument(
        "--news_ckpt_dir",
        type=str,
        required=True,
        help="news_expert 模型目录，/home/pc/WorkSpace/liutao/FinSA-MoE/models/news_expert",
    )
    parser.add_argument(
        "--forum_ckpt_dir",
        type=str,
        required=True,
        help="forum_expert 模型目录，/home/pc/WorkSpace/liutao/FinSA-MoE/models/forum_expert",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="合并后输出的新目录，/home/pc/WorkSpace/liutao/FinSA-MoE/models/moe_init",
    )

    args = parser.parse_args()
    merge_expert_adapters(
        news_ckpt_dir=args.news_ckpt_dir,
        forum_ckpt_dir=args.forum_ckpt_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
