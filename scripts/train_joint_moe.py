import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils_seed import set_global_seed
from src.trainer_joint_moe import JointMoETrainer
import yaml

def main():
    config_path = r"/home/pc/WorkSpace/liutao/FinSA-MoE/config/training_config.yaml"
    init_checkpoint = r"/home/pc/WorkSpace/liutao/FinSA-MoE/models/moe_init"
    news_data_dir = r"/home/pc/WorkSpace/liutao/FinSA-MoE/data/processed/news"
    forum_data_dir = r"/home/pc/WorkSpace/liutao/FinSA-MoE/data/processed/forum"
    base_output_dir = r"/home/pc/WorkSpace/liutao/FinSA-MoE/models/seed/moe_joint_seed13"

    seeds = [13, 21, 42, 87, 123]   # 随机种子列表

    for seed in seeds:
        # 读 config -> 写入 seed
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("training", {})
        cfg["training"]["seed"] = int(seed)

        out_dir = f"{base_output_dir}_seed{seed}"
        os.makedirs(out_dir, exist_ok=True)
        tmp_cfg_path = os.path.join(out_dir, "training_config_used.yaml")
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        trainer = JointMoETrainer(
            config_path=tmp_cfg_path,
            init_checkpoint=init_checkpoint,
        )
        trainer.train_joint(
            news_data_dir=news_data_dir,
            forum_data_dir=forum_data_dir,
            output_dir=out_dir,
        )

if __name__ == "__main__":
    main()
