import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import GLMFinancialTrainer


def main():
    trainer = GLMFinancialTrainer(
        config_path= r"/home/pc/WorkSpace/liutao/FinSA-MoE/config/training_config.yaml",
        dataset_type= "news"
    )

    trainer.train(
        data_dir= r"/home/pc/WorkSpace/liutao/FinSA-MoE/data/processed/news",
        output_dir= r"/home/pc/WorkSpace/liutao/FinSA-MoE/models/news_expert",
    )


if __name__ == "__main__":
    main()