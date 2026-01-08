"""
main.py - FinSA-MoE项目主程序
推理运行示例：python main.py --mode inference --data_type forum --model_path ./models/moe_joint/moe
最好使用
"""
import argparse
import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(description='FinSA-MoE金融情感分析')
    parser.add_argument('--mode', choices=['train', 'test', 'inference'],
                        required=True, help='运行模式')
    parser.add_argument('--data_type', choices=['news', 'forum'],
                        default='forum', help='数据类型')
    parser.add_argument('--model_path', type=str,
                        default='./models/moe_joint/moe', help='模型路径')

    args = parser.parse_args()

    if args.mode == 'train':
        from src.train import GLMFinancialTrainer
        trainer = GLMFinancialTrainer(
            model_name=args.model_path,
            dataset_type=args.data_type
        )
        # 加载数据并训练
        print("开始训练...")

    elif args.mode == 'test':
        from scripts.test_model import test_model
        test_model(args.model_path)

    elif args.mode == 'inference':
        from src.inference import ProductionInferenceEngine
        engine = ProductionInferenceEngine(args.model_path)
        # 交互式推理
        while True:
            text = input("\n请输入文本 (输入'quit'退出): ")
            if text == 'quit':
                break
            result = engine.predict_single(text)
            print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.2%})")


if __name__ == "__main__":
    main()