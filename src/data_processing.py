# 数据预处理：处理新闻文本和股吧评论的csv文件，并分别按7:2:2的比例分为训练集:验证集:测试集

import pandas as pd
import os
import random
from collections import defaultdict
import numpy as np


def process_stock_news_files(folder_path, output_folder):
    """
    处理股票新闻、股评CSV文件，进行均衡的数据集划分

    Args:
        folder_path: 包含CSV文件的文件夹路径
        output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)

    # 按情感分类存储数据
    sentiment_data = {
        '利好': [],
        '利空': [],
        '中性': []
    }

    print("开始读取CSV文件...")

    # 读取所有CSV文件
    file_count = 0
    total_records = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 处理每条记录
                for _, row in df.iterrows():
                    # 合并新闻标题和摘要作为text
   
                    text = str(row['text']) if pd.notna(row['text']) else ""

                    # 获取情感标签
                    sentiment = str(row['label']) if pd.notna(row['label']) else "中性"

                    # 根据情感分类存储
                    if sentiment in sentiment_data:
                        sentiment_data[sentiment].append({
                            'text': text,
                            'sentiment': sentiment
                        })
                    else:
                        # 如果情感标签不在预期范围内，归为中性
                        sentiment_data['中性'].append({
                            'text': text,
                            'sentiment': '中性'
                        })

                    total_records += 1

                file_count += 1
                if file_count % 500 == 0:
                    print(f"已处理 {file_count} 个文件，累计 {total_records} 条记录")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                continue

    print(f"\n文件读取完成！")
    print(f"总共处理文件: {file_count} 个")
    print(f"总记录数: {total_records} 条")

    # 统计各类别数量
    positive_count = len(sentiment_data['利好'])
    negative_count = len(sentiment_data['利空'])
    neutral_count = len(sentiment_data['中性'])

    print(f"利好消息: {positive_count} 条")
    print(f"利空消息: {negative_count} 条")
    print(f"中性消息: {neutral_count} 条")

    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)

    # 打乱每个情感类别的数据
    for sentiment in sentiment_data:
        random.shuffle(sentiment_data[sentiment])

    # 新闻数据集定义划分比例
    # train_ratio = 0.98
    # valid_ratio = 0.01
    # test_ratio = 0.01
    train_ratio = 0.6
    valid_ratio = 0.2
    test_ratio = 0.2

    # 为每个情感类别分别划分数据集
    train_data = []
    valid_data = []
    test_data = []

    for sentiment, data_list in sentiment_data.items():
        n_total = len(data_list)

        # 计算每个集合的大小
        n_valid = max(1, int(n_total * valid_ratio))
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_valid - n_test

        print(f"\n{sentiment}消息划分:")
        print(f"  训练集: {n_train} 条")
        print(f"  验证集: {n_valid} 条")
        print(f"  测试集: {n_test} 条")

        # 划分数据
        train_data.extend(data_list[:n_train])
        valid_data.extend(data_list[n_train:n_train + n_valid])
        test_data.extend(data_list[n_train + n_valid:])

    # 打乱各数据集（保持类别均衡）
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

    print(f"\n最终数据集大小:")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(valid_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # 为每个数据集添加递增的ID并保存
    def save_dataset(data, filename):
        df = pd.DataFrame(data)
        # 添加递增ID
        df['id'] = range(1, len(df) + 1)
        # 重命名列
        df = df[['id', 'text', 'sentiment']]
        df = df.rename(columns={'sentiment': 'label'})
        # 保存文件
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已保存: {filename}")

    # 保存三个数据集
    save_dataset(train_data, 'train.csv')
    save_dataset(valid_data, 'valid.csv')
    save_dataset(test_data, 'test.csv')

    # 验证数据集中的类别分布
    def check_distribution(data, dataset_name):
        sentiment_counts = defaultdict(int)
        for item in data:
            sentiment_counts[item['sentiment']] += 1

        print(f"\n{dataset_name}类别分布:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  {sentiment}: {count} 条 ({percentage:.2f}%)")

    check_distribution(train_data, "训练集")
    check_distribution(valid_data, "验证集")
    check_distribution(test_data, "测试集")

    print(f"\n处理完成！所有文件已保存到: {output_folder}")


# 使用示例
if __name__ == "__main__":
    # 设置新闻数据集输入文件夹路径（包含5000多个CSV文件）
    # input_folder = r"C:\software\pycharm\PyCharm 2024.1.3\Project\FinSA-MoE\data\raw\news"

    # 设置新闻数据集输出文件夹路径
    # output_folder = r"C:\software\pycharm\PyCharm 2024.1.3\Project\FinSA-MoE\data\processed\news"
    
    # 设置股评数据集输入文件夹路径
    input_folder = r"../data/raw/forum"

    # 设置股评数据集输出文件夹路径
    output_folder = r"../data/processed/forum"

    # 处理文件
    process_stock_news_files(input_folder, output_folder)