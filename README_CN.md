[English](./README.md) | 简体中文
# **金融情感分析中面向领域异质性的自适应专家专门化**


## 整体框架
FinSA-MoE（Financial Sentiment Analysis Mixture-of-Experts，金融情感分析专家混合模型），这是一种参数高效的自适应专家专门化框架，旨在解决金融情感分析中的领域异质性问题（见图 I）。FinSA-MoE 基于共享的主干模型，通过 低秩适配（Low-Rank Adaptation，LoRA） 构建领域特定的专家，并采用 专家混合（Mixture-of-Experts，MoE）架构，以实现对异构金融文本来源的自适应专家选择。此外，我们还引入了 FinNF，这是一个大规模中文金融情感数据集，包含 166 万篇带有情感极性标注的金融新闻和股票评论。该数据集为在异质数据环境下评估情感分析模型提供了一个全面的基准。实验结果表明，FinSA-MoE 在多项评测指标上均稳定优于传统深度学习模型和现有的金融领域大语言模型，充分证明了自适应专家专门化机制在大规模金融情感分析任务中的有效性。
<p align="center">
  <img src="./img/FinSA-MoE.svg" width="600"/>
</p>
<p align="center">
  图 1  FinSA-MoE 整体框架
</p>


## 数据集说明
本文所使用的 **FinNF 数据集** 已开源。请将下载并解压后的数据文件放置于以下目录中：
```text
FinSA-MoE/data
```
*如果想训练其他领域的专家，可以使用其他相关数据集进行微调，字段要保证有id,text,label*
### 下载方式
FinNF 数据集可通过以下途径下载：<br>
👉 [Google Drive](https://drive.google.com/drive/folders/1NqjRtXBjntKkiNlxBkvgnzRnrbw4M0PY?usp=sharing)<br>
👉 [百度网盘](https://pan.baidu.com/s/1P7tps9G-8rcEBrslOTyXtQ?pwd=8888)|提取码：8888<br>

### 📊 数据集统计📊
我们分别对新闻和股评数据集采用98%/1%/1%与60%/20%/20% 的划分比例。FinNF 数据集的情绪极性分布见图2：
<p align="center">
  <img src="./img/Data.png" width="600"/>
</p>
<p align="center">
  图 2  FinNF数据集情绪极性分布
</p>

## 环境要求
python版本：>=3.11

通过git将项目代码下载到本地：
```bash
git clone https://github.com/chtkg/FinSA-MoE.git
cd FinSA-MoE
```
依次使用下面语句进行环境配置：
```bash
conda create -n myenv python=3.11 -y   # myenv为新建的虚拟环境名称
conda activate myenv      # 激活虚拟环境
pip install -r requirements.txt   # 安装库
```
## 单专家微调
运行如下代码:
```bash
python scripts/train_news_expert.py    # 也可以使用train_forum_expert.py，这两个文件一样
```
会生成LoRA微调后的模型。

## MoE联合训练
首先合并经过LoRA微调后的几个子模型，运行如下代码：
```bash
python scripts/merge_expert_lora.py    # 会生成MoE初始模型
```
引入一个路由器，使FinSA-MoE框架具备“自动选择合适专家”的能力，我们将新闻和股评数据的训练集、验证集、测试集混合用于联合训练，并使用领域标签作为监督信号（数据集字段：domain）。前500 steps，我们先冻结单专家微调阶段的新闻专家和股评专家的全部参数，仅训练Router和分类头，随后解冻所有专家 LoRA 参数与路由器联合训练，路由器参数和两个 LoRA 专家参数共同接受梯度更新。<br>
生成moe_init模型文件后，运行如下代码:
```bash
python scripts/train_joint_moe.py   # 会生成仅训练Router和分类头的模型：router_warmup；和最终模型：moe_joint
```

## 评估和推理
运行如下代码评估模型性能：
```bash
python scripts/evaluate.py --model_path 模型路径 --dataset_type news --data_path 数据集路径
```
使用FinSA-MoE/main.py可以进行推理，也可以使用FinSA-MoE/src/inference.py进行推理
```bash
python main.py --mode inference --data_type forum --model_path 模型路径
python src/inference.py
```
<p align="center">
  <img src="./img/inference.png" width="600"/>
</p>
<p align="center">
  图 3 推理
</p>







