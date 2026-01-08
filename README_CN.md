[English](./README.md) | 简体中文
# FinSA-MoE：面向跨领域中文金融情绪识别的参数高效混合专家框架

## 整体框架
FinSA-MoE（Financial Sentiment Analysis-Mixture-of-Experts），一种面向跨领域中文金融情绪识别的参数高效混合专家框架，如图1所示。该框架通过对GLM-Z1-9B模型的LoRA微调，构建了面向金融新闻与股票评论的领域专家，并引入Mixture-of-Experts(MoE)架构以自适应选择专家，实现跨领域情绪识别的高效泛化。同时，我们构建了FinNF数据集，包含166万条带情感极性的高质量中文金融新闻与股评文本，为FinSA-MoE框架的微调与评测提供了数据基础。实验结果表明，FinSA-MoE在准确率、精确率、召回率及F1值等指标上均显著优于传统深度学习模型与现有金融大语言模型，在高噪声、语义模糊的文本中表现出更强的稳健性与泛化能力。
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
会生成LoRA微调后的模型，FinSA-MoE经过FinNF新闻和股评数据集微调后，分别在测试集上的表现如图3、4所示：
<p align="center">
  <img src="./img/FinSA-MoE vs. Baselines on the FinNF News Test Set.png" width="600"/>
</p>
<p align="center">
  图 3  FinSA-MoE vs. Baselines在FinNF新闻测试集上的表现
</p>
<p align="center">
  <img src="./img/FinSA-MoE vs. Baselines on the FinNF Forum Test Set.png" width="600"/>
</p>
<p align="center">
  图 3  FinSA-MoE vs. Baselines在FinNF股评测试集上的表现
</p>

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

## 评估
运行如下代码：
```bash
python scripts/evaluate.py --model_path 模型路径 --dataset_type news --data_path 数据集路径
```






