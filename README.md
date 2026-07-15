English | [简体中文](./README_CN.md)
# FinSA-MoE: A Parameter-Efficient Mixture-of-Experts Framework for Cross-Domain Chinese Financial Sentiment Analysis

## Overall Framework
FinSA-MoE (Financial Sentiment Analysis Mixture-of-Experts), a parameter-efficient adaptive expert specialization framework designed for financial sentiment analysis under domain heterogeneity(see Fig.1). This framework introduces domain-specific experts tailored for financial news and stock commentary via LoRA fine-tuning, combined with a Mixture-of-Experts (MoE) architecture that dynamically routes inputs to the most relevant expert, enhancing cross-domain generalization in heterogeneous financial text sources. We also introduce the FinNF dataset, which includes 1.66 million high-quality financial news articles and stock commentary annotated with sentiment polarity, providing a comprehensive benchmark for evaluating the FinSA-MoE framework. Experimental results demonstrate that FinSA-MoE outperforms traditional deep learning models and existing financial LLMs across multiple evaluation metrics, including accuracy, precision, recall, and F1 score, and demonstrates robustness and cross-domain generalization in high-noise, semantically ambiguous settings.
<p align="center">
  <img src="./img/figure1.png" width="600"/>
</p>
<p align="center">
Figure 1: Overall architecture of the proposed FinSA-MoE framework.
</p>


## Dataset Description
The **FinNF dataset** used in this work has been publicly released. Please place the downloaded and extracted data files in the following directory:
```text
FinSA-MoE/data
```
*If you want to train experts for other domains, you can fine-tune the model using other relevant datasets. The dataset fields should include id, text, and label.*
### Download Links
The FinNF dataset can be downloaded from the following sources:<br>
👉 [Google Drive](https://drive.google.com/drive/folders/1NqjRtXBjntKkiNlxBkvgnzRnrbw4M0PY?usp=sharing)<br>
👉 [Baidu Netdisk](https://pan.baidu.com/s/1P7tps9G-8rcEBrslOTyXtQ?pwd=8888)|Extraction code：8888<br>

### 📊 Dataset Statistics 📊
We split the news dataset and the stock forum dataset using ratios of 98%/1%/1% and 80%/10%/10%, respectively. The sentiment polarity distribution of the FinNF dataset is shown in Figure 2.
<p align="center">
  <img src="./img/data.png" width="600"/>
</p>
<p align="center">
  Figure 2  Sentiment polarity distribution of the FinNF dataset
</p>

## Environment Requirements
Python version：>=3.11

Download the project code via Git：
```bash
git clone https://github.com/chtkg/FinSA-MoE.git
cd FinSA-MoE
```
Configure the environment using the following commands:
```bash
conda create -n myenv python=3.11 -y   # myenv is the name of the virtual environment
conda activate myenv      # activate the virtual environment
pip install -r requirements.txt   # install dependencies
```
## Single-Expert Fine-Tuning
Run the following command:
```bash
python scripts/train_news_expert.py    # you can also use train_forum_expert.py; the two files are identical
```
This will generate a LoRA fine-tuned model. 


## MoE Joint Training
First, merge the sub-models fine-tuned with LoRA by running the following command:
```bash
python scripts/merge_expert_lora.py    # generates the initial MoE model
```
A router is introduced to enable the FinSA-MoE framework to automatically select appropriate experts. We mix the training, validation, and test sets of the news and forum datasets for joint training, and use domain labels as supervision signals (dataset field: domain).
During the first 500 steps, all parameters of the news and forum experts from the single-expert fine-tuning stage are frozen, and only the router and classification head are trained. Subsequently, all expert LoRA parameters are unfrozen and jointly trained with the router. The router parameters and the two LoRA expert parameters are updated together via backpropagation. <br>
After generating the moe_init model file, run the following command:
```bash
python scripts/train_joint_moe.py   # generates the model trained only with router and classification head: router_warmup; and the final model: moe_joint
```

## Evaluation and Inference
Run the following command to evaluate model performance:
```bash
python scripts/evaluate.py --model_path model_path --dataset_type news --data_path dataset_path
```
nference can be performed using FinSA-MoE/main.py or FinSA-MoE/src/inference.py:
```bash
python main.py --mode inference --data_type forum --model_path model_path
python src/inference.py
```
<p align="center">
  <img src="./img/inference.png" width="600"/>
</p>
<p align="center">
  Figure 3 Inference
</p

