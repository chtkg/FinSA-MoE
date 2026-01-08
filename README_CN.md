[English](./README.md) | 简体中文
# FinSA-MoE：面向跨领域中文金融情绪识别的参数高效混合专家框架
FinSA-MoE（Financial Sentiment Analysis-Mixture-of-Experts），一种面向跨领域中文金融情绪识别的参数高效混合专家框架。该框架通过对GLM-Z1-9B模型的LoRA微调，构建了面向金融新闻与股票评论的领域专家，并引入Mixture-of-Experts(MoE)架构以自适应选择专家，实现跨领域情绪识别的高效泛化。同时，我们构建了FinNF数据集，包含166万条带情感极性的高质量中文金融新闻与股评文本，为FinSA-MoE框架的微调与评测提供了数据基础。实验结果表明，FinSA-MoE在准确率、精确率、召回率及F1值等指标上均显著优于传统深度学习模型与现有金融大语言模型，在高噪声、语义模糊的文本中表现出更强的稳健性与泛化能力。
<p align="center">
  <img src="./img/FinSA-MoE.svg" width="600"/>
</p>
<p align="center">
  图 1  FinSA-MoE 整体框架
</p>


