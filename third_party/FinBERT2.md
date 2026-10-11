# FinBERT2 —— 上游依赖说明

`sentiment/` 不附带语言模型，而是调用熵简科技开源的 **FinBERT2** 套件完成金融文本的
情绪编码。FinBERT2 是独立开源项目，本仓库**不重新分发**其源码与权重。

## 上游信息

| 项目 | 地址 |
|---|---|
| 代码仓库 | https://github.com/valuesimplex/FinBERT2 |
| 预训练模型 | https://huggingface.co/valuesimplex-ai-lab/FinBERT2-base |
| 论文 | Xu et al. *FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap in Finance-Specific Deployment of Large Language Models*. KDD '25. |
| 论文 DOI | https://doi.org/10.1145/3711896.3737219 |
| 许可 | MIT License, Copyright (c) 2024 mengyougithub |

本项目实际使用的部分是 FinBERT2 套件中的 `Fin-labeler` 下游任务框架：
以 `FinBERT2-base` 为底座，在自建的二分类数据集上微调情绪分类头，
推理时输出 softmax 概率并取 `score = P(正面) − P(负面)`。

## 如何准备模型

1. 克隆上游仓库：

   ```bash
   git clone https://github.com/valuesimplex/FinBERT2.git
   ```

2. 按上游 `Fin-labeler/` 下的微调脚本训练情绪分类模型。

3. 把微调后的模型目录放到本仓库 `models/FinBERT2-base_SC_2/`，
   或用环境变量指向它：

   ```bash
   export FINBERT_MODEL_DIR=/path/to/FinBERT2-base_SC_2
   ```

模型权重为 GB 级，不随仓库分发。

## 引用

如果本项目对你有帮助，请一并引用上游工作：

```bibtex
@inproceedings{xu2025finbert2,
  author    = {Xu Xuan and Wen Fufang and Chu Beilin and Fu Zhibing and Lin Qinhong
               and Liu Jiaqi and Fei Binjie and Li Yu and Zhou Linna and Yang Zhongliang},
  title     = {FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap
               in Finance-Specific Deployment of Large Language Models},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery
               and Data Mining V.2 (KDD '25)},
  year      = {2025},
  doi       = {10.1145/3711896.3737219},
  url       = {https://doi.org/10.1145/3711896.3737219}
}
```
