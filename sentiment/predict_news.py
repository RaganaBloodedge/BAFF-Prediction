"""FinBERT2 情绪推理示例：对给定新闻标题输出情绪标签与概率分布。

用法：
    python sentiment/predict_news.py                  # 使用内置示例新闻
    python sentiment/predict_news.py "某条新闻标题" ...  # 对指定文本推理
"""

import sys
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import resolve_model_dir  # noqa: E402

LABELS = {0: "负面", 1: "正面"}

SAMPLE_NEWS = [
    "国家能源局介绍，我国充电设施发展提质升级，呈现三大显著特点。",
    "某科技公司季度盈利超预期，股价上涨。",
    "宏观经济不确定性增加，投资者谨慎。",
]


def main():
    texts = sys.argv[1:] or SAMPLE_NEWS

    model_dir = resolve_model_dir()
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)

    for text, prob in zip(texts, probs):
        label = LABELS[int(torch.argmax(prob))]
        print(f"新闻: {text}")
        print(f"情感: {label}, 概率分布: {prob.tolist()}")
        print("-" * 60)


if __name__ == "__main__":
    main()
