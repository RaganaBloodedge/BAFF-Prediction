"""用 FinBERT2 对新闻标题打情绪分，聚合成日度情绪文件。

    data/news_{code}.csv  →  data/sentiment_{code}.csv

情绪分定义：score = P(正面) − P(负面) ∈ [−1, 1]；
同一交易日的多条新闻取算术平均作为当日情绪分。

模型目录优先取环境变量 FINBERT_MODEL_DIR，默认为 models/FinBERT2-base_SC_2，
权重不在本仓库内（见 third_party/FinBERT2.md）。

用法：
    python sentiment/build_sentiment_from_csv.py
    python sentiment/build_sentiment_from_csv.py 000547 600010
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, load_stocks, resolve_model_dir  # noqa: E402

# 二分类输出列顺序：0=负面，1=正面
NEG_LABEL, POS_LABEL = 0, 1


def load_model(model_dir: Path | str | None = None):
    """加载 FinBERT2 微调模型与 tokenizer。"""
    model_dir = Path(model_dir) if model_dir else resolve_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"找不到情绪模型目录: {model_dir}\n"
            "请先按 third_party/FinBERT2.md 微调 FinBERT2 并放入该目录，"
            "或用环境变量 FINBERT_MODEL_DIR 指向已有权重。"
        )
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_scores(texts: List[str], tokenizer, model, device,
                   batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    """对一批文本推理，返回 score = P(正面) − P(负面)，范围约 [−1, 1]。"""
    if not texts:
        return np.zeros((0,), dtype=np.float32)

    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        probs_np = probs.cpu().numpy()
        all_scores.append(probs_np[:, POS_LABEL] - probs_np[:, NEG_LABEL])

    return np.concatenate(all_scores, axis=0)


def build_daily_sentiment(code: str, input_csv: str | Path | None = None,
                          output_csv: str | Path | None = None,
                          text_col: str = "text",
                          date_col: str = "trade_date") -> str:
    """从新闻 CSV 生成按交易日聚合的日度情绪文件，返回输出路径。"""
    input_csv = Path(input_csv) if input_csv else DATA_DIR / f"news_{code}.csv"
    output_csv = Path(output_csv) if output_csv else DATA_DIR / f"sentiment_{code}.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"找不到新闻 CSV 文件: {input_csv}")

    df = pd.read_csv(input_csv)
    if date_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"{input_csv} 必须包含列: {date_col}, {text_col}")

    work = df[[date_col, text_col]].dropna()
    if work.empty:
        raise ValueError(f"{input_csv} 中没有有效的 {date_col}/{text_col} 数据")

    tokenizer, model, device = load_model()
    scores = predict_scores(work[text_col].astype(str).tolist(),
                            tokenizer, model, device, batch_size=32)

    graded = pd.DataFrame({
        date_col: work[date_col].astype(str).values,
        "sentiment_score": scores,
    })
    daily = (graded.groupby(date_col, as_index=False)["sentiment_score"]
             .mean().sort_values(date_col))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[{code}] 已生成日度情绪文件: {output_csv}（{len(daily)} 个交易日）")
    return str(output_csv)


def main():
    parser = argparse.ArgumentParser(description="用 FinBERT2 生成日度情绪分")
    parser.add_argument("codes", nargs="*", help="股票代码，默认取 configs/stocks.txt")
    args = parser.parse_args()

    codes = args.codes or [c.split(".")[0] for c in load_stocks()]
    for code in codes:
        build_daily_sentiment(code.split(".")[0])


if __name__ == "__main__":
    main()
