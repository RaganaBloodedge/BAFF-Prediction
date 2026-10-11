"""统计“情绪方向与真实价格趋势一致”的比例。

对每只股票：当日情绪为正且价格上行、或情绪为负且价格下行记为一致。
输入 results/aggregated/all_results_aggregated.csv，输出
results/aggregated/trend_same_statistics.csv 与 trend_same_barplot.png
（对应论文 Table 6）。

用法：
    python analysis/trend_consistency.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import AGGREGATED_DIR  # noqa: E402


def main():
    df = pd.read_csv(AGGREGATED_DIR / "all_results_aggregated.csv")

    df["validation_date"] = pd.to_datetime(df["validation_date"])
    df = df.sort_values(["ts_code", "validation_date"])

    df["price_change"] = df.groupby("ts_code")["true_price"].diff()
    df["price_up"] = df["price_change"] > 0
    df["price_down"] = df["price_change"] < 0
    df["sentiment_positive"] = df["sentiment_score"] > 0
    df["sentiment_negative"] = df["sentiment_score"] < 0
    df["trend_same"] = ((df["price_up"] & df["sentiment_positive"])
                        | (df["price_down"] & df["sentiment_negative"]))

    valid = df[df["price_change"].notna()]

    group_stats = valid.groupby("ts_code").agg(
        same_count=("trend_same", "sum"),
        total_valid=("trend_same", "count"),
    ).reset_index()
    group_stats["same_ratio"] = group_stats["same_count"] / group_stats["total_valid"]

    total_row = pd.DataFrame({
        "ts_code": ["ALL"],
        "same_count": [valid["trend_same"].sum()],
        "total_valid": [len(valid)],
        "same_ratio": [valid["trend_same"].sum() / len(valid)],
    })
    result = pd.concat([group_stats, total_row], ignore_index=True)

    result.to_csv(AGGREGATED_DIR / "trend_same_statistics.csv",
                  index=False, encoding="utf-8-sig")

    # 柱状图：各股票趋势一致率，ALL 用橙色突出
    plt.figure(figsize=(12, 6))
    colors = ["steelblue"] * (len(result) - 1) + ["darkorange"]
    bars = plt.bar(result["ts_code"], result["same_ratio"], color=colors, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f"{height:.2%}", ha="center", va="bottom", fontsize=9)

    plt.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="50% Baseline")
    plt.xlabel("Stock Code", fontsize=12)
    plt.ylabel("Rate", fontsize=12)
    plt.title("Rate of Price and Sentiment Sharing the Same Trend", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 0.7)
    plt.legend()
    plt.tight_layout()

    out_png = AGGREGATED_DIR / "trend_same_barplot.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print("柱状图已保存为", out_png)
    print(result)


if __name__ == "__main__":
    main()
