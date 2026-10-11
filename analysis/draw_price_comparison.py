"""为每只股票绘制 真实价 / 基线预测 / 情绪调制预测 三线对比图。

输入 results/aggregated/all_results_aggregated.csv，
输出 figures/comparison/{code}_price_comparison.png。

用法：
    python analysis/draw_price_comparison.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import AGGREGATED_DIR, FIGURES_DIR  # noqa: E402


def main():
    save_dir = FIGURES_DIR / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(AGGREGATED_DIR / "all_results_aggregated.csv")
    df["validation_date"] = pd.to_datetime(df["validation_date"])
    df = df.sort_values(["ts_code", "validation_date"])

    for code in df["ts_code"].unique():
        subset = df[df["ts_code"] == code]

        plt.figure(figsize=(12, 6))
        plt.plot(subset["validation_date"], subset["true_price"],
                 label="True Price", marker="o", linestyle="-", linewidth=2)
        plt.plot(subset["validation_date"], subset["pred_price_base"],
                 label="Pred Price Base", marker="s", linestyle="--")
        plt.plot(subset["validation_date"], subset["pred_price_sent"],
                 label="Pred Price Sent", marker="^", linestyle=":")

        plt.title(f"Price Prediction Comparison for {code}")
        plt.xlabel("Validation Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / f"{code}_price_comparison.png", dpi=150)
        plt.close()

    print(f"所有图片已保存至文件夹：{save_dir}")


if __name__ == "__main__":
    main()
