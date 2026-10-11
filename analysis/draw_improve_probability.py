"""绘制各股票改善概率的横向条形图。

数据来自 results/aggregated/count.csv（由 analysis/improve_probability.py 生成），
输出 figures/comparison/improve_probability.png。

用法：
    python analysis/draw_improve_probability.py
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
    df = pd.read_csv(AGGREGATED_DIR / "count.csv")
    df = df.sort_values("improve_prob", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df["ts_code"], df["improve_prob"], color="steelblue")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=9)

    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, label="50% Baseline")
    ax.set_title("Improve Probability Contrast", fontsize=14, pad=15)
    ax.set_xlabel("Improve Probability", fontsize=12)
    ax.set_ylabel("Stock Code", fontsize=12)
    ax.set_xlim(0, 0.65)
    ax.legend()
    plt.tight_layout()

    save_dir = FIGURES_DIR / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "improve_probability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("图片已保存至", out)


if __name__ == "__main__":
    main()
