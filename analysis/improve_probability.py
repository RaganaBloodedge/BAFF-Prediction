"""统计每只股票的改善概率：AE_base > AE_sent 的样本占比。

输入 results/aggregated/all_results_aggregated.csv，输出
results/aggregated/count.csv（对应论文 Table 5）。

用法：
    python analysis/improve_probability.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import AGGREGATED_DIR  # noqa: E402


def main():
    df = pd.read_csv(AGGREGATED_DIR / "all_results_aggregated.csv")

    total = df.groupby("ts_code").size().reset_index(name="total")
    improve = (df[df["ae_base"] > df["ae_sent"]]
               .groupby("ts_code").size().reset_index(name="improve"))

    result = total.merge(improve, on="ts_code", how="left").fillna(0)
    result["improve_prob"] = result["improve"] / result["total"]
    result = result[["ts_code", "improve_prob"]]

    out = AGGREGATED_DIR / "count.csv"
    result.to_csv(out, index=False)
    print(f"结果已保存至 {out}")
    print(result)


if __name__ == "__main__":
    main()
