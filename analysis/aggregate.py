"""汇总每日预测结果，计算每只股票的 MAE/MRE 与情绪改善量。

遍历 results/{validation_date}/ 下的单日结果 CSV，合并为
results/aggregated/all_results_aggregated.csv，并输出
results/aggregated/stock_MAE_MRE.csv 与每只股票的 re_base / re_sent 误差曲线。

用法：
    python analysis/aggregate.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import AGGREGATED_DIR, RESULTS_DIR  # noqa: E402


def load_all_results(result_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """读取所有单日结果目录（跳过 aggregated/），合并成一个 DataFrame。"""
    dfs = []
    for folder in sorted(Path(result_dir).iterdir()):
        if not folder.is_dir() or folder.name == "aggregated":
            continue
        for file in sorted(folder.glob("*.csv")):
            try:
                dfs.append(pd.read_csv(file))
            except Exception as e:
                print("读取失败:", file, e)

    if not dfs:
        raise ValueError(f"{result_dir} 下未读取到任何单日结果 csv")

    data = pd.concat(dfs, ignore_index=True)
    # validation_date 可能是 “20260327.0” 这类浮点字符串，先截断再解析
    data["validation_date"] = data["validation_date"].astype(str).str.split(".").str[0]
    data["validation_date"] = pd.to_datetime(data["validation_date"], format="%Y%m%d")
    return data.sort_values(["ts_code", "validation_date"]).reset_index(drop=True)


def compute_stock_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """按股票聚合 AE/RE 均值，并计算 sentiment_improve = MAE_base − MAE_sent。"""
    metrics = data.groupby("ts_code").agg({
        "ae_base": "mean",
        "ae_sent": "mean",
        "re_base": "mean",
        "re_sent": "mean",
    }).rename(columns={
        "ae_base": "MAE_base",
        "ae_sent": "MAE_sent",
        "re_base": "MRE_base",
        "re_sent": "MRE_sent",
    }).reset_index()

    metrics["sentiment_improve"] = metrics["MAE_base"] - metrics["MAE_sent"]
    return metrics


def plot_each_stock(data: pd.DataFrame, save_dir: Path = AGGREGATED_DIR) -> None:
    """为每只股票绘制 re_base / re_sent 随时间变化的误差曲线。"""
    for ts_code, df in data.groupby("ts_code"):
        plt.figure(figsize=(8, 4))
        plt.plot(df["validation_date"], df["re_base"], label="re_base")
        plt.plot(df["validation_date"], df["re_sent"], label="re_sent")
        plt.title(ts_code)
        plt.ylabel("Relative Error")
        plt.xlabel("Date")
        plt.legend()
        plt.savefig(save_dir / f"{ts_code}_re_curve.png", dpi=150)
        plt.close()


def main():
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all_results()
    data.to_csv(AGGREGATED_DIR / "all_results_aggregated.csv", index=False)

    metrics = compute_stock_metrics(data)
    metrics.to_csv(AGGREGATED_DIR / "stock_MAE_MRE.csv", index=False)

    plot_each_stock(data)
    print("图像已保存至:", AGGREGATED_DIR)
    print(metrics)


if __name__ == "__main__":
    main()
