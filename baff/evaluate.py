"""机制评估：把预测明细表转换成各机制的误差与统计量。

全部以 ``results/predictions.csv`` 为输入，因此**不需要重训网络**——换一个
机制或换一个 α 只是重算一遍价格列。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import metrics
from .mechanism import SIGNAL_TRANSFORMS, apply_mechanism

GROUP_COLS = ("seed", "val_date")


def apply_signal(
    preds: pd.DataFrame,
    transform: str = "raw",
    group_cols: Sequence[str] = GROUP_COLS,
) -> pd.Series:
    """按指定变换生成参与调制的信号列。

    ``cs_zscore`` 按 (seed, val_date) 做横截面标准化——只用当日截面信息，
    不含未来数据。
    """
    if transform not in SIGNAL_TRANSFORMS:
        raise KeyError(f"未知信号变换: {transform}，可选: {sorted(SIGNAL_TRANSFORMS)}")
    sentiment = preds["sentiment"].astype(float)

    if transform == "raw":
        return sentiment.copy()
    if transform == "tanh":
        return np.tanh(sentiment)
    # cs_zscore
    def _z(block: pd.Series) -> pd.Series:
        sd = block.std()
        return (block - block.mean()) / sd if sd > 0 else pd.Series(0.0, index=block.index)

    return preds.groupby(list(group_cols), sort=False)["sentiment"].transform(_z)


def evaluate(
    preds: pd.DataFrame,
    mechanism: str,
    transform: str = "raw",
    **params: object,
) -> pd.DataFrame:
    """返回逐样本明细，含基线误差、调制后误差及其差值。

    ``improvement = ae_base - ae_cand``，**正值表示调制更优**。
    """
    df = preds.copy()
    df["signal"] = apply_signal(df, transform)
    df["adj_return"] = apply_mechanism(
        mechanism, df["pred_return"].to_numpy(), df["signal"].to_numpy(), **params
    )
    # 基线与调制都走「前收 × (1 + 收益率)」这条同构路径。若基线直接用
    # pred_close，α=0 时两条路径的浮点舍入不同，会产出 1e-12 量级的假改善。
    df["base_close"] = df["prev_close"] * (1.0 + df["pred_return"])
    df["adj_close"] = df["prev_close"] * (1.0 + df["adj_return"])

    df["ae_base"] = (df["true_close"] - df["base_close"]).abs()
    df["ae_cand"] = (df["true_close"] - df["adj_close"]).abs()
    # 相对误差，便于跨股票比较（价格量级差异极大）
    denom = df["true_close"].replace(0, np.nan)
    df["re_base"] = df["ae_base"] / denom * 100.0
    df["re_cand"] = df["ae_cand"] / denom * 100.0
    df["improvement"] = df["ae_base"] - df["ae_cand"]
    df["transform"] = transform
    df["mechanism"] = mechanism
    return df


def by_stock(detail: pd.DataFrame) -> pd.DataFrame:
    """按 (股票, 种子) 聚合，与论文表 4 的口径一致。"""
    g = detail.groupby(["ts_code", "seed"], sort=False)
    out = pd.DataFrame({
        "MAE_base": g["ae_base"].mean(),
        "MAE_cand": g["ae_cand"].mean(),
        "MRE_base": g["re_base"].mean(),
        "MRE_cand": g["re_cand"].mean(),
        "n_days": g["ae_base"].size(),
    }).reset_index()
    out["improvement"] = out["MAE_base"] - out["MAE_cand"]
    # 相对口径：股票价格量级从个位数到三位数，绝对 MAE 的均值会被高价股主导
    out["improvement_pct"] = out["improvement"] / out["MAE_base"] * 100.0
    out["improved"] = out["improvement"] > 0
    return out


def summarize(detail: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    """把明细汇总成一行可直接进结果表/图的统计量。

    同时给出「逐股票改善量」与「逐样本改善量」两个视角的均值、bootstrap
    置信区间与配对检验 p 值——前者对应论文口径，后者样本量更大、更敏感。
    """
    from . import stats as st

    stocks = by_stock(detail)
    per_stock = stocks["improvement"].to_numpy()
    per_stock_pct = stocks["improvement_pct"].to_numpy()
    per_sample = detail["improvement"].to_numpy()

    def _pack(values: np.ndarray, prefix: str) -> dict:
        mean, lo, hi = st.bootstrap_ci(values, n_boot=n_boot, seed=seed)
        test = st.paired_test(values, np.zeros_like(values), method="wilcoxon")
        return {
            f"{prefix}mean": mean,
            f"{prefix}ci_low": lo,
            f"{prefix}ci_high": hi,
            f"{prefix}pos_ratio": float((values > 0).mean()),
            f"{prefix}pvalue": float(test.pvalue),
        }

    row: dict = {
        "mechanism": detail["mechanism"].iloc[0],
        "transform": detail["transform"].iloc[0],
        "n_samples": int(len(detail)),
        "n_stocks": int(detail["ts_code"].nunique()),
        "MAE_base": float(detail["ae_base"].mean()),
        "MAE_cand": float(detail["ae_cand"].mean()),
        "RMSE_base": float(np.sqrt((detail["ae_base"] ** 2).mean())),
        "RMSE_cand": float(np.sqrt((detail["ae_cand"] ** 2).mean())),
    }
    row.update(_pack(per_stock, "stock_"))
    row.update(_pack(per_stock_pct, "stockpct_"))
    row.update(_pack(per_sample, "sample_"))
    return row


def directional_stats(detail: pd.DataFrame) -> dict:
    """方向层面的表现：调制是否改善了涨跌方向判断。"""
    true_ret = detail["true_return"].to_numpy()
    base_ret = detail["pred_return"].to_numpy()
    cand_ret = detail["adj_return"].to_numpy()
    return {
        "dir_acc_base": metrics.directional_accuracy(true_ret, base_ret),
        "dir_acc_cand": metrics.directional_accuracy(true_ret, cand_ret),
        "ic_base": metrics.information_coefficient(base_ret, true_ret),
        "ic_cand": metrics.information_coefficient(cand_ret, true_ret),
        "sentiment_ic": metrics.information_coefficient(
            detail["signal"].to_numpy(), true_ret
        ),
    }
