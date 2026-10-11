"""统计检验工具。

论文原文只报告了逐股 MAE 的差值（``sentiment_improve``），没有回答
「这点改善是否超出随机波动」——本模块补上这一步。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class TestResult:
    """配对检验结果。"""

    n: int
    mean_diff: float
    median_diff: float
    stat: float
    pvalue: float
    method: str

    def as_dict(self) -> dict:
        return asdict(self)


def paired_test(
    candidate,
    baseline,
    method: str = "wilcoxon",
    alternative: str = "two-sided",
) -> TestResult:
    """配对检验：候选误差与基线误差是否存在系统性差异。

    参数
    ----
    candidate, baseline:
        逐样本（或逐股票的聚合）误差序列，长度相同、一一配对。
        差值定义为 ``baseline - candidate``，故 **正值表示候选更优**。
    method:
        ``"wilcoxon"``（默认，不假设正态，样本量小时更稳）或 ``"ttest"``。
    """
    c = np.asarray(candidate, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if c.size != b.size:
        raise ValueError(f"两序列长度必须一致: {c.size} vs {b.size}")
    if c.size < 2:
        raise ValueError("配对检验至少需要 2 个样本")

    diff = b - c
    if np.allclose(diff, 0):
        # scipy 在全零差值上会给出 nan，这里显式返回不显著
        return TestResult(c.size, 0.0, 0.0, 0.0, 1.0, method)

    if method == "wilcoxon":
        try:
            res = sps.wilcoxon(diff, alternative=alternative)
        except ValueError:
            # 全为同一符号时 wilcoxon 会退化，改用符号检验的近似
            res = sps.binomtest(int((diff > 0).sum()), diff.size, 0.5)
    elif method == "ttest":
        res = sps.ttest_rel(b, c, alternative=alternative)
    else:
        raise ValueError(f"未知检验方法: {method}")

    return TestResult(
        n=int(c.size),
        mean_diff=float(diff.mean()),
        median_diff=float(np.median(diff)),
        stat=float(res.statistic),
        pvalue=float(res.pvalue),
        method=method,
    )


def bootstrap_ci(
    values,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """对均值的 bootstrap 置信区间。

    返回 ``(均值, 下界, 上界)``。用于衡量改善量的估计精度——16 只股票的
    均值如果区间跨越 0，就不能声称有效。
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("bootstrap 至少需要 2 个样本")
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须位于 (0, 1)")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def summarize(values, prefix: str = "", n_boot: int = 10000, seed: int = 0) -> dict:
    """把一组逐样本差异汇总成可直接写进表格的字典。"""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    mean, lo, hi = bootstrap_ci(arr, n_boot=n_boot, seed=seed)
    test = paired_test(arr, np.zeros_like(arr), method="wilcoxon")
    return {
        f"{prefix}n": int(arr.size),
        f"{prefix}mean": mean,
        f"{prefix}ci_low": lo,
        f"{prefix}ci_high": hi,
        f"{prefix}positive_ratio": float((arr > 0).mean()),
        f"{prefix}pvalue": test.pvalue,
    }
