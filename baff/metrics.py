"""预测评估指标。

全部为纯函数，输入一维 ``numpy`` 数组，输出标量，便于单测与在面板数据上聚合。

单位约定：MAE / RMSE 与价格同量纲；MRE 为百分比；方向准确率与 IC 为无量纲。
"""

from __future__ import annotations

import numpy as np


def _as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("输入为空数组")
    return arr


def mae(y_true, y_pred) -> float:
    """平均绝对误差。"""
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """均方根误差。"""
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mre(y_true, y_pred) -> float:
    """平均相对误差（百分比）。

    ``y_true`` 中出现 0 的样本会被跳过——否则相对误差无定义。
    """
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    mask = y_true != 0
    if not mask.any():
        raise ValueError("y_true 全为 0，相对误差无定义")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def directional_accuracy(return_true, return_pred) -> float:
    """方向准确率：预测涨跌方向与真实方向一致的样本占比。

    收益率为 0 的样本方向无意义，予以剔除。
    """
    rt, rp = _as_1d(return_true), _as_1d(return_pred)
    mask = (rt != 0) & (rp != 0)
    if not mask.any():
        raise ValueError("样本方向全为 0，方向准确率无定义")
    return float(np.mean(np.sign(rt[mask]) == np.sign(rp[mask])))


def information_coefficient(pred, target) -> float:
    """IC：预测值与真实值的 Spearman 秩相关。

    比 Pearson 更稳健，不假设线性关系；样本量小或存在极值时优先使用。
    """
    from scipy.stats import spearmanr

    a, b = _as_1d(pred), _as_1d(target)
    if a.size != b.size:
        raise ValueError(f"长度不一致: {a.size} vs {b.size}")
    if a.size < 3:
        raise ValueError("Spearman 相关至少需要 3 个样本")
    rho, _ = spearmanr(a, b)
    return float(rho)


def improvement(base_metric: float, candidate_metric: float) -> float:
    """改善量 = 基线误差 − 候选误差。正值表示候选更优（误差更小）。"""
    return float(base_metric - candidate_metric)
