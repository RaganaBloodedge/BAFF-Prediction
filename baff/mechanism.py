"""行为调制机制族。

论文把调制写死成一处公式：``m = 1 + α·sign(r_pred)·s``，α 固定 0.5。
本模块把它拆成两个可独立替换的部分，使"机制本身好不好"变成可做消融的问题：

1. **信号变换** ``transform(signal) -> signal'``
   决定情绪分以什么形式参与调制。原始取值严重偏正（720 个样本中 44% > 0.9，
   仅 19.6% 为负），直接使用会让调制因子长期贴在上下界附近、失去逐日区分度。

2. **调制机制** ``fn(pred_returns, signal', **params) -> adj_returns``
   决定预测收益率如何被信号改变。``sign`` 即论文原机制；``soft`` 用
   ``tanh`` 把方向一致性从二值改成连续，保留"预测幅度大且方向一致"这一
   更强证据；``magnitude`` 去掉方向一致性，用于检验方向项是否真的在起作用。

所有机制满足同一组边界性质，并由此构成单测契约：

- ``α = 0`` 或 ``signal = 0`` 时退化为纯价格预测（``adj == pred``）；
- ``pred_returns = 0`` 时输出为 0；
- 输出与输入等长，且为有限值。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_ALPHA = 0.5
# 日收益率量级约 1e-2，tau 取 0.02 对应约一个日收益标准差，
# 使 tanh 在正常收益区间内保持区分度而不过早饱和。
DEFAULT_TAU = 0.02

ArrayLike = Sequence[float] | np.ndarray


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _as_1d(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _check_pair(pred: np.ndarray, signal: np.ndarray) -> None:
    if pred.size != signal.size:
        raise ValueError(f"预测与信号长度不一致: {pred.size} vs {signal.size}")
    if pred.size == 0:
        raise ValueError("输入为空数组")


# --------------------------------------------------------------------------- #
# 信号变换
# --------------------------------------------------------------------------- #
def transform_raw(signal: ArrayLike, groups: ArrayLike | None = None) -> np.ndarray:
    """原样使用情绪分。保留原始量纲，也是论文的做法。"""
    return _as_1d(signal).copy()


def transform_tanh(signal: ArrayLike, groups: ArrayLike | None = None) -> np.ndarray:
    """压到 ``(-1, 1)``，抑制极端自信的那批近 ±1 取值。"""
    return np.tanh(_as_1d(signal))


def transform_cs_zscore(signal: ArrayLike, groups: ArrayLike | None = None) -> np.ndarray:
    """横截面 z-score：在同一个交易日内跨股票标准化。

    只用到当日截面信息，不含未来数据。物理含义是"该股当天的新闻情绪相对
    同批股票更正面还是更负面"，比绝对取值更贴近交易决策。
    未提供 ``groups`` 时退化为整体 z-score。
    """
    s = _as_1d(signal)
    if groups is None:
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else np.zeros_like(s)

    g = np.asarray(groups).reshape(-1)
    if g.size != s.size:
        raise ValueError(f"分组标签长度不一致: {g.size} vs {s.size}")

    out = np.zeros_like(s)
    for key in np.unique(g):
        m = g == key
        block = s[m]
        sd = block.std()
        out[m] = (block - block.mean()) / sd if sd > 0 else 0.0
    return out


SIGNAL_TRANSFORMS: dict[str, Callable[..., np.ndarray]] = {
    "raw": transform_raw,
    "tanh": transform_tanh,
    "cs_zscore": transform_cs_zscore,
}


# --------------------------------------------------------------------------- #
# 调制机制
# --------------------------------------------------------------------------- #
def no_modulation(pred_returns: ArrayLike, signal: ArrayLike, **_: object) -> np.ndarray:
    """纯价格基线：情绪分不参与调制，但仍要求长度一致。"""
    pred, s = _as_1d(pred_returns), _as_1d(signal)
    _check_pair(pred, s)
    return pred.copy()


def sign_modulation(
    pred_returns: ArrayLike,
    signal: ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    **_: object,
) -> np.ndarray:
    """论文原机制 ``m = 1 + α·sign(r)·s``。

    方向一致（``sign(r)·s > 0``）时 ``m > 1``，涨跌幅被放大；方向矛盾时
    ``m < 1``，预测被抑制。
    """
    pred, s = _as_1d(pred_returns), _as_1d(signal)
    _check_pair(pred, s)
    return (1.0 + alpha * np.sign(pred) * s) * pred


def soft_modulation(
    pred_returns: ArrayLike,
    signal: ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    tau: float = DEFAULT_TAU,
    **_: object,
) -> np.ndarray:
    """连续方向一致性 ``m = 1 + α·tanh(r/τ)·s``。

    把 ``sign(r)`` 换成 ``tanh(r/τ)``：预测幅度越大、置信越强时权重越高，
    幅度趋近 0 时调制自动关闭——而 ``sign`` 在 ``r`` 任意微小时都会全额作用。
    """
    if tau <= 0:
        raise ValueError(f"tau 必须为正，收到 {tau}")
    pred, s = _as_1d(pred_returns), _as_1d(signal)
    _check_pair(pred, s)
    return (1.0 + alpha * np.tanh(pred / tau) * s) * pred


def magnitude_modulation(
    pred_returns: ArrayLike,
    signal: ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    **_: object,
) -> np.ndarray:
    """去掉方向一致性，只按情绪强弱缩放：``m = 1 + α·s``。

    作为对照机制——若它与 ``sign`` 表现相当，说明方向项（论文机制的核心
    卖点）并未贡献信息。
    """
    pred, s = _as_1d(pred_returns), _as_1d(signal)
    _check_pair(pred, s)
    return (1.0 + alpha * s) * pred


MECHANISMS: dict[str, Callable[..., np.ndarray]] = {
    "none": no_modulation,
    "sign": sign_modulation,
    "soft": soft_modulation,
    "magnitude": magnitude_modulation,
}


def apply_mechanism(
    name: str,
    pred_returns: ArrayLike,
    signal: ArrayLike,
    **params: object,
) -> np.ndarray:
    """按名称施加调制机制，额外的关键字参数透传给机制本身。"""
    if name not in MECHANISMS:
        raise KeyError(f"未知机制: {name}，可选: {sorted(MECHANISMS)}")
    return MECHANISMS[name](pred_returns, signal, **params)


# --------------------------------------------------------------------------- #
# 数据驱动的调制强度
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlphaFit:
    """在给定样本上按目标函数拟合出的最优 α。"""

    alpha: float
    loss: float
    grid: tuple[float, ...]


def fit_alpha(
    pred_returns: ArrayLike,
    signal: ArrayLike,
    true_returns: ArrayLike,
    mechanism: str = "sign",
    grid: Sequence[float] = tuple(np.round(np.arange(0.0, 1.05, 0.05), 2)),
) -> AlphaFit:
    """在网格上选使预测收益 MSE 最小的 α。

    论文把 α 固定为 0.5 未做任何寻优。这个函数回答："就算给你最优的 α，
    机制能好到哪去"——即机制的上界，而非某个任意取值的表现。
    """
    pred, s = _as_1d(pred_returns), _as_1d(signal)
    true = _as_1d(true_returns)
    if not (pred.size == s.size == true.size):
        raise ValueError("pred_returns / signal / true_returns 长度必须一致")

    best_alpha, best_loss = float(grid[0]), float("inf")
    for a in grid:
        adj = apply_mechanism(mechanism, pred, s, alpha=float(a))
        loss = float(np.mean((adj - true) ** 2))
        if loss < best_loss:
            best_alpha, best_loss = float(a), loss
    return AlphaFit(alpha=best_alpha, loss=best_loss, grid=tuple(float(a) for a in grid))


def describe() -> str:
    """机制族的一句话说明，供文档与 CLI 帮助复用。"""
    lines = ["信号变换:", *[f"  - {k}" for k in SIGNAL_TRANSFORMS],
             "调制机制:", *[f"  - {k}" for k in MECHANISMS]]
    return "\n".join(lines)
