"""数据加载与对齐。

数据有两种落盘布局，本模块都支持：

- **按日期分区**（回测主用）：``root/{YYYYMMDD}/{kind}_{code}.csv``
  每个目录是"截至当日"的一帧快照，回测按目录顺序逐个解析。
- **平铺**：``root/{kind}_{code}.csv``
  单只股票单次实验的简化布局。

三类文件都含 ``trade_date`` 列，可据此对齐。价格文件由 Tushare 导出，
按日期**降序**排列，加载时统一翻转为升序。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(RuntimeError):
    """数据缺失或格式不符。"""


# --------------------------------------------------------------------------- #
# 单文件加载
# --------------------------------------------------------------------------- #
def load_price(path: Path | str) -> pd.DataFrame:
    """读取价格 CSV，按 ``trade_date`` 升序返回，并校验必需列。"""
    path = Path(path)
    if not path.exists():
        raise DataError(f"找不到价格文件: {path}")
    df = pd.read_csv(path)
    required = {"ts_code", "trade_date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise DataError(f"{path} 缺少列: {sorted(missing)}")
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date").reset_index(drop=True)


def load_sentiment(path: Path | str) -> pd.DataFrame:
    """读取日度情绪 CSV，返回 ``trade_date`` / ``sentiment_score`` 两列。"""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"找不到情绪文件: {path}，请先运行 sentiment/build_sentiment_from_csv.py"
        )
    df = pd.read_csv(path)
    required = {"trade_date", "sentiment_score"}
    missing = required - set(df.columns)
    if missing:
        raise DataError(f"{path} 缺少列: {sorted(missing)}")
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 快照
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Snapshot:
    """某只股票在某个验证日的一帧数据。"""

    ts_code: str
    val_date: str
    dates: tuple[str, ...]
    closes: np.ndarray
    sentiment: dict[str, float]

    @property
    def n_obs(self) -> int:
        return len(self.dates)

    def sentiment_on(self, date: str, default: float = 0.0) -> float:
        """取某日情绪分，缺失记 0（中性），与原始实现保持一致。"""
        return float(self.sentiment.get(str(date), default))

    def last_sentiment(self) -> float:
        return self.sentiment_on(self.val_date)


def discover_dates(root: Path | str) -> list[str]:
    """扫描日期分区目录，返回升序的 ``YYYYMMDD`` 列表。"""
    root = Path(root)
    if not root.is_dir():
        raise DataError(f"数据根目录不存在: {root}")
    dates = [
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8
    ]
    if not dates:
        raise DataError(f"{root} 下没有找到 YYYYMMDD 形式的日期目录")
    return sorted(dates)


def load_snapshot(root: Path | str, ts_code: str, date: str) -> Snapshot:
    """加载指定日期目录下的单只股票快照。"""
    root = Path(root)
    day_dir = root / str(date)
    if not day_dir.is_dir():
        raise DataError(f"日期目录不存在: {day_dir}")

    code = ts_code.split(".")[0]
    price_df = load_price(day_dir / f"price_{code}.csv")
    sent_df = load_sentiment(day_dir / f"sentiment_{code}.csv")

    return Snapshot(
        ts_code=ts_code,
        val_date=str(date),
        dates=tuple(price_df["trade_date"].tolist()),
        closes=price_df["close"].to_numpy(dtype=np.float64),
        sentiment=dict(
            zip(sent_df["trade_date"].tolist(),
                sent_df["sentiment_score"].astype(float),
                strict=True)
        ),
    )


def iter_snapshots(
    root: Path | str,
    stocks: Sequence[str],
    dates: Sequence[str] | None = None,
    skip_missing: bool = True,
) -> Iterator[Snapshot]:
    """按 (日期, 股票) 顺序迭代全部快照。

    ``skip_missing=True`` 时跳过任一文件缺失的组合并继续——历史数据里
    个别股票在某些日期缺文件是常态，不应中断整轮回测。
    """
    root = Path(root)
    if dates is None:
        dates = discover_dates(root)
    for date in dates:
        for ts_code in stocks:
            try:
                yield load_snapshot(root, ts_code, date)
            except DataError:
                if not skip_missing:
                    raise


def available_snapshots(root: Path | str, stocks: Sequence[str],
                        dates: Sequence[str] | None = None) -> int:
    """统计可用的 (日期, 股票) 组合数，回测前用于估算耗时。"""
    return sum(1 for _ in iter_snapshots(root, stocks, dates))
