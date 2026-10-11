"""回测引擎。

对每个 (交易日, 股票) 组合做一次"截至当日重训、预测当日"的独立实验，
模拟真实部署中无法预知未来数据的约束。

**关键设计**：调制是纯后处理，与 LSTM 训练无关。因此引擎只负责产出
「原始预测 + 当日情绪分」并缓存成一张表；所有调制机制、α 取值、信号变换
都在这张表上做后处理，秒级完成——不需要为每个 α 重训一次网络。

产出列：

=========================  ================================================
``ts_code``                股票代码
``val_date``               验证日（该次实验的预测目标日）
``seed``                   随机种子
``prev_close``             验证日前一交易日收盘价
``true_close``             验证日真实收盘价
``pred_close``             LSTM 预测收盘价
``sentiment``              验证日情绪分
``pred_return``            ``(pred_close - prev_close) / prev_close``
``true_return``            ``(true_close - prev_close) / prev_close``
``model``                  产生该行的模型（``price_lstm`` / ``price_lstm_concat``）
=========================  ================================================
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Snapshot, iter_snapshots
from .train import Scaler, TrainConfig, build_supervised, predict_window, train_model

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "ts_code", "val_date", "seed", "model",
    "prev_close", "true_close", "pred_close",
    "sentiment", "pred_return", "true_return",
]


@dataclass(frozen=True)
class BacktestConfig:
    """回测设置。"""

    train: TrainConfig = TrainConfig()
    seeds: tuple[int, ...] = (0,)
    use_sentiment_input: bool = False
    min_train_obs: int = 30

    def model_name(self) -> str:
        return "price_lstm_concat" if self.use_sentiment_input else "price_lstm"


def forecast_one(snapshot: Snapshot, cfg: BacktestConfig) -> dict | None:
    """对单个快照训练并预测一点，返回结果行；数据不足时返回 None。"""
    closes = snapshot.closes
    window = cfg.train.window
    val_idx = len(closes) - 1

    if val_idx < cfg.min_train_obs or val_idx - window < 0:
        logger.debug("跳过 %s @ %s：样本不足", snapshot.ts_code, snapshot.val_date)
        return None

    train_closes_raw = closes[:val_idx]
    if train_closes_raw.size <= window:
        return None

    scaler = Scaler.fit(train_closes_raw)
    train_scaled = scaler.transform(train_closes_raw)

    # 特征拼接模式下，情绪序列与价格按日期对齐；缺失日记 0（中性）
    sentiment_series = None
    if cfg.use_sentiment_input:
        sentiment_series = np.array(
            [snapshot.sentiment_on(d) for d in snapshot.dates], dtype=np.float64
        )

    x, y = build_supervised(
        train_scaled,
        sentiment_series[:val_idx] if sentiment_series is not None else None,
        window,
    )
    model = train_model(x, y, cfg.train, use_sentiment=cfg.use_sentiment_input)

    window_closes = scaler.transform(closes[val_idx - window:val_idx])
    window_sent = (
        sentiment_series[val_idx - window:val_idx] if sentiment_series is not None else None
    )
    pred_scaled = predict_window(model, window_closes, window_sent)
    pred_close = float(scaler.inverse(pred_scaled))

    prev_close = float(closes[val_idx - 1])
    true_close = float(closes[val_idx])

    return {
        "ts_code": snapshot.ts_code,
        "val_date": snapshot.val_date,
        "seed": cfg.train.seed,
        "model": cfg.model_name(),
        "prev_close": prev_close,
        "true_close": true_close,
        "pred_close": pred_close,
        "sentiment": snapshot.last_sentiment(),
        "pred_return": (pred_close - prev_close) / prev_close,
        "true_return": (true_close - prev_close) / prev_close,
    }


def run_backtest(
    data_root: Path | str,
    stocks: Sequence[str],
    dates: Sequence[str] | None = None,
    cfg: BacktestConfig | None = None,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """跑完整回测，返回预测明细表。"""
    cfg = cfg or BacktestConfig()
    rows: list[dict] = []
    total = 0

    snapshots = list(iter_snapshots(data_root, stocks, dates, skip_missing=skip_missing))
    logger.info(
        "回测开始：%d 个快照 x %d 个种子，模型 %s",
        len(snapshots), len(cfg.seeds), cfg.model_name(),
    )

    for snapshot in snapshots:
        for seed in cfg.seeds:
            run_cfg = BacktestConfig(
                train=TrainConfig(**{**cfg.train.as_dict(), "seed": seed}),
                seeds=(seed,),
                use_sentiment_input=cfg.use_sentiment_input,
                min_train_obs=cfg.min_train_obs,
            )
            row = forecast_one(snapshot, run_cfg)
            if row is not None:
                rows.append(row)
        total += 1
        if total % 50 == 0:
            logger.info("进度 %d/%d 快照", total, len(snapshots))

    if not rows:
        raise RuntimeError("回测没有产生任何结果，请检查数据目录与股票池是否匹配")

    df = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    logger.info("回测完成：%d 行结果，覆盖 %d 只股票 / %d 个交易日",
                len(df), df["ts_code"].nunique(), df["val_date"].nunique())
    return df
