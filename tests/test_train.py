"""训练与回测的冒烟测试。

需要 torch；未安装时整文件跳过，因此核心测试套件在无 torch 的 CI 上也能跑绿。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="需要 torch 才能跑训练相关测试")

from baff import backtest  # noqa: E402
from baff import train as T  # noqa: E402
from baff.data import Snapshot  # noqa: E402

pytestmark = pytest.mark.torch


def _synthetic_series(n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(0)
    return 10.0 + np.cumsum(rng.normal(0, 0.1, size=n))


def test_set_seed_makes_training_deterministic():
    x = torch.randn(16, 5, 1)
    y = torch.randn(16)
    cfg = T.TrainConfig(window=5, epochs=5, seed=123)

    T.set_seed(cfg.seed)
    m1 = T.train_model(x, y, cfg)
    p1 = T.predict_window(m1, np.zeros(5), None)

    T.set_seed(cfg.seed)
    m2 = T.train_model(x, y, cfg)
    p2 = T.predict_window(m2, np.zeros(5), None)

    assert p1 == pytest.approx(p2), "固定种子后同一数据两次训练必须给出相同结果"


def test_different_seed_changes_result():
    x = torch.randn(32, 5, 1)
    y = torch.randn(32)
    a = T.train_model(x, y, T.TrainConfig(window=5, epochs=5, seed=1))
    b = T.train_model(x, y, T.TrainConfig(window=5, epochs=5, seed=2))
    pa = T.predict_window(a, np.zeros(5), None)
    pb = T.predict_window(b, np.zeros(5), None)
    assert pa != pytest.approx(pb)


def test_build_supervised_shapes_price_only():
    series = _synthetic_series(30)
    x, y = T.build_supervised(series, None, window=20)
    assert x.shape == (10, 20, 1)
    assert y.shape == (10,)


def test_build_supervised_shapes_with_sentiment():
    series = _synthetic_series(30)
    sent = np.linspace(-1, 1, 30)
    x, y = T.build_supervised(series, sent, window=20)
    assert x.shape == (10, 20, 2)


def test_build_supervised_rejects_short_series():
    with pytest.raises(ValueError, match="不足以构造窗口"):
        T.build_supervised(np.arange(5.0), None, window=20)


def test_build_supervised_rejects_length_mismatch():
    with pytest.raises(ValueError, match="不一致"):
        T.build_supervised(_synthetic_series(30), np.zeros(10), window=20)


def test_scaler_roundtrip():
    values = _synthetic_series(50)
    scaler = T.Scaler.fit(values)
    np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values)


def test_scaler_handles_constant_series():
    scaler = T.Scaler.fit(np.full(10, 3.0))
    assert scaler.std == 1.0
    np.testing.assert_allclose(scaler.transform(np.full(10, 3.0)), 0.0)


def test_training_reduces_loss():
    """训练确实在优化——20 个 epoch 后训练损失应明显低于初始。"""
    series = _synthetic_series(80)
    x, y = T.build_supervised(series, None, window=10)

    def loss_of(epochs: int) -> float:
        model = T.train_model(x, y, T.TrainConfig(window=10, epochs=epochs, seed=0))
        model.eval()
        with torch.no_grad():
            pred = model(x).squeeze(-1)
        return float(torch.mean((pred - y) ** 2).item())

    assert loss_of(40) < loss_of(1)


def test_forecast_one_produces_expected_row():
    snap = Snapshot(
        ts_code="000001.SZ",
        val_date="20260301",
        dates=tuple(f"2026{i:04d}" for i in range(1000, 1050)),
        closes=_synthetic_series(50),
        sentiment={"20260301": 0.5},
    )
    cfg = backtest.BacktestConfig(
        train=T.TrainConfig(window=20, epochs=3, seed=0), seeds=(0,)
    )
    row = backtest.forecast_one(snap, cfg)
    assert row is not None
    assert set(backtest.OUTPUT_COLUMNS).issubset(row)
    assert row["ts_code"] == "000001.SZ"
    assert row["val_date"] == "20260301"
    assert row["sentiment"] == pytest.approx(0.5)
    assert np.isfinite(row["pred_close"])


def test_forecast_one_returns_none_when_too_short():
    snap = Snapshot(
        ts_code="000001.SZ",
        val_date="20260301",
        dates=tuple(f"2026{i:04d}" for i in range(1000, 1010)),
        closes=_synthetic_series(10),
        sentiment={},
    )
    cfg = backtest.BacktestConfig(train=T.TrainConfig(window=20, epochs=1))
    assert backtest.forecast_one(snap, cfg) is None
