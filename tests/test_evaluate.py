"""评估层测试：用合成预测表锁定评估口径。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baff import evaluate


def make_preds() -> pd.DataFrame:
    """2 只股票 × 2 个验证日 × 1 个种子。"""
    rows = []
    spec = [
        ("A.SZ", "20260301", 10.0, 11.0, 10.0, 0.6),
        ("A.SZ", "20260302", 11.0, 10.5, 11.5, -0.4),
        ("B.SZ", "20260301", 20.0, 21.0, 20.5, 0.2),
        ("B.SZ", "20260302", 21.0, 20.0, 20.8, -0.8),
    ]
    for ts_code, val_date, prev, true, pred, sent in spec:
        rows.append({
            "ts_code": ts_code,
            "val_date": val_date,
            "seed": 0,
            "model": "price_lstm",
            "prev_close": prev,
            "true_close": true,
            "pred_close": pred,
            "sentiment": sent,
            "pred_return": (pred - prev) / prev,
            "true_return": (true - prev) / prev,
        })
    return pd.DataFrame(rows)


def test_baseline_mechanism_reproduces_ae_base():
    """mechanism=none 时调制后误差必须等于基线误差，improvement 恒为 0。"""
    detail = evaluate.evaluate(make_preds(), "none", "raw")
    np.testing.assert_allclose(detail["ae_cand"], detail["ae_base"])
    np.testing.assert_allclose(detail["improvement"], 0.0)


def test_improvement_is_base_minus_candidate():
    detail = evaluate.evaluate(make_preds(), "sign", "raw", alpha=0.5)
    np.testing.assert_allclose(
        detail["improvement"], detail["ae_base"] - detail["ae_cand"]
    )


def test_adjusted_price_matches_formula():
    """调制后价格 = 前收 × (1 + 调制后收益率)。"""
    detail = evaluate.evaluate(make_preds(), "sign", "raw", alpha=0.5)
    expected = detail["prev_close"] * (1 + detail["adj_return"])
    np.testing.assert_allclose(detail["adj_close"], expected)


def test_relative_error_is_percentage():
    detail = evaluate.evaluate(make_preds(), "none", "raw")
    np.testing.assert_allclose(
        detail["re_base"], detail["ae_base"] / detail["true_close"] * 100.0
    )


def test_by_stock_aggregates_per_stock():
    detail = evaluate.evaluate(make_preds(), "sign", "raw", alpha=0.5)
    agg = evaluate.by_stock(detail)
    assert set(agg["ts_code"]) == {"A.SZ", "B.SZ"}
    assert (agg["n_days"] == 2).all()
    # 每只股票一行：MAE 是两日 AE 的均值
    a_row = agg[agg["ts_code"] == "A.SZ"].iloc[0]
    manual = detail[detail["ts_code"] == "A.SZ"]["ae_base"].mean()
    assert a_row["MAE_base"] == pytest.approx(manual)


def test_cs_zscore_signal_has_zero_cross_sectional_mean():
    detail = evaluate.evaluate(make_preds(), "sign", "cs_zscore", alpha=0.5)
    for _, block in detail.groupby("val_date"):
        assert abs(block["signal"].mean()) < 1e-12


def test_cs_zscore_constant_cross_section_is_neutral():
    """某日所有股票情绪相同 → 截面标准差为 0 → 信号为 0 → 不调制。"""
    preds = make_preds()
    preds.loc[preds["val_date"] == "20260301", "sentiment"] = 0.5
    detail = evaluate.evaluate(preds, "sign", "cs_zscore", alpha=0.5)
    day = detail[detail["val_date"] == "20260301"]
    np.testing.assert_allclose(day["adj_return"], day["pred_return"])
    np.testing.assert_allclose(day["improvement"], 0.0)


def test_summarize_reports_expected_fields():
    detail = evaluate.evaluate(make_preds(), "sign", "raw", alpha=0.5)
    row = evaluate.summarize(detail)
    for key in ("mechanism", "transform", "n_samples", "n_stocks", "MAE_base",
                "MAE_cand", "stock_mean", "stock_pvalue", "sample_mean"):
        assert key in row
    assert row["n_stocks"] == 2
    assert row["n_samples"] == 4


def test_directional_stats_keys():
    detail = evaluate.evaluate(make_preds(), "sign", "raw", alpha=0.5)
    out = evaluate.directional_stats(detail)
    assert set(out) == {"dir_acc_base", "dir_acc_cand", "ic_base", "ic_cand", "sentiment_ic"}


def test_unknown_transform_raises():
    with pytest.raises(KeyError):
        evaluate.apply_signal(make_preds(), "no-such-transform")
