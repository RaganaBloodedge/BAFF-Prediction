"""数据加载层测试：用临时目录构造最小数据集。"""

from __future__ import annotations

import pandas as pd
import pytest

from baff import data


def _write_day(root, date: str, code: str = "000001", closes=None, scores=None) -> None:
    day = root / date
    day.mkdir(parents=True, exist_ok=True)
    closes = closes if closes is not None else [10.0, 10.5, 11.0, 10.8]
    # 日期序列的**最后一天**必须是 date，否则快照取不到当日情绪
    n = len(closes)
    base = int(date)
    dates = [str(base - (n - 1 - i)) for i in range(n)]
    pd.DataFrame({
        "ts_code": [f"{code}.SZ"] * len(closes),
        "trade_date": dates,
        "close": closes,
    }).to_csv(day / f"price_{code}.csv", index=False)

    scores = scores if scores is not None else [0.1, -0.2, 0.3, 0.4]
    pd.DataFrame({
        "trade_date": dates,
        "sentiment_score": scores,
    }).to_csv(day / f"sentiment_{code}.csv", index=False)


def test_load_price_sorts_ascending(tmp_path):
    p = tmp_path / "price.csv"
    pd.DataFrame({
        "ts_code": ["A", "A", "A"],
        "trade_date": ["20260303", "20260301", "20260302"],
        "close": [3.0, 1.0, 2.0],
    }).to_csv(p, index=False)
    df = data.load_price(p)
    assert df["trade_date"].tolist() == ["20260301", "20260302", "20260303"]
    assert df["close"].tolist() == [1.0, 2.0, 3.0]


def test_load_price_missing_file(tmp_path):
    with pytest.raises(data.DataError, match="找不到价格文件"):
        data.load_price(tmp_path / "nope.csv")


def test_load_price_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"ts_code": ["A"], "trade_date": ["20260301"]}).to_csv(p, index=False)
    with pytest.raises(data.DataError, match="缺少列"):
        data.load_price(p)


def test_load_sentiment_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"trade_date": ["20260301"]}).to_csv(p, index=False)
    with pytest.raises(data.DataError, match="缺少列"):
        data.load_sentiment(p)


def test_discover_dates_sorted_and_filtered(tmp_path):
    for d in ("20260301", "20260227", "not-a-date", "20260302"):
        (tmp_path / d).mkdir()
    assert data.discover_dates(tmp_path) == ["20260227", "20260301", "20260302"]


def test_discover_dates_empty_raises(tmp_path):
    with pytest.raises(data.DataError, match="没有找到"):
        data.discover_dates(tmp_path)


def test_load_snapshot_fields(tmp_path):
    _write_day(tmp_path, "20260301")
    snap = data.load_snapshot(tmp_path, "000001.SZ", "20260301")
    assert snap.ts_code == "000001.SZ"
    assert snap.val_date == "20260301"
    assert snap.n_obs == 4
    assert snap.closes[-1] == pytest.approx(10.8)
    assert snap.last_sentiment() == pytest.approx(0.4)


def test_snapshot_missing_sentiment_defaults_to_neutral(tmp_path):
    _write_day(tmp_path, "20260301")
    snap = data.load_snapshot(tmp_path, "000001.SZ", "20260301")
    assert snap.sentiment_on("20990101") == 0.0


def test_iter_snapshots_skips_missing(tmp_path):
    _write_day(tmp_path, "20260301", code="000001")
    _write_day(tmp_path, "20260302", code="000002")
    # 每只股票各只在一个日期有数据 → 应有 2 个可用快照，缺的不报错
    snaps = list(data.iter_snapshots(tmp_path, ["000001.SZ", "000002.SZ"]))
    assert len(snaps) == 2
    keys = {(s.val_date, s.ts_code) for s in snaps}
    assert keys == {("20260301", "000001.SZ"), ("20260302", "000002.SZ")}


def test_iter_snapshots_raises_when_not_skipping(tmp_path):
    _write_day(tmp_path, "20260301", code="000001")
    with pytest.raises(data.DataError):
        list(data.iter_snapshots(tmp_path, ["999999.SZ"], skip_missing=False))


def test_available_snapshots_counts(tmp_path):
    _write_day(tmp_path, "20260301", code="000001")
    _write_day(tmp_path, "20260302", code="000001")
    assert data.available_snapshots(tmp_path, ["000001.SZ"]) == 2
