"""统计检验工具的行为测试。"""

from __future__ import annotations

import numpy as np
import pytest

from baff import stats


def test_paired_test_detects_clear_difference():
    rng = np.random.default_rng(0)
    baseline = rng.normal(1.0, 0.1, size=60)
    candidate = baseline - 0.3  # 候选误差系统性更小
    res = stats.paired_test(candidate, baseline)
    assert res.n == 60
    assert res.mean_diff == pytest.approx(0.3, abs=1e-9)
    assert res.pvalue < 0.01


def test_paired_test_no_difference_is_not_significant():
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = stats.paired_test(baseline, baseline)
    assert res.pvalue == 1.0
    assert res.mean_diff == 0.0


def test_paired_test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        stats.paired_test([1, 2, 3], [1, 2])


def test_paired_test_needs_at_least_two():
    with pytest.raises(ValueError):
        stats.paired_test([1.0], [2.0])


def test_paired_test_ttest_alternative():
    rng = np.random.default_rng(3)
    baseline = rng.normal(0, 1, size=40)
    res = stats.paired_test(baseline - 0.5, baseline, method="ttest")
    assert res.method == "ttest"
    assert res.pvalue < 0.05


def test_paired_test_unknown_method():
    with pytest.raises(ValueError):
        stats.paired_test([1, 2, 3], [2, 3, 4], method="anova")


def test_paired_test_handles_all_positive_diff():
    """差值为全正时 wilcoxon 会退化，必须仍返回有限 p 值。"""
    candidate = np.array([0.1, 0.2, 0.3, 0.4])
    baseline = np.array([1.0, 1.0, 1.0, 1.0])
    res = stats.paired_test(candidate, baseline)
    assert np.isfinite(res.pvalue)
    assert res.pvalue <= 1.0


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(11)
    values = rng.normal(0.05, 0.1, size=200)
    mean, lo, hi = stats.bootstrap_ci(values, seed=1)
    assert lo < mean < hi
    assert mean == pytest.approx(values.mean())


def test_bootstrap_ci_is_reproducible():
    values = np.linspace(-1, 1, 50)
    a = stats.bootstrap_ci(values, seed=42)
    b = stats.bootstrap_ci(values, seed=42)
    assert a == b


def test_bootstrap_ci_straddles_zero_for_noisy_null():
    # 对称样本，均值恰为 0 —— 避免随机抽样导致的偶发失败
    values = np.linspace(-1.0, 1.0, 41)
    _, lo, hi = stats.bootstrap_ci(values, seed=2)
    assert lo < 0 < hi


def test_summarize_reports_expected_keys():
    values = np.array([0.1, 0.2, -0.05, 0.3, 0.0])
    out = stats.summarize(values, prefix="x_")
    for key in ("x_n", "x_mean", "x_ci_low", "x_ci_high", "x_positive_ratio", "x_pvalue"):
        assert key in out
    assert out["x_n"] == 5
    assert out["x_positive_ratio"] == pytest.approx(0.6)
