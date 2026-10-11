"""指标函数的数值正确性测试。"""

from __future__ import annotations

import numpy as np
import pytest

from baff import metrics


def test_mae_hand_checked():
    assert metrics.mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)


def test_rmse_hand_checked():
    assert metrics.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))


def test_mre_hand_checked():
    # |(100-110)/100| = 10%, |(200-190)/200| = 5%  →  均值 7.5%
    assert metrics.mre([100, 200], [110, 190]) == pytest.approx(7.5)


def test_mre_skips_zero_truth():
    """真实值为 0 的样本必须跳过，而不是产生 inf。"""
    assert metrics.mre([0, 100], [5, 110]) == pytest.approx(10.0)


def test_mre_all_zero_raises():
    with pytest.raises(ValueError):
        metrics.mre([0, 0], [1, 1])


def test_perfect_prediction_is_zero_error():
    y = [1.5, 2.5, 3.5]
    assert metrics.mae(y, y) == 0.0
    assert metrics.rmse(y, y) == 0.0
    assert metrics.mre(y, y) == 0.0


def test_directional_accuracy_hand_checked():
    assert metrics.directional_accuracy([1, -1, 1, -1], [1, 1, -1, -1]) == 0.5


def test_directional_accuracy_ignores_flat():
    """收益率为 0 的方向没有意义，应剔除后再算。"""
    assert metrics.directional_accuracy([1, -1, 0], [1, 1, 1]) == 0.5


def test_information_coefficient_monotone():
    """严格单调同增时 IC 应为 1。"""
    assert metrics.information_coefficient([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_information_coefficient_inverted():
    assert metrics.information_coefficient([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_information_coefficient_needs_three_points():
    with pytest.raises(ValueError):
        metrics.information_coefficient([1, 2], [3, 4])


def test_improvement_sign_convention():
    """improvement 正值 = 候选误差更小 = 更好。"""
    assert metrics.improvement(1.0, 0.8) == pytest.approx(0.2)
    assert metrics.improvement(0.8, 1.0) < 0
