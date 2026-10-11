"""调制机制族的契约测试。

这些测试锁定的是**机制的数学性质**，而非某个具体数值——只要性质成立，
重构机制实现时不会悄悄改变语义。
"""

from __future__ import annotations

import numpy as np
import pytest

from baff import mechanism as M

RET = np.array([0.012, -0.020, 0.031, -0.005, 0.008])
SIG = np.array([0.8, -0.6, 0.3, 0.9, -0.2])


# --------------------------------------------------------------------------- #
# 边界性质：所有机制都必须满足
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(M.MECHANISMS))
def test_alpha_zero_is_identity(name):
    """alpha=0 时必须完全退化为纯价格预测。"""
    out = M.apply_mechanism(name, RET, SIG, alpha=0.0)
    np.testing.assert_allclose(out, RET, atol=1e-12)


@pytest.mark.parametrize("name", sorted(M.MECHANISMS))
def test_zero_signal_is_identity(name):
    """情绪为 0（中性）时不得改变预测。"""
    out = M.apply_mechanism(name, RET, np.zeros_like(SIG), alpha=0.5)
    np.testing.assert_allclose(out, RET, atol=1e-12)


@pytest.mark.parametrize("name", sorted(M.MECHANISMS))
def test_zero_return_stays_zero(name):
    """预测收益率为 0 时，调制后仍为 0——不应凭空造出涨跌。"""
    out = M.apply_mechanism(name, np.zeros_like(RET), SIG, alpha=0.5)
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(M.MECHANISMS))
def test_shape_and_finiteness(name):
    out = M.apply_mechanism(name, RET, SIG, alpha=0.5)
    assert out.shape == RET.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("name", sorted(M.MECHANISMS))
def test_length_mismatch_raises(name):
    with pytest.raises(ValueError):
        M.apply_mechanism(name, RET, SIG[:-1], alpha=0.5)


# --------------------------------------------------------------------------- #
# 机制语义
# --------------------------------------------------------------------------- #
def test_sign_amplifies_on_agreement():
    """方向一致（sign(r)·s > 0）时绝对值被放大。"""
    r = np.array([0.02, -0.02])
    s = np.array([0.5, -0.5])  # 与各自的预测方向一致
    out = M.apply_mechanism("sign", r, s, alpha=0.5)
    assert abs(out[0]) > abs(r[0])
    assert abs(out[1]) > abs(r[1])
    np.testing.assert_allclose(out, r * 1.25)


def test_sign_shrinks_on_conflict():
    """方向矛盾时被抑制。"""
    r = np.array([0.02])
    s = np.array([-0.5])
    out = M.apply_mechanism("sign", r, s, alpha=0.5)
    assert abs(out[0]) < abs(r[0])
    np.testing.assert_allclose(out, r * 0.75)


def test_sign_flips_sign_only_when_factor_negative():
    """alpha·s < -1 时调制因子为负，预测方向会被翻转——这是机制的已知副作用。"""
    r = np.array([0.02])
    s = np.array([-1.0])
    out = M.apply_mechanism("sign", r, s, alpha=1.5)
    assert out[0] < 0


def test_soft_matches_sign_when_return_large():
    """收益率远大于 tau 时，soft 应收敛到 sign。"""
    r = np.array([1.0, -1.0])
    s = np.array([0.5, 0.5])
    soft = M.apply_mechanism("soft", r, s, alpha=0.5, tau=0.02)
    sign = M.apply_mechanism("sign", r, s, alpha=0.5)
    np.testing.assert_allclose(soft, sign, rtol=1e-6)


def test_soft_disables_modulation_near_zero_return():
    """收益率趋近 0 时 soft 的调制自动关闭，而 sign 仍全额作用。"""
    r = np.array([1e-9])
    s = np.array([0.9])
    soft = M.apply_mechanism("soft", r, s, alpha=0.5, tau=0.02)
    sign = M.apply_mechanism("sign", r, s, alpha=0.5)
    assert abs(soft[0] - r[0]) < abs(sign[0] - r[0])


def test_magnitude_ignores_direction():
    """magnitude 机制对方向不敏感：同情绪下涨跌预测被同等缩放。"""
    r = np.array([0.02, -0.02])
    s = np.array([0.5, 0.5])
    out = M.apply_mechanism("magnitude", r, s, alpha=0.5)
    np.testing.assert_allclose(np.abs(out), np.abs(r) * 1.25)


def test_negative_alpha_rejected_by_tau_validation_only():
    """tau 非正必须报错。"""
    with pytest.raises(ValueError):
        M.apply_mechanism("soft", RET, SIG, alpha=0.5, tau=0.0)


# --------------------------------------------------------------------------- #
# 信号变换
# --------------------------------------------------------------------------- #
def test_cs_zscore_is_zero_mean_within_group():
    signal = np.array([0.9, 0.1, 0.5, -0.2, 0.4, 0.8])
    groups = np.array(["d1", "d1", "d1", "d2", "d2", "d2"])
    out = M.transform_cs_zscore(signal, groups)
    for g in ("d1", "d2"):
        block = out[groups == g]
        assert abs(block.mean()) < 1e-12
        assert abs(block.std() - 1.0) < 1e-9


def test_cs_zscore_constant_group_becomes_zero():
    """组内取值全相同时标准差为 0，应返回全 0 而非 nan。"""
    out = M.transform_cs_zscore(np.array([0.5, 0.5]), np.array(["d1", "d1"]))
    np.testing.assert_allclose(out, 0.0)
    assert np.all(np.isfinite(out))


def test_transform_raw_is_a_copy():
    src = np.array([0.1, 0.2])
    out = M.transform_raw(src)
    out[0] = 99.0
    assert src[0] == 0.1, "transform_raw 不得修改输入"


# --------------------------------------------------------------------------- #
# α 拟合
# --------------------------------------------------------------------------- #
def test_fit_alpha_recovers_planted_value():
    """构造一组由 α*=0.3 生成的观测，拟合应能找回它。"""
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.02, size=400)
    s = rng.uniform(-1, 1, size=400)
    true_alpha = 0.3
    target = M.apply_mechanism("sign", r, s, alpha=true_alpha)
    fit = M.fit_alpha(r, s, target, mechanism="sign",
                      grid=np.round(np.arange(0, 1.01, 0.05), 2))
    assert abs(fit.alpha - true_alpha) <= 0.05


def test_fit_alpha_hits_boundary_when_no_signal():
    """完全无信号时最优解落在网格边界或 0，不应崩溃。"""
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, size=100)
    s = rng.uniform(-1, 1, size=100)
    fit = M.fit_alpha(r, s, r, mechanism="sign")
    assert fit.loss >= 0.0
    assert fit.alpha in fit.grid


def test_unknown_mechanism_raises():
    with pytest.raises(KeyError):
        M.apply_mechanism("does-not-exist", RET, SIG)
