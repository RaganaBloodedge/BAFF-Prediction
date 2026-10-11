"""配置加载测试。"""

from __future__ import annotations

import textwrap

from baff import config as C


def test_builtin_defaults_have_expected_shape():
    assert C.BUILTIN_DEFAULTS["backtest"]["window"] == 20
    assert 0.5 in C.BUILTIN_DEFAULTS["ablations"]["alpha_grid"]
    assert C.BUILTIN_DEFAULTS["paper_window"]["days"] == 22


def test_missing_file_falls_back_to_defaults(tmp_path):
    cfg = C.load_config(tmp_path / "nope.yaml")
    assert cfg == C.BUILTIN_DEFAULTS


def test_repo_config_is_valid():
    """仓库里那份 experiment.yaml 必须能读、能覆盖默认值。"""
    cfg = C.load_config()
    assert cfg["backtest"]["epochs"] > 0
    assert cfg["backtest"]["seeds"]
    assert len(cfg["ablations"]["alpha_grid"]) > 1


def test_override_merges_deeply(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        backtest:
          epochs: 5
        paper_window:
          days: 3
    """), encoding="utf-8")
    cfg = C.load_config(p)
    # 覆盖的键变了
    assert cfg["backtest"]["epochs"] == 5
    assert cfg["paper_window"]["days"] == 3
    # 同层未被覆盖的键保留
    assert cfg["backtest"]["window"] == 20
    assert cfg["paper_window"]["start"] == "20260227"


def test_malformed_yaml_falls_back(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("backtest: [unclosed\n  - :::\n", encoding="utf-8")
    cfg = C.load_config(p)
    assert cfg["backtest"]["window"] == 20


def test_non_mapping_yaml_falls_back(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    cfg = C.load_config(p)
    assert cfg == C.BUILTIN_DEFAULTS


def test_load_config_does_not_mutate_defaults(tmp_path):
    """加载过配置后，内置默认值不能被污染。"""
    p = tmp_path / "c.yaml"
    p.write_text("backtest:\n  epochs: 1\n", encoding="utf-8")
    C.load_config(p)
    assert C.BUILTIN_DEFAULTS["backtest"]["epochs"] == 200
