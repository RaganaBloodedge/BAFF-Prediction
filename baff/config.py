"""实验配置。

``configs/experiment.yaml`` 是实验设置的事实来源，脚本的默认值由它提供；
未安装 PyYAML 或文件缺失时退回内置默认值，保证在最小依赖下也能跑通。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .paths import EXPERIMENT_CONFIG

BUILTIN_DEFAULTS: dict[str, Any] = {
    "backtest": {
        "window": 20,
        "hidden_dim": 64,
        "num_layers": 2,
        "epochs": 200,
        "batch_size": 32,
        "lr": 0.001,
        "seeds": [0],
        "threads": 1,
    },
    "ablations": {
        "alpha_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        "tau_grid": [0.005, 0.01, 0.02, 0.05],
        "mechanisms": ["sign", "soft", "magnitude"],
        "transforms": ["raw", "cs_zscore"],
    },
    "paper_window": {
        "start": "20260227",
        "days": 22,
    },
}


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    """递归合并，override 中的叶子值覆盖 base。"""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | str | None = None) -> dict:
    """读取配置并与内置默认值深合并。

    任何读取/解析失败都退回默认值——实验不应因为一个 YAML 缩进错误就跑不起来。
    """
    config = copy.deepcopy(BUILTIN_DEFAULTS)
    target = Path(path) if path else EXPERIMENT_CONFIG
    if not target.exists():
        return config

    try:
        import yaml
    except ImportError:
        return config

    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - 配置损坏时退回默认值而不是崩溃
        return config

    if not isinstance(loaded, Mapping):
        return config
    return _deep_merge(config, loaded)
