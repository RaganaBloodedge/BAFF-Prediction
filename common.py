"""兼容层：路径常量与股票池由 :mod:`baff.paths` 提供。

``fetch/`` 与 ``sentiment/`` 下的脚本通过本模块取路径，调用方式保持不变；
新代码建议直接 ``from baff import paths``。
"""

from __future__ import annotations

from baff.paths import (  # noqa: F401
    AGGREGATED_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DEFAULT_MODEL_DIR,
    FIGURES_DIR,
    RAW_DATA_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    STOCKS_FILE,
    load_stocks,
    resolve_model_dir,
)

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "AGGREGATED_DIR",
    "FIGURES_DIR",
    "CONFIG_DIR",
    "STOCKS_FILE",
    "DEFAULT_MODEL_DIR",
    "load_stocks",
    "resolve_model_dir",
]
