"""仓库级路径常量与股票池加载。

所有路径都以本文件所在目录向上两级（即仓库根）解析，因此在任意工作目录下
执行都能正确定位 data/、results/、figures/。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

RESULTS_DIR = REPO_ROOT / "results"
AGGREGATED_DIR = RESULTS_DIR / "aggregated"
ABLATION_DIR = RESULTS_DIR / "ablations"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"

FIGURES_DIR = REPO_ROOT / "figures"
CONFIG_DIR = REPO_ROOT / "configs"
STOCKS_FILE = CONFIG_DIR / "stocks.txt"
EXPERIMENT_CONFIG = CONFIG_DIR / "experiment.yaml"

# FinBERT2 微调后的二分类模型目录。权重体积大，不随仓库分发；
# 可用环境变量 FINBERT_MODEL_DIR 覆盖。
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "FinBERT2-base_SC_2"


def load_stocks(path: Path | str = STOCKS_FILE) -> list[str]:
    """从配置文件读取股票池，忽略空行与 ``#`` 之后的注释。"""
    codes: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            codes.append(line)
    return codes


def resolve_model_dir() -> Path:
    """情绪模型目录：优先环境变量 ``FINBERT_MODEL_DIR``，否则用仓库内 models/。"""
    env = os.environ.get("FINBERT_MODEL_DIR")
    return Path(env) if env else DEFAULT_MODEL_DIR
