"""仓库级共享工具：路径常量与股票池加载。

所有脚本都以本文件所在目录作为仓库根来解析路径，因此在任意工作目录下
执行都能正确定位 data/、results/、figures/。
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
AGGREGATED_DIR = RESULTS_DIR / "aggregated"
CONFIG_DIR = REPO_ROOT / "configs"
STOCKS_FILE = CONFIG_DIR / "stocks.txt"

# FinBERT2 微调后的二分类模型目录。权重体积大，不随仓库分发；
# 可用环境变量 FINBERT_MODEL_DIR 覆盖。
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "FinBERT2-base_SC_2"


def load_stocks(path: Path | str = STOCKS_FILE) -> list[str]:
    """从配置文件读取股票池，忽略空行与 # 之后的注释。"""
    codes: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            codes.append(line)
    return codes


def resolve_model_dir() -> Path:
    """情绪模型目录：优先环境变量 FINBERT_MODEL_DIR，否则用仓库内 models/。"""
    env = os.environ.get("FINBERT_MODEL_DIR")
    return Path(env) if env else DEFAULT_MODEL_DIR
