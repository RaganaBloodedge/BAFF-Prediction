"""LSTM 训练与推理。

与原实现的三点差异：

1. **固定随机种子** —— 原实现没有设置种子，同一份数据重跑会得到不同的
   MAE，"结果不可复现"是它作为作品最硬的伤。这里在 python / numpy / torch
   三处同步固定。
2. **去掉 DataLoader** —— 单个训练集只有三十余个样本，DataLoader 的
   多进程开销大于收益；直接张量索引既快又完全确定。
3. **标准化器显式对象化** —— 训练集统计量封装成 :class:`Scaler` 并在
   推理时复用，避免"用验证集统计量做标准化"这一类隐蔽泄漏。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .models import PriceLSTM, PriceLSTMConcat


def set_seed(seed: int) -> None:
    """固定 python / numpy / torch 三处随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainConfig:
    """训练超参。默认值与论文 3.2 节保持一致，便于对齐比较。"""

    window: int = 20
    hidden_dim: int = 64
    num_layers: int = 2
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0

    def as_dict(self) -> dict:
        return {
            "window": self.window,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Scaler:
    """只用训练集统计量的标准化器。"""

    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> Scaler:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        std = float(arr.std())
        return cls(mean=float(arr.mean()), std=std if std > 0 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean


def build_supervised(
    closes: np.ndarray,
    sentiment: np.ndarray | None,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造监督样本。

    返回 ``x`` 形状 ``(N, W, C)``、``y`` 形状 ``(N,)``；``sentiment`` 为 None
    时 ``C = 1``（纯价格），否则 ``C = 2``（价格 + 情绪，特征拼接基线）。
    """
    closes = np.asarray(closes, dtype=np.float32).reshape(-1)
    if closes.size <= window:
        raise ValueError(f"序列长度 {closes.size} 不足以构造窗口 {window}")

    xs = [closes[i:i + window] for i in range(closes.size - window)]
    ys = [closes[i + window] for i in range(closes.size - window)]

    if sentiment is None:
        x = np.stack(xs)[..., None]          # (N, W, 1)
    else:
        sent = np.asarray(sentiment, dtype=np.float32).reshape(-1)
        if sent.size != closes.size:
            raise ValueError(f"情绪序列长度 {sent.size} 与价格 {closes.size} 不一致")
        sent_win = [sent[i:i + window] for i in range(closes.size - window)]
        x = np.stack([np.stack([c, s], axis=-1) for c, s in zip(xs, sent_win, strict=True)])

    return (torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(np.asarray(ys, dtype=np.float32)))


def build_model(cfg: TrainConfig, use_sentiment: bool) -> nn.Module:
    if use_sentiment:
        return PriceLSTMConcat(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
    return PriceLSTM(input_dim=1, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
    use_sentiment: bool = False,
) -> nn.Module:
    """在全量训练样本上训练，返回训练好的模型。"""
    set_seed(cfg.seed)
    model = build_model(cfg, use_sentiment)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    n = x.shape[0]
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            pred = model(x[idx]).squeeze(-1)
            loss = criterion(pred, y[idx])
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_window(
    model: nn.Module,
    window_closes: np.ndarray,
    window_sentiment: np.ndarray | None,
) -> float:
    """对单个窗口做一步预测，返回**标准化空间**的标量。"""
    model.eval()
    c = np.asarray(window_closes, dtype=np.float32).reshape(-1)
    if window_sentiment is None:
        x = c[:, None]
    else:
        s = np.asarray(window_sentiment, dtype=np.float32).reshape(-1)
        x = np.stack([c, s], axis=-1)
    tensor = torch.from_numpy(x[None, ...].astype(np.float32))
    return float(model(tensor).squeeze().item())
