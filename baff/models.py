"""价格预测网络定义。

两个模型共用同一套训练/推理接口，区别只在输入通道数：

- :class:`PriceLSTM`          —— 单通道（收盘价），对应 BAFF 的价格流基线。
- :class:`PriceLSTMConcat`    —— 双通道（收盘价 + 当日情绪分），对应论文在
  第 1.4 节讨论过、但原文未实跑的「特征级拼接」对照组。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PriceLSTM(nn.Module):
    """单变量收盘价 LSTM：输入 ``(B, W, 1)``，输出 ``(B, 1)``。"""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PriceLSTMConcat(nn.Module):
    """特征拼接基线：把情绪分作为与收盘价并列的第二个输入通道。

    输入 ``(B, W, 2)``，最后一维为 ``[收盘价, 情绪分]``。用于回答论文中
    悬而未决的对照问题——把情绪拼进特征空间，是否比把它当作调制信号更好。
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_model(name: str, **kwargs) -> nn.Module:
    """按名称构造模型，供配置驱动的实验使用。"""
    registry = {
        "price_lstm": PriceLSTM,
        "price_lstm_concat": PriceLSTMConcat,
    }
    if name not in registry:
        raise KeyError(f"未知模型: {name}，可选: {sorted(registry)}")
    return registry[name](**kwargs)
