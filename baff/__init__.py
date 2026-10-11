"""BAFF: Behavior-Aware Forecasting Framework.

价格流用 LSTM 处理历史收盘价，文本流用 FinBERT2 把每日财经新闻标题编码为
日度情绪分；情绪分不进入价格特征空间，而是作为行为调制信号，按方向一致性
放大或抑制 LSTM 给出的预测收益率。

本包是整个项目的核心实现，被 fetch/、sentiment/、forecast/、experiments/
各层复用。
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
