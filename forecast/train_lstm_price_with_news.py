"""BAFF 主实验：在单只股票上训练价格 LSTM，比较纯价格预测与行为调制预测。

流程：
  1. 读 data/price_{code}.csv，取最后一日为验证日，其余为训练集；
  2. 仅用收盘价训练 PriceLSTM（window=20，hidden=64，2 层，200 epoch）；
  3. 由 LSTM 预测价换算收益率 r_pred，用当日情绪分 s 做方向一致性调制
     m = 1 + α·sign(r_pred)·s，α = 0.5，得 r_adj = m·r_pred；
  4. 输出基线与调制后的 AE/RE，落盘
     results/{validation_date}/{code}_{timestamp}.csv
     figures/{validation_date}/{code}_forecast.png

前置数据由 fetch/ 与 sentiment/ 两阶段产出，两者都以 trade_date 对齐。

用法：
    python forecast/train_lstm_price_with_news.py
    python forecast/train_lstm_price_with_news.py 000547.SZ
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, FIGURES_DIR, RESULTS_DIR, load_stocks  # noqa: E402

from dataset import StockDataset  # noqa: E402
from lstm_price import PriceLSTM  # noqa: E402


def load_price(price_csv: Path | str) -> pd.DataFrame:
    """读取 Tushare daily 导出的价格 CSV，按 trade_date 升序。"""
    if not Path(price_csv).exists():
        raise FileNotFoundError(f"找不到价格数据文件: {price_csv}")
    df = pd.read_csv(price_csv)
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date").reset_index(drop=True)


def load_sentiment(sentiment_csv: Path | str) -> pd.DataFrame:
    """读取日度情绪文件，需包含 trade_date 与 sentiment_score 两列。"""
    if not Path(sentiment_csv).exists():
        raise FileNotFoundError(
            f"找不到情绪数据文件: {sentiment_csv}，"
            f"请先运行 sentiment/build_sentiment_from_csv.py 生成。"
        )
    df = pd.read_csv(sentiment_csv)
    if "trade_date" not in df.columns or "sentiment_score" not in df.columns:
        raise ValueError(f"{sentiment_csv} 必须包含列: trade_date, sentiment_score")
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def run_for_stock(
    ts_code: str,
    data_dir: Path | str | None = None,
    fig_dir: Path | str | None = None,
    alpha: float = 0.5,
) -> dict:
    """对单只股票完成训练与两种预测，落盘结果并返回指标字典。"""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    code = ts_code.split(".")[0]
    price_csv = data_dir / f"price_{code}.csv"
    sentiment_csv = data_dir / f"sentiment_{code}.csv"

    # ========== 1. 价格数据：只用收盘价训练 ==========
    price_df = load_price(price_csv)
    prices_raw = price_df["close"].values.reshape(-1, 1)  # (T, 1)

    window = 20
    all_dates = price_df["trade_date"].values.astype(str)

    val_date = all_dates[-1]
    val_idx = int(np.where(all_dates == val_date)[0][0])

    train_raw = prices_raw[:val_idx]                       # 训练集
    test_raw = prices_raw[val_idx - window: val_idx + 1]   # 验证点及其窗口
    split = val_idx

    # 用训练集统计量做标准化
    mean = train_raw.mean()
    std = train_raw.std()
    if std == 0:
        std = 1.0

    train_data = (train_raw - mean) / std
    test_data = (test_raw - mean) / std

    train_set = StockDataset(train_data, window=window)
    test_set = StockDataset(test_data, window=window)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    # ========== 2. 仅基于价格训练 LSTM ==========
    model = PriceLSTM(input_dim=1, hidden_dim=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            y = y.squeeze(-1)
            y_hat = model(x).squeeze(-1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch + 1}, Train Loss (price only): {total_loss / len(train_set):.6f}")

    # ========== 3. 纯价格预测 ==========
    def predict(ds: StockDataset) -> np.ndarray:
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        preds = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                preds.append(model(x).squeeze(-1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    train_preds = predict(train_set) * std + mean
    test_preds = predict(test_set) * std + mean

    train_true = prices_raw[window:split]
    test_true = prices_raw[split:]

    # ========== 4. 行为调制：用情绪分调整预测收益率 ==========
    sent_df = load_sentiment(sentiment_csv)
    sent_map = {row.trade_date: float(row.sentiment_score) for row in sent_df.itertuples()}

    test_dates = price_df["trade_date"].values[split:]
    sentiment_scores = np.array(
        [sent_map.get(str(d), 0.0) for d in test_dates], dtype=np.float32
    )

    # 前一交易日真实价，用于把价格换算成收益率
    prev_true_prices = np.concatenate([[prices_raw[split - 1, 0]], test_true[:-1, 0]])

    pred_returns = (test_preds - prev_true_prices) / prev_true_prices
    direction = np.sign(pred_returns)
    modulation_factor = 1 + alpha * direction * sentiment_scores
    adj_returns = modulation_factor * pred_returns
    test_preds_with_sent = prev_true_prices * (1 + adj_returns)

    # ========== 5. 指标：纯价格 vs 情绪调制 ==========
    test_true_item = test_true[0, 0]
    ae_base = abs(test_true_item - test_preds[0])
    ae_sent = abs(test_true_item - test_preds_with_sent[0])
    re_base = ae_base / test_true_item * 100
    re_sent = ae_sent / test_true_item * 100

    print(f"\n[{ts_code}] 纯价格 AE={ae_base:.4f}, RE={re_base:.2f}% | "
          f"情绪调制 AE={ae_sent:.4f}, RE={re_sent:.2f}%")

    # ========== 6. 可视化 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(prices_raw, label="True Price", color="black")
    ax.plot(range(window, split), train_preds,
            label="Train Fit (Price only)", color="blue", alpha=0.8)

    x_base = [split - 1, split]
    ax.plot(x_base, [train_preds[-1], test_preds[0]],
            label="Test Pred (Price only)", color="orange", linestyle="--", alpha=0.9)
    ax.plot(x_base, [train_preds[-1], test_preds_with_sent[0]],
            label="Test Pred (Price × Sentiment)", color="red")

    ax.axvline(x=split, linestyle="--", color="gray", label="Train/Test Split")
    ax.legend()
    ax.set_title(f"{ts_code} - LSTM Price Forecasting with FinBERT2 Sentiment")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Close Price")
    fig.tight_layout()

    fig_dir = Path(fig_dir) if fig_dir else FIGURES_DIR / str(val_date)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{ts_code}_forecast.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图片已保存至: {fig_path}")

    # ========== 7. 落盘单日结果 ==========
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / str(val_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{ts_code}_{now_str}.csv"

    pd.DataFrame({
        "ts_code": [ts_code],
        "validation_date": [price_df["trade_date"].values[split]],
        "alpha": [alpha],
        "true_price": [test_true_item],
        "pred_price_base": [test_preds[0]],
        "pred_price_sent": [test_preds_with_sent[0]],
        "sentiment_score": [sentiment_scores[0]],
        "re_base": [re_base],
        "re_sent": [re_sent],
        "ae_base": [ae_base],
        "ae_sent": [ae_sent],
    }).to_csv(save_path, index=False)
    print(f"结果已保存至: {save_path}")

    return {"ts_code": ts_code, "ae_base": ae_base, "ae_sent": ae_sent,
            "re_base": re_base, "re_sent": re_sent}


def main():
    stocks = sys.argv[1:] if len(sys.argv) > 1 else load_stocks()
    for ts_code in stocks:
        run_for_stock(ts_code)


if __name__ == "__main__":
    main()
