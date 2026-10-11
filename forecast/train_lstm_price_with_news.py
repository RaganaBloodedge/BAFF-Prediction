"""单只股票单日演练：训练价格 LSTM，并比较基线与行为调制预测。

这是最小可运行示例，用来快速验证数据链路与调制机制是否正常。批量回测与
消融实验请用 ``experiments/`` 下的脚本。

用法::

    python forecast/train_lstm_price_with_news.py                 # 股票池全部，取最新日期
    python forecast/train_lstm_price_with_news.py 000547.SZ
    python forecast/train_lstm_price_with_news.py 000547.SZ --date 20260327
    python forecast/train_lstm_price_with_news.py --list-dates
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baff import paths  # noqa: E402
from baff.backtest import BacktestConfig, forecast_one  # noqa: E402
from baff.data import DataError, discover_dates, load_snapshot  # noqa: E402
from baff.evaluate import evaluate  # noqa: E402
from baff.paths import load_stocks  # noqa: E402
from baff.train import TrainConfig  # noqa: E402

logger = logging.getLogger("forecast")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BAFF 单股演练")
    p.add_argument("stocks", nargs="*", help="股票代码，默认取 configs/stocks.txt")
    p.add_argument("--data-root", default=str(paths.RAW_DATA_DIR),
                   help="数据目录（含 YYYYMMDD 分区）")
    p.add_argument("--date", default=None, help="验证日，默认取最新一天")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--list-dates", action="store_true", help="列出可用日期后退出")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s  %(message)s",
    )

    try:
        dates = discover_dates(args.data_root)
    except DataError as exc:
        logger.error("%s", exc)
        return 2

    if args.list_dates:
        print("\n".join(dates))
        return 0

    date = args.date or dates[-1]
    stocks = args.stocks or load_stocks()
    cfg = BacktestConfig(
        train=TrainConfig(window=args.window, epochs=args.epochs, seed=args.seed),
        seeds=(args.seed,),
    )

    rows = []
    for ts_code in stocks:
        try:
            snapshot = load_snapshot(args.data_root, ts_code, date)
        except DataError as exc:
            logger.warning("跳过 %s：%s", ts_code, exc)
            continue

        row = forecast_one(snapshot, cfg)
        if row is None:
            logger.warning("跳过 %s：训练样本不足", ts_code)
            continue
        rows.append(row)

    if not rows:
        logger.error("没有产生任何结果")
        return 1

    import pandas as pd

    preds = pd.DataFrame(rows)
    detail = evaluate(preds, "sign", "raw", alpha=args.alpha)

    cols = ["ts_code", "prev_close", "true_close", "pred_close",
            "sentiment", "ae_base", "ae_cand", "improvement"]
    print(f"\n验证日 {date}（alpha={args.alpha}，seed={args.seed}）")
    print(detail[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    improved = int((detail["improvement"] > 0).sum())
    print(f"\n改善 {improved}/{len(detail)} 只；"
          f"平均改善 {detail['improvement'].mean():+.4f}")

    out_dir = paths.RESULTS_DIR / "single"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}_alpha{args.alpha}.csv"
    detail.to_csv(out_path, index=False)
    logger.info("明细已写入 %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
