"""跑一次回测，把原始预测缓存成 ``results/predictions.csv``。

后续所有消融都在这张表上做后处理，无需重训。

用法::

    # 主回测（纯价格 LSTM），参数取 configs/experiment.yaml 的默认值
    python experiments/run_backtest.py

    # 多种子，用于估计训练随机性带来的方差
    python experiments/run_backtest.py --seeds 0,1,2

    # 特征拼接基线（情绪直接作为输入通道，需要重训）
    python experiments/run_backtest.py --sentiment-input --out results/predictions_concat.csv

    # 只跑论文窗口（2026-02-27 起 22 个交易日）
    python experiments/run_backtest.py --max-dates 22
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baff import paths  # noqa: E402
from baff.backtest import BacktestConfig, run_backtest  # noqa: E402
from baff.config import load_config  # noqa: E402
from baff.data import discover_dates  # noqa: E402
from baff.paths import load_stocks  # noqa: E402
from baff.train import TrainConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # 先解析 --config，再用它填其余默认值
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    bt = load_config(known.config)["backtest"]

    p = argparse.ArgumentParser(
        description="BAFF 回测：训练 LSTM 并缓存原始预测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=known.config, help="实验配置 YAML")
    p.add_argument("--data-root", default=str(paths.RAW_DATA_DIR),
                   help="数据根目录（含 YYYYMMDD 分区）")
    p.add_argument("--out", default=str(paths.PREDICTIONS_CSV), help="输出 CSV 路径")
    p.add_argument("--seeds", default=",".join(str(s) for s in bt["seeds"]),
                   help="随机种子，逗号分隔")
    p.add_argument("--dates", default=None, help="限定日期，逗号分隔，默认全部")
    p.add_argument("--max-dates", type=int, default=None, help="只取前 N 个交易日")
    p.add_argument("--epochs", type=int, default=bt["epochs"])
    p.add_argument("--window", type=int, default=bt["window"])
    p.add_argument("--hidden-dim", type=int, default=bt["hidden_dim"])
    p.add_argument("--num-layers", type=int, default=bt["num_layers"])
    p.add_argument("--sentiment-input", action="store_true",
                   help="特征拼接基线：情绪作为第二输入通道")
    p.add_argument("--threads", type=int, default=bt["threads"],
                   help="torch 线程数；模型极小，单线程通常更快")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    torch.set_num_threads(args.threads)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    else:
        dates = discover_dates(args.data_root)
        if args.max_dates:
            dates = dates[:args.max_dates]

    stocks = load_stocks()
    cfg = BacktestConfig(
        train=TrainConfig(
            window=args.window,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            seed=seeds[0],
        ),
        seeds=seeds,
        use_sentiment_input=args.sentiment_input,
    )

    logging.info("数据 %s，%d 个交易日 x %d 只股票 x %d 个种子",
                 args.data_root, len(dates), len(stocks), len(seeds))
    df = run_backtest(args.data_root, stocks, dates, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logging.info("已写出 %d 行 → %s", len(df), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
