"""消融实验：α 扫描、机制族对比、特征拼接基线、统计检验。

输入是 ``results/predictions.csv``（纯后处理，不重训）。输出全部落在
``results/ablations/`` 与 ``figures/ablations/``。

用法::

    python experiments/run_ablations.py
    python experiments/run_ablations.py --concat results/predictions_concat.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baff import evaluate, paths  # noqa: E402

logger = logging.getLogger("ablations")

ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
TAU_GRID = (0.005, 0.01, 0.02, 0.05)


def alpha_sweep(preds: pd.DataFrame) -> pd.DataFrame:
    """在 sign 机制上扫 α，同时给出原始与横截面标准化两种信号。"""
    rows = []
    for transform in ("raw", "cs_zscore"):
        for alpha in ALPHA_GRID:
            detail = evaluate.evaluate(preds, "sign", transform, alpha=alpha)
            row = evaluate.summarize(detail)
            row["alpha"] = alpha
            rows.append(row)
    return pd.DataFrame(rows)


def mechanism_comparison(preds: pd.DataFrame) -> pd.DataFrame:
    """机制族横向对比。每个机制各取论文默认 α 与扫出来的最优 α 两档。"""
    rows = []

    none_detail = evaluate.evaluate(preds, "none", "raw")
    rows.append({**evaluate.summarize(none_detail), "variant": "baseline(no sentiment)"})

    for transform in ("raw", "cs_zscore"):
        for mechanism in ("sign", "magnitude"):
            for tag, alpha in (("alpha=0.5", 0.5), ("alpha=best", None)):
                if alpha is None:
                    sweep = [
                        evaluate.summarize(
                            evaluate.evaluate(preds, mechanism, transform, alpha=a)
                        )["stock_mean"]
                        for a in ALPHA_GRID
                    ]
                    alpha = ALPHA_GRID[int(np.argmax(sweep))]
                detail = evaluate.evaluate(preds, mechanism, transform, alpha=alpha)
                row = evaluate.summarize(detail)
                row.update(alpha=alpha, variant=f"{mechanism}/{transform}/{tag}")
                rows.append(row)

    for transform in ("raw", "cs_zscore"):
        for tau in TAU_GRID:
            detail = evaluate.evaluate(preds, "soft", transform, alpha=0.5, tau=tau)
            row = evaluate.summarize(detail)
            row.update(alpha=0.5, tau=tau, variant=f"soft/{transform}/tau={tau}")
            rows.append(row)

    return pd.DataFrame(rows)


def concat_baseline(preds: pd.DataFrame, concat_path: Path) -> pd.DataFrame:
    """特征拼接 vs 行为调制：两条路线在同一评估口径下的对比。"""
    rows = [{
        "route": "behaviour_modulation(paper, alpha=0.5)",
        "MAE": evaluate.evaluate(preds, "sign", "raw", alpha=0.5)["ae_cand"].mean(),
    }, {
        "route": "no_sentiment(baseline)",
        "MAE": (preds["true_close"] - preds["pred_close"]).abs().mean(),
    }]

    if concat_path.exists():
        concat = pd.read_csv(concat_path)
        rows.append({
            "route": "feature_concat(sentiment as input channel)",
            "MAE": (concat["true_close"] - concat["pred_close"]).abs().mean(),
        })
    else:
        logger.warning("未找到拼接基线 %s，跳过该项（先跑 run_backtest.py --sentiment-input）",
                       concat_path)
    return pd.DataFrame(rows)


def plot_alpha_curve(sweep: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for transform, block in sweep.groupby("transform"):
        ax.plot(block["alpha"], block["stock_mean"], marker="o", label=transform)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("alpha (modulation strength)")
    ax.set_ylabel("mean MAE improvement (per stock)")
    ax.set_title("Sensitivity of behaviour modulation to alpha")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_bars(comp: pd.DataFrame, out: Path) -> None:
    data = comp.dropna(subset=["stock_mean"]).sort_values("stock_mean")
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.34 * len(data))))
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in data["stock_mean"]]
    ax.barh(data["variant"], data["stock_mean"], color=colors)
    for y, (lo, hi) in enumerate(zip(
        data["stock_ci_low"], data["stock_ci_high"], strict=True
    )):
        ax.plot([lo, hi], [y, y], color="black", linewidth=1.2)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("mean MAE improvement (per stock), 95% bootstrap CI")
    ax.set_title("Mechanism comparison")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BAFF 消融实验")
    p.add_argument("--predictions", default=str(paths.PREDICTIONS_CSV))
    p.add_argument("--concat", default=str(paths.RESULTS_DIR / "predictions_concat.csv"))
    p.add_argument("--out-dir", default=str(paths.ABLATION_DIR))
    p.add_argument("--fig-dir", default=str(paths.FIGURES_DIR / "ablations"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s  %(message)s")

    preds = pd.read_csv(args.predictions)
    logger.info("读入 %d 行预测，覆盖 %d 只股票 / %d 个交易日 / %d 个种子",
                len(preds), preds["ts_code"].nunique(),
                preds["val_date"].nunique(), preds["seed"].nunique())

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = alpha_sweep(preds)
    sweep.to_csv(out_dir / "alpha_sweep.csv", index=False)
    plot_alpha_curve(sweep, fig_dir / "alpha_curve.png")

    comp = mechanism_comparison(preds)
    comp.to_csv(out_dir / "mechanism_comparison.csv", index=False)
    plot_mechanism_bars(comp, fig_dir / "mechanism_bars.png")

    concat = concat_baseline(preds, Path(args.concat))
    concat.to_csv(out_dir / "concat_baseline.csv", index=False)

    # 论文口径的逐股表：基线（α=0.5, sign, raw）
    detail = evaluate.evaluate(preds, "sign", "raw", alpha=0.5)
    per_stock = evaluate.by_stock(detail)
    per_stock.to_csv(out_dir / "per_stock_paper_mechanism.csv", index=False)

    directional = pd.DataFrame([{**evaluate.directional_stats(detail), "alpha": 0.5}])
    directional.to_csv(out_dir / "directional.csv", index=False)

    print("\n=== α 扫描（sign, raw）===")
    print(sweep[sweep["transform"] == "raw"][
        ["alpha", "stock_mean", "stockpct_mean", "stock_pos_ratio", "stock_pvalue"]
    ].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== 机制族对比（按 stock_mean 排序）===")
    cols = ["variant", "stock_mean", "stockpct_mean", "stock_pos_ratio", "stock_pvalue"]
    print(comp[cols].sort_values("stock_mean", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== 特征拼接 vs 行为调制（整体 MAE）===")
    print(concat.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 方向层面 ===")
    print(directional.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    logger.info("产物已写入 %s 与 %s", out_dir, fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
