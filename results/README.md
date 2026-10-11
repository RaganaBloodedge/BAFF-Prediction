# results/

实验结果。分三块，对应两条不同的实验流程。

## `aggregated/` —— 论文原始流程的结果

由 `analysis/` 下的脚本从 `results/{YYYYMMDD}/` 的单日结果汇总而来，与论文
表格一一对应：

| 文件 | 生成脚本 | 对应论文 |
|---|---|---|
| `all_results_aggregated.csv` | `analysis/aggregate.py` | 全量明细（Table 3 / Table 4 的来源） |
| `stock_MAE_MRE.csv` | `analysis/aggregate.py` | Table 4：逐股 MAE/MRE 与 `sentiment_improve` |
| `count.csv` | `analysis/improve_probability.py` | Table 5：改善概率 |
| `trend_same_statistics.csv` | `analysis/trend_consistency.py` | Table 6：趋势一致率 |
| `trend_same_barplot.png` | `analysis/trend_consistency.py` | 趋势一致率柱状图 |
| `re_curves/` | `analysis/aggregate.py` | 逐股 `re_base` / `re_sent` 误差曲线 |

数据覆盖 2026-02-27 → 2026-03-27，16 只股票。

这套数字来自论文写作时的运行，**未固定随机种子，无法逐位复现**。

## `predictions.csv` —— 升级流程的原始预测缓存

由 `experiments/run_backtest.py` 产出。每行是一次「截至验证日重训、预测
验证日」的独立实验：

```
ts_code, val_date, seed, model, prev_close, true_close, pred_close,
sentiment, pred_return, true_return
```

关键设计：**调制是纯后处理**。这张表只保存 LSTM 的原始预测，所有机制变体、
α 取值、信号变换都在它上面重算，不需要重训网络——换一个实验设定只要几秒。

训练过程固定了随机种子（见 `configs/experiment.yaml` 的 `seeds`），因此这张表
可以逐位复现。

## `ablations/` —— 消融与显著性检验

由 `experiments/run_ablations.py` 产出：

| 文件 | 内容 |
|---|---|
| `alpha_sweep.csv` | 调制强度 α 从 0 到 1 的敏感度曲线 |
| `mechanism_comparison.csv` | 机制族对比，含 bootstrap 置信区间与配对检验 p 值 |
| `concat_baseline.csv` | 特征拼接 vs 行为调制的整体 MAE 对比 |
| `per_stock_paper_mechanism.csv` | 论文机制（sign, α=0.5）下的逐股结果 |
| `directional.csv` | 方向准确率与 IC |

对应的图在 `figures/ablations/`。

改善量一律定义为

```
improvement = MAE_base − MAE_candidate
```

**正值表示调制优于纯价格基线。**

每日单点结果（`results/{date}/{code}_{timestamp}.csv`）数量多、体积大，
未随仓库分发。
