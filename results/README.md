# 实验结果

`aggregated/` 下的文件由 `analysis/` 中的脚本汇总每日结果而来，与论文表格一一对应。

| 文件 | 生成脚本 | 对应论文 |
|---|---|---|
| `all_results_aggregated.csv` | `analysis/aggregate.py` | 全量明细（Table 3 / Table 4 的来源） |
| `stock_MAE_MRE.csv` | `analysis/aggregate.py` | Table 4：每只股票的 MAE/MRE 与 `sentiment_improve` |
| `count.csv` | `analysis/improve_probability.py` | Table 5：改善概率 |
| `trend_same_statistics.csv` | `analysis/trend_consistency.py` | Table 6：情绪与价格趋势一致率 |
| `trend_same_barplot.png` | `analysis/trend_consistency.py` | 趋势一致率柱状图 |
| `re_curves/` | `analysis/aggregate.py` | 每只股票的 `re_base` / `re_sent` 误差曲线 |

数据覆盖 2026-02-27 → 2026-03-27，16 只股票。

每日单点结果（`results/{date}/{code}_{timestamp}.csv`）由主实验直接产出，
数量多、体积大，未随仓库分发。
