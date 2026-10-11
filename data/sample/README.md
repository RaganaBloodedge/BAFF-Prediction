# 样例数据

`20260227/` 是 2026-02-27 单个交易日的完整快照，对应论文 Table 3 的单日样例，
包含 16 只股票 × 3 类文件：

| 文件 | 列 | 说明 |
|---|---|---|
| `price_{code}.csv` | ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount | Tushare 日线 |
| `news_{code}.csv` | trade_date, text | 同花顺公司新闻标题（按日期去重） |
| `sentiment_{code}.csv` | trade_date, sentiment_score | FinBERT2 日度情绪分，∈ [−1, 1] |

三类文件都以 `trade_date`（yyyyMMdd）对齐，因此情绪与价格可按日期直接配对。

完整实验数据由 `fetch/` 与 `sentiment/` 两阶段脚本重新抓取生成，按
`data/{yyyyMMdd}/` 分目录存放——每个目录是"截至当日"的一帧快照；
`experiments/run_backtest.py` 默认从这个分区布局（`data/raw/`）逐日读取。
原始数据体积大且版权归各数据源所有，不随仓库分发，本仓库只保留这一天
作为格式参考。
