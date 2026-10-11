# 样例数据

`20260227/` 是 2026-02-27 单个交易日的完整快照，对应论文 Table 3 的单日样例，
包含 16 只股票 × 3 类文件：

| 文件 | 列 | 说明 |
|---|---|---|
| `price_{code}.csv` | ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount | Tushare 日线 |
| `news_{code}.csv` | trade_date, text | 同花顺公司新闻标题（按日期去重） |
| `sentiment_{code}.csv` | trade_date, sentiment_score | FinBERT2 日度情绪分，∈ [−1, 1] |

三类文件都以 `trade_date`（yyyyMMdd）对齐，因此情绪与价格可按日期直接配对。

完整实验数据由 `fetch/` 与 `sentiment/` 两阶段的脚本重新抓取生成，按
`data/{yyyyMMdd}/` 分目录存放；本仓库只保留这一天作为格式参考。
