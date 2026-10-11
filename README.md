# BAFF: Behavior-Aware Forecasting Framework

**基于行为调制机制的短期股票价格预测** —— 价格 LSTM × FinBERT2 新闻情绪

BAFF 是一个模块化的双流预测框架。价格流用 LSTM 处理历史收盘价序列，文本流用
FinBERT2 把每日财经新闻标题编码为日度情绪分。两者**不做特征拼接**：情绪分不进入
价格特征空间，而是作为**行为调制信号**，按方向一致性放大或抑制 LSTM 给出的预测收益率。

本仓库是硕士毕业论文（ITMO University，2026）的实验实现。

## 核心机制

LSTM 先在历史收盘价上给出下一日价格预测，换算成收益率：

```
r_pred = (p̂_{t+1} − p_t) / p_t
```

取其方向 `d = sign(r_pred)`，与当日情绪分 `s ∈ [−1, 1]` 构造方向一致性调制因子：

```
m      = 1 + α · d · s          (α = 0.5)
r_adj  = m · r_pred
p̂_adj  = p_t · (1 + r_adj)
```

情绪方向与预测方向一致（`d·s > 0`）时 `m > 1`，涨跌幅被放大；方向矛盾（`d·s < 0`）
时 `m < 1`，预测被抑制。`α` 为固定超参，实验取 0.5。

用收益率而非价格作为调制对象，是为了跨股票尺度不变、更贴近交易决策；
选择"调制"而非"特征拼接"，动机见论文第 1.4 节对特征级融合四类缺陷的讨论
（时间错位、缺乏结构化建模依据、忽视跨模态交互、缺乏行为金融现实性）。

## 目录结构

```
BAFF-Prediction/
├── baff/                              核心包（可安装、可测试）
│   ├── mechanism.py                   调制机制族：信号变换 × 调制函数
│   ├── backtest.py                    回测引擎（逐日重训，缓存原始预测）
│   ├── evaluate.py                    机制评估：误差 / 改善量 / 显著性汇总
│   ├── train.py                       LSTM 训练（固定种子）与标准化
│   ├── data.py                        数据加载（日期分区 / 平铺两种布局）
│   ├── metrics.py                     MAE / RMSE / MRE / 方向准确率 / IC
│   ├── stats.py                       Wilcoxon 配对检验 + bootstrap CI
│   ├── models.py                      PriceLSTM / PriceLSTMConcat
│   ├── config.py                      实验配置加载
│   └── paths.py                       路径常量与股票池
├── experiments/                       升级后的实验入口
│   ├── run_backtest.py                回测 → results/predictions.csv
│   └── run_ablations.py               α 扫描 / 机制对比 / 显著性检验
├── fetch/                             数据抓取
│   ├── fetch_price.py                 Tushare 日线 → data/price_{code}.csv
│   └── fetch_news_ths.py              同花顺公司新闻标题 → data/news_{code}.csv
├── sentiment/                         情绪编码
│   ├── build_sentiment_from_csv.py    批量新闻 → 日度情绪分 data/sentiment_{code}.csv
│   └── predict_news.py                单条新闻推理示例
├── forecast/                          单股演练 CLI（薄包装，复用 baff 包）
├── analysis/                          论文原始流程的聚合与绘图脚本
├── tests/                             pytest 单元测试（95 个用例）
├── configs/                           stocks.txt + experiment.yaml
├── data/sample/                       单日样例（论文 Table 3）
├── results/                           论文结果 + 可复现结果（见 results/README.md）
├── figures/                           生成的图表
├── docs/EXPERIMENTS.md                升级实验的设计与结论
├── third_party/FinBERT2.md            上游依赖说明
└── run_pipeline.sh                    一键流水线
```

## 数据流

实验按日增量滚动执行，模拟真实部署：每天只用截至当日的数据，保证输入与预测目标的时间一致。

```
① fetch_price.py        →  data/price_{code}.csv        (Tushare，近 90 日)
② fetch_news_ths.py     →  data/news_{code}.csv         (同花顺，标题)
③ sentiment/build_sentiment_from_csv.py
                        →  data/sentiment_{code}.csv    (FinBERT2 日度情绪分)
④ forecast/train_lstm_price_with_news.py
                        →  results/{date}/{code}_{ts}.csv
                        →  figures/{date}/{code}_forecast.png
⑤ analysis/*            →  results/aggregated/*
```

三类文件都以 `trade_date` 对齐，情绪与价格可按日期直接配对。

## 快速开始

```bash
pip install -r requirements.txt
```

两个前置条件：

**1. Tushare 凭据**

```bash
export TUSHARE_TOKEN=<your_token>
# PowerShell: $env:TUSHARE_TOKEN = "<your_token>"
```

**2. FinBERT2 情绪模型**

权重为 GB 级，不随仓库分发。按 [`third_party/FinBERT2.md`](third_party/FinBERT2.md) 准备后，
放到 `models/FinBERT2-base_SC_2/`，或用 `FINBERT_MODEL_DIR` 指向它。

跑通一天的全流程：

```bash
bash run_pipeline.sh
```

跑完整的可复现实验（固定随机种子）：

```bash
python experiments/run_backtest.py      # 逐日重训回测，约 16 分钟
python experiments/run_ablations.py     # α 扫描 / 机制对比 / 显著性检验，秒级
pytest tests -q                         # 95 个单元测试
```

实验设计与结论见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

也可以单只股票分步试跑：

```bash
python fetch/fetch_price.py 000547.SZ
python fetch/fetch_news_ths.py 000547.SZ
python sentiment/build_sentiment_from_csv.py 000547
python forecast/train_lstm_price_with_news.py 000547.SZ
python analysis/aggregate.py
```

## 实验设置

| 项目 | 取值 |
|---|---|
| 股票池 | 16 只，覆盖金融、消费、医药、能源资源、半导体与信息技术五类行业 |
| 回溯窗口 W | 20 个交易日 |
| LSTM | 输入维度 1 / 隐藏维度 64 / 2 层 |
| 训练 | 200 epoch，batch 32，Adam，lr 1e-3，MSE 损失 |
| 情绪模型 | FinBERT2-base 微调（二分类） |
| 情绪分 | `score = P(正面) − P(负面) ∈ [−1, 1]`，按交易日取均值 |
| 调制强度 α | 0.5 |
| 实验窗口 | 2026-02-27 → 2026-03-27 |
| 评估基线 | 相同价格输入、相同参数下的纯价格 LSTM |

## 结果

月度聚合（对应论文 Table 4，单位同原始价格；`sentiment_improve = MAE_base − MAE_sent`，正值表示调制带来改善）：

| 代码 | MAE_base | MAE_sent | sentiment_improve |
|---|---|---|---|
| 000547.SZ | 1.50 | 1.41 | **+0.0938** |
| 000858.SZ | 1.56 | 1.85 | −0.2907 |
| 002155.SZ | 2.36 | 2.37 | −0.0052 |
| 002460.SZ | 1.73 | 1.75 | −0.0121 |
| 002594.SZ | 2.23 | 2.27 | −0.0467 |
| 300308.SZ | 17.72 | 16.54 | **+1.1807** |
| 300394.SZ | 13.90 | 14.02 | −0.1250 |
| 301308.SZ | 17.49 | 17.94 | −0.4541 |
| 600010.SH | 0.05 | 0.06 | −0.0061 |
| 600276.SH | 1.44 | 1.43 | **+0.0103** |
| 600759.SH | 0.55 | 0.54 | **+0.0122** |
| 600879.SH | 1.02 | 1.12 | −0.0950 |
| 601318.SH | 0.91 | 0.95 | −0.0363 |
| 601899.SH | 1.20 | 1.19 | **+0.0054** |
| 603618.SH | 1.30 | 1.23 | **+0.0670** |
| 603986.SH | 11.31 | 13.05 | −1.7431 |

结论：

- 聚合指标上，行为调制整体**未超过**纯价格基线：16 只中 6 只 `sentiment_improve` 为正，
  7 只改善概率超过 0.5，且无一只表现出稳定显著的改善。
- 情绪与真实价格趋势一致率整体仅 **53%**（对应论文 Table 6），接近随机水平；
  个股分化明显（最好 70%，最差 30%）。
- 基线与调制后的相对误差曲线走势高度相似，说明调制未改变误差的整体结构。

实验期恰逢地缘政治不确定性上升，宏观因子主导定价时，公司层面新闻情绪的解释力被系统性削弱。
更完整的讨论见论文第 3.5 节。

### 可复现实验（2026-10 升级）

固定随机种子、把验证窗口扩到 45 个交易日、补上统计检验后，结论比论文更细：

- 论文机制（sign, α=0.5）的改善 +0.026/股，Wilcoxon p=0.60，**不显著**；
  扫描 α ∈ [0, 1] 的最优值（≈0.4）也只有 +0.030（p=0.50）；
- 去掉方向一致性（只按情绪强弱缩放）后调制**显著恶化**（p<0.001）——
  机制里真正干活的是方向一致性项，不是情绪强度；
- 特征拼接对照组实跑：拼接 MAE 4.72 > 基线 4.63 > 调制 4.61，论文
  "调制优于拼接"的机制选择**首次获得实证支持**；
- 根因发现：情绪分 44% 的样本 > 0.9、仅 19.6% 为负，区分度严重不足——
  这是调制效果有限的更底层原因，比"宏观环境"的解释更可操作。

详见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 上游依赖与致谢

情绪编码环节复用熵简科技开源的 **FinBERT2** 套件，该套件以 MIT 协议独立开源，
本仓库不重新分发其源码与权重。准备方式与引用格式见
[`third_party/FinBERT2.md`](third_party/FinBERT2.md) 与 [`NOTICE`](NOTICE)。

股票行情来自 Tushare，新闻标题来自同花顺个股页；数据版权归各数据源所有，
本仓库只提供抓取脚本，不分发原始数据。

## 局限

- 日度情绪分用等权平均聚合，假设当日各条新闻等权且独立，丢失新闻间交互。
- 情绪分分布严重偏正（44% 的样本 > 0.9），区分度有限——这比调制机制本身
  更底层，校准情绪分可能收益更大（[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §2）。
- 调制强度已做 α 全域扫描与显著性检验（本仓库升级），但任何取值都不显著；
  逐股、逐时段的自适应 α 留待后续。
- 情绪与价格的时间对齐存在隐含假设，需进一步验证是否存在信息错位。
- 全部结果基于单随机种子；多种子方差估计是自然的下一步
  （[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §7）。

## License

MIT —— 见 [`LICENSE`](LICENSE)。
