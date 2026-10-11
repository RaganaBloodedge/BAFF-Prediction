#!/usr/bin/env bash
# BAFF 流水线
#
# 分成两条独立流程：
#   data        —— 数据链路：行情抓取 → 新闻抓取 → 情绪打分
#                  需要 TUSHARE_TOKEN 与 FinBERT2 模型（见 third_party/FinBERT2.md）
#   experiments —— 实验链路：回测 → 消融。只需要 data 产出的 CSV，不再训练情绪模型
#
# 用法：
#   bash run_pipeline.sh              # 默认跑实验链路
#   bash run_pipeline.sh data         # 只跑数据链路
#   bash run_pipeline.sh all          # 两条都跑
#
# 可用 PYTHON=<解释器路径> 覆盖默认的 python。
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

run_data() {
  echo "==> 行情抓取（Tushare）"
  "$PY" fetch/fetch_price.py
  echo "==> 新闻抓取（同花顺）"
  "$PY" fetch/fetch_news_ths.py
  echo "==> 情绪打分（FinBERT2）"
  "$PY" sentiment/build_sentiment_from_csv.py
}

run_experiments() {
  echo "==> 回测：训练价格 LSTM 并缓存原始预测"
  "$PY" experiments/run_backtest.py
  echo "==> 回测：特征拼接基线（情绪作为输入通道）"
  "$PY" experiments/run_backtest.py --sentiment-input \
      --out results/predictions_concat.csv
  echo "==> 消融：α 扫描 / 机制对比 / 显著性检验"
  "$PY" experiments/run_ablations.py
}

case "${1:-experiments}" in
  data)        run_data ;;
  experiments) run_experiments ;;
  all)         run_data; run_experiments ;;
  *)           echo "用法: $0 [data|experiments|all]"; exit 2 ;;
esac
