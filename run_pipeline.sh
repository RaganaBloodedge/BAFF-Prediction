#!/usr/bin/env bash
# BAFF 每日滚动流水线：抓取 → 情绪打分 → 训练预测 → 聚合分析
#
# 运行前：
#   1. 准备 Tushare 凭据：export TUSHARE_TOKEN=<your_token>
#   2. 准备 FinBERT2 微调模型：见 third_party/FinBERT2.md
set -euo pipefail
cd "$(dirname "$0")"

python fetch/fetch_price.py
python fetch/fetch_news_ths.py
python sentiment/build_sentiment_from_csv.py
python forecast/train_lstm_price_with_news.py
python analysis/aggregate.py
python analysis/improve_probability.py
python analysis/trend_consistency.py
python analysis/draw_price_comparison.py
python analysis/draw_improve_probability.py
