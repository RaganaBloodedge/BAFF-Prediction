"""从 Tushare 拉取日线行情，落盘为 data/price_{code}.csv。

凭据通过环境变量 TUSHARE_TOKEN 提供，不写进源码：
    bash:       export TUSHARE_TOKEN=<your_token>
    PowerShell: $env:TUSHARE_TOKEN = "<your_token>"

用法：
    python fetch/fetch_price.py                 # 抓取 configs/stocks.txt 中的全部股票
    python fetch/fetch_price.py 000547.SZ ...   # 只抓指定股票
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import tushare as ts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, load_stocks  # noqa: E402


def _pro_api():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "未设置环境变量 TUSHARE_TOKEN，无法访问 Tushare。\n"
            "  bash:       export TUSHARE_TOKEN=<your_token>\n"
            '  PowerShell: $env:TUSHARE_TOKEN = "<your_token>"'
        )
    ts.set_token(token)
    return ts.pro_api()


def _default_dates(days: int = 90) -> tuple[str, str]:
    """返回截至今天的最近 days 天的起止日期（yyyyMMdd）。"""
    end = datetime.now()
    start = end - timedelta(days=days)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def fetch_price(ts_code: str, start_date: str | None = None,
                end_date: str | None = None, days: int = 90) -> str:
    """拉取单只股票日线并保存到 data/price_{code}.csv，返回文件路径。"""
    if start_date is None or end_date is None:
        start_date, end_date = _default_dates(days)
    pro = _pro_api()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    code = ts_code.split(".")[0]
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    path = DATA_DIR / f"price_{code}.csv"
    df.to_csv(path, index=False)
    return str(path)


def fetch_price_batch(ts_codes: list[str], days: int = 90) -> list[str]:
    """批量拉取，单只失败不影响其余股票。"""
    start_date, end_date = _default_dates(days)
    pro = _pro_api()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for ts_code in ts_codes:
        code = ts_code.split(".")[0]
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            path = DATA_DIR / f"price_{code}.csv"
            df.to_csv(path, index=False)
            paths.append(str(path))
            print(f"[{ts_code}] 已保存 {len(df)} 条至 {path}")
        except Exception as e:
            print(f"[{ts_code}] 拉取失败: {e}")
    return paths


def main():
    stocks = sys.argv[1:] if len(sys.argv) > 1 else load_stocks()
    fetch_price_batch(stocks, days=90)


if __name__ == "__main__":
    main()
