"""从同花顺个股页抓取“公司新闻”标题，落盘为 data/news_{code}.csv。

数据源：https://stockpage.10jqka.com.cn/{code}/ 的“公司新闻”区块，仅取标题，
不请求新闻正文。页面结构变更时解析规则需要相应调整。

用法：
    python fetch/fetch_news_ths.py
    python fetch/fetch_news_ths.py 000547.SZ ...
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, load_stocks  # noqa: E402


def _ths_code(ts_code: str) -> str:
    """Tushare 风格 ts_code（000001.SZ）→ 同花顺 6 位代码（000001）。"""
    return ts_code.split(".")[0]


def url_for_code(code: str) -> str:
    """同花顺个股首页链接。"""
    return f"https://stockpage.10jqka.com.cn/{code}/"


def fetch_news_for_ts_code(ts_code: str, max_items: int = 200) -> pd.DataFrame:
    """解析“公司新闻”区块，返回 [trade_date, text] 两列的 DataFrame。

    遍历页面所有 <ul>/<li>，取其中 <a> 的文本作为标题，并从 li 文本中
    匹配两种时间格式：① MM-DD HH:MM（补当前年份）② YYYY-MM-DD。
    最后以 (trade_date, title) 去重。
    """
    code = _ths_code(ts_code)
    url = url_for_code(code)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")

    records: List[Dict[str, Any]] = []
    dt_pattern_short = re.compile(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2})")
    dt_pattern_full = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
    current_year = datetime.now().year

    seen = set()
    for ul in soup.find_all("ul"):
        for li in ul.find_all("li"):
            a = li.find("a")
            if not a:
                continue

            title = a.get_text(strip=True)
            if not title or len(title) < 2:
                continue

            li_text = li.get_text(" ", strip=True)
            m = dt_pattern_short.search(li_text)
            if m:
                year, month, day = current_year, int(m.group(1)), int(m.group(2))
            else:
                m = dt_pattern_full.search(li_text)
                if not m:
                    continue
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))

            try:
                dt = datetime(year, month, day)
            except ValueError:
                continue

            trade_date = dt.strftime("%Y%m%d")
            key = (trade_date, title)
            if key in seen:
                continue
            seen.add(key)
            records.append({"trade_date": trade_date, "text": title})

    if not records:
        raise RuntimeError(
            f"在 {url} 的“公司新闻”区块中未解析到任何新闻，"
            f"可能是页面结构变更或暂时无新闻。"
        )

    df = pd.DataFrame(records).drop_duplicates().sort_values("trade_date")
    if max_items is not None and len(df) > max_items:
        df = df.tail(max_items)
    return df


def fetch_and_save_news(ts_code: str, max_items: int = 300) -> str:
    """抓取单只股票新闻并保存到 data/news_{code}.csv，返回输出路径。"""
    df = fetch_news_for_ts_code(ts_code, max_items=max_items)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = DATA_DIR / f"news_{_ths_code(ts_code)}.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return str(output_csv)


def main():
    stocks = sys.argv[1:] if len(sys.argv) > 1 else load_stocks()
    for ts_code in stocks:
        path = fetch_and_save_news(ts_code, max_items=300)
        print(f"[{ts_code}] 已抓取并保存至: {path}")


if __name__ == "__main__":
    main()
