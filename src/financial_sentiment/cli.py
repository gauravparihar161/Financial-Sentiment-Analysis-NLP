"""Command-line backtest entry point."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from .backtest import run_backtest
from .news import fetch_gdelt
from .sentiment import score_headlines


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest FinBERT news signals without look-ahead bias.")
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--query", default="Tesla")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    news = fetch_gdelt(args.query, args.start, args.end)
    scores = score_headlines(news["headline"])
    news = pd.concat([news.reset_index(drop=True), scores], axis=1)
    prices = yf.download(args.ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)
    prices = prices.reset_index()
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [column[0] for column in prices.columns]
    aligned, metrics = run_backtest(news, prices)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    news.to_csv(output / "scored_news.csv", index=False)
    aligned.to_csv(output / "aligned_backtest.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

