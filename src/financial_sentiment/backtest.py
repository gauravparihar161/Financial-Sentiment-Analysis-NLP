"""Leakage-safe alignment and evaluation utilities."""

from __future__ import annotations

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def build_daily_signals(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate timestamped scored news into UTC calendar-day features."""
    required = {"published_at", "sentiment_score"}
    if missing := required - set(news.columns):
        raise ValueError(f"News data is missing columns: {sorted(missing)}")
    frame = news.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame["date"] = frame["published_at"].dt.tz_localize(None).dt.normalize()
    return frame.groupby("date", as_index=False).agg(
        sentiment_score=("sentiment_score", "mean"), article_count=("sentiment_score", "size")
    )


def prepare_market_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute next-session return and forward five-session realised volatility."""
    frame = prices.copy()
    if "Date" not in frame or "Close" not in frame:
        raise ValueError("Market data requires Date and Close columns.")
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.sort_values("Date").drop_duplicates("Date")
    frame["daily_return"] = frame["Close"].pct_change()
    frame["next_return"] = frame["Close"].shift(-1).div(frame["Close"]).sub(1)
    frame["forward_volatility_5d"] = frame["daily_return"].rolling(5).std().shift(-5)
    return frame.dropna(subset=["next_return", "forward_volatility_5d"])


def align_signals(signals: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Attach a signal only to the *next* trading session, avoiding same-day leakage."""
    trading_days = market[["Date"]].drop_duplicates().sort_values("Date")
    signal_days = signals.sort_values("date").copy()
    signal_days["available_date"] = signal_days["date"] + pd.Timedelta(days=1)
    aligned = pd.merge_asof(
        signal_days, trading_days, left_on="available_date", right_on="Date", direction="forward"
    )
    return aligned.dropna(subset=["Date"]).merge(market, on="Date", how="inner")


def evaluate(aligned: pd.DataFrame) -> dict[str, float | int]:
    """Evaluate rank/linear association and positive-signal directional precision."""
    clean = aligned.dropna(subset=["sentiment_score", "next_return", "forward_volatility_5d"])
    if len(clean) < 3:
        raise ValueError("At least three aligned trading days are required for evaluation.")
    signal = clean["sentiment_score"]
    positive = clean[signal > 0]
    return {
        "observations": len(clean),
        "pearson_sentiment_vs_return": float(pearsonr(signal, clean["next_return"]).statistic),
        "spearman_sentiment_vs_return": float(spearmanr(signal, clean["next_return"]).statistic),
        "pearson_sentiment_vs_volatility": float(pearsonr(signal, clean["forward_volatility_5d"]).statistic),
        "positive_signal_precision": float((positive["next_return"] > 0).mean()) if len(positive) else 0.0,
    }


def run_backtest(news: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Build daily features, align them to future market outcomes, and evaluate."""
    market = prepare_market_data(prices)
    aligned = align_signals(build_daily_signals(news), market)
    return aligned, evaluate(aligned)

