import pandas as pd

from financial_sentiment.backtest import align_signals, build_daily_signals, prepare_market_data


def test_signal_is_attached_to_next_trading_day():
    prices = pd.DataFrame({"Date": pd.bdate_range("2025-01-02", periods=12), "Close": range(100, 112)})
    news = pd.DataFrame({"published_at": ["2025-01-03T18:00:00Z"], "sentiment_score": [0.8]})
    aligned = align_signals(build_daily_signals(news), prepare_market_data(prices))
    assert aligned.iloc[0]["Date"] == pd.Timestamp("2025-01-06")
    assert aligned.iloc[0]["sentiment_score"] == 0.8


def test_daily_signal_uses_mean_score_and_count():
    news = pd.DataFrame(
        {"published_at": ["2025-01-03T02:00:00Z", "2025-01-03T18:00:00Z"], "sentiment_score": [0.9, -0.3]}
    )
    signals = build_daily_signals(news)
    assert signals.iloc[0]["article_count"] == 2
    assert signals.iloc[0]["sentiment_score"] == 0.3

