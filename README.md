# Financial News Sentiment Backtester

An end-to-end NLP project that tests whether financial-news sentiment contains information about **future** stock returns and volatility. It uses FinBERT for scoring, GDELT for historical headline retrieval, yfinance for market data, and a deliberately leakage-safe backtest.

## Why this is different from the original notebook

- FinBERT is used only to score text; it is not treated as a data source.
- Historical backtests use GDELT instead of a one-month NewsAPI window.
- A signal from day *t* is joined to the next available trading session, rather than correlated with the already-known same-day close.
- The output includes directional precision and return/volatility correlations, not a single misleading price-level correlation.
- NewsAPI is supported only for live/recent data through `NEWSAPI_KEY`; secrets are never committed.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
financial-sentiment --ticker TSLA --query Tesla --start 2025-01-01 --end 2025-06-30
```

The command writes reproducible inputs and outputs to `artifacts/`:

- `scored_news.csv` — raw historical headlines plus FinBERT outputs
- `aligned_backtest.csv` — next-session outcomes joined to prior available signals
- `metrics.json` — summary evaluation metrics

## API

```bash
uvicorn financial_sentiment.api:app --reload
curl http://127.0.0.1:8000/health
```

`POST /score` accepts `{"headlines":["Tesla reports stronger margins"]}` and returns FinBERT scores.

## Validation principles

This is an exploratory backtest, not investment advice. Evaluate multiple tickers and non-overlapping time windows, compare against a zero-signal baseline, and report uncertainty before making predictive claims. The project does not use a signal published after the market outcome it is evaluated against.

## Development

```bash
ruff check src tests
pytest
docker build -t financial-sentiment .
docker run -p 8000:8000 financial-sentiment
```

GitHub Actions runs linting and unit tests on every pull request.

