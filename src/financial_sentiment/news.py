"""Historical and live news adapters.

GDELT is used for historical retrieval; NewsAPI is deliberately limited to live/recent use.
"""

from __future__ import annotations

from datetime import date
import os

import httpx
import pandas as pd


def _normalise(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["published_at", "headline", "source"])
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    return frame.dropna(subset=["published_at", "headline"]).drop_duplicates("headline")


def fetch_gdelt(query: str, start: date, end: date, max_records: int = 250) -> pd.DataFrame:
    """Fetch historical headlines from GDELT's public DOC 2.0 endpoint.

    GDELT's API is capped per query, so callers should use sensible date ranges and record
    the resulting raw dataset for reproducibility.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": f"{start:%Y%m%d}000000",
        "enddatetime": f"{end:%Y%m%d}235959",
    }
    response = httpx.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
    response.raise_for_status()
    articles = response.json().get("articles", [])
    return _normalise(
        [
            {
                "published_at": article.get("seendate"),
                "headline": article.get("title"),
                "source": article.get("domain"),
            }
            for article in articles
        ]
    )


def fetch_newsapi(query: str, api_key: str | None = None, page_size: int = 100) -> pd.DataFrame:
    """Fetch recent headlines from NewsAPI. The key must be supplied via NEWSAPI_KEY."""
    key = api_key or os.getenv("NEWSAPI_KEY")
    if not key:
        raise ValueError("NEWSAPI_KEY is required for live NewsAPI retrieval.")
    response = httpx.get(
        "https://newsapi.org/v2/everything",
        params={"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": page_size},
        headers={"X-Api-Key": key},
        timeout=30,
    )
    response.raise_for_status()
    return _normalise(
        [
            {
                "published_at": article.get("publishedAt"),
                "headline": article.get("title"),
                "source": (article.get("source") or {}).get("name"),
            }
            for article in response.json().get("articles", [])
        ]
    )

