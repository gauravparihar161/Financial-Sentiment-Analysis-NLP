"""FinBERT inference with batched, probability-weighted scoring."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


LABEL_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def score_headlines(headlines: Iterable[str], batch_size: int = 16) -> pd.DataFrame:
    """Return FinBERT labels, confidence, and a signed confidence-weighted score."""
    from transformers import pipeline

    texts = list(headlines)
    if not texts:
        return pd.DataFrame(columns=["label", "confidence", "sentiment_score"])
    classifier = pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
    results = classifier(texts, batch_size=batch_size)
    frame = pd.DataFrame(results).rename(columns={"label": "label", "score": "confidence"})
    frame["label"] = frame["label"].str.lower()
    frame["sentiment_score"] = frame["label"].map(LABEL_SCORE) * frame["confidence"]
    return frame[["label", "confidence", "sentiment_score"]]

