"""Small API surface for scoring new headlines after a backtest has been validated."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .sentiment import score_headlines

app = FastAPI(title="Financial Sentiment API", version="0.1.0")


class HeadlineRequest(BaseModel):
    headlines: list[str] = Field(min_length=1, max_length=100)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(request: HeadlineRequest) -> dict[str, list[dict[str, str | float]]]:
    scores = score_headlines(request.headlines)
    return {"results": scores.to_dict(orient="records")}

