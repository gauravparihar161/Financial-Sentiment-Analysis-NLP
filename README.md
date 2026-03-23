# 📈 Financial Sentiment Analysis Pipeline

## 🎯 Project Overview
Can we predict stock volatility using AI? This project built an end-to-end data pipeline to analyze the relationship between **Tesla (TSLA)** stock prices and global news sentiment during the volatile **March 2026** news cycle.



## 🛠️ The Tech Stack
* **Language:** Python 3.10
* **AI Model:** FinBERT (HuggingFace Transformers)
* **Data Sources:** NewsAPI (Headlines), yfinance (Market Data)
* **Analysis:** Pandas, Matplotlib, Scipy (Pearson Correlation)

## 🔄 Data Analyst Lifecycle
1. **Data Ingestion:** Automated extraction of 30 days of market data and news headlines via REST APIs.
2. **Sentiment Engineering:** Used a domain-specific BERT model (FinBERT) to classify unstructured text into Positive, Negative, and Neutral scores.
3. **Data Integration:** Merged time-series price data with aggregated daily sentiment scores.
4. **Statistical Validation:** Calculated Pearson’s $r$ and P-values to determine the reliability of the correlation.

## 📊 Results & Insights
* **Correlation found:** $r = 0.36$ (Moderate Positive)
* **Key Insight:** During the Tesla "FSD Probe" in March 2026, negative sentiment spikes preceded price drops with a 24-hour lag.
* **Conclusion:** While NVIDIA showed low sensitivity to daily news, Tesla's "High-Beta" nature makes it a prime candidate for sentiment-based risk modeling.

## 🚀 How to Run
1. Clone this repo.
2. Install requirements: `pip install yfinance transformers pandas matplotlib`
3. Run the `.ipynb` notebook in Google Colab or Jupyter.
