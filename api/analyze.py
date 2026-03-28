from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from stock import (  # noqa: E402
    basic_analysis,
    calculate_technical_indicators,
    evaluate_stock,
    fetch_news_sentiment_from_gnews,
    fetch_news_sentiment_from_newsapi,
    fetch_news_sentiment_from_yfinance,
    get_stock_data,
)

try:
    import pandas_ta as ta  # type: ignore
except ImportError:
    ta = None

app = Flask(__name__)

DEFAULT_PERIOD = "6mo"
ALLOWED_PERIODS = {"6mo", "1y", "2y", "max"}
PERIOD_ALIASES = {
    "6mo": "6mo",
    "6months": "6mo",
    "6m": "6mo",
    "1y": "1y",
    "1year": "1y",
    "2y": "2y",
    "2years": "2y",
    "max": "max",
}

TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")


def _clean_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if pd.isna(value):
        return None
    return None


def _series_to_list(series: pd.Series | None) -> list[int | float | None]:
    if series is None:
        return []
    return [_clean_number(value) for value in series.tolist()]


def _normalize_period(raw_period: str | None) -> str:
    if not raw_period:
        return DEFAULT_PERIOD
    cleaned = raw_period.strip().lower().replace(" ", "")
    normalized = PERIOD_ALIASES.get(cleaned, cleaned)
    return normalized if normalized in ALLOWED_PERIODS else DEFAULT_PERIOD


def _weighted_sentiment(news_rows: list[tuple[float, float, list[str], str]]) -> float | None:
    if not news_rows:
        return None

    weighted_sum = sum(sentiment * weight for sentiment, weight, _, _ in news_rows)
    total_weight = sum(weight for _, weight, _, _ in news_rows)
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _build_indicator_series(df: pd.DataFrame) -> dict[str, list[int | float | None]]:
    close = df["Close"]
    volume = df["Volume"]

    sma_20 = close.rolling(window=20).mean() if len(df) >= 20 else None
    sma_50 = close.rolling(window=50).mean() if len(df) >= 50 else None
    sma_200 = close.rolling(window=200).mean() if len(df) >= 200 else None
    volume_sma_20 = volume.rolling(window=20).mean() if len(df) >= 20 else None

    rsi_series = None
    macd_line = None
    macd_signal = None
    macd_hist = None

    if ta is not None and len(df) >= 14:
        try:
            rsi_series = df.ta.rsi(length=14)
            if len(df) >= 34:
                macd_df = df.ta.macd(fast=12, slow=26, signal=9, append=False)
                if macd_df is not None and not macd_df.empty:
                    macd_line = macd_df.get("MACD_12_26_9")
                    macd_signal = macd_df.get("MACDs_12_26_9")
                    macd_hist = macd_df.get("MACDh_12_26_9")
        except Exception:
            pass

    return {
        "sma20": _series_to_list(sma_20),
        "sma50": _series_to_list(sma_50),
        "sma200": _series_to_list(sma_200),
        "volume_sma20": _series_to_list(volume_sma_20),
        "rsi14": _series_to_list(rsi_series),
        "macd": _series_to_list(macd_line),
        "macd_signal": _series_to_list(macd_signal),
        "macd_hist": _series_to_list(macd_hist),
    }


@app.get("/")
def analyze_stock():
    raw_ticker = (request.args.get("ticker") or "").strip().upper()
    raw_period = request.args.get("period")
    period = _normalize_period(raw_period)

    if not raw_ticker:
        return jsonify({"status": "error", "error": "ticker query parameter is required."}), 400
    if not TICKER_PATTERN.match(raw_ticker):
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Invalid ticker format. Use letters, numbers, dot, or hyphen.",
                }
            ),
            400,
        )

    historical_data, current_price, company_fundamentals, error = get_stock_data(raw_ticker, period=period)
    if error:
        return jsonify({"status": "error", "ticker": raw_ticker, "error": error}), 400
    if historical_data is None or historical_data.empty or company_fundamentals is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "ticker": raw_ticker,
                    "error": "No historical data available for this ticker.",
                }
            ),
            404,
        )

    technical_indicators = calculate_technical_indicators(historical_data)
    company_long_name = company_fundamentals.get("longName")

    news_api_key = os.environ.get("NEWS_API_KEY")
    gnews_api_key = os.environ.get("GNEWS_API_KEY")

    all_news_rows: list[tuple[float, float, list[str], str]] = []
    all_news_titles: list[str] = []

    if news_api_key:
        newsapi_rows, newsapi_titles = fetch_news_sentiment_from_newsapi(
            raw_ticker, news_api_key, company_long_name
        )
        all_news_rows.extend(newsapi_rows)
        all_news_titles.extend(newsapi_titles)

    if gnews_api_key:
        gnews_rows, gnews_titles = fetch_news_sentiment_from_gnews(
            raw_ticker, gnews_api_key, company_long_name
        )
        all_news_rows.extend(gnews_rows)
        all_news_titles.extend(gnews_titles)

    yf_news_rows, yf_news_titles = fetch_news_sentiment_from_yfinance(raw_ticker, company_long_name)
    all_news_rows.extend(yf_news_rows)
    all_news_titles.extend(yf_news_titles)

    deduped_titles = list(dict.fromkeys(all_news_titles))
    overall_news_sentiment = _weighted_sentiment(all_news_rows)

    basic_recommendation, basic_confidence, basic_reason = basic_analysis(
        historical_data, overall_news_sentiment, deduped_titles
    )

    swing_analysis = evaluate_stock(
        historical_data=historical_data,
        technical_indicators=technical_indicators,
        company_fundamentals=company_fundamentals,
        overall_news_sentiment=overall_news_sentiment,
        current_price=current_price,
        all_time_high=company_fundamentals.get("ath_from_period"),
        all_news_articles_data=all_news_rows,
        all_news_titles=deduped_titles,
    )

    indicator_series = _build_indicator_series(historical_data)
    chart_data = {
        "dates": historical_data.index.strftime("%Y-%m-%d").tolist(),
        "open": _series_to_list(historical_data["Open"]),
        "high": _series_to_list(historical_data["High"]),
        "low": _series_to_list(historical_data["Low"]),
        "close": _series_to_list(historical_data["Close"]),
        "volume": _series_to_list(historical_data["Volume"]),
        **indicator_series,
    }

    warnings: list[str] = []
    if ta is None:
        warnings.append("pandas-ta is not installed. RSI and MACD chart series may be unavailable.")
    if not news_api_key:
        warnings.append("NEWS_API_KEY is not configured. NewsAPI headlines are skipped.")
    if not gnews_api_key:
        warnings.append("GNEWS_API_KEY is not configured. GNews headlines are skipped.")

    payload = {
        "status": "ok",
        "meta": {
            "ticker": raw_ticker,
            "period": period,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "company_name": company_long_name or raw_ticker,
        },
        "quote": {
            "current_price": _clean_number(current_price),
            "currency": company_fundamentals.get("currency") or "USD",
            "exchange": company_fundamentals.get("exchange"),
        },
        "fundamentals": {
            "market_cap": _clean_number(company_fundamentals.get("marketCap")),
            "trailing_pe": _clean_number(company_fundamentals.get("trailingPE")),
            "eps_growth": _clean_number(company_fundamentals.get("earningsGrowth")),
            "roe": _clean_number(company_fundamentals.get("returnOnEquity")),
            "debt_to_equity": _clean_number(company_fundamentals.get("debtToEquity")),
            "ath_from_period": _clean_number(company_fundamentals.get("ath_from_period")),
        },
        "indicators": {
            "sma_5": _clean_number(technical_indicators.get("SMA_5")),
            "sma_10": _clean_number(technical_indicators.get("SMA_10")),
            "sma_20": _clean_number(technical_indicators.get("SMA_20")),
            "rsi": _clean_number(technical_indicators.get("RSI")),
            "macd": _clean_number(technical_indicators.get("MACD")),
            "macd_signal": _clean_number(technical_indicators.get("MACD_Signal")),
            "macd_hist": _clean_number(technical_indicators.get("MACD_Hist")),
            "volume_sma_5": _clean_number(technical_indicators.get("Volume_SMA_5")),
        },
        "news": {
            "overall_sentiment": _clean_number(overall_news_sentiment),
            "titles": deduped_titles[:20],
            "article_count": len(all_news_rows),
        },
        "analysis": {
            "basic": {
                "recommendation": basic_recommendation,
                "confidence": int(basic_confidence),
                "reason": basic_reason,
            },
            "swing_trader": swing_analysis.get("swing_trader", {}),
        },
        "charts": chart_data,
        "warnings": warnings,
    }

    return jsonify(payload)
