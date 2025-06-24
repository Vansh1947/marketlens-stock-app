"""
This module provides comprehensive stock analysis functionalities, including
data fetching, technical indicator calculation, sentiment analysis from news,
and generating trading recommendations. It now focuses exclusively on swing trading,
including dynamic alert generation, refined news sentiment analysis with source weighting,
and a dedicated swing recommendation system.
"""
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
import re # Added for regex in news filtering

import logging

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conditional imports for external APIs
try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("Yfinance library not found. Stock data functionalities will be skipped.")

try:
    import pandas_ta as ta
except ImportError:
    ta = None
    logger.warning("Pandas-ta library not found. Technical indicators will be skipped. Please install it by running: pip install pandas-ta")

try:
    from newsapi import NewsApiClient
    from newsapi.newsapi_exception import NewsAPIException
except ImportError:
    NewsApiClient = None # type: ignore
    class MockNewsAPIException(Exception):
        pass
    NewsAPIException = MockNewsAPIException # type: ignore
    logger.warning("Newsapi-python library not found. NewsAPI functionalities will be skipped.")

try:
    import feedparser
except ImportError:
    feedparser = None
    logger.warning("Feedparser library not found. RSS feed functionalities will be skipped.")

try:
    from gnews import GNews
except ImportError:
    GNews = None
    logger.warning("Gnews library not found. GNews functionalities will be skipped.")


# --- IMPORTANT: INSTALL NECESSARY LIBRARIES (for direct script execution) ---
# If you encounter ModuleNotFoundError for the libraries below, run:
# pip install yfinance textblob newsapi-python feedparser gnews pandas-ta
# --- END OF INSTALLATION INSTRUCTIONS ---


# --- CONFIGURATION (BEST PRACTICE: USING ENVIRONMENT VARIABLES) ---
# Set these environment variables in your system's environment (NOT directly in code):
# export NEWS_API_KEY="your_actual_news_api_key_here"
# export OPENAI_API_KEY="sk-proj-..."
# stock.py will now only read from environment variables if run directly

# Initialize GNews client
gnews_client = None # Initialize to None
if GNews:  # Check if the GNews library is installed
    gnews_client = GNews(max_results=20, period='7d') # Default to 20 results from last 7 days
    logger.info("GNews client initialized (max_results=20, period=7d).")
else: # Ensure this 'else' is aligned with the 'if GNews:' above
    logger.warning("Gnews library not found. GNews functionalities will be skipped.")

# Define the base Google News RSS URL (will be made ticker-specific dynamically)
BASE_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"

# --- ANALYSIS THRESHOLDS (Constants for clarity and easy modification) ---
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70
RSI_BULLISH_NEUTRAL_THRESHOLD = 55 # For growth stock check
PE_RATIO_UNDERVALUED_THRESHOLD = 15
PE_RATIO_OVERVALUED_THRESHOLD = 30
POSITIVE_SENTIMENT_THRESHOLD = 0.1 # For sentiment classification: >0.1 is Positive
NEGATIVE_SENTIMENT_THRESHOLD = 0.0 # For sentiment classification: <0.0 is Negative
EPS_GROWTH_STRONG_THRESHOLD = 0.30 
EPS_GROWTH_NEGATIVE_THRESHOLD = -0.1
VOLUME_HIGH_THRESHOLD_MULTIPLIER = 1.5 # Current volume > 1.5 * SMA_5_Volume
# New fundamental thresholds
ROE_GOOD_THRESHOLD = 0.15 # 15%
DEBT_TO_EQUITY_LOW_THRESHOLD = 0.5 # Lower is better
DEBT_TO_EQUITY_HIGH_THRESHOLD = 1.5 # Higher is worse
# Swing Trader Thresholds
SWING_RSI_LOWER_BUY = 30
SWING_RSI_UPPER_BUY = 45
SWING_SENTIMENT_BUY_THRESHOLD = 0.2
SWING_ATH_DISCOUNT_BUY = 0.95 # Current price < 95% of ATH
SWING_RSI_SELL = 70
SWING_SENTIMENT_SELL_THRESHOLD = -0.2
SWING_ATH_PREMIUM_SELL = 0.98 # Current price >= 98% of ATH

# Long-Term Investor Thresholds
LT_PE_BUY_THRESHOLD = 40
LT_EPS_GROWTH_BUY_THRESHOLD = 0.0 # EPS Growth > 0
LT_ROE_BUY_THRESHOLD = 0.15 # 15%
LT_DE_BUY_THRESHOLD = 0.80 # 80%

# New Alert Thresholds (can be refined)
MACD_HIST_BULLISH_THRESHOLD = 0.5
MACD_HIST_BEARISH_THRESHOLD = -0.5
PRICE_NEAR_ATH_THRESHOLD = 0.95 # 95% of ATH
NEWS_SENTIMENT_STRONG_NEGATIVE_THRESHOLD = -0.4
NEWS_SENTIMENT_STRONG_POSITIVE_THRESHOLD = 0.4

# New News Refinement Constants
SOURCE_WEIGHTS = {
    "Bloomberg": 1.0,
    "Reuters": 0.9,
    "CNBC": 0.8,
    "The Wall Street Journal": 0.95,
    "Associated Press": 0.85,
    "Yahoo Finance": 0.7,
    "MarketWatch": 0.7,
    "Seeking Alpha": 0.6,
    "Reddit": 0.4,
    "Unknown Blog": 0.2,
    "Google News": 0.7, # Default for GNews/RSS if specific source not found
    "NewsAPI": 0.7, # Default for NewsAPI if specific source not found
}

# --- END OF THRESHOLDS ---

# --- Recommendation Bands based on Final Score ---
def classify_recommendation(final_score: float) -> str:
    """Classifies a final score into a recommendation string."""
    if final_score >= 80:
        return "Strong Buy"
    elif 60 <= final_score < 80:
        return "Buy"
    elif 40 <= final_score < 60:
        return "Hold"
    elif 20 <= final_score < 40: # This is now "Sell"
        return "Sell"
    else:  # < 20
        return "Strong Sell"

# --- TECHNICAL INDICATOR CALCULATIONS ---
def calculate_technical_indicators(historical_data: pd.DataFrame) -> dict:
    """
    Calculates various technical indicators for the given historical stock data.

    Args:
        historical_data (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns.

    Returns:
        dict: A dictionary containing calculated technical indicators.
    """
    if len(historical_data) < 5:
        logger.warning("Insufficient data for calculating technical indicators. Returning empty dict.")
        return {}
    if ta is None:
        return {}
    df = historical_data.copy()
    indicators = {}

    # Ensure enough data for indicators
    if len(df) < 200: # Max window size for SMAs
        # pandas-ta might also print its own warnings if data is insufficient for certain indicators
        logger.warning(f"Not enough historical data ({len(df)} rows) for some indicators (e.g., SMA200 needs 200). Some indicators might be None.")

    # Simple Moving Averages
    indicators['SMA_5'] = df['Close'].rolling(window=5).mean().iloc[-1] if len(df) >= 5 else None
    indicators['SMA_10'] = df['Close'].rolling(window=10).mean().iloc[-1] if len(df) >= 10 else None # Added for swing
    indicators['SMA_20'] = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
    indicators['SMA_50'] = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
    indicators['SMA_200'] = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None

    # Relative Strength Index (RSI)
    # pandas-ta handles data length checks internally
    rsi_series = df.ta.rsi(length=14)
    if rsi_series is not None and not rsi_series.empty:
        indicators['RSI'] = rsi_series.iloc[-1]
    else:
        indicators['RSI'] = None

    # Moving Average Convergence Divergence (MACD)
    # pandas-ta returns a DataFrame for MACD
    # Standard MACD (12,26,9) needs about 34 periods for full calculation.
    # pandas-ta will return NaNs if data is insufficient.
    macd_df = df.ta.macd(fast=12, slow=26, signal=9, append=False) # Use append=False to get only the MACD columns
    if macd_df is not None and not macd_df.empty:
        # Columns are typically named like 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
        indicators['MACD'] = macd_df.iloc[-1].get(f'MACD_12_26_9')
        indicators['MACD_Signal'] = macd_df.iloc[-1].get(f'MACDs_12_26_9')
        indicators['MACD_Hist'] = macd_df.iloc[-1].get(f'MACDh_12_26_9') # Added histogram
    else:
        indicators['MACD'] = None
        indicators['MACD_Signal'] = None
        indicators['MACD_Hist'] = None

    # Volume Simple Moving Average
    indicators['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean().iloc[-1] if len(df) >= 5 else None

    return indicators

# --- BASIC TECHNICAL ANALYSIS ---
def basic_analysis(historical_data: pd.DataFrame, news_sentiment: float = None) -> tuple:
    """
    Performs basic stock analysis based on technical indicators and news sentiment.
    NOTE: This function is kept for compatibility with the basic analysis section in stock_app.py.
    The comprehensive analysis is now handled by `enhanced_analysis` and `evaluate_stock`.

    Args:
        historical_data (pd.DataFrame): DataFrame with historical stock data.
        news_sentiment (float, optional): Sentiment score of news (-1 to 1). Defaults to None.

    Returns:
        tuple: (Recommendation: str, Confidence: int, Reason: str)
    """
    if historical_data.empty:
        return "Hold", 0, "Insufficient historical data for analysis."

    technical_indicators = calculate_technical_indicators(historical_data)

    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    reasons = []    
    
    # SMA Crossover
    sma_5 = technical_indicators.get('SMA_5')
    sma_20 = technical_indicators.get('SMA_20')
    if sma_5 is not None and sma_20 is not None:
        if sma_5 > sma_20: # Bullish
            buy_signals += 1
            reasons.append("5-day SMA above 20-day SMA (Bullish Crossover)")
        elif sma_5 < sma_20: # Bearish
            sell_signals += 1
            reasons.append("5-day SMA below 20-day SMA (Bearish Crossover)")
        else:
            hold_signals += 1
            reasons.append("5-day and 20-day SMAs are close (Neutral Crossover)")

    # RSI
    rsi_value = technical_indicators.get('RSI') 
    if rsi_value is not None:
        if rsi_value < RSI_OVERSOLD_THRESHOLD: buy_signals += 1; reasons.append(f"RSI ({rsi_value:.2f}) indicates oversold condition")
        elif rsi_value > RSI_OVERBOUGHT_THRESHOLD: sell_signals += 1; reasons.append(f"RSI ({rsi_value:.2f}) indicates overbought condition")
        else: hold_signals += 1; reasons.append(f"RSI ({rsi_value:.2f}) is neutral")
    # MACD
    macd_value = technical_indicators.get('MACD')
    macd_signal_value = technical_indicators.get('MACD_Signal')
    if macd_value is not None and macd_signal_value is not None:
        if macd_value > macd_signal_value:
            buy_signals += 1 # Bullish crossover
            reasons.append("MACD above MACD Signal (Bullish MACD Crossover)")
        elif macd_value < macd_signal_value:
            sell_signals += 1 # Bearish crossover
            reasons.append("MACD below MACD Signal (Bearish MACD Crossover)")
        else:
            hold_signals += 1
            reasons.append("MACD and MACD Signal are close (Neutral MACD)")

    # News Sentiment
    if news_sentiment is not None:
        if news_sentiment > POSITIVE_SENTIMENT_THRESHOLD: # > 0.1 (new threshold)
            buy_signals += 1
            reasons.append(f"Positive news sentiment ({news_sentiment:.2f})")
        elif news_sentiment < NEGATIVE_SENTIMENT_THRESHOLD: # < 0.0 (new threshold)
            sell_signals += 1
            reasons.append(f"Negative news sentiment ({news_sentiment:.2f})")
        else: # Neutral
            hold_signals += 1
            reasons.append(f"Neutral news sentiment ({news_sentiment:.2f})")

    total_signals = buy_signals + sell_signals + hold_signals 
    if total_signals == 0: return "Hold", 0, "No conclusive signals from available data." 
    if buy_signals > sell_signals: final_confidence = int((buy_signals / total_signals) * 100)
    elif sell_signals > buy_signals: final_confidence = int((sell_signals / total_signals) * 100)
    else: final_confidence = 50 # Neutral confidence
    final_confidence = max(0, min(final_confidence, 100)) # Ensure it's within 0-100
    if buy_signals > sell_signals and buy_signals >= hold_signals: return "Buy", final_confidence, "Primary signals suggest Buy: " + "; ".join(reasons)
    elif sell_signals > buy_signals and sell_signals >= hold_signals: return "Sell", final_confidence, "Primary signals suggest Sell: " + "; ".join(reasons)
    else: return "Hold", final_confidence, "Mixed or neutral signals: " + "; ".join(reasons)

# --- CATEGORIZED SCORING ENGINE ---

def _calculate_technical_score(technical_indicators: dict, historical_data: pd.DataFrame) -> tuple[float, dict]:
    """Calculates a normalized technical score (0-100) and provides a breakdown."""
    score = 50.0
    breakdown = {}
    
    rsi_val = technical_indicators.get('RSI')
    if rsi_val is not None:
        if rsi_val < RSI_OVERSOLD_THRESHOLD:
            points, details = 15, f"Oversold at {rsi_val:.2f}"
        elif rsi_val > RSI_OVERBOUGHT_THRESHOLD:
            points, details = -15, f"Overbought at {rsi_val:.2f}"
        else:
            points, details = 0, f"Neutral at {rsi_val:.2f}"
        score += points
        breakdown["RSI"] = {"points": points, "details": details}
    
    macd_val = technical_indicators.get('MACD')
    macd_signal_val = technical_indicators.get('MACD_Signal')
    if macd_val is not None and macd_signal_val is not None:
        if macd_val > macd_signal_val:
            points, details = 15, "Bullish"
        elif macd_val < macd_signal_val:
            points, details = -15, "Bearish"
        else:
            points, details = 0, "Neutral"
        score += points
        breakdown["MACD Crossover"] = {"points": points, "details": details}
    
    sma_5_val = technical_indicators.get('SMA_5')
    sma_20_val = technical_indicators.get('SMA_20')
    if sma_5_val is not None and sma_20_val is not None:
        if sma_5_val > sma_20_val:
            points, details = 10, "Bullish"
        elif sma_5_val < sma_20_val:
            points, details = -10, "Bearish"
        else:
            points, details = 0, "Neutral"
        score += points
        breakdown["Short-term SMA Cross"] = {"points": points, "details": details}
    
    sma_50_val = technical_indicators.get('SMA_50')
    sma_200_val = technical_indicators.get('SMA_200')
    if sma_50_val is not None and sma_200_val is not None:
        if sma_50_val > sma_200_val:
            points, details = 10, "Golden Cross"
        else:
            points, details = -15, "Death Cross"
        score += points
        breakdown["Long-term SMA Cross"] = {"points": points, "details": details}
    
    # Volume Activity
    volume_sma_5 = technical_indicators.get('Volume_SMA_5')
    current_volume = historical_data['Volume'].iloc[-1] if not historical_data.empty else None
    
    if volume_sma_5 is not None and current_volume is not None:
        if current_volume > volume_sma_5 * VOLUME_HIGH_THRESHOLD_MULTIPLIER:
            points = 5
            details = f"Current volume {current_volume:,.0f} significantly above 5-day SMA {volume_sma_5:,.0f}"
        else:
            points = 0
            details = f"Current volume {current_volume:,.0f} near 5-day SMA {volume_sma_5:,.0f}"
        score += points
        breakdown["Volume Activity"] = {"points": points, "details": details}
    else:
        breakdown["Volume Activity"] = {"points": 0, "details": "N/A"}
    return max(0, min(100, score)), breakdown

def _calculate_fundamental_score(company_fundamentals: dict) -> tuple[float, dict]:
    """Calculates a normalized fundamental score (0-100) and provides a breakdown."""
    score = 50.0
    breakdown = {}
    
    pe_ratio = company_fundamentals.get('trailingPE') if company_fundamentals else None
    eps_growth = company_fundamentals.get('earningsGrowth') if company_fundamentals else None
    
    if pe_ratio is not None and not np.isinf(pe_ratio):
        if pe_ratio < PE_RATIO_UNDERVALUED_THRESHOLD:
            points, details = 15, f"Undervalued at {pe_ratio:.2f}"
        elif pe_ratio > PE_RATIO_OVERVALUED_THRESHOLD:
            points, details = -10, f"Overvalued at {pe_ratio:.2f}"
        else:
            points, details = 0, f"Neutral at {pe_ratio:.2f}"
        score += points
        breakdown["P/E Ratio"] = {"points": points, "details": details}
    
    if eps_growth is not None:
        if eps_growth > EPS_GROWTH_STRONG_THRESHOLD:
            points, details = 25, f"Strong at {eps_growth:.2%}"
        elif eps_growth < -0.30:
            points, details = -30, f"Decline at {eps_growth:.2%}"
        elif eps_growth < 0:
            points, details = -15, f"Negative at {eps_growth:.2%}"
        else:
            points, details = 0, f"Neutral at {eps_growth:.2%}"
        score += points
        breakdown["EPS Growth"] = {"points": points, "details": details}
    
    # Return on Equity (ROE)
    return_on_equity = company_fundamentals.get('returnOnEquity')
    if return_on_equity is not None:
        if return_on_equity > ROE_GOOD_THRESHOLD: # e.g., > 15%
            points, details = 15, f"Strong at {return_on_equity:.2%}"
        elif return_on_equity < 0: # Negative ROE
            points, details = -15, f"Negative at {return_on_equity:.2%}"
        else:
            points, details = 0, f"Neutral at {return_on_equity:.2%}"
        score += points
        breakdown["Return on Equity (ROE)"] = {"points": points, "details": details}
    else:
        breakdown["Return on Equity (ROE)"] = {"points": 0, "details": "N/A"}
    
    # Debt to Equity
    debt_to_equity = company_fundamentals.get('debtToEquity')
    if debt_to_equity is not None and not np.isinf(debt_to_equity):
        if debt_to_equity < DEBT_TO_EQUITY_LOW_THRESHOLD: # e.g., < 0.5
            points, details = 10, f"Low Debt at {debt_to_equity:.2f}"
        elif debt_to_equity > DEBT_TO_EQUITY_HIGH_THRESHOLD: # e.g., > 1.5
            points, details = -10, f"High Debt at {debt_to_equity:.2f}"
        else:
            points, details = 0, f"Moderate Debt at {debt_to_equity:.2f}"
        score += points
        breakdown["Debt to Equity"] = {"points": points, "details": details}
    else:
        breakdown["Debt to Equity"] = {"points": 0, "details": "N/A"}
    
    return max(0, min(100, score)), breakdown

def _calculate_sentiment_score(news_sentiment: float | None) -> tuple[float, dict]:
    """Calculates a normalized sentiment score (0-100) and provides a breakdown."""
    score = 50.0
    breakdown = {}
    
    if news_sentiment is not None: # Sentiment rule: <0 = Negative, 0-0.1 = Neutral, >0.1 = Positive
        if news_sentiment > POSITIVE_SENTIMENT_THRESHOLD: # > 0.1
            points = 20 # Positive impact
            sentiment_label = "Positive"
        elif news_sentiment < NEGATIVE_SENTIMENT_THRESHOLD: # < 0.0
            points = -20 # Negative impact
            sentiment_label = "Negative"
        else: # 0.0 to 0.1 (inclusive of 0.0)
            points = 0
            sentiment_label = "Neutral"
        score += points
        breakdown["Overall News Sentiment"] = {"points": points, "details": f"{news_sentiment:.2f} ({sentiment_label})"}
    else:
        breakdown["Overall News Sentiment"] = {"points": 0, "details": "N/A"}
    
    return max(0, min(100, score)), breakdown

# --- DYNAMIC ALERTS GENERATION ---
def generate_dynamic_alerts(rsi: float | None, macd_hist: float | None, sma_5: float | None, sma_10: float | None, current_volume: float | None, volume_sma_5: float | None, sma_50: float | None,
                            sma_200: float | None, current_price: float | None,
                            news_sentiment: float | None, ath_price: float | None) -> list[str]:
    """
    Generates dynamic alerts based on various technical and sentiment conditions.
    """
    alerts = []

    # RSI Alerts
    if rsi is not None:
        if rsi < RSI_OVERSOLD_THRESHOLD: # RSI < 30
            alerts.append("🔻 Oversold – possible rebound soon.")
        elif rsi > RSI_OVERBOUGHT_THRESHOLD: # RSI > 70
            alerts.append("⚠️ Overbought – price may drop soon.")
        else: # RSI is neutral
            alerts.append(f"ℹ️ RSI Neutral ({rsi:.2f}) – no strong momentum signal.")
    
    # Short-term SMA Cross Alerts (SMA_5 / SMA_10)
    if sma_5 is not None and sma_10 is not None:
        if sma_5 > sma_10:
            alerts.append("📈 5-day SMA above 10-day SMA – bullish short-term trend.")
        elif sma_5 < sma_10:
            alerts.append("📉 5-day SMA below 10-day SMA – bearish short-term trend.")

    # Volume Alerts
    if current_volume is not None and volume_sma_5 is not None:
        if current_volume > volume_sma_5 * VOLUME_HIGH_THRESHOLD_MULTIPLIER:
            alerts.append("📈 High volume spike – confirming trend.")
        elif current_volume < volume_sma_5 * 0.5: # Very low volume
            alerts.append("📉 Low volume – lack of interest/momentum.")
    
    # MACD Alerts (using histogram for crossover detection)
    if macd_hist is not None:
        if macd_hist < MACD_HIST_BEARISH_THRESHOLD: # MACD line crosses below signal
            alerts.append("🧨 Bearish MACD crossover.")
        elif macd_hist > MACD_HIST_BULLISH_THRESHOLD: # MACD line crosses above signal
            alerts.append("🚀 Bullish MACD crossover.")

    # SMA Alerts (Golden/Death Cross)
    # Price Action Alerts (Near ATH)
    if current_price is not None and ath_price is not None and ath_price > 0:
        if current_price >= PRICE_NEAR_ATH_THRESHOLD * ath_price:
            alerts.append("📈 Near 1-year high – check valuation.")

    # News Sentiment Alerts
    if news_sentiment is not None:
        if news_sentiment < NEWS_SENTIMENT_STRONG_NEGATIVE_THRESHOLD: # sentiment < -0.4
            alerts.append("📰 Negative news trend – caution advised.")
        elif news_sentiment > NEWS_SENTIMENT_STRONG_POSITIVE_THRESHOLD: # sentiment > 0.4
            alerts.append("📢 Positive sentiment momentum – market optimism.")

    return alerts


# --- SWING TRADER RECOMMENDATION SYSTEM ---
def evaluate_swing(data: dict) -> dict:
    """
    Evaluates stock for swing trading based on technicals, sentiment, and ATH.
    """
    # Extract data, providing defaults for safety
    sma_5 = data.get("SMA_5")
    sma_10 = data.get("SMA_10") # Added for swing
    rsi = data.get("RSI")
    macd = data.get("MACD")
    macd_signal = data.get("MACD_Signal")
    macd_hist = data.get("MACD_Hist") # For histogram slope
    volume_sma_5 = data.get("Volume_SMA_5")
    current_volume = data.get("current_volume")
    sentiment = data.get("sentiment", 0.0) # News sentiment
    current_price = data.get("current_price")
    ath = data.get("ATH")
    all_alerts = data.get("all_alerts", [])

    recommendation = "Hold"
    swing_specific_alerts = []
    
    # Check if critical data is missing for a meaningful analysis
    if any(x is None for x in [sma_5, sma_10, rsi, macd, macd_signal, macd_hist, volume_sma_5, current_volume, current_price, ath]):
        return {
            "recommendation": "Hold",
            "confidence": 0, 
            "alerts": ["⚠️ Insufficient data for swing analysis"]
        }

    # --- Calculate individual factor scores (-1 to +1) ---
    scores = {}

    # MACD Score (30%)
    macd_score = 0
    if macd is not None and macd_signal is not None:
        if macd > macd_signal and macd_hist > 0: # Bullish crossover with positive histogram
            macd_score = 1
        elif macd < macd_signal and macd_hist < 0: # Bearish crossover with negative histogram
            macd_score = -1
        elif macd > macd_signal: # Bullish crossover, but histogram might be flattening
            macd_score = 0.5
        elif macd < macd_signal: # Bearish crossover, but histogram might be flattening
            macd_score = -0.5
    scores["MACD"] = macd_score

    # RSI Score (20%)
    rsi_score = 0
    if rsi is not None:
        if rsi < RSI_OVERSOLD_THRESHOLD:
            rsi_score = 1 # Oversold, potential rebound
        elif rsi > RSI_OVERBOUGHT_THRESHOLD:
            rsi_score = -1 # Overbought, potential drop
        elif rsi >= 50:
            rsi_score = 0.5 # Bullish momentum
        elif rsi < 50:
            rsi_score = -0.5 # Bearish momentum
    scores["RSI"] = rsi_score

    # SMA_10 Score (20%) - Using SMA_5 / SMA_10 cross
    sma_10_score = 0
    if sma_5 is not None and sma_10 is not None:
        if sma_5 > sma_10:
            sma_10_score = 1 # Bullish short-term cross
        elif sma_5 < sma_10:
            sma_10_score = -1 # Bearish short-term cross
    scores["SMA_10"] = sma_10_score

    # Volume Score (10%) - Volume Spike
    volume_score = 0
    if current_volume is not None and volume_sma_5 is not None:
        if current_volume > volume_sma_5 * VOLUME_HIGH_THRESHOLD_MULTIPLIER:
            volume_score = 1 # High volume confirms trend
        elif current_volume < volume_sma_5 * 0.5: # Very low volume
            volume_score = -0.5 # Lack of interest/momentum
    scores["Volume"] = volume_score

    # News Sentiment Score (20%)
    sentiment_score = 0
    if sentiment is not None:
        if sentiment > POSITIVE_SENTIMENT_THRESHOLD:
            sentiment_score = 1
        elif sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            sentiment_score = -1
    scores["News"] = sentiment_score

    # --- Calculate weighted sum and final confidence ---
    weighted_sum = (
        scores["MACD"] * 0.30 +
        scores["RSI"] * 0.20 +
        scores["SMA_10"] * 0.20 +
        scores["Volume"] * 0.10 +
        scores["News"] * 0.20
    )
    
    # Scale to 0-100: sum(weighted_scores) * 50 + 50
    confidence = int(weighted_sum * 50 + 50)
    confidence = max(0, min(100, confidence)) # Ensure 0-100 range

    # --- Determine Recommendation ---
    if confidence >= 70:
        recommendation = "Buy"
    elif confidence <= 30:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    # --- Filter relevant alerts ---
    for alert in all_alerts: # Filter alerts for swing trading relevance
        if any(keyword in alert for keyword in ["RSI", "MACD", "Oversold", "Overbought", "Bullish", "Bearish", "Positive sentiment", "Negative news trend", "SMA", "volume"]):
            swing_specific_alerts.append(alert)
    
    if not swing_specific_alerts and recommendation == "Hold":
        swing_specific_alerts.append("ℹ️ No strong short-term signals for swing trading.")
    elif not swing_specific_alerts and recommendation == "Buy":
        swing_specific_alerts.append("🔔 Strong swing buy signals detected.")
    elif not swing_specific_alerts and recommendation == "Sell":
        swing_specific_alerts.append("⚠️ Strong swing sell signals detected.")

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "alerts": swing_specific_alerts
    }

def evaluate_stock(
    historical_data: pd.DataFrame,
    technical_indicators: dict,
    company_fundamentals: dict,
    overall_news_sentiment: float | None,
    current_price: float | None,
    all_time_high: float | None # This is the ATH for the selected period
) -> dict:
    """
    Orchestrates the swing trader recommendation system.
    """
    # Extract data for dynamic alerts
    rsi = technical_indicators.get('RSI')
    macd_hist = technical_indicators.get('MACD_Hist')
    sma_5 = technical_indicators.get('SMA_5')
    sma_10 = technical_indicators.get('SMA_10')
    sma_50 = technical_indicators.get('SMA_50') # Kept for general alerts, though not used in swing confidence
    sma_200 = technical_indicators.get('SMA_200') # Kept for general alerts, though not used in swing confidence
    current_volume = historical_data['Volume'].iloc[-1] if not historical_data.empty else None
    volume_sma_5 = technical_indicators.get('Volume_SMA_5')

    # Generate all dynamic alerts once
    all_dynamic_alerts = generate_dynamic_alerts(rsi, macd_hist, sma_5, sma_10, current_volume, volume_sma_5, sma_50, sma_200,
                                                 current_price, overall_news_sentiment, all_time_high)

    data_for_eval = {
        "SMA_5": technical_indicators.get('SMA_5'),
        "SMA_10": technical_indicators.get('SMA_10'), # Added for swing
        "SMA_20": technical_indicators.get('SMA_20'),
        "SMA_50": technical_indicators.get('SMA_50'),
        "SMA_200": technical_indicators.get('SMA_200'),
        "RSI": technical_indicators.get('RSI'),
        "MACD": technical_indicators.get('MACD'),
        "MACD_Signal": technical_indicators.get('MACD_Signal'),
        "MACD_Hist": technical_indicators.get('MACD_Hist'), # Added for swing
        "Volume_SMA_5": technical_indicators.get('Volume_SMA_5'),
        "current_volume": current_volume, # Pass current volume
        "PE_ratio": company_fundamentals.get('trailingPE'), # Kept for enhanced_analysis
        "EPS_growth": company_fundamentals.get('earningsGrowth'), # Kept for enhanced_analysis
        "ROE": company_fundamentals.get('returnOnEquity'), # Kept for enhanced_analysis
        "D_E_ratio": company_fundamentals.get('debtToEquity'), # Kept for enhanced_analysis
        "sentiment": overall_news_sentiment,
        "current_price": current_price,
        "ATH": all_time_high,
        "all_alerts": all_dynamic_alerts # Pass all alerts to sub-functions
    }
    
    swing_result = evaluate_swing(data_for_eval)

    return {
        "swing_trader": swing_result
    }

# --- ADVANCED ANALYSIS ---
def enhanced_analysis(historical_data: pd.DataFrame, technical_indicators: dict, company_fundamentals: dict, news_sentiment: float | None) -> tuple:
    """
    Performs an enhanced, categorized, and simplified stock analysis.
    This function now focuses on providing category scores and breakdowns,
    as the main recommendations are handled by `evaluate_stock`.

    Returns:
        tuple: (Master_Breakdown, Category_Scores, Final_Score_Value)
    """
    # 1. Calculate scores for each category
    tech_score, tech_breakdown = _calculate_technical_score(technical_indicators, historical_data)
    fund_score, fund_breakdown = _calculate_fundamental_score(company_fundamentals)
    sent_score, sent_breakdown = _calculate_sentiment_score(news_sentiment) # Alerts are now generated by generate_dynamic_alerts

    category_scores = {
        "Technical": tech_score,
        "Fundamental": fund_score,
        "Sentiment": sent_score,
    }

    # 2. Calculate final weighted score (using a simple average, no sector-specific weights)
    final_score_value = (tech_score + fund_score + sent_score) / 3

    # 3. Combine all breakdowns for a full report
    master_breakdown = {
        "Technical Analysis": tech_breakdown,
        "Fundamental Analysis": fund_breakdown,
        "Sentiment Analysis": sent_breakdown,
    }

    # Note: Recommendation, Confidence_Level, and Alerts are now handled by evaluate_stock
    # and its sub-functions (evaluate_swing, evaluate_long_term)
    return master_breakdown, category_scores, final_score_value

# --- UTILITY FUNCTIONS ---
def analyze_sentiment(text: str) -> float:
    """
    Analyzes the sentiment of a given text using TextBlob.

    Args:
        text (str): The input text.

    Returns:
        float: Sentiment polarity score (-1.0 to 1.0).
    """
    if not isinstance(text, str) or not text.strip(): # Check for empty or whitespace-only strings
        return 0.0 # Return neutral sentiment for non-string or empty input
    return TextBlob(text).sentiment.polarity

def get_source_weight(source_name: str) -> float:
    """Returns the weight for a given news source, defaulting to a lower weight if unknown."""
    # Normalize source name for better matching (e.g., remove "The", "Inc.", convert to lower)
    normalized_name = source_name.lower().replace("the ", "").replace("inc.", "").strip()
    
    # Simple mapping for common variations
    if "bloomberg" in normalized_name: return SOURCE_WEIGHTS["Bloomberg"]
    if "reuters" in normalized_name: return SOURCE_WEIGHTS["Reuters"]
    if "cnbc" in normalized_name: return SOURCE_WEIGHTS["CNBC"]
    if "wall street journal" in normalized_name: return SOURCE_WEIGHTS["The Wall Street Journal"]
    if "associated press" in normalized_name: return SOURCE_WEIGHTS["Associated Press"]
    if "yahoo finance" in normalized_name: return SOURCE_WEIGHTS["Yahoo Finance"]
    if "marketwatch" in normalized_name: return SOURCE_WEIGHTS["MarketWatch"]
    if "seeking alpha" in normalized_name: return SOURCE_WEIGHTS["Seeking Alpha"]
    if "reddit" in normalized_name: return SOURCE_WEIGHTS["Reddit"]

    # Fallback to general weights if specific match not found
    if "newsapi" in normalized_name: return SOURCE_WEIGHTS["NewsAPI"]
    if "google news" in normalized_name: return SOURCE_WEIGHTS["Google News"] # For GNews/RSS
    return SOURCE_WEIGHTS.get(source_name, SOURCE_WEIGHTS["Unknown Blog"])

def fetch_news_sentiment_from_newsapi(ticker_symbol: str, api_key: str | None, company_name: str | None) -> tuple[list[tuple[float, float]], list[str]]:
    """
    Fetches recent news articles for a given ticker symbol from NewsAPI
    filters them, and returns a list of (sentiment, weight) tuples and titles.
    """
    if not api_key:
        logger.warning("NewsAPI key not provided. Skipping NewsAPI fetch.")
        return [], []
    
    if not NewsApiClient: # Check if the library was successfully imported
        # A warning is already printed at import time.
        return [], []

    newsapi_client = NewsApiClient(api_key=api_key)
    weighted_sentiments = [] # List of (sentiment, weight) tuples
    news_titles = []
    try:
        # Fetch news from the last 7 days (free tier usually limits to 30 days history)
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        articles_response = newsapi_client.get_everything(q=ticker_symbol,
                                                          language='en',
                                                          sort_by='relevancy',
                                                          from_param=from_date,
                                                          to=to_date, 
                                                          page_size=20)

        if articles_response and articles_response['articles']:
            for article in articles_response['articles']:
                title = article.get('title', '')
                description = article.get('description', '') or ""
                source_name = article.get('source', {}).get('name', 'NewsAPI') # Default to NewsAPI if source name missing
                
                article_text = title + " " + description
                
                if is_stock_mentioned(article_text, ticker_symbol, company_name):
                    sentiment = analyze_sentiment(article_text)
                    weight = get_source_weight(source_name)
                    weighted_sentiments.append((sentiment, weight))
                    news_titles.append(title)
            
            if weighted_sentiments:
                logger.info(f"Fetched {len(news_titles)} relevant articles from NewsAPI for {ticker_symbol}.")
                return weighted_sentiments, news_titles # Return list of (sentiment, weight)
        logger.info(f"No relevant news found for {ticker_symbol} from NewsAPI.")
        return [], []
    except NewsAPIException as e: # type: ignore [misc] # misc because NewsAPIException could be the mock
        error_details = str(e) # Standard way to get exception message.
        # Check for common error substrings if specific codes/methods aren't available
        # on the exception object (especially if it's the mock or an older library version).
        # Also, try to use get_code() if available for more specific handling.
        is_rate_limited = 'rateLimited' in error_details.lower() or ('get_code' in dir(e) and callable(getattr(e, 'get_code')) and e.get_code() == 'rateLimited')
        is_api_key_invalid = 'apiKeyInvalid' in error_details.lower() or \
                             'unauthorized' in error_details.lower() or \
                             ('get_code' in dir(e) and callable(getattr(e, 'get_code')) and e.get_code() == 'apiKeyInvalid')
        
        if is_rate_limited:
            logger.warning(f"NewsAPI Rate Limit Exceeded: {error_details}. Please wait before trying again.")
        elif is_api_key_invalid:
            logger.error(f"NewsAPI Key Invalid/Unauthorized: {error_details}. Please check your NEWS_API_KEY environment variable.")
        else:
            code_info = f"(Code: {e.get_code()})" if ('get_code' in dir(e) and callable(getattr(e, 'get_code'))) else ""
            logger.error(f"Error fetching news from NewsAPI for {ticker_symbol} {code_info}: {error_details}")
        return [], []
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching NewsAPI data for {ticker_symbol}: {e}")
        return [], []

def is_stock_mentioned(news_article_text: str, ticker: str, company_name: str | None) -> bool:
    """
    Checks if the news article text explicitly mentions the stock ticker or company name.
    """
    if not news_article_text:
        return False
    
    keywords = [ticker.upper()]
    if company_name:
        # Add full name and first word of the company name (e.g., "Alphabet" from "Alphabet Inc.")
        keywords.extend([company_name, company_name.split(' ')[0]])
        # Add common variations for well-known companies
        if "Alphabet" in company_name:
            keywords.append("Google")
            keywords.append("$GOOG")
        elif "Apple" in company_name:
            keywords.append("$AAPL")
        elif "Reliance" in company_name: # For RELIANCE.NS
            keywords.append("Reliance Industries")
            keywords.append("RIL")

    # Create a regex pattern for whole word matching to avoid false positives (e.g., "APPL" in "application")
    # Escape special characters in ticker for regex
    ticker_escaped = re.escape(ticker)
    patterns = [r'\b' + re.escape(k) + r'\b' for k in keywords if k] # Whole word match
    patterns.append(r'\$' + ticker_escaped) # Match $TICKER

    combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
    
    return bool(combined_pattern.search(news_article_text))

def fetch_news_sentiment_from_rss(rss_url: str, ticker_symbol: str, company_name: str | None) -> tuple[list[tuple[float, float]], list[str]]:
    """
    Fetches news from an RSS feed, filters by ticker symbol,
    and returns a list of (sentiment, weight) tuples and titles.
    """
    if feedparser is None:
        return [], []

    weighted_sentiments = []
    news_titles = []
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            return [], []

        for entry in feed.entries:
            title = entry.get('title', '')
            summary = entry.get('summary', '') # Use .get() for safety
            source_name = entry.get('source', {}).get('title', feed.get('feed', {}).get('title', 'Google News')) # Fallback to feed title
            article_text = title + " " + summary

            if is_stock_mentioned(article_text, ticker_symbol, company_name):
                sentiment = analyze_sentiment(article_text)
                weight = get_source_weight(source_name)
                weighted_sentiments.append((sentiment, weight))
                news_titles.append(title)

        if weighted_sentiments:
            logger.info(f"Fetched {len(news_titles)} relevant articles from RSS for {ticker_symbol}.")
            return weighted_sentiments, news_titles # Return list of (sentiment, weight)
        logger.info(f"No relevant news found for {ticker_symbol} in RSS feed.")
        return [], []
    except Exception as e:
        logger.error(f"Error fetching RSS news from {rss_url}: {e}")
        return [], []

def fetch_news_sentiment_from_gnews(ticker_symbol: str, api_key: str | None, company_name: str | None) -> tuple[list[tuple[float, float]], list[str]]: # type: ignore
    """
    Fetches news for a given ticker symbol from GNews and calculates sentiment.
        ticker_symbol (str): The stock ticker symbol (often for .NS market).
        api_key (str | None): The GNews API key (though the gnews library might not strictly require it).
    # The gnews library (v0.4.1) doesn't strictly require an API key for basic usage,
    # but we keep the key parameter for consistency or future library versions.
    Returns:
        tuple: (Average sentiment: float | None, List of (sentiment, weight) tuples: list, List of news titles: list)
    """
    if not gnews_client:
        logger.warning("GNews client not initialized. Cannot fetch news from GNews.")
        return [], [] # Return empty list of (sentiment, weight)
    
    query = f"{ticker_symbol} stock news"
    weighted_sentiments = []
    news_titles = []
    try:
        # Assuming gnews_client.get_news(query) returns a list of article-like objects
        # Each object is expected to have 'title' and 'description' attributes or similar.
        # Adjust period and max_results as needed and if supported by the library.
        # Some gnews libraries might require setting period, country, etc. during client initialization
        # or on the get_news method.
        news_items = gnews_client.get_news(query)

        if news_items:
            for item in news_items:
                title = item.get('title', '')
                description = item.get('description', '') or item.get('text', '') or ""
                source_name = item.get('publisher', {}).get('title', 'Google News') # Default to Google News
                article_text = title + " " + description
                if is_stock_mentioned(article_text, ticker_symbol, company_name):
                    sentiment = analyze_sentiment(article_text)
                    weight = get_source_weight(source_name)
                    weighted_sentiments.append((sentiment, weight))
                    news_titles.append(title)
            if weighted_sentiments:
                logger.info(f"Fetched {len(news_titles)} relevant articles from GNews for {ticker_symbol}.")
                return weighted_sentiments, news_titles # Return list of (sentiment, weight)
        logger.info(f"No relevant news found for {ticker_symbol} from GNews.")
        return [], []
    except Exception as e:
        logger.error(f"Error fetching or processing GNews for {ticker_symbol}: {e}")
        return [], []

def get_stock_data(ticker_symbol: str, period: str = "max") -> tuple[pd.DataFrame | None, float | None, dict | None, str | None]:
    """
    Fetches historical stock data and basic company fundamentals using yfinance.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        period (str): The period for which to fetch historical data (e.g., "1y", "2y", "max").

    Returns:
        tuple: (historical_data: pd.DataFrame, current_price: float,
                company_fundamentals: dict, error_message: str)
               Returns (None, None, None, error_message) on failure.
    """
    if yf is None:
        return None, None, None, "Yfinance library not available. Cannot fetch stock data." # Return error message

    try:
        stock = yf.Ticker(ticker_symbol)
        # Attempt to fetch company info first to validate ticker and get fundamentals
        company_fundamentals = stock.info

        # A common sign of an invalid/delisted ticker is an empty info dict or missing key financial fields
        if not company_fundamentals or \
           (company_fundamentals.get('regularMarketPrice') is None and \
            company_fundamentals.get('longName') is None and \
            company_fundamentals.get('marketCap') is None): # Check for essential fields
            return None, None, None, f"No valid data or fundamentals found for '{ticker_symbol}'. It might be an invalid ticker, delisted, or data is unavailable."

        # Fetch historical data for the specified period
        historical_data = stock.history(period=period)

        if historical_data.empty:
            # We might have fundamentals, but no historical data for the specified period
            current_price_from_info = company_fundamentals.get('regularMarketPrice') or company_fundamentals.get('currentPrice')
            # It's unusual to have fundamentals but no historical data for a valid, active ticker
            # but we return what we have along with a message if historical data is empty.
            return None, current_price_from_info, company_fundamentals, f"No historical data found for '{ticker_symbol}' for the period '{period}'. Some fundamental data might be available."

        # Ensure 'Close' column exists and has data before accessing iloc[-1]
        if 'Close' in historical_data.columns and not historical_data.empty:
            current_price_from_history = historical_data['Close'].iloc[-1]
            # Calculate All-Time High from the fetched historical data period
            # This is the high for the selected period, not the true all-time high unless period='max'
            ath_from_period = historical_data['High'].max() # Use 'High' column for ATH
            # Add ATH to company_fundamentals for easy access in other functions
            company_fundamentals['ath_from_period'] = ath_from_period
            company_fundamentals['period_used_for_ath'] = period # Store the period for context

        else:
            # Fallback if 'Close' is missing or empty, though unlikely if historical_data itself is not empty
            current_price_from_history = company_fundamentals.get('regularMarketPrice') or company_fundamentals.get('currentPrice')
            if current_price_from_history is None: # If no price found at all
                 return historical_data, None, company_fundamentals, f"Historical data fetched but 'Close' price is missing for {ticker_symbol}."

        return historical_data, current_price_from_history, company_fundamentals, None # Success
    except Exception as e:
        return None, None, None, f"Error fetching data for {ticker_symbol}: {type(e).__name__} - {e}"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    while True:
        ticker_input = input("Please enter the stock ticker symbol to analyze (e.g., AAPL, GOOG): ")
        if ticker_input.strip():
            TICKER = ticker_input.strip().upper()
            break
        else:
            logger.warning("No ticker symbol entered. Please try again.")

    logger.info(f"Starting comprehensive stock analysis script for {TICKER}...")

    # 1. Get Stock Data
    historical_data, current_price, company_fundamentals, error = get_stock_data(TICKER, period="max") # Use "max" for ATH

    if error:
        logger.error(f"Error: {error}")
    elif historical_data is None or historical_data.empty:
        logger.warning(f"Could not retrieve sufficient data for {TICKER}. Exiting.")
    else:
        # Get company long name for news filtering
        company_long_name = company_fundamentals.get('longName')

        print(f"\n--- Data for {TICKER} ---")
        print(f"Current Price: ${current_price:.2f}")

        # 2. Calculate Technical Indicators
        technical_indicators = calculate_technical_indicators(historical_data)
        logger.info("\n--- Technical Indicators ---")
        # Display only relevant indicators for brevity in console output
        display_indicators = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'Volume_SMA_5', 'MACD_Hist']
        for k in display_indicators:
            v = technical_indicators.get(k)
            if v is not None:
                if isinstance(v, (int, float)):
                    print(f"{k}: {v:.2f}")
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: N/A")
        
        # Add ATH to fundamentals for consistency with app
        if 'ath_from_period' not in company_fundamentals:
            company_fundamentals['ath_from_period'] = historical_data['High'].max()
            company_fundamentals['period_used_for_ath'] = "max"


        # 3. Fetch News Sentiment (from NewsAPI and RSS)
        # Retrieve API keys from environment for direct script execution
        env_news_api_key = os.environ.get("NEWS_API_KEY")
        env_gnews_api_key = os.environ.get("GNEWS_API_KEY")

        # Each fetch function now returns a list of (sentiment, weight) tuples and titles
        newsapi_weighted_sentiments, newsapi_titles = fetch_news_sentiment_from_newsapi(TICKER, env_news_api_key, company_long_name)
        ticker_specific_rss_url = BASE_GOOGLE_NEWS_RSS_URL.format(ticker=TICKER)
        rss_weighted_sentiments, rss_titles = fetch_news_sentiment_from_rss(ticker_specific_rss_url, TICKER, company_long_name)
        gnews_weighted_sentiments, gnews_titles = fetch_news_sentiment_from_gnews(TICKER, env_gnews_api_key, company_long_name)

        all_weighted_sentiments = []
        all_news_titles = []

        all_weighted_sentiments.extend(newsapi_weighted_sentiments)
        all_news_titles.extend(newsapi_titles)
        all_weighted_sentiments.extend(rss_weighted_sentiments)
        all_news_titles.extend(rss_titles)
        all_weighted_sentiments.extend(gnews_weighted_sentiments)
        all_news_titles.extend(gnews_titles)

        overall_news_sentiment = None
        if all_weighted_sentiments:
            # Calculate weighted average
            total_sentiment_score = sum(s * w for s, w in all_weighted_sentiments)
            total_weight = sum(w for s, w in all_weighted_sentiments)
            if total_weight > 0:
                overall_news_sentiment = total_sentiment_score / total_weight
            else:
                overall_news_sentiment = None # Avoid division by zero if all weights are zero

        logger.info("\n--- News Sentiment ---")
        if overall_news_sentiment is not None:
            logger.info(f"Overall News Sentiment: {overall_news_sentiment:.2f}")
            logger.info("Recent News Titles (sample):")
            # Convert to set to get unique titles, then back to list for slicing
            for i, title in enumerate(list(set(all_news_titles))[:10]):
                logger.info(f"  - {title}")
        else:
            logger.warning("Could not fetch news sentiment from any source.")

        # 4. Basic Analysis (using overall news sentiment)
        basic_recommendation, basic_confidence, basic_reason = basic_analysis(historical_data, overall_news_sentiment)
        logger.info(f"\n--- Basic Analysis for {TICKER} ---")
        logger.info(f"Recommendation: {basic_recommendation} (Confidence: {basic_confidence}%)")
        logger.info(f"Reason: {basic_reason}")

        # 5. Enhanced Analysis (Category Scores & Breakdown)
        master_breakdown, category_scores, final_score_value = enhanced_analysis(
            historical_data,
            technical_indicators,
            company_fundamentals,
            overall_news_sentiment
        )

        logger.info(f"\n--- Enhanced Analysis Breakdown for {TICKER} ---")
        logger.info("Category Scores:")
        for category, score in category_scores.items():
            logger.info(f"  - {category}: {score:.2f}/100")

        logger.info("Reason Breakdown:")
        for category, details in master_breakdown.items():
            logger.info(f"  - {category}:")
            for reason, value in details.items():
                logger.info(f"    - {reason}: {value}")
        
        # Swing Trader Recommendation System
        logger.info(f"\n--- Swing Trader Recommendation System for {TICKER} ---")
        # ATH is now part of company_fundamentals
        all_time_high_for_period = company_fundamentals.get('ath_from_period')
        swing_analysis_results = evaluate_stock(
            historical_data, technical_indicators, company_fundamentals, overall_news_sentiment, current_price, all_time_high_for_period
        )
        logger.info("\nSwing Trader Recommendation:")
        logger.info(f"  Recommendation: {swing_analysis_results['swing_trader']['recommendation']}")
        logger.info(f"  Confidence: {swing_analysis_results['swing_trader']['confidence']}%")
        logger.info("  Alerts:")
        for alert in swing_analysis_results['swing_trader']['alerts']:
            logger.info(f"    - {alert}")


    logger.info("\nScript finished.")
