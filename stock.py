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
import random # For sampling headlines

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

# Long-Term Investor Thresholds (These are no longer used in the main app logic but kept for reference)
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

# New: News Keywords for granular sentiment analysis
BULLISH_KEYWORDS_SCORES = {
    "beats expectations": 0.3,
    "record revenue": 0.4,
    "buyback": 0.3,
    "dividend increase": 0.25,
    "raises forecast": 0.2,
    "acquires": 0.25,
    "partners with": 0.2,
    "positive outlook": 0.2,
    "new product": 0.2,
    "launch": 0.2,
    "strong earnings": 0.3,
    "growth": 0.15,
    "expansion": 0.15,
    "upgraded to buy": 0.35,
    "price target raised": 0.3,
    "initiated with outperform": 0.25,
    "strategic alliance": 0.2,
    "merger deal": 0.25,
    "introduces new product": 0.2,
    "launches AI platform": 0.25,
    "unveils service": 0.2,
    "leads the market": 0.2,
    "dominates segment": 0.25,
    "increases market share": 0.2,
    "FDA approval": 0.4,
    "license granted": 0.35,
    "greenlight from": 0.3,
    "CEO buys shares": 0.3,
    "board increases stake": 0.25,
    "patent granted": 0.3,
    "AI innovation": 0.35,
    "technology breakthrough": 0.3,
    "upbeat outlook": 0.2,
    "expecting strong growth": 0.25,
}

BEARISH_KEYWORDS_SCORES = {
    "misses expectations": -0.4,
    "lower guidance": -0.3,
    "lawsuit": -0.25,
    "investigation": -0.25,
    "data breach": -0.3,
    "resignation": -0.2,
    "cuts forecast": -0.2,
    "profit warning": -0.35,
    "layoffs": -0.2,
    "downgrade": -0.25,
    "debt": -0.15,
    "decline": -0.15,
    "recall": -0.2,
    "lower revenue": -0.35,
    "profit slump": -0.3,
    "loss widened": -0.3,
    "downgraded to sell": -0.4,
    "target lowered": -0.35,
    "initiated with underperform": -0.3,
    "under investigation": -0.3,
    "SEC probe": -0.35,
    "executive leaves": -0.25,
    "management shuffle": -0.2,
    "liquidity concerns": -0.35,
    "debt pressure": -0.3,
    "cash burn": -0.3,
    "missed payment": -0.4,
    "exec sells stake": -0.3,
    "shareholders exit": -0.25,
    "large sell-off": -0.3,
    "announces layoffs": -0.25,
    "plant shutdown": -0.3,
    "reduces workforce": -0.2,
    "loses market share": -0.25,
    "fierce competition": -0.2,
    "falling behind": -0.2,
    "expects decline": -0.25,
    "uncertain outlook": -0.2,
    "IPO withdrawn": -0.3,
    "project postponed": -0.25,
    "regulatory delay": -0.25,
}

# Consolidate and deduplicate keywords, taking the first score if duplicates exist
BULLISH_KEYWORDS_SCORES = {k: v for k, v in BULLISH_KEYWORDS_SCORES.items()}
BEARISH_KEYWORDS_SCORES = {k: v for k, v in BEARISH_KEYWORDS_SCORES.items()}

# Map themes to example keywords for alert generation (simplified for now)
NEWS_THEMES = {
    "Earnings Beat": ["beats expectations", "record revenue", "strong earnings", "profit surge"],
    "Analyst Upgrade": ["upgraded to buy", "price target raised", "initiated with outperform"],
    "Buyback/Dividend": ["declares dividend", "announces buyback", "dividend increase", "capital return"],
    "Acquisition/Partnership": ["strategic alliance", "merger deal", "partnership with", "acquires"],
    "Product Launch": ["introduces new product", "launches AI platform", "unveils service", "new product", "launch"],
    "Market Leadership": ["leads the market", "dominates segment", "increases market share"],
    "Regulatory Approval": ["FDA approval", "license granted", "greenlight from"],
    "Insider Buying": ["CEO buys shares", "board increases stake"],
    "Innovation": ["patent granted", "AI innovation", "technology breakthrough"],
    "Growth Forecasts": ["raises guidance", "upbeat outlook", "expecting strong growth", "raises forecast"],

    "Earnings Miss": ["misses expectations", "lower revenue", "profit slump", "loss widened"],
    "Analyst Downgrade": ["downgraded to sell", "target lowered", "initiated with underperform"],
    "Regulatory Risk": ["under investigation", "SEC probe", "data breach", "lawsuit filed"],
    "Resignations": ["CEO steps down", "executive leaves", "management shuffle", "resignation"],
    "Debt/Cash Problems": ["liquidity concerns", "debt pressure", "cash burn", "missed payment", "debt"],
    "Insider Selling": ["exec sells stake", "shareholders exit", "large sell-off"],
    "Layoffs/Closures": ["announces layoffs", "plant shutdown", "reduces workforce", "layoffs"],
    "Competitive Pressure": ["loses market share", "fierce competition", "falling behind"],
    "Negative Forecasts": ["cuts guidance", "expects decline", "uncertain outlook", "cuts forecast", "profit warning"],
    "Delisting/Delays": ["IPO withdrawn", "project postponed", "regulatory delay", "recall"],
}

# Reverse map keywords to themes for easier lookup
KEYWORD_TO_THEME = {}
for theme, keywords in NEWS_THEMES.items():
    for keyword in keywords:
        KEYWORD_TO_THEME[keyword] = theme

def analyze_news_keywords(text: str) -> tuple[float, list[str]]:
    """
    Analyzes text for predefined bullish and bearish keywords and returns a score
    and a list of matched themes.
    """
    text_lower = text.lower()
    keyword_score = 0.0
    matched_themes = set() # Use a set to avoid duplicate themes

    for keyword, score in BULLISH_KEYWORDS_SCORES.items():
        if keyword in text_lower:
            keyword_score += score
            if keyword in KEYWORD_TO_THEME:
                matched_themes.add(KEYWORD_TO_THEME[keyword])
            else:
                matched_themes.add(f"Bullish: {keyword}") # Fallback if keyword not in themes

    for keyword, score in BEARISH_KEYWORDS_SCORES.items():
        if keyword in text_lower:
            keyword_score += score
            if keyword in KEYWORD_TO_THEME:
                matched_themes.add(KEYWORD_TO_THEME[keyword])
            else:
                matched_themes.add(f"Bearish: {keyword}") # Fallback if keyword not in themes
    
    # Cap the keyword score to prevent it from dominating too much
    keyword_score = max(-1.0, min(1.0, keyword_score)) # Cap between -1 and 1

    return keyword_score, list(matched_themes)

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
    # The warning is for SMA200, which is the longest period.
    if len(df) < 200: 
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
def basic_analysis(historical_data: pd.DataFrame, news_sentiment: float = None, all_news_titles: list[str] = None) -> tuple:
    """
    Performs basic stock analysis based on technical indicators and news sentiment.
    This function is kept for a quick, general overview.

    Args:
        historical_data (pd.DataFrame): DataFrame with historical stock data.
        news_sentiment (float, optional): Sentiment score of news (-1 to 1). Defaults to None.
        all_news_titles (list[str], optional): List of news headlines. Defaults to None.

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

    # News Sentiment with Headlines
    if news_sentiment is not None:
        news_headlines_str = ""
        if all_news_titles:
            # Use a random sample of headlines for the basic analysis reason
            sample_headlines = random.sample(all_news_titles, min(len(all_news_titles), 2))
            news_headlines_str = f" (e.g., '{' / '.join(sample_headlines)}')"

        if news_sentiment > POSITIVE_SENTIMENT_THRESHOLD: # > 0.1 (new threshold)
            buy_signals += 1
            reasons.append(f"Positive news sentiment ({news_sentiment:.2f}){news_headlines_str}")
        elif news_sentiment < NEGATIVE_SENTIMENT_THRESHOLD: # < 0.0 (new threshold)
            sell_signals += 1
            reasons.append(f"Negative news sentiment ({news_sentiment:.2f}){news_headlines_str}")
        else: # Neutral
            hold_signals += 1
            reasons.append(f"Neutral news sentiment ({news_sentiment:.2f}){news_headlines_str}")

    total_signals = buy_signals + sell_signals + hold_signals 
    if total_signals == 0: return "Hold", 0, "No conclusive signals from available data." 
    if buy_signals > sell_signals: final_confidence = int((buy_signals / total_signals) * 100)
    elif sell_signals > buy_signals: final_confidence = int((sell_signals / total_signals) * 100)
    else: final_confidence = 50 # Neutral confidence
    final_confidence = max(0, min(final_confidence, 100)) # Ensure it's within 0-100
    if buy_signals > sell_signals and buy_signals >= hold_signals: return "Buy", final_confidence, "Primary signals suggest Buy: " + "; ".join(reasons)
    elif sell_signals > buy_signals and sell_signals >= hold_signals: return "Sell", final_confidence, "Primary signals suggest Sell: " + "; ".join(reasons)
    else: return "Hold", final_confidence, "Mixed or neutral signals: " + "; ".join(reasons)

# --- DYNAMIC ALERTS GENERATION ---
def generate_dynamic_alerts(rsi: float | None, macd_hist: float | None, sma_5: float | None, sma_10: float | None, current_volume: float | None, volume_sma_5: float | None, sma_50: float | None,
                            sma_200: float | None, current_price: float | None,
                            overall_news_sentiment: float | None, ath_price: float | None, all_news_articles_data: list[tuple[float, float, list[str], str]]) -> list[str]:
    """
    Generates dynamic alerts based on various technical and sentiment conditions.
    all_news_articles_data: List of (sentiment, weight, themes, title) for each article.
    """
    alerts = []
    processed_news_alerts = set() # Use a set to avoid duplicate news alerts from different articles triggering the same theme

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

    # SMA Alerts (Golden/Death Cross - still useful for context, even if not direct swing signal)
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200:
            alerts.append("✨ Golden Cross detected (SMA50 > SMA200) – long-term bullish context.")
        elif sma_50 < sma_200:
            alerts.append("☠️ Death Cross detected (SMA50 < SMA200) – long-term bearish context.")

    # Price Action Alerts (Near ATH)
    if current_price is not None and ath_price is not None and ath_price > 0:
        if current_price >= PRICE_NEAR_ATH_THRESHOLD * ath_price:
            alerts.append("📈 Near 1-year high – check valuation.")

    # Helper to append news headlines to alerts
    def append_headlines_to_alert(alert_msg: str, headlines: list[str]) -> str:
        if headlines:
            # Use a random sample of headlines for the alert
            sample_headlines = random.sample(headlines, min(len(headlines), 2))
            return f"{alert_msg} (e.g., '{' / '.join(sample_headlines)}')"
        return alert_msg

    # News Alerts (more granular based on themes from individual articles)
    if all_news_articles_data:
        for article_sentiment, article_weight, article_themes, article_title in all_news_articles_data:
            article_headlines = [article_title] # Use the specific article's title for its alert

            bullish_themes_in_article = set()
            bearish_themes_in_article = set()
            mixed_themes_in_article = set()

            for theme in article_themes:
                is_bullish = any(k in BULLISH_KEYWORDS_SCORES for k in NEWS_THEMES.get(theme, []) if k in BULLISH_KEYWORDS_SCORES)
                is_bearish = any(k in BEARISH_KEYWORDS_SCORES for k in NEWS_THEMES.get(theme, []) if k in BEARISH_KEYWORDS_SCORES)

                if is_bullish and not is_bearish:
                    bullish_themes_in_article.add(theme)
                elif is_bearish and not is_bullish:
                    bearish_themes_in_article.add(theme)
                else:
                    mixed_themes_in_article.add(theme)
            
            # Prioritize alerts from this single article to avoid direct contradictions
            # If an article has both strong bullish and strong bearish themes, prioritize bearish
            if bearish_themes_in_article:
                for theme in sorted(list(bearish_themes_in_article)):
                    processed_news_alerts.add(append_headlines_to_alert(f"📰 News: {theme} detected – strong negative signal.", article_headlines))
            elif bullish_themes_in_article:
                for theme in sorted(list(bullish_themes_in_article)):
                    processed_news_alerts.add(append_headlines_to_alert(f"📢 News: {theme} detected – strong positive signal.", article_headlines))
            
            # Mixed themes are less critical, but still informative
            if mixed_themes_in_article:
                for theme in sorted(list(mixed_themes_in_article)):
                    processed_news_alerts.add(append_headlines_to_alert(f"ℹ️ News: {theme} detected – mixed sentiment.", article_headlines))
    
    # Add all unique processed news alerts to the main alerts list
    alerts.extend(list(processed_news_alerts))

    # Fallback for general sentiment if no specific themes were matched across all articles
    if not processed_news_alerts and overall_news_sentiment is not None:
        if overall_news_sentiment < NEWS_SENTIMENT_STRONG_NEGATIVE_THRESHOLD: # sentiment < -0.4
            alerts.append(append_headlines_to_alert("📰 Overall news sentiment is strongly negative – caution advised.", all_news_titles))
        elif overall_news_sentiment > NEWS_SENTIMENT_STRONG_POSITIVE_THRESHOLD: # sentiment > 0.4
            alerts.append(append_headlines_to_alert("📢 Overall news sentiment is strongly positive – market optimism.", all_news_titles))
        else:
            alerts.append(append_headlines_to_alert("ℹ️ Overall news sentiment is neutral.", all_news_titles))

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
        # This filtering is now less critical as generate_dynamic_alerts is more targeted
        # but still good to have for robustness if alert types change.
        if any(keyword in alert for keyword in ["RSI", "MACD", "Oversold", "Overbought", "Bullish", "Bearish", "Positive sentiment", "Negative news trend", "SMA", "volume", "News:"]):
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
    all_time_high: float | None,
    all_news_articles_data: list[tuple[float, float, list[str], str]], # Changed from all_matched_news_themes
    all_news_titles: list[str] # Pass all news titles for alerts
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

    # Generate all dynamic alerts once, passing all news titles and detailed article data
    all_dynamic_alerts = generate_dynamic_alerts(rsi, macd_hist, sma_5, sma_10, current_volume, volume_sma_5, sma_50, sma_200,
                                                 current_price, overall_news_sentiment, all_time_high, all_news_articles_data, all_news_titles)

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
        # Fundamental data is no longer used for scoring in swing_trader, but kept in company_fundamentals
        # for potential future use or display in other sections if needed.
        "PE_ratio": company_fundamentals.get('trailingPE'), 
        "EPS_growth": company_fundamentals.get('earningsGrowth'), 
        "ROE": company_fundamentals.get('returnOnEquity'), 
        "D_E_ratio": company_fundamentals.get('debtToEquity'), 
        "sentiment": overall_news_sentiment,
        "current_price": current_price,
        "ATH": all_time_high,
        "all_alerts": all_dynamic_alerts # Pass all alerts to sub-functions
    }
    
    swing_result = evaluate_swing(data_for_eval)

    return {
        "swing_trader": swing_result
    }

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

def fetch_news_sentiment_from_newsapi(ticker_symbol: str, api_key: str | None, company_name: str | None) -> tuple[list[tuple[float, float, list[str], str]], list[str]]:
    """
    Fetches recent news articles for a given ticker symbol from NewsAPI
    filters them, and returns a list of (sentiment, weight, themes, title) tuples and titles.
    """
    if not api_key:
        logger.warning("NewsAPI key not provided. Skipping NewsAPI fetch.")
        return [], []
    
    if not NewsApiClient: # Check if the library was successfully imported
        # A warning is already printed at import time.
        return [], []

    newsapi_client = NewsApiClient(api_key=api_key)
    results = [] # List of (sentiment, weight, themes, title) tuples
    all_titles_for_overall_display = [] # Flat list of all titles for overall display
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
                    raw_sentiment = analyze_sentiment(article_text)
                    keyword_sentiment, matched_themes = analyze_news_keywords(article_text)
                    combined_sentiment = max(-1.0, min(1.0, raw_sentiment + keyword_sentiment))
                    weight = get_source_weight(source_name)
                    results.append((combined_sentiment, weight, matched_themes, title))
                    all_titles_for_overall_display.append(title)
            
            if results:
                logger.info(f"Fetched {len(all_titles_for_overall_display)} relevant articles from NewsAPI for {ticker_symbol}.")
                return results, all_titles_for_overall_display
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

def fetch_news_sentiment_from_rss(rss_url: str, ticker_symbol: str, company_name: str | None) -> tuple[list[tuple[float, float, list[str], str]], list[str]]:
    """
    Fetches news from an RSS feed, filters by ticker symbol,
    and returns a list of (sentiment, weight, themes, title) tuples and titles.
    """
    if feedparser is None:
        return [], []

    results = []
    all_titles_for_overall_display = []
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
                raw_sentiment = analyze_sentiment(article_text)
                keyword_sentiment, matched_themes = analyze_news_keywords(article_text)
                combined_sentiment = max(-1.0, min(1.0, raw_sentiment + keyword_sentiment))
                weight = get_source_weight(source_name)
                results.append((combined_sentiment, weight, matched_themes, title))
                all_titles_for_overall_display.append(title)

        if results:
            logger.info(f"Fetched {len(all_titles_for_overall_display)} relevant articles from RSS for {ticker_symbol}.")
            return results, all_titles_for_overall_display
        logger.info(f"No relevant news found for {ticker_symbol} in RSS feed.")
        return [], []
    except Exception as e:
        logger.error(f"Error fetching RSS news from {rss_url}: {e}")
        return [], []

def fetch_news_sentiment_from_gnews(ticker_symbol: str, api_key: str | None, company_name: str | None) -> tuple[list[tuple[float, float, list[str], str]], list[str]]: # type: ignore
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
    results = []
    all_titles_for_overall_display = []
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
                    raw_sentiment = analyze_sentiment(article_text)
                    keyword_sentiment, matched_themes = analyze_news_keywords(article_text)
                    combined_sentiment = max(-1.0, min(1.0, raw_sentiment + keyword_sentiment))
                    weight = get_source_weight(source_name)
                    results.append((combined_sentiment, weight, matched_themes, title))
                    all_titles_for_overall_display.append(title)
            if results:
                logger.info(f"Fetched {len(all_titles_for_overall_display)} relevant articles from GNews for {ticker_symbol}.")
                return results, all_titles_for_overall_display
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
        logger.error(f"Error fetching stock data: {error}")
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

        all_news_articles_data = [] # List of (sentiment, weight, themes, title) for each article
        all_news_titles_for_overall_display = [] # Flat list of all titles for overall display

        newsapi_results, newsapi_titles = fetch_news_sentiment_from_newsapi(TICKER, env_news_api_key, company_long_name)
        all_news_articles_data.extend(newsapi_results)
        all_news_titles_for_overall_display.extend(newsapi_titles)

        ticker_specific_rss_url = BASE_GOOGLE_NEWS_RSS_URL.format(ticker=TICKER)
        rss_results, rss_titles = fetch_news_sentiment_from_rss(ticker_specific_rss_url, TICKER, company_long_name)
        all_news_articles_data.extend(rss_results)
        all_news_titles_for_overall_display.extend(rss_titles)

        gnews_results, gnews_titles = fetch_news_sentiment_from_gnews(TICKER, env_gnews_api_key, company_long_name)
        all_news_articles_data.extend(gnews_results)
        all_news_titles_for_overall_display.extend(gnews_titles)

        overall_news_sentiment = None
        if all_news_articles_data:
            # Calculate weighted average from all articles' data
            total_sentiment_score = sum(s * w for s, w, _, _ in all_news_articles_data)
            total_weight = sum(w for s, w, _, _ in all_news_articles_data)
            if total_weight > 0:
                overall_news_sentiment = total_sentiment_score / total_weight
            else:
                overall_news_sentiment = None # Avoid division by zero if all weights are zero

        logger.info("\n--- News Sentiment ---")
        if overall_news_sentiment is not None:
            logger.info(f"Overall News Sentiment: {overall_news_sentiment:.2f}")
            logger.info("Recent News Titles (sample):")
            # Convert to set to get unique titles, then back to list for slicing
            for i, title in enumerate(list(set(all_news_titles_for_overall_display))[:10]):
                logger.info(f"  - {title}")
        else:
            logger.warning("Could not fetch news sentiment from any source.")

        # 4. Basic Analysis (using overall news sentiment)
        basic_recommendation, basic_confidence, basic_reason = basic_analysis(historical_data, overall_news_sentiment, all_news_titles_for_overall_display)
        logger.info(f"\n--- Basic Analysis for {TICKER} ---")
        logger.info(f"Recommendation: {basic_recommendation} (Confidence: {basic_confidence}%)")
        logger.info(f"Reason: {basic_reason}")

        # Swing Trader Recommendation System
        logger.info(f"\n--- Swing Trader Recommendation System for {TICKER} ---")
        # ATH is now part of company_fundamentals
        all_time_high_for_period = company_fundamentals.get('ath_from_period')
        swing_analysis_results = evaluate_stock(
            historical_data, technical_indicators, company_fundamentals, overall_news_sentiment, current_price, all_time_high_for_period, all_news_articles_data, all_news_titles_for_overall_display
        )
        logger.info("\nSwing Trader Recommendation:")
        logger.info(f"  Recommendation: {swing_analysis_results['swing_trader']['recommendation']}")
        logger.info(f"  Confidence: {swing_analysis_results['swing_trader']['confidence']}%")
        logger.info("  Alerts:")
        for alert in swing_analysis_results['swing_trader']['alerts']:
            logger.info(f"    - {alert}")


    logger.info("\nScript finished.")
