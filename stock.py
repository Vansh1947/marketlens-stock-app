"""
This module provides comprehensive stock analysis functionalities, including
data fetching, technical indicator calculation, sentiment analysis from news,
and generating trading recommendations.
"""
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta


# Conditional imports for external APIs
try:
    import yfinance as yf
except ImportError:
    yf = None
    print("Warning: 'yfinance' library not found. Stock data functionalities will be skipped.")

try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("Warning: 'pandas-ta' library not found. Technical indicators will be skipped.")
    print("Please install it by running: pip install pandas-ta")

try:
    from newsapi import NewsApiClient
    from newsapi.newsapi_exception import NewsAPIException
except ImportError:
    NewsApiClient = None # type: ignore
    # Define a simple mock for NewsAPIException if the library is not found.
    # This mock won't have specific methods like get_code() or get_message().
    # The error handling block will need to rely on str(e) or e.args.
    class MockNewsAPIException(Exception):
        pass
    NewsAPIException = MockNewsAPIException # type: ignore
    print("Warning: 'newsapi-python' library not found. NewsAPI functionalities will be skipped.")

try:
    import feedparser
except ImportError:
    feedparser = None
    print("Warning: 'feedparser' library not found. RSS feed functionalities will be skipped.")

try:
    from gnews import GNews # Assuming this is the library for gnews
except ImportError:
    GNews = None
    print("Warning: 'gnews' library not found. GNews functionalities will be skipped.")


# --- IMPORTANT: INSTALL NECESSARY LIBRARIES ---
# If you encounter ModuleNotFoundError for the libraries below, run:
# pip install yfinance textblob newsapi-python feedparser gnews pandas-ta
# --- END OF INSTALLATION INSTRUCTIONS ---


# --- CONFIGURATION (BEST PRACTICE: USING ENVIRONMENT VARIABLES) ---
# Set these environment variables in your system's environment (NOT directly in code):
# export NEWS_API_KEY="your_actual_news_api_key_here"
# export OPENAI_API_KEY="sk-proj-..."
# stock.py will now only read from environment variables if run directly

# --- END OF THRESHOLDS ---

# Initialize GNews client
gnews_client = None # Initialize to None
if GNews:  # Check if the GNews library is installed
    gnews_client = GNews(max_results=20, period='7d')
    print("GNews client initialized (max_results=20, period=7d).")
else: # Ensure this 'else' is aligned with the 'if GNews:' above
    print("Warning: 'gnews' library not found. GNews functionalities will be skipped.")

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

def _calculate_confidence_level(technical_indicators: dict, company_fundamentals: dict, news_sentiment: float | None, historical_data: pd.DataFrame) -> int:
    """
    Calculates a confidence level based on a weighted sum of individual indicator contributions.
    The score is normalized to a 0-100 range, where 50 is neutral, 100 is strong buy confidence,
    and 0 is strong sell confidence.
    """
    confidence_raw_score = 0

    # Technical Indicators Contributions
    sma_5 = technical_indicators.get('SMA_5')
    sma_20 = technical_indicators.get('SMA_20')
    if sma_5 is not None and sma_20 is not None:
        if sma_5 > sma_20:
            confidence_raw_score += 10 # Bullish short-term cross
        elif sma_5 < sma_20:
            confidence_raw_score -= 10 # Bearish short-term cross

    sma_50 = technical_indicators.get('SMA_50')
    sma_200 = technical_indicators.get('SMA_200')
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200:
            confidence_raw_score += 15 # Golden Cross (strong bullish)
        elif sma_50 < sma_200:
            confidence_raw_score -= 15 # Death Cross (strong bearish)

    rsi_val = technical_indicators.get('RSI')
    if rsi_val is not None:
        if rsi_val < RSI_OVERSOLD_THRESHOLD:
            confidence_raw_score += 10 # Oversold (bullish)
        elif rsi_val > RSI_OVERBOUGHT_THRESHOLD:
            confidence_raw_score -= 10 # Overbought (bearish)

    macd_val = technical_indicators.get('MACD')
    macd_signal_val = technical_indicators.get('MACD_Signal')
    if macd_val is not None and macd_signal_val is not None:
        if macd_val > macd_signal_val:
            confidence_raw_score += 10 # Bullish MACD crossover
        elif macd_val < macd_signal_val:
            confidence_raw_score -= 10 # Bearish MACD crossover

    # Fundamental Indicators Contributions
    pe_ratio = company_fundamentals.get('trailingPE')
    if pe_ratio is not None and not np.isinf(pe_ratio):
        if pe_ratio < PE_RATIO_UNDERVALUED_THRESHOLD:
            confidence_raw_score += 10 # Undervalued P/E
        elif pe_ratio > PE_RATIO_OVERVALUED_THRESHOLD:
            confidence_raw_score -= 10 # Overvalued P/E

    eps_growth = company_fundamentals.get('earningsGrowth') # Year over Year EPS Growth
    if eps_growth is not None:
        if eps_growth > EPS_GROWTH_STRONG_THRESHOLD:
            confidence_raw_score += 15 # Strong EPS Growth
        elif eps_growth < EPS_GROWTH_NEGATIVE_THRESHOLD:
            confidence_raw_score -= 15 # Negative EPS Growth

    return_on_equity = company_fundamentals.get('returnOnEquity')
    if return_on_equity is not None:
        if return_on_equity > ROE_GOOD_THRESHOLD: # e.g., > 15%
            confidence_raw_score += 15 # Strong ROE
        elif return_on_equity < 0: # Negative ROE is a bad sign
            confidence_raw_score -= 15 # Negative ROE

    debt_to_equity = company_fundamentals.get('debtToEquity')
    if debt_to_equity is not None and not np.isinf(debt_to_equity):
        if debt_to_equity < DEBT_TO_EQUITY_LOW_THRESHOLD:
            confidence_raw_score += 10 # Low Debt/Equity
        elif debt_to_equity > DEBT_TO_EQUITY_HIGH_THRESHOLD:
            confidence_raw_score -= 10 # High Debt/Equity

    # Volume Contribution
    volume_sma_5 = technical_indicators.get('Volume_SMA_5')
    current_volume = historical_data['Volume'].iloc[-1] if not historical_data.empty else None

    if volume_sma_5 is not None and current_volume is not None:
        if current_volume > volume_sma_5 * VOLUME_HIGH_THRESHOLD_MULTIPLIER:
            confidence_raw_score += 5 # High volume confirms trend, adds confidence

    # Sentiment Contribution
    if news_sentiment is not None:
        if news_sentiment > POSITIVE_SENTIMENT_THRESHOLD: # > 0.1
            confidence_raw_score += 10 # Positive news
        elif news_sentiment < NEGATIVE_SENTIMENT_THRESHOLD: # < 0.0
            confidence_raw_score -= 10 # Negative news

    # Normalize the raw score to a 0-100 range.
    # Max theoretical positive contribution: 10+15+10+10 (tech) + 10+15+15+10 (fund) + 10 (sent) + 5 (volume) = 120
    # Max theoretical negative contribution: -10-15-10-10 (tech) -10-15-15-10 (fund) -10 (sent) = -115
    # So, the raw score range is approximately -115 to +120.
    clamped_raw_score = max(-115, min(120, confidence_raw_score))
    confidence_level = int((clamped_raw_score + 115) / 235 * 100) # Adjusted denominator
    return max(0, min(100, confidence_level)) # Clamp between 0 and 100

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
        print("Insufficient data for calculating technical indicators. Returning empty dict.")
        return {}
    if ta is None:
        return {}

    df = historical_data.copy()
    indicators = {}

    # Ensure enough data for indicators
    if len(df) < 200: # Max window size for SMAs
        # pandas-ta might also print its own warnings if data is insufficient for certain indicators
        # but this general warning is still good.
        print(f"Warning: Not enough historical data ({len(df)} rows) for some indicators (e.g., SMA200 needs 200).")

    # Simple Moving Averages
    indicators['SMA_5'] = df['Close'].rolling(window=5).mean().iloc[-1] if len(df) >= 5 else None
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
def analyze_stock(historical_data: pd.DataFrame, news_sentiment: float = None) -> tuple:
    """
    Performs basic stock analysis based on technical indicators and news sentiment.

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
    # company_fundamentals is not used in basic analysis, but passed for consistency
    # with enhanced_analysis signature if it were to be extended.
    # The enhanced_analysis will have full fundamentals.
    # For basic, we'll just pass an empty dict if not available.
    company_fundamentals_for_basic = {} # Placeholder for basic analysis

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
    rsi_value = technical_indicators.get('RSI') # No change here, logic is fine
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

    total_signals = buy_signals + sell_signals + hold_signals # No change here, logic is fine
    if total_signals == 0: return "Hold", 0, "No conclusive signals from available data." # Default confidence to 0 if no signals
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

def _calculate_sentiment_score(news_sentiment: float | None) -> tuple[float, dict, list]:
    """Calculates a normalized sentiment score (0-100) and provides a breakdown and alerts."""
    score = 50.0
    breakdown = {}
    alerts = []
    
    if news_sentiment is not None: # Sentiment rule: <0 = Negative, 0-0.1 = Neutral, >0.1 = Positive
        if news_sentiment > POSITIVE_SENTIMENT_THRESHOLD: # > 0.1
            points = 20 # Positive impact
            alerts.append(f"Notice: Positive news sentiment detected ({news_sentiment:.2f}).")
            sentiment_label = "Positive"
        elif news_sentiment < NEGATIVE_SENTIMENT_THRESHOLD: # < 0.0
            points = -20 # Negative impact
            alerts.append(f"Alert: Negative news sentiment detected ({news_sentiment:.2f}).")
            sentiment_label = "Negative"
        else: # 0.0 to 0.1 (inclusive of 0.0)
            points = 0
            sentiment_label = "Neutral"
        score += points
        breakdown["Overall News Sentiment"] = {"points": points, "details": f"{news_sentiment:.2f} ({sentiment_label})"}
    else:
        breakdown["Overall News Sentiment"] = {"points": 0, "details": "N/A"}
    
    return max(0, min(100, score)), breakdown, alerts

# --- ADVANCED ANALYSIS ---
def enhanced_analysis(historical_data: pd.DataFrame, technical_indicators: dict, company_fundamentals: dict, news_sentiment: float | None) -> tuple:
    """
    Performs an enhanced, categorized, and simplified stock analysis.

    Returns:
        tuple: (Recommendation, Confidence_Level, Alerts, Master_Breakdown, Category_Scores, Final_Score_Value)
    """
    # 1. Calculate scores for each category
    tech_score, tech_breakdown = _calculate_technical_score(technical_indicators, historical_data)
    fund_score, fund_breakdown = _calculate_fundamental_score(company_fundamentals)
    sent_score, sent_breakdown, alerts = _calculate_sentiment_score(news_sentiment)

    category_scores = {
        "Technical": tech_score,
        "Fundamental": fund_score,
        "Sentiment": sent_score,
    }

    # 2. Calculate final weighted score (using a simple average, no sector-specific weights)
    # This simplifies the logic as requested.
    final_score_value = (tech_score + fund_score + sent_score) / 3

    # 3. Classify recommendation and calculate confidence
    recommendation = classify_recommendation(final_score_value)
    confidence_level = _calculate_confidence_level(technical_indicators, company_fundamentals, news_sentiment, historical_data)

    # 4. Combine all breakdowns for a full report
    master_breakdown = {
        "Technical Analysis": tech_breakdown,
        "Fundamental Analysis": fund_breakdown,
        "Sentiment Analysis": sent_breakdown,
    }

    return recommendation, confidence_level, alerts, master_breakdown, category_scores, final_score_value

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

def fetch_news_sentiment_from_newsapi(ticker_symbol: str, api_key: str | None) -> tuple[float | None, list[str]]:
    """
    Fetches recent news articles for a given ticker symbol from NewsAPI
    and calculates the average sentiment.

    Returns:
        tuple: (Average sentiment: float | None, List of news titles: list)
    """
    if not api_key:
        print("NewsAPI key not provided. Skipping NewsAPI fetch.")
        return None, []
    
    if not NewsApiClient: # Check if the library was successfully imported
        # A warning is already printed at import time.
        return None, []

    newsapi_client = NewsApiClient(api_key=api_key)
    all_articles = []
    try:
        # Fetch news from the last 7 days (free tier usually limits to 30 days history)
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        articles_response = newsapi_client.get_everything(q=ticker_symbol,
                                                          language='en',
                                                          sort_by='relevancy',
                                                          from_param=from_date,
                                                          to=to_date,
                                                          page_size=20) # Max 20 articles per request for free tier

        if articles_response and articles_response['articles']:
            all_articles = articles_response['articles']
            sentiments = [analyze_sentiment(article.get('title', '') + " " + (article.get('description', '') or ""))
                          for article in all_articles if article.get('title') or article.get('description')]
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                news_titles = [article.get('title', 'No Title') for article in all_articles]
                print(f"Fetched {len(news_titles)} articles from NewsAPI for {ticker_symbol}.")
                return avg_sentiment, news_titles
        print(f"No recent news found for {ticker_symbol} from NewsAPI.")
        return None, []
    except NewsAPIException as e: # type: ignore [misc] # misc because NewsAPIException could be the mock
        error_details = str(e) # Standard way to get exception message.
        # Check for common error substrings if specific codes/methods aren't available
        # on the exception object (especially if it's the mock or an older library version).
        # Also, try to use get_code() if available for more specific handling.
        is_rate_limited = 'rateLimited' in error_details.lower() or ('get_code' in dir(e) and callable(e.get_code) and e.get_code() == 'rateLimited')
        is_api_key_invalid = 'apiKeyInvalid' in error_details.lower() or \
                             'unauthorized' in error_details.lower() or \
                             ('get_code' in dir(e) and callable(e.get_code) and e.get_code() == 'apiKeyInvalid')

        if is_rate_limited:
            print(f"NewsAPI Rate Limit Exceeded: {error_details}. Please wait before trying again.")
        elif is_api_key_invalid:
            print(f"NewsAPI Key Invalid/Unauthorized: {error_details}. Please check your NEWS_API_KEY environment variable.")
        else:
            code_info = f"(Code: {e.get_code()})" if ('get_code' in dir(e) and callable(e.get_code)) else ""
            print(f"Error fetching news from NewsAPI for {ticker_symbol} {code_info}: {error_details}")
        return None, []
    except Exception as e:
        print(f"An unexpected error occurred while fetching NewsAPI data for {ticker_symbol}: {e}")
        return None, []

def fetch_news_sentiment_from_rss(rss_url: str, ticker_symbol: str) -> tuple[float | None, list[str]]:
    """
    Fetches news from an RSS feed, filters by ticker symbol,
    and calculates sentiment. Requires 'feedparser'.
    """
    if feedparser is None:
        # print("Feedparser not available. Cannot fetch RSS news.") # Silenced
        return None, []

    relevant_articles = []
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            # print(f"No entries found in RSS feed: {rss_url}") # Silenced
            return None, []

        for entry in feed.entries:
            title = entry.get('title', '')
            summary = entry.get('summary', '') # Use .get() for safety
            content = title + " " + summary

            # Simple check: see if ticker symbol is in title or summary (case-insensitive)
            if ticker_symbol.lower() in content.lower():
                relevant_articles.append(entry)

        if relevant_articles:
            sentiments = [analyze_sentiment(entry.get('title', '') + " " + (entry.get('summary', '') or ""))
                          for entry in relevant_articles]
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                news_titles = [entry.get('title', 'No Title') for entry in relevant_articles]
                # print(f"Fetched {len(news_titles)} relevant articles from RSS for {ticker_symbol}.") # Silenced
                return avg_sentiment, news_titles
        # print(f"No relevant news found for {ticker_symbol} in RSS feed.") # Silenced
        return None, []
    except Exception as e:
        # print(f"Error fetching RSS news from {rss_url}: {e}") # Silenced
        return None, []

def fetch_news_sentiment_from_gnews(ticker_symbol: str, api_key: str | None) -> tuple[float | None, list[str]]: # type: ignore
    """
    Fetches news for a given ticker symbol from GNews and calculates sentiment.
        ticker_symbol (str): The stock ticker symbol (often for .NS market).
        api_key (str | None): The GNews API key (though the gnews library might not strictly require it).
    # The gnews library (v0.4.1) doesn't strictly require an API key for basic usage,
    # but we keep the key parameter for consistency or future library versions.
    Returns:
        tuple: (Average sentiment: float | None, List of news titles: list)
    """
    if not gnews_client:
        print("GNews client not initialized. Cannot fetch news from GNews.")
        return None, []
    
    query = f"{ticker_symbol} stock news"
    try:
        # Assuming gnews_client.get_news(query) returns a list of article-like objects
        # Each object is expected to have 'title' and 'description' attributes or similar.
        # Adjust period and max_results as needed and if supported by the library.
        # Some gnews libraries might require setting period, country, etc. during client initialization
        # or on the get_news method.
        # gnews_client.period = '7d' # Example: if the library supports setting period
        # gnews_client.max_results = 10 # Example: if the library supports setting max results
        news_items = gnews_client.get_news(query)

        if news_items:
            sentiments = [analyze_sentiment(item.get('title', '') + " " + (item.get('description', '') or item.get('text', '')))
                          for item in news_items if item.get('title') or item.get('description') or item.get('text')]
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                news_titles = [item.get('title', 'No Title') for item in news_items]
                print(f"Fetched {len(news_titles)} articles from GNews for {ticker_symbol}.")
                return avg_sentiment, news_titles
        print(f"No recent news found for {ticker_symbol} from GNews.")
        return None, []
    except Exception as e:
        print(f"Error fetching or processing GNews for {ticker_symbol}: {e}")
        return None, []

def get_stock_data(ticker_symbol: str) -> tuple[pd.DataFrame | None, float | None, dict | None, str | None]:
    """
    Fetches historical stock data and basic company fundamentals using yfinance.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        tuple: (historical_data: pd.DataFrame, current_price: float,
                company_fundamentals: dict, error_message: str)
               Returns (None, None, None, error_message) on failure.
    """
    if yf is None:
        return None, None, None, "Yfinance library not available. Cannot fetch stock data."

    try:
        stock = yf.Ticker(ticker_symbol)
        # Attempt to fetch company info first to validate ticker and get fundamentals
        company_fundamentals = stock.info

        # A common sign of an invalid/delisted ticker is an empty info dict or missing key financial fields
        if not company_fundamentals or \
           (company_fundamentals.get('regularMarketPrice') is None and \
            company_fundamentals.get('longName') is None and \
            company_fundamentals.get('marketCap') is None):
            return None, None, None, f"No valid data or fundamentals found for '{ticker_symbol}'. It might be an invalid ticker, delisted, or data is unavailable."

        # Fetch 1 year of daily historical data for comprehensive analysis
        historical_data = stock.history(period="2y")

        if historical_data.empty:
            # We might have fundamentals, but no historical data for the specified period
            current_price_from_info = company_fundamentals.get('regularMarketPrice') or company_fundamentals.get('currentPrice')
            # It's unusual to have fundamentals but no historical data for a valid, active ticker over "1y"
            # but we return what we have along with a message.
            return None, current_price_from_info, company_fundamentals, f"No historical data found for '{ticker_symbol}' for the period '1y'. Some fundamental data might be available."

        # Ensure 'Close' column exists and has data before accessing iloc[-1]
        if 'Close' in historical_data.columns and not historical_data['Close'].empty:
            current_price_from_history = historical_data['Close'].iloc[-1]
        else:
            # Fallback if 'Close' is missing or empty, though unlikely if historical_data itself is not empty
            current_price_from_history = company_fundamentals.get('regularMarketPrice') or company_fundamentals.get('currentPrice')
            if current_price_from_history is None:
                 return historical_data, None, company_fundamentals, f"Historical data fetched but 'Close' price is missing for {ticker_symbol}."

        return historical_data, current_price_from_history, company_fundamentals, None
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
            print("No ticker symbol entered. Please try again.")

    # TICKER = args.ticker.strip().upper() # Convert to uppercase for consistency # Old argparse way
   

    print(f"Starting comprehensive stock analysis script for {TICKER}...")

    # 1. Get Stock Data
    historical_data, current_price, company_fundamentals, error = get_stock_data(TICKER)

    if error:
        print(f"\nError: {error}")
    elif historical_data is None or historical_data.empty:
        print(f"\nCould not retrieve sufficient data for {TICKER}. Exiting.")
    else:
        print(f"\n--- Data for {TICKER} ---")
        print(f"Current Price: ${current_price:.2f}")

        # 2. Calculate Technical Indicators
        technical_indicators = calculate_technical_indicators(historical_data)
        print("\n--- Technical Indicators ---")
        if technical_indicators:
            for k, v in technical_indicators.items():
                if v is not None:
                    print(f"{k}: {v:.2f}")
                else:
                    print(f"{k}: Not enough data")
        else:
            print("Technical indicators skipped (pandas-ta not available or insufficient data).")


        # 3. Fetch News Sentiment (from NewsAPI and RSS)
        # Retrieve API keys from environment for direct script execution
        env_news_api_key = os.environ.get("NEWS_API_KEY")
        env_gnews_api_key = os.environ.get("GNEWS_API_KEY")
        newsapi_sentiment, newsapi_titles = fetch_news_sentiment_from_newsapi(TICKER, env_news_api_key) # Pass key

        # Dynamically create ticker-specific RSS URL
        ticker_specific_rss_url = BASE_GOOGLE_NEWS_RSS_URL.format(ticker=TICKER)
        rss_sentiment, rss_titles = fetch_news_sentiment_from_rss(ticker_specific_rss_url, TICKER)

        # Combine sentiments and titles
        combined_sentiments = []
        # Fetch GNews sentiment (Pass key, though library might not use it)
        gnews_sentiment, gnews_titles = fetch_news_sentiment_from_gnews(TICKER, env_gnews_api_key)


        combined_news_titles = []

        if newsapi_sentiment is not None:
            combined_sentiments.append(newsapi_sentiment)
            combined_news_titles.extend(newsapi_titles)
        if rss_sentiment is not None:
            combined_sentiments.append(rss_sentiment)
            combined_news_titles.extend(rss_titles)
        if gnews_sentiment is not None: # Add GNews results
            combined_sentiments.append(gnews_sentiment)
            combined_news_titles.extend(gnews_titles) # Correctly add GNews titles
        overall_news_sentiment = None
        if combined_sentiments:
            overall_news_sentiment = np.mean(combined_sentiments)

        print("\n--- News Sentiment ---")
        if overall_news_sentiment is not None:
            print(f"Overall News Sentiment: {overall_news_sentiment:.2f}")
            print("Recent News Titles (sample):")
            # Convert to set to get unique titles, then back to list for slicing
            for i, title in enumerate(list(set(combined_news_titles))[:10]):
                print(f"  - {title}")
        else:
            print("Could not fetch news sentiment from any source.")

        # 4. Basic Analysis (using overall news sentiment)
        basic_recommendation, basic_confidence, basic_reason = analyze_stock(historical_data, overall_news_sentiment)
        print(f"\n--- Basic Analysis for {TICKER} ---")
        print(f"Recommendation: {basic_recommendation} (Confidence: {basic_confidence}%)")
        print(f"Reason: {basic_reason}")

        # 5. Enhanced Analysis
        enhanced_recommendation, confidence_level, alerts, breakdown, category_scores, final_score_value = enhanced_analysis(
            historical_data,
            technical_indicators,
            company_fundamentals,
            overall_news_sentiment
        )

        print(f"\n--- Enhanced Analysis for {TICKER} ---")
        print("Category Scores:")
        for category, score in category_scores.items():
            print(f"  - {category}: {score:.2f}/100")

        print(f"Recommendation: {enhanced_recommendation} (Final Score: {final_score_value:.2f}, Confidence: {confidence_level}%)")
        print("Reason Breakdown:")
        for category, details in breakdown.items():
            print(f"  - {category}:")
            for reason, value in details.items():
                print(f"    - {reason}: {value}")
        if alerts:
            print("Alerts:")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print("No specific alerts.")

    print("\nScript finished.")