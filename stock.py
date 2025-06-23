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
POSITIVE_SENTIMENT_THRESHOLD = 0.25 # Renamed from VERY_POSITIVE_SENTIMENT_THRESHOLD for consistency
SLIGHTLY_POSITIVE_SENTIMENT_THRESHOLD = 0.10
NEUTRAL_SENTIMENT_LOWER_BOUND = -0.10 # If sentiment is >= this and < SLIGHTLY_POSITIVE, it's Neutral
NEGATIVE_SENTIMENT_THRESHOLD = -0.10 # Anything below this is considered Negative
EPS_GROWTH_STRONG_THRESHOLD = 0.30 
EPS_GROWTH_NEGATIVE_THRESHOLD = -0.1
# --- END OF THRESHOLDS ---



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
    confidence_score = 0 # For dynamic confidence

    # SMA Crossover
    sma_5 = technical_indicators.get('SMA_5')
    sma_20 = technical_indicators.get('SMA_20')
    if sma_5 is not None and sma_20 is not None:
        if sma_5 > sma_20:
            buy_signals += 1
            reasons.append("5-day SMA above 20-day SMA (Bullish Crossover)")
            confidence_score += 20
        elif sma_5 < sma_20:
            sell_signals += 1
            reasons.append("5-day SMA below 20-day SMA (Bearish Crossover)")
            confidence_score += 15 # Sell signals also contribute to confidence in the signal
        else:
            hold_signals += 1
            reasons.append("5-day and 20-day SMAs are close (Neutral Crossover)")
            confidence_score += 5

    # RSI
    rsi_value = technical_indicators.get('RSI')
    if rsi_value is not None:
        if rsi_value < RSI_OVERSOLD_THRESHOLD:
            buy_signals += 1
            reasons.append(f"RSI ({rsi_value:.2f}) indicates oversold condition")
            confidence_score += 20
        elif rsi_value > RSI_OVERBOUGHT_THRESHOLD:
            sell_signals += 1
            reasons.append(f"RSI ({rsi_value:.2f}) indicates overbought condition")
            confidence_score += 20
        else:
            hold_signals += 1
            reasons.append(f"RSI ({rsi_value:.2f}) is neutral")
            confidence_score += 10

    # MACD
    macd_value = technical_indicators.get('MACD')
    macd_signal_value = technical_indicators.get('MACD_Signal')
    if macd_value is not None and macd_signal_value is not None:
        if macd_value > macd_signal_value:
            buy_signals += 1
            reasons.append("MACD above MACD Signal (Bullish MACD Crossover)")
            confidence_score += 25
        elif macd_value < macd_signal_value:
            sell_signals += 1
            reasons.append("MACD below MACD Signal (Bearish MACD Crossover)")
            confidence_score += 20
        else:
            hold_signals += 1
            reasons.append("MACD and MACD Signal are close (Neutral MACD)")
            confidence_score += 5

    # News Sentiment
    if news_sentiment is not None:
        if news_sentiment >= POSITIVE_SENTIMENT_THRESHOLD:
            buy_signals += 1
            reasons.append(f"Positive news sentiment ({news_sentiment:.2f})")
            confidence_score += 20
        elif news_sentiment >= SLIGHTLY_POSITIVE_SENTIMENT_THRESHOLD:
            buy_signals += 0.5 # Fractional signal, or adjust confidence
            reasons.append(f"Slightly positive news sentiment ({news_sentiment:.2f})")
            confidence_score += 10
        elif news_sentiment < NEGATIVE_SENTIMENT_THRESHOLD: # Using the new NEGATIVE_SENTIMENT_THRESHOLD
            sell_signals += 1
            reasons.append(f"Negative news sentiment ({news_sentiment:.2f})")
            confidence_score += 15
        else: # Neutral
            hold_signals += 1
            reasons.append(f"Neutral news sentiment ({news_sentiment:.2f})")
            confidence_score += 5

    total_signals = buy_signals + sell_signals + hold_signals
    if total_signals == 0: # No valid indicators to base a decision on
        return "Hold", 0, "No conclusive signals from available data." # Default confidence to 0 if no signals

    final_confidence = max(0, min(int(confidence_score), 100)) # Cap confidence

    if buy_signals > sell_signals and buy_signals >= hold_signals:
        return "Buy", final_confidence, "Primary signals suggest Buy: " + "; ".join(reasons)
    elif sell_signals > buy_signals and sell_signals >= hold_signals:
        return "Sell", final_confidence, "Primary signals suggest Sell: " + "; ".join(reasons)
    else:
        return "Hold", final_confidence, "Mixed or neutral signals: " + "; ".join(reasons)

# --- ADVANCED ANALYSIS ---
def enhanced_analysis(stock_symbol: str, historical_data: pd.DataFrame, technical_indicators: dict,
                      company_fundamentals: dict, news_sentiment: float,
                      social_media_sentiment: float, market_news: list) -> tuple:
    """
    Performs an enhanced stock analysis combining technical, fundamental, and sentiment data.

    Args:
        stock_symbol (str): The ticker symbol of the stock.
        historical_data (pd.DataFrame): DataFrame with historical stock data.
        technical_indicators (dict): Dictionary of calculated technical indicators.
        company_fundamentals (dict): Dictionary of company fundamental data.
        news_sentiment (float): Sentiment score of news (-1 to 1).
        social_media_sentiment (float): Sentiment score from social media (-1 to 1).
        market_news (list): List of relevant market news headlines/summaries.

    Returns:
        tuple: (Recommendation: str, Confidence: int, Reason: str, Alerts: list)
    """
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    alerts = []
    reasons = []
    confidence_score = 50 # Start with a neutral base confidence

    # Technical Indicators
    rsi_val = technical_indicators.get('RSI')
    if rsi_val is not None:
        if rsi_val < RSI_OVERSOLD_THRESHOLD:
            buy_signals += 1
            reasons.append(f"RSI ({rsi_val:.2f}) oversold")
            confidence_score += 10 # Adjusted from +15
        elif rsi_val > RSI_OVERBOUGHT_THRESHOLD:
            sell_signals += 1
            reasons.append(f"RSI ({rsi_val:.2f}) overbought")
            confidence_score -= 10 # Negative impact for overbought
        elif RSI_BULLISH_NEUTRAL_THRESHOLD <= rsi_val <= 65: # Moderate Bullish RSI (55-65)
            confidence_score += 5
            reasons.append(f"RSI ({rsi_val:.2f}) indicates moderate bullish strength")
        elif rsi_val > 65 and rsi_val < RSI_OVERBOUGHT_THRESHOLD: # Still bullish but closer to overbought (65-70)
            confidence_score += 7 # Adjusted from +10
            reasons.append(f"RSI ({rsi_val:.2f}) in bullish zone, approaching overbought")
        else:
            hold_signals += 1
            reasons.append(f"RSI ({rsi_val:.2f}) neutral")
    macd_val = technical_indicators.get('MACD')
    macd_signal_val = technical_indicators.get('MACD_Signal')
    macd_bullish_crossover = False
    if macd_val is not None and macd_signal_val is not None:
        if macd_val > macd_signal_val:
            buy_signals += 1
            reasons.append("MACD bullish crossover")
            confidence_score += 10 
            macd_bullish_crossover = True
        elif macd_val < macd_signal_val:
            sell_signals += 1
            reasons.append("MACD bearish crossover")
            confidence_score -= 10 
        else:
            hold_signals += 1
            reasons.append("MACD neutral")

    # SMA5 vs SMA20
    sma_5_val = technical_indicators.get('SMA_5') # Assuming SMA_5 is calculated and available
    sma_20_val = technical_indicators.get('SMA_20')
    if sma_5_val is not None and sma_20_val is not None:
        if sma_5_val > sma_20_val:
            buy_signals +=1 
            reasons.append("SMA_5 > SMA_20 (Short-term bullish)")
            confidence_score +=10 
        elif sma_5_val < sma_20_val:
            sell_signals +=1
            reasons.append("SMA_5 < SMA_20 (Short-term bearish)")
            confidence_score -=5 

    sma_50_val = technical_indicators.get('SMA_50')
    sma_200_val = technical_indicators.get('SMA_200')
    death_cross_active = False
    if sma_50_val is not None and sma_200_val is not None:
        if sma_50_val > sma_200_val:
            buy_signals += 1
            reasons.append("50-day SMA above 200-day SMA (Golden Cross)")
            confidence_score += 5 # Golden cross has some positive impact
        else:
            death_cross_active = True # Potential Death Cross
            # sell_signals += 1 # We will handle its impact conditionally
            # reasons.append("50-day SMA below 200-day SMA (Death Cross)") # Add reason later if applied

    # Company Fundamentals
    pe_ratio = None
    eps_growth = None
    sector = None
    if company_fundamentals: 
        pe_ratio = company_fundamentals.get('trailingPE')
        eps_growth = company_fundamentals.get('earningsGrowth')
        sector = company_fundamentals.get('sector') # Attempt to get sector
    # The rest of the logic handles pe_ratio and eps_growth being None gracefully


    if pe_ratio is not None and not np.isinf(pe_ratio):
        if pe_ratio < PE_RATIO_UNDERVALUED_THRESHOLD: # e.g. < 15
            buy_signals += 1
            reasons.append(f"Low P/E Ratio ({pe_ratio:.2f})")
            confidence_score += 5
            # Check for tech sector context even when undervalued
            if sector and "technology" in sector.lower() and pe_ratio < 20:
                reasons.append(f"P/E ({pe_ratio:.2f}) is especially low for Technology sector.")
                confidence_score += 5
        elif pe_ratio > PE_RATIO_OVERVALUED_THRESHOLD: # e.g. > 30
            # sell_signals += 1 # High P/E for growth stock might be normal
            reasons.append(f"High P/E Ratio ({pe_ratio:.2f})")
            confidence_score -= 5 # High P/E is generally a slight negative for confidence
        else: # P/E is in the neutral range
            hold_signals += 1
            reasons.append(f"Neutral P/E Ratio ({pe_ratio:.2f})")
    else:
        reasons.append("P/E Ratio not available or infinite.")

    strong_eps_growth = False
    if eps_growth is not None:
        if eps_growth > EPS_GROWTH_STRONG_THRESHOLD:
            buy_signals += 1
            reasons.append(f"Strong EPS Growth ({eps_growth:.2%})")
            confidence_score += 10 
            strong_eps_growth = True
        elif eps_growth < EPS_GROWTH_NEGATIVE_THRESHOLD:
            sell_signals += 1
            reasons.append(f"Negative EPS Growth ({eps_growth:.2%})")
            # Confidence adjustment for very negative EPS growth handled below
        else:
            hold_signals += 1
            reasons.append(f"Neutral EPS Growth ({eps_growth:.2%})")
    else:
        reasons.append("EPS Growth not available.")

    news_is_neutral_or_positive = False
    news_sentiment_value_for_logic = 0.0 # Default to neutral if None
    if news_sentiment is not None:
        news_sentiment_value_for_logic = news_sentiment

    # Specific penalty for very negative EPS growth
    if eps_growth is not None and eps_growth < -0.30: # e.g., less than -30%
        reasons.append(f"Significant EPS Decline ({eps_growth:.2%}) heavily impacts outlook.")
        confidence_score -= 20 # Adjusted from -25, still a major penalty

    # News Sentiment
    if news_sentiment is not None:
        if news_sentiment >= POSITIVE_SENTIMENT_THRESHOLD: # >= 0.25
            buy_signals += 1
            reasons.append(f"Positive news sentiment ({news_sentiment:.2f})")
            confidence_score += 10 # Adjusted from +15
            news_is_neutral_or_positive = True
        elif news_sentiment >= SLIGHTLY_POSITIVE_SENTIMENT_THRESHOLD: # >= 0.10
            buy_signals += 0.5 # Or consider it a weaker buy signal
            reasons.append(f"Slightly positive news sentiment ({news_sentiment:.2f})")
            confidence_score += 5 
            news_is_neutral_or_positive = True
        elif news_sentiment < NEGATIVE_SENTIMENT_THRESHOLD: # < -0.10
            sell_signals += 1
            reasons.append(f"Negative news sentiment ({news_sentiment:.2f})")
            confidence_score -= 5 
        else: # Neutral (-0.10 <= sentiment < 0.10)
            hold_signals += 1
            reasons.append(f"Neutral news sentiment ({news_sentiment:.2f})")
            news_is_neutral_or_positive = True # Neutral is also counted for deprioritizing death cross
    else:
        reasons.append("News sentiment not available.")

    # Deprioritize Death Cross for growth stocks
    if death_cross_active:
        # Conditions to deprioritize: RSI > 55, MACD bullish, News Neutral/Positive, Strong EPS Growth
        deprioritize = (
            (rsi_val is not None and rsi_val > RSI_BULLISH_NEUTRAL_THRESHOLD) and
            macd_bullish_crossover and # MACD must be bullish
            (news_sentiment_value_for_logic >= NEUTRAL_SENTIMENT_LOWER_BOUND) and # News is neutral or positive
            strong_eps_growth
        )
        if deprioritize:
            reasons.append("Death Cross observed but deprioritized due to strong growth signals.")
            # No confidence penalty if deprioritized, maybe even a slight positive for resilience
            # confidence_score += 0 
        else:
            sell_signals += 1
            reasons.append("50-day SMA below 200-day SMA (Death Cross)")
            confidence_score -= 10 # Stronger penalty for active, non-deprioritized Death Cross

    # Social Media Sentiment (Placeholder - requires external integration)
    if social_media_sentiment is not None:
        # Using the new sentiment scale for consistency
        if social_media_sentiment >= POSITIVE_SENTIMENT_THRESHOLD:
            buy_signals += 0.5 # Social media might be weighted less
            reasons.append(f"Positive social media sentiment ({social_media_sentiment:.2f})")
            confidence_score += 5
        elif social_media_sentiment >= SLIGHTLY_POSITIVE_SENTIMENT_THRESHOLD:
            reasons.append(f"Slightly positive social media sentiment ({social_media_sentiment:.2f})")
            confidence_score += 3
        elif social_media_sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            sell_signals += 0.5
            reasons.append(f"Negative social media sentiment ({social_media_sentiment:.2f})")
            confidence_score -= 5
        else: # Neutral
            reasons.append(f"Neutral social media sentiment ({social_media_sentiment:.2f})")
    else:
        reasons.append("Social media sentiment not available.")

    # Market News Alerts
    for news in market_news:
        # Simplified keyword matching for alerts
        if any(keyword in news.lower() for keyword in ["risk", "volatility", "uncertainty", "downside"]):
            alerts.append(f"Alert: Market risk/volatility indicated: '{news}'")
        if any(keyword in news.lower() for keyword in ["drop", "crash", "recession", "bankrupt"]):
            alerts.append(f"Alert: Potential market downturn mentioned: '{news}'")
        if any(keyword in news.lower() for keyword in ["fraud", "scandal", "investigation"]):
            alerts.append(f"Alert: Company-specific negative news: '{news}'")
        if any(keyword in news.lower() for keyword in ["growth", "expansion", "profit", "innovat"]):
            alerts.append(f"Alert: Positive company/market news: '{news}'")

    total_signals = buy_signals + sell_signals + hold_signals
    if total_signals == 0:
        return "Hold", 0, "No conclusive signals from available data.", alerts

    final_confidence = max(0, min(int(confidence_score), 100)) # Cap confidence

    # Determine recommendation based on signal counts first, then adjust with confidence bands
    if buy_signals > sell_signals and buy_signals >= hold_signals:
        recommendation = "Buy"
    elif sell_signals > buy_signals and sell_signals >= hold_signals:
        recommendation = "Sell"
    else: # hold_signals are dominant or signals are very mixed
        recommendation = "Hold"

    # Now, let the confidence score refine the "strength" of this recommendation
    # This part is more about interpreting the confidence score in context of the recommendation
    # The actual recommendation (Buy/Sell/Hold) is primarily from signal counts.
    # The confidence score reflects how strong that signal is.
    # Example: If recommendation is "Buy" and confidence is 30%, it's a "Weak Buy".
    # The previous logic of setting recommendation based on confidence bands was a bit circular.

    # Let's ensure the confidence isn't misleadingly high if signals are very contradictory
    # or if a strong negative (like massive EPS drop) has pulled it down.
    # If recommendation is "Buy" but confidence is low (e.g. < 40), it's a weak buy.
    # If recommendation is "Sell" but confidence is low (e.g. < 40), it's a weak sell.

    # The override for dominant sell signals can remain if desired:
    # if sell_signals > buy_signals + hold_signals and final_confidence < 50 :
    #     recommendation = "Sell" # This could still be useful
        
    return recommendation, final_confidence, f"Recommendation based on weighted analysis: {'; '.join(reasons)}", alerts

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

def extract_financial_events(content: str) -> list[str]:
    """
    Extracts potential financial events from text content.

    Args:
        content (str): The text content to analyze.

    Returns:
        list: A list of identified financial event types.
    """
    events = []
    if "earnings" in content.lower() or "quarterly results" in content.lower() or "revenue" in content.lower() or "profit" in content.lower():
        events.append("Earnings Report")
    if "merger" in content.lower() or "acquisition" in content.lower() or "takeover" in content.lower():
        events.append("Merger/Acquisition")
    if "layoff" in content.lower() or "job cuts" in content.lower() or "restructuring" in content.lower():
        events.append("Layoffs/Restructuring")
    if "dividend" in content.lower():
        events.append("Dividend Announcement")
    if "product launch" in content.lower() or "innovation" in content.lower() or "new technology" in content.lower():
        events.append("Product/Innovation News")
    if "lawsuit" in content.lower() or "regulatory" in content.lower() or "fine" in content.lower():
        events.append("Legal/Regulatory Issue")
    return events

def assess_impact(events: list[str], sentiment: float) -> tuple[dict, list[str]]:
    """
    Assesses the potential short-term and long-term impact of financial events
    based on their sentiment.

    Args:
        events (list): A list of identified financial event types.
        sentiment (float): The sentiment score associated with the events (-1 to 1).

    Returns:
        tuple: (Impact dictionary: dict, Alerts list: list)
               Impact dictionary has 'short_term' and 'long_term' keys.
    """
    impact = {"short_term": "Neutral", "long_term": "Neutral"}
    alerts = []

    # Prioritize specific events
    if "Legal/Regulatory Issue" in events:
        if sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            impact["short_term"] = "Highly Negative"
            impact["long_term"] = "Potentially Negative"
            alerts.append("Alert: Legal/Regulatory issue with negative sentiment. High risk.")
        elif sentiment > POSITIVE_SENTIMENT_THRESHOLD:
            impact["short_term"] = "Neutral"
            impact["long_term"] = "Neutral"
            alerts.append("Legal/Regulatory issue: Resolved or positive outcome implied.")
        else:
            alerts.append("Legal/Regulatory issue: Unclear impact, requires close monitoring.")
        return impact, alerts # Override other impacts if this is present

    if "Earnings Report" in events:
        if sentiment > POSITIVE_SENTIMENT_THRESHOLD:
            impact["short_term"] = "Positive"
            impact["long_term"] = "Positive"
            alerts.append("Earnings Beat: Positive outlook.")
        elif sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            impact["short_term"] = "Negative"
            impact["long_term"] = "Negative"
            alerts.append("Earnings Miss: Negative outlook.")
        else:
            alerts.append("Earnings Report: Neutral sentiment.")

    if "Merger/Acquisition" in events:
        if sentiment > POSITIVE_SENTIMENT_THRESHOLD:
            impact["short_term"] = "Positive"
            impact["long_term"] = "Positive"
            alerts.append("Merger/Acquisition: Potentially positive for growth.")
        elif sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            impact["short_term"] = "Negative"
            impact["long_term"] = "Negative"
            alerts.append("Merger/Acquisition: Potentially negative (e.g., overpayment, integration issues).")
        else:
            alerts.append("Merger/Acquisition: Mixed sentiment, watch for details.")

    if "Layoffs/Restructuring" in events:
        if sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            impact["short_term"] = "Negative"
            impact["long_term"] = "Negative"
            alerts.append("Layoffs/Restructuring: Indicates potential issues or cost-cutting.")
        elif sentiment > POSITIVE_SENTIMENT_THRESHOLD: # Sometimes layoffs are seen positively for efficiency
            impact["short_term"] = "Neutral to Positive"
            impact["long_term"] = "Neutral to Positive"
            alerts.append("Layoffs/Restructuring: Market views as positive for efficiency.")
        else:
            alerts.append("Layoffs/Restructuring: Neutral sentiment, requires further analysis.")

    if "Product/Innovation News" in events:
        if sentiment > POSITIVE_SENTIMENT_THRESHOLD:
            impact["short_term"] = "Positive"
            impact["long_term"] = "Positive"
            alerts.append("New Product/Innovation: Potential for future growth.")
        else:
            alerts.append("Product/Innovation News: Watch for market adoption and reception.")

    return impact, alerts

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

def generate_signal(impact: dict) -> str:
    """
    Generates a simple 'Buy', 'Sell', or 'Hold' signal based on impact assessment.

    Args:
        impact (dict): Dictionary with 'short_term' and 'long_term' impact.

    Returns:
        str: 'Buy', 'Sell', or 'Hold'.
    """
    if "Highly Negative" in impact.values() or "Negative" in impact.values():
        return 'Sell'
    elif "Positive" in impact.values() or "Neutral to Positive" in impact.values():
        return 'Buy'
    else:
        return 'Hold'

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
        # Placeholder for social_media_sentiment - would need integration with a social media API
        social_media_sentiment_placeholder = 0.1 # Example value

        enhanced_recommendation, enhanced_confidence, enhanced_reason, alerts = enhanced_analysis(
            TICKER,
            historical_data,
            technical_indicators,
            company_fundamentals,
            overall_news_sentiment,
            social_media_sentiment_placeholder,
            combined_news_titles # Using combined news titles for alerts
        )

        print(f"\n--- Enhanced Analysis for {TICKER} ---")
        print(f"Recommendation: {enhanced_recommendation} (Confidence: {enhanced_confidence}%)")
        print(f"Reason: {enhanced_reason}")
        if alerts:
            print("Alerts:")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print("No specific alerts.")

        # Example of financial event extraction and impact assessment
        # Use a more comprehensive sample news content for event detection
        sample_news_content_event = f"""
        {TICKER} announced strong Q1 earnings, beating analyst expectations on both revenue and profit,
        driven by robust advertising growth. However, the company also hinted at potential restructuring
        in its cloud division and faces an ongoing anti-trust lawsuit from the DOJ.
        Analysts remain bullish, but the legal issue adds uncertainty.
        """
        events = extract_financial_events(sample_news_content_event)
        sentiment_for_event = analyze_sentiment(sample_news_content_event)
        impact, event_alerts = assess_impact(events, sentiment_for_event)
        event_signal = generate_signal(impact)

        print("\n--- Financial Event Analysis Example ---")
        print(f"Sample News: '{sample_news_content_event}'")
        print(f"Identified Events: {events}")
        print(f"Sentiment for Event: {sentiment_for_event:.2f}")
        print(f"Assessed Impact: {impact}")
        print(f"Event-based Signal: {event_signal}")
        if event_alerts:
            print("Event Alerts:")
            for alert in event_alerts:
                print(f"  - {alert}")

    print("\nScript finished.")