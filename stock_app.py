import streamlit as st
import os
st.set_page_config(
    page_title="MarketLens - Stock Analysis",
    page_icon="🔍",  # Lens emoji or 📈
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import all your functions from stock.py, including new ones for news refinement, basic_analysis, and updated parameters for generate_dynamic_alerts
# Assuming stock.py is in the same directory or accessible via PYTHONPATH.
# Note: The return type of news fetching functions changed to include matched themes.
from stock import (get_stock_data, calculate_technical_indicators, fetch_news_sentiment_from_newsapi,
    fetch_news_sentiment_from_gnews, fetch_news_sentiment_from_yfinance, analyze_sentiment, basic_analysis, evaluate_stock,
    is_stock_mentioned, get_source_weight
 )
import numpy as np # Needed for np.mean if sentiments are combined
import plotly.graph_objects as go
import pandas as pd # Import pandas
from plotly.subplots import make_subplots

# Conditional pandas_ta import for plotting specific indicators
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    # stock.py already warns about pandas-ta missing for core calculations.
    # Plotting functions will handle pandas-ta (aliased as ta) being None.

# --- Caching Decorators for Streamlit ---
# Cache stock data for 1 hour.
@st.cache_data(ttl=3600)
def cached_get_stock_data(ticker_symbol, period):
    return get_stock_data(ticker_symbol, period)

# Cache technical indicators for 1 hour.
@st.cache_data(ttl=3600)
def cached_calculate_technical_indicators(historical_data):
    return calculate_technical_indicators(historical_data)

# Cache news sentiment for 4 hours (adjust TTL as needed for freshness vs API limits).
@st.cache_data(ttl=14400)
def cached_fetch_news_sentiment_from_newsapi(ticker_symbol, api_key, company_name):
    return fetch_news_sentiment_from_newsapi(ticker_symbol, api_key, company_name)

@st.cache_data(ttl=14400)
def cached_fetch_news_sentiment_from_yfinance(ticker_symbol, company_name):
    return fetch_news_sentiment_from_yfinance(ticker_symbol, company_name)
@st.cache_data(ttl=14400)
def cached_fetch_news_sentiment_from_gnews(ticker_symbol, api_key, company_name):
    return fetch_news_sentiment_from_gnews(ticker_symbol, api_key, company_name)

# Re-read environment variables inside the app or pass them
# Use st.secrets first, fallback to environment variables
APP_NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", os.environ.get("NEWS_API_KEY"))
APP_GNEWS_API_KEY = st.secrets.get("GNEWS_API_KEY", os.environ.get("GNEWS_API_KEY"))
    
# --- Plotting Functions ---
def create_price_volume_chart(df_hist, ticker):
    """Creates a Candlestick price chart with SMAs and a Volume bar chart."""
    if df_hist is None or df_hist.empty:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3]) # 70% for price, 30% for volume

    # Candlestick chart for price
    fig.add_trace(go.Candlestick(x=df_hist.index,
                                 open=df_hist['Open'],
                                 high=df_hist['High'],
                                 low=df_hist['Low'],
                                 close=df_hist['Close'],
                                 name='Price (OHLC)'),
                  row=1, col=1)

    # SMAs - calculated directly from historical_data for plotting
    if len(df_hist) >= 20:
        sma_20 = df_hist['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=df_hist.index, y=sma_20, mode='lines', name='SMA 20', line=dict(width=1, color='#FFA726')), row=1, col=1) # Specific Orange
    if len(df_hist) >= 50:
        sma_50 = df_hist['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=df_hist.index, y=sma_50, mode='lines', name='SMA 50', line=dict(width=1, color='#AB47BC')), row=1, col=1) # Specific Purple
    if len(df_hist) >= 200: # Check if data is sufficient for SMA 200
        sma_200 = df_hist['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=df_hist.index, y=sma_200, mode='lines', name='SMA 200', line=dict(width=1.5, color='#42A5F5')), row=1, col=1) # Specific Blue

    # Volume chart
    fig.add_trace(go.Bar(x=df_hist.index, y=df_hist['Volume'], name='Volume', marker_color='rgba(100,100,100,0.5)'),
                  row=2, col=1)

    # Add Volume SMA to the volume chart
    if len(df_hist) >= 20:
        volume_sma_20 = df_hist['Volume'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=df_hist.index, y=volume_sma_20, mode='lines', name='Volume SMA 20', line=dict(width=1, color='#636EFA')), row=2, col=1)

    fig.update_layout(
        title_text=f'{ticker} - Price, SMAs, and Volume',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        legend_title_text='Legend'
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

def create_rsi_chart(df_hist, ticker):
    """Creates an RSI chart if pandas-ta is available and data is sufficient."""
    if not ta or df_hist is None or df_hist['Close'].isnull().all() or len(df_hist) < 14: # RSI typically needs 14 periods
        return None
    try:
        rsi_series = df_hist.ta.rsi(length=14)
        if rsi_series is None or rsi_series.isnull().all():
            st.write("RSI could not be calculated (possibly insufficient data).")
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, mode='lines', name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", annotation_text="Overbought (70)", annotation_position="bottom right") # Specific Red
        fig.add_hline(y=30, line_dash="dash", line_color="#66BB6A", annotation_text="Oversold (30)", annotation_position="bottom right") # Specific Green
        fig.update_layout(title_text=f'{ticker} - Relative Strength Index (RSI)',
                          xaxis_title='Date', yaxis_title='RSI', yaxis_range=[0,100])
        return fig
    except Exception as e:
        st.error(f"Error creating RSI chart with pandas-ta: {e}")
        return None

def create_macd_chart(df_hist, ticker):
    """Creates a MACD chart if pandas-ta is available and data is sufficient."""
    if not ta or df_hist is None or df_hist['Close'].isnull().all() or len(df_hist) < 34: # MACD (12,26,9) needs ~34 periods
        return None
    try:
        fast_period, slow_period, signal_period = 12, 26, 9
        # Use append=False so it doesn't modify df_hist, returns only MACD columns
        macd_df = df_hist.ta.macd(fast=fast_period, slow=slow_period, signal=signal_period, append=False)

        if macd_df is None or macd_df.empty:
            st.write("MACD could not be calculated (possibly insufficient data).")
            return None

        # Define column names based on pandas-ta convention
        macd_line_col = f'MACD_{fast_period}_{slow_period}_{signal_period}'
        signal_line_col = f'MACDs_{fast_period}_{slow_period}_{signal_period}'
        hist_col = f'MACDh_{fast_period}_{slow_period}_{signal_period}'

        if not all(col in macd_df.columns for col in [macd_line_col, signal_line_col, hist_col]):
            st.error(f"MACD columns not found in pandas-ta output. Expected: {macd_line_col}, {signal_line_col}, {hist_col}")
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df[macd_line_col], mode='lines', name='MACD Line', line=dict(color='#29B6F6')))
        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df[signal_line_col], mode='lines', name='Signal Line', line=dict(color='#FFA726')))
        
        macd_hist_values = macd_df[hist_col].dropna() # Drop NaNs for color calculation and plotting
        if not macd_hist_values.empty:
            colors = ['#26A69A' if val >= 0 else '#EF5350' for val in macd_hist_values]
            fig.add_trace(go.Bar(x=macd_hist_values.index, y=macd_hist_values, name='MACD Histogram', marker_color=colors))

        fig.update_layout(title_text=f'{ticker} - MACD', xaxis_title='Date', yaxis_title='MACD Value')
        return fig
    except Exception as e:
        st.error(f"Error creating MACD chart with pandas-ta: {e}")
        return None

# Sidebar for API Key Status
st.sidebar.title("Configuration Status")
if APP_NEWS_API_KEY:
    st.sidebar.success("NEWS_API_KEY Loaded")
else:
    st.sidebar.error("NEWS_API_KEY Missing")
if APP_GNEWS_API_KEY:
    st.sidebar.success("GNEWS_API_KEY Loaded")
else:
    st.sidebar.error("GNEWS_API_KEY Missing (for GNews)")
# OpenAI API Key status removed
st.sidebar.markdown("---") # Separator in sidebar
st.sidebar.info("Ensure API keys are set for full functionality.")

# --- Main Title with Styling ---

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🔍 MarketLens 📈</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #696969;'>Your Clear View on Stock Performance & News</p>", unsafe_allow_html=True)
st.markdown("---") # Horizontal rule below title

col1, col2 = st.columns([3, 1])
with col1:
    # Input for the stock ticker
    ticker_input_raw = st.text_input("Enter Stock Ticker (e.g., GOOG, AAPL, RELIANCE.NS):", "GOOG")
with col2:
    # Input for the analysis period
    analysis_period = st.selectbox(
        "Select Analysis Period (for ATH):",
        ("6 Months", "1 Year", "2 Years", "Max"),
        index=0  # Default to '6 Months' for ATH
    )

# Map user-friendly names to yfinance period strings
period_map = {
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "Max": "max"
}
selected_period_yf = period_map[analysis_period]

# Button to trigger analysis
if st.button("Analyze Stock"):
    if not ticker_input_raw.strip():
        st.warning("Please enter a stock ticker symbol.")
    else:
        # Clean and uppercase the ticker symbol for consistent processing
        ticker_symbol_processed = ticker_input_raw.strip().upper()
        
        with st.spinner(f"Analyzing {ticker_symbol_processed} for period: {analysis_period}..."):
            # 1. Get Stock Data
            # Use cached version
            historical_data, current_price, company_fundamentals, error = cached_get_stock_data(ticker_symbol_processed, period=selected_period_yf)

        if error:
            st.error(f"Error fetching data for {ticker_symbol_processed}: {error}")
        elif historical_data is None or historical_data.empty:
            st.warning(f"Could not retrieve sufficient data for {ticker_symbol_processed}.")
        else:
            # Get company long name for news filtering
            company_long_name = company_fundamentals.get('longName')

            st.markdown(f"<h3 style='color: #4682B4;'>📊 Data for {ticker_symbol_processed}</h3>", unsafe_allow_html=True)
            st.metric(label="Current Price", value=f"${current_price:.2f}" if current_price is not None else "N/A")

            # --- PLOTTING CHARTS ---
            st.markdown("---")  # Visual separator
            st.markdown("<h3 style='color: #4682B4;'>📊 Price Chart & Volume</h3>", unsafe_allow_html=True)
            fig_price_volume = create_price_volume_chart(historical_data, ticker_symbol_processed)
            if fig_price_volume:
                st.plotly_chart(fig_price_volume, use_container_width=True)
            else:
                st.write("Could not generate price/volume chart.")

            if ta: # Only attempt to plot pandas-ta based charts if ta is available
                st.subheader("Technical Indicator Charts")
                with st.expander("📈 RSI Chart", expanded=False): # Added emoji
                    fig_rsi = create_rsi_chart(historical_data, ticker_symbol_processed)
                    if fig_rsi:
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    else:
                        st.write("Could not generate RSI chart (not enough data or pandas-ta issue).")

                with st.expander("📊 MACD Chart", expanded=False): # Added emoji, removed one level of nesting
                    fig_macd = create_macd_chart(historical_data, ticker_symbol_processed)
                    if fig_macd:
                        st.plotly_chart(fig_macd, use_container_width=True)
                    else:
                        st.write("Could not generate MACD chart (not enough data or pandas-ta issue).")
            else:
                st.info("pandas-ta library not found. Advanced technical indicator charts (RSI, MACD) are unavailable for plotting.")

            # 2. Calculate Technical Indicators
            st.markdown("---") # Visual separator
            with st.expander("⚙️ Key Indicators (Last Values)", expanded=False):
                # Use cached version
                technical_indicators = cached_calculate_technical_indicators(historical_data)
                
                # Safely display technical indicators
                sma_5 = technical_indicators.get('SMA_5')
                st.write(f"**SMA_5:** {f'{sma_5:.2f}' if sma_5 is not None else 'N/A'}")

                sma_10 = technical_indicators.get('SMA_10')
                st.write(f"**SMA_10:** {f'{sma_10:.2f}' if sma_10 is not None else 'N/A'}") # Display SMA_10

                sma_20 = technical_indicators.get('SMA_20')
                st.write(f"**SMA_20:** {f'{sma_20:.2f}' if sma_20 is not None else 'N/A'}")

                rsi = technical_indicators.get('RSI')
                st.write(f"**RSI:** {f'{rsi:.2f}' if rsi is not None else 'N/A'}")

                volume_sma_5 = technical_indicators.get('Volume_SMA_5')
                st.write(f"**Volume_SMA_5:** {f'{volume_sma_5:,.0f}' if volume_sma_5 is not None else 'N/A'}")
                
                macd_val = technical_indicators.get('MACD')
                macd_signal_val = technical_indicators.get('MACD_Signal')
                macd_hist_val = technical_indicators.get('MACD_Hist')
                if macd_val is not None and macd_signal_val is not None:
                    st.write(f"**MACD:** {macd_val:.2f} (Signal: {macd_signal_val:.2f}, Hist: {macd_hist_val:.2f})")
                else:
                    st.write("**MACD:** N/A")
                
                st.markdown("---")
                st.markdown("**Key Fundamental Metrics:**")

                # Safely display fundamental indicators
                pe_ratio = company_fundamentals.get('trailingPE')
                st.write(f"**P/E Ratio:** {f'{pe_ratio:.2f}' if isinstance(pe_ratio, (int, float)) else 'N/A'}")
                eps_growth = company_fundamentals.get('earningsGrowth')
                st.write(f"**EPS Growth (YoY):** {f'{eps_growth:.2%}' if isinstance(eps_growth, (int, float)) else 'N/A'}")
                roe = company_fundamentals.get('returnOnEquity')
                st.write(f"**Return on Equity (ROE):** {f'{roe:.2%}' if isinstance(roe, (int, float)) else 'N/A'}")
                debt_to_equity = company_fundamentals.get('debtToEquity')
                st.write(f"**Debt to Equity:** {f'{debt_to_equity:.2f}' if isinstance(debt_to_equity, (int, float)) else 'N/A'}")

            # 3. Fetch News Sentiment (from NewsAPI and GNews)
            all_news_articles_data = [] # List of (sentiment, weight, themes, title) for each article
            all_news_titles_for_overall_display = [] # Flat list of all titles for overall display

            # Fetch NewsAPI sentiment if key is available
            if APP_NEWS_API_KEY: # Check if the NewsAPI key is configured
                newsapi_results, newsapi_titles = cached_fetch_news_sentiment_from_newsapi(ticker_symbol_processed, APP_NEWS_API_KEY, company_long_name)
                all_news_articles_data.extend(newsapi_results)
                all_news_titles_for_overall_display.extend(newsapi_titles)

            # Attempt GNews if key is present (useful for broader coverage or specific regions like India)
            # Check if the GNews API key is configured
            if APP_GNEWS_API_KEY:
                st.write(f"Attempting to fetch news from GNews for {ticker_symbol_processed}...") # User feedback
                gnews_results, gnews_titles = cached_fetch_news_sentiment_from_gnews(ticker_symbol_processed, APP_GNEWS_API_KEY, company_long_name) # Pass the API key
                all_news_articles_data.extend(gnews_results)
                all_news_titles_for_overall_display.extend(gnews_titles)

            # Fallback to yfinance news for broader coverage, especially for Indian stocks
            yfinance_results, yfinance_titles = cached_fetch_news_sentiment_from_yfinance(ticker_symbol_processed, company_long_name)
            all_news_articles_data.extend(yfinance_results)
            all_news_titles_for_overall_display.extend(yfinance_titles)

            overall_news_sentiment = None
            if all_news_articles_data: # Check if there's any news data
                total_sentiment_score = sum(s * w for s, w, _, _ in all_news_articles_data) # Unpack the tuple
                total_weight = sum(w for s, w, _, _ in all_news_articles_data) # Unpack the tuple
                if total_weight > 0:
                    overall_news_sentiment = total_sentiment_score / total_weight
                else:
                    overall_news_sentiment = None # Avoid division by zero if all weights are zero

            # De-duplicate news titles for overall display
            if all_news_titles_for_overall_display:
                unique_titles = list(dict.fromkeys(all_news_titles_for_overall_display)) # More efficient deduplication
                all_news_titles_for_overall_display = unique_titles

            st.markdown("---") # Visual separator
            with st.expander("📰 News Sentiment Details", expanded=False): # Added emoji
                if overall_news_sentiment is not None: # Only display if sentiment was calculated
                    st.write(f"**Overall Weighted News Sentiment:** {overall_news_sentiment:.2f}")
                    st.write("Recent News Titles (sample):")
                    # Display more unique titles (up to 10)
                    for i, title in enumerate(all_news_titles_for_overall_display[:10]):
                        st.write(f"• {title}") # Use bullet points
                # If after all sources, there's still no overall sentiment
                if overall_news_sentiment is None:
                    st.write("No news sentiment could be determined from available sources.")
                st.caption("Sentiment is based on news titles and descriptions, filtered for relevance and weighted by source credibility.")

            st.markdown("---") # Visual separator
            # 4. Basic Analysis (Kept for compatibility)
            with st.expander("🔬 Basic Analysis (Technical + News Sentiment)", expanded=False): # Added emoji
                basic_recommendation, basic_confidence, basic_reason = basic_analysis(historical_data, overall_news_sentiment, all_news_titles_for_overall_display)
                st.write(f"**Recommendation:** {basic_recommendation} (Confidence: {basic_confidence}%)")
                st.write(f"**Reason:** {basic_reason}")
            
            # New: Swing Trader Recommendation System
            st.markdown("---") # Visual separator
            st.markdown("<h3 style='color: #4682B4;'>🎯 Swing Trader Recommendation System</h3>", unsafe_allow_html=True)
            
            # ATH is calculated in get_stock_data and passed via company_fundamentals for simplicity
            # It's stored as 'ath_from_period'
            all_time_high_for_period = company_fundamentals.get('ath_from_period')
            # Pass the collected news themes to evaluate_stock
            swing_analysis_results = evaluate_stock(
                historical_data, technical_indicators, company_fundamentals, overall_news_sentiment, current_price, all_time_high_for_period, all_news_articles_data, all_news_titles_for_overall_display # Pass all news titles
            ) # This now returns only swing_trader results
            
            st.subheader("📈 Swing Trader Recommendation")
            st.write(f"**Recommendation:** {swing_analysis_results['swing_trader']['recommendation']}")
            st.write(f"**Confidence:** {swing_analysis_results['swing_trader']['confidence']}%")
            if swing_analysis_results["swing_trader"].get("alerts"):
                st.info("Swing Trader Alerts:")
                for alert in swing_analysis_results["swing_trader"]["alerts"]:
                    st.write(f"- {alert}")