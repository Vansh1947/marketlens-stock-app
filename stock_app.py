import streamlit as st
import os
import sys # Add this import
st.set_page_config(
    page_title="MarketLens - Stock Analysis",
    page_icon="🔍",  # Lens emoji or 📈
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import all your functions from stock.py
# Assuming stock.py is in the same directory or accessible via PYTHONPATH
from stock import (get_stock_data, calculate_technical_indicators, fetch_news_sentiment_from_newsapi,
    fetch_news_sentiment_from_gnews, analyze_sentiment, analyze_stock, enhanced_analysis,
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

# Input for the stock ticker
ticker_input_raw = st.text_input("Enter Stock Ticker (e.g., GOOG, AAPL, RELIANCE.NS):", "GOOG")

# Button to trigger analysis
if st.button("Analyze Stock"):
    if not ticker_input_raw.strip():
        st.warning("Please enter a stock ticker symbol.")
    else:
        # Clean and uppercase the ticker symbol for consistent processing
        ticker_symbol_processed = ticker_input_raw.strip().upper()
        st.info(f"Analyzing {ticker_symbol_processed}...")

        # 1. Get Stock Data
        historical_data, current_price, company_fundamentals, error = get_stock_data(ticker_symbol_processed)

        if error:
            st.error(f"Error fetching data for {ticker_symbol_processed}: {error}")
        elif historical_data is None or historical_data.empty:
            st.warning(f"Could not retrieve sufficient data for {ticker_symbol_processed}.")
        else:
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

                # Initialize fig_macd before the expander
                fig_macd = None 
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
            with st.expander("⚙️ Key Indicators (Last Values)", expanded=False): # Changed title
                technical_indicators = calculate_technical_indicators(historical_data)
                
                # Safely display technical indicators
                sma_5 = technical_indicators.get('SMA_5')
                st.write(f"**SMA_5:** {f'{sma_5:.2f}' if sma_5 is not None else 'N/A'}")

                sma_20 = technical_indicators.get('SMA_20')
                st.write(f"**SMA_20:** {f'{sma_20:.2f}' if sma_20 is not None else 'N/A'}")

                rsi = technical_indicators.get('RSI')
                st.write(f"**RSI:** {f'{rsi:.2f}' if rsi is not None else 'N/A'}")

                volume_sma_5 = technical_indicators.get('Volume_SMA_5')
                st.write(f"**Volume_SMA_5:** {f'{volume_sma_5:,.0f}' if volume_sma_5 is not None else 'N/A'}")
                
                macd_val = technical_indicators.get('MACD')
                macd_signal_val = technical_indicators.get('MACD_Signal')
                if macd_val is not None and macd_signal_val is not None:
                    st.write(f"**MACD:** {macd_val:.2f} (Signal: {macd_signal_val:.2f})")
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
            # Initialize newsapi_client for Streamlit app context if not already done in stock.py
            newsapi_sentiment, newsapi_titles = None, [] # Ensure they are defined
            # Fetch NewsAPI sentiment if key is available
            if APP_NEWS_API_KEY: # Check if the NewsAPI key is configured
                newsapi_sentiment, newsapi_titles = fetch_news_sentiment_from_newsapi(ticker_symbol_processed, APP_NEWS_API_KEY)
            # else: # Commented out as the sidebar already indicates missing key.
                # Sidebar already indicates missing key. No prominent warning in main UI.

            gnews_sentiment, gnews_titles = None, [] # Ensure defined
            # Attempt GNews if key is present (useful for broader coverage or specific regions like India)
            # Check if the GNews API key is configured
            if APP_GNEWS_API_KEY: 
                st.write(f"Attempting to fetch news from GNews for {ticker_symbol_processed}...") # User feedback
                gnews_sentiment, gnews_titles = fetch_news_sentiment_from_gnews(ticker_symbol_processed, APP_GNEWS_API_KEY) # Pass the API key
            # else: # Commented out as the sidebar already indicates missing key.
                # Sidebar already indicates missing key.

            combined_sentiments = []
            combined_news_titles = []

            if newsapi_sentiment is not None:
                combined_sentiments.append(newsapi_sentiment)
                combined_news_titles.extend(newsapi_titles)
            if gnews_sentiment is not None:
                combined_sentiments.append(gnews_sentiment)
                combined_news_titles.extend(gnews_titles)

            # De-duplicate news titles
            if combined_news_titles:
                unique_titles = list(dict.fromkeys(combined_news_titles)) # More efficient deduplication
                combined_news_titles = unique_titles

            overall_news_sentiment = None
            if combined_sentiments:
                overall_news_sentiment = np.mean(combined_sentiments)

            st.markdown("---") # Visual separator
            with st.expander("📰 News Sentiment Details", expanded=False): # Added emoji
                if overall_news_sentiment is not None:
                    st.write(f"**Overall News Sentiment:** {overall_news_sentiment:.2f}")
                    st.write("Recent News Titles (sample):")
                    # Display more unique titles (up to 10)
                    for i, title in enumerate(combined_news_titles[:10]):
                        st.write(f"• {title}") # Use bullet points
                # If after all sources, there's still no overall sentiment
                if overall_news_sentiment is None:
                    st.write("No news sentiment could be determined from available sources.")
                st.caption("Sentiment is based on news titles and descriptions.")

            # DeepSeek News Summary section removed
            st.markdown("---") # Visual separator
            # 4. Basic Analysis
            with st.expander("🔬 Basic Analysis (Technical + News Sentiment)", expanded=False): # Added emoji
                basic_recommendation, basic_confidence, basic_reason = analyze_stock(historical_data, overall_news_sentiment)
                st.write(f"**Recommendation:** {basic_recommendation} (Confidence: {basic_confidence}%)")
                st.write(f"**Reason:** {basic_reason}")


            # 5. Enhanced Analysis (Simplified)
            st.markdown("---")  # Visual separator
            enhanced_recommendation, confidence_level, alerts, breakdown, category_scores, final_score_value = enhanced_analysis(
                historical_data,
                technical_indicators,
                company_fundamentals,
                overall_news_sentiment
            )
            # Detailed Analysis Breakdown (collapsible)
            with st.expander("Detailed Analysis Breakdown", expanded=False):
                for category, details in breakdown.items():
                    st.markdown(f"**{category}**")
                    for reason, item_details in details.items():
                        points = item_details.get('points', 0)
                        details_text = item_details.get('details', 'N/A')
                        if "points" in item_details:
                            display_value = f"{points:+.0f} ({details_text})"
                        else:
                            display_value = details_text
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• **{reason}:** `{display_value}`")
            if alerts:
                st.warning("Important Alerts:")
                for alert in alerts:
                    st.write(f"  - {alert}")

            st.markdown("---") # Visual separator

            # --- KEY ANALYSIS SUMMARY ---
            with st.container(border=True):
                st.subheader(f"KEY ANALYSIS SUMMARY FOR {ticker_symbol_processed}")
                st.metric(label="Final Recommendation", value=enhanced_recommendation)
                st.metric(label="Confidence Level", value=f"{confidence_level}%")
                # Simplified score calculation details
                st.markdown(f"**Final Score Value:** {final_score_value:.2f}")

                # Construct a concise reason string for the summary
                summary_reasons = []
                
                def get_summary_from_item(item_dict, default_text="Neutral"):
                    points = item_dict.get('points', 0)
                    details = item_dict.get('details', '')
                    if points > 0:
                        return f"+{points} ({details.split(' at ')[0]})"
                    elif points < 0:
                        return f"{points} ({details.split(' at ')[0]})"
                    return f"0 ({default_text})"

                # Technical highlights
                tech_breakdown = breakdown.get("Technical Analysis", {})
                if "Short-term SMA Cross" in tech_breakdown:
                    summary_reasons.append(f"Short-term SMA: {get_summary_from_item(tech_breakdown['Short-term SMA Cross'])}")
                if "Long-term SMA Cross" in tech_breakdown:
                    summary_reasons.append(f"Long-term SMA: {get_summary_from_item(tech_breakdown['Long-term SMA Cross'])}")
                if "RSI" in tech_breakdown:
                    summary_reasons.append(f"RSI: {get_summary_from_item(tech_breakdown['RSI'])}")
                if "MACD Crossover" in tech_breakdown:
                    summary_reasons.append(f"MACD: {get_summary_from_item(tech_breakdown['MACD Crossover'])}")
                if "Volume Activity" in tech_breakdown:
                    summary_reasons.append(f"Volume: {'High' if tech_breakdown['Volume Activity'].get('points', 0) > 0 else 'Normal'}")
                # Fundamental highlights
                fund_breakdown = breakdown.get("Fundamental Analysis", {})
                if "P/E Ratio" in fund_breakdown:
                    summary_reasons.append(f"P/E Ratio: {get_summary_from_item(fund_breakdown['P/E Ratio'])}")
                if "EPS Growth" in fund_breakdown:
                    summary_reasons.append(f"EPS Growth: {get_summary_from_item(fund_breakdown['EPS Growth'])}")
                if "Return on Equity (ROE)" in fund_breakdown:
                    summary_reasons.append(f"ROE: {get_summary_from_item(fund_breakdown['Return on Equity (ROE)'])}")
                if "Debt to Equity" in fund_breakdown:
                    summary_reasons.append(f"Debt/Equity: {get_summary_from_item(fund_breakdown['Debt to Equity'])}")
                # Sentiment highlight
                sent_breakdown = breakdown.get("Sentiment Analysis", {})
                if "Overall News Sentiment" in sent_breakdown:
                    sentiment_details = sent_breakdown["Overall News Sentiment"].get("details", "")
                    if "Positive" in sentiment_details: summary_reasons.append("News Sentiment: Positive")
                    elif "Negative" in sentiment_details: summary_reasons.append("News Sentiment: Negative")
                    else: summary_reasons.append("News Sentiment: Neutral")

                # Limit to top 5 reasons, prioritizing the order they are added
                final_summary_reason_str = "Recommendation based on analysis: " + "; ".join(summary_reasons[:5])
                if len(summary_reasons) > 5:
                    final_summary_reason_str += "; ..." # Indicate more reasons if truncated
                st.markdown(f"**Primary Reasons:** {final_summary_reason_str}")
                
                if alerts: # Display critical alerts from enhanced analysis again for emphasis
                    st.warning("Important Alerts to Consider (from Enhanced Analysis):")
                    for alert in alerts:
                        st.write(f" - {alert}")