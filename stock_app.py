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
from stock import ( # Removed global client imports
    get_stock_data, calculate_technical_indicators, fetch_news_sentiment_from_newsapi, # type: ignore
    fetch_news_sentiment_from_gnews, analyze_sentiment, analyze_stock, enhanced_analysis, # type: ignore
    extract_financial_events, assess_impact, generate_signal, # type: ignore
    calculate_pivot_points, forecast_short_term_trend, detect_double_top_bottom, # type: ignore
    detect_sma_crossover_pattern, calculate_risk_reward # type: ignore
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

            # --- PLOTTING CHARTS (Feature 9: Clean, Interactive UI) ---
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

            # 2. Calculate Technical Indicators (Feature 9: Clean, Interactive UI)
            st.markdown("---") # Visual separator
            with st.expander("⚙️ Technical Indicators (Last Values)", expanded=False): # Added emoji
                technical_indicators = calculate_technical_indicators(historical_data)
                if technical_indicators:
                    for k, v in technical_indicators.items():
                        if v is not None:
                            st.write(f"{k}: {v:.2f}")
                        else:
                            st.write(f"{k}: Not enough data")
                else:
                    st.write("Technical indicators skipped (pandas-ta not available or insufficient data).")


            # 3. Fetch News Sentiment (Feature 4: Real Financial News Filtering)
            # Initialize newsapi_client for Streamlit app context if not already done in stock.py
            newsapi_sentiment, newsapi_items = None, [] # Ensure they are defined, now list of dicts
            # Fetch NewsAPI sentiment if key is available
            if APP_NEWS_API_KEY: # Check if the NewsAPI key is configured
                newsapi_sentiment, newsapi_items = fetch_news_sentiment_from_newsapi(ticker_symbol_processed, APP_NEWS_API_KEY)
            # else: # Commented out as the sidebar already indicates missing key.
                # Sidebar already indicates missing key. No prominent warning in main UI.

            gnews_sentiment, gnews_items = None, [] # Ensure defined, now list of dicts
            # Attempt GNews if key is present (useful for broader coverage or specific regions like India)
            # Check if the GNews API key is configured
            if APP_GNEWS_API_KEY: 
                st.write(f"Attempting to fetch news from GNews for {ticker_symbol_processed}...") # User feedback
                gnews_sentiment, gnews_items = fetch_news_sentiment_from_gnews(ticker_symbol_processed, APP_GNEWS_API_KEY) # Pass the API key
            # else: # Commented out as the sidebar already indicates missing key.
                # Sidebar already indicates missing key.

            combined_sentiments = []
            combined_news_items = [] # This will now store list of dicts with 'title' and 'date'

            if newsapi_sentiment is not None:
                combined_sentiments.append(newsapi_sentiment)
                if newsapi_items: combined_news_items.extend(newsapi_items)
            if gnews_sentiment is not None:
                combined_sentiments.append(gnews_sentiment)
                if gnews_items: combined_news_items.extend(gnews_items)

            # De-duplicate news items based on title, preserving date
            if combined_news_items:
                seen_titles = set()
                unique_items = []
                for item in combined_news_items:
                    # Ensure 'title' key exists before accessing
                    title = item.get('title')
                    if title and title not in seen_titles:
                        unique_items.append(item)
                        seen_titles.add(title)
                combined_news_items = unique_items

            overall_news_sentiment = None
            if combined_sentiments:
                overall_news_sentiment = np.mean(combined_sentiments)

            # Extract just the titles for functions that expect a list of strings
            combined_news_titles_only = [item.get('title', '') for item in combined_news_items]

            st.markdown("---") # Visual separator
            with st.expander("📰 News Sentiment Details", expanded=False): # Added emoji
                if overall_news_sentiment is not None:
                    st.write(f"**Overall News Sentiment:** {overall_news_sentiment:.2f}")
                    st.write("Recent News Titles (sample):")
                    # Display more unique titles (up to 10)
                    for item in combined_news_items[:10]: # Iterate over dicts to display date
                        st.write(f"• **{item.get('date', 'No Date')}:** {item.get('title', 'No Title')}")
                # If after all sources, there's still no overall sentiment
                if overall_news_sentiment is None:
                    st.write("No news sentiment could be determined from available sources.")
                st.caption("Sentiment is based on news titles and descriptions.")

            # DeepSeek News Summary section removed
            st.markdown("---") # Visual separator (Feature 9: Clean, Interactive UI)
            # 4. Basic Analysis
            with st.expander("🔬 Basic Analysis (Technical + News Sentiment)", expanded=False): # Added emoji
                basic_recommendation, basic_confidence, basic_reason = analyze_stock(historical_data, overall_news_sentiment)
                st.write(f"**Recommendation:** {basic_recommendation} (Confidence: {basic_confidence}%)")
                st.write(f"**Reason:** {basic_reason}")


            # 5. Enhanced Analysis
            st.markdown("---") # Visual separator (Feature 9: Clean, Interactive UI)
            social_media_sentiment_input = None # Explicitly None as it's a placeholder
            enhanced_recommendation, enhanced_confidence, enhanced_reason, alerts = enhanced_analysis(
                ticker_symbol_processed,
                historical_data,
                technical_indicators,
                company_fundamentals,
                overall_news_sentiment,
                social_media_sentiment_input,
                combined_news_titles_only # Pass list of strings
            )
            st.markdown("<h3 style='color: #4682B4;'>✨ Enhanced Analysis</h3>", unsafe_allow_html=True) # Styled subheader
            st.metric(label="Enhanced Recommendation", value=f"{enhanced_recommendation}", help=f"Confidence: {enhanced_confidence}%")
            st.write(f"**Reason:** {enhanced_reason}")
            if alerts:
                st.warning("Alerts:")
                for alert in alerts:
                    st.write(f"  - {alert}")
            else:
                st.write("No specific alerts.")

            # Financial Event Impact Analysis (Feature 4: Real Financial News Filtering)
            st.markdown("---") # Visual separator
            st.markdown("<h3 style='color: #4682B4;'>📰 Financial Event Impact Analysis</h3>", unsafe_allow_html=True)

            news_item_for_event_analysis = ""

            if combined_news_titles_only: # Use the list of titles for this
                news_item_for_event_analysis = combined_news_titles_only[0] # Use the first actual news title
                is_sample_news_for_event_analysis = False
            else:
                # No live news was fetched. Since we are in a block where historical_data is valid,
                # we can use a sample snippet for demonstration.
                is_sample_news_for_event_analysis = True
                news_item_for_event_analysis = f"""
                {ticker_symbol_processed} reported mixed Q2 results. While revenue saw a slight increase,
                net profit declined due to rising operational costs. The company announced a
                new strategic partnership aimed at expanding into new markets and is also
                exploring cost-cutting measures.
                """
            
            if news_item_for_event_analysis.strip():  # Ensure there's content to analyze
                if is_sample_news_for_event_analysis:
                    st.info("No live news fetched. Displaying event analysis with a sample news snippet.")
                st.markdown(f"**Analyzing News Snippet:** `{news_item_for_event_analysis}`")
                st.caption("Note: Event analysis is based on the news title/snippet. Full article content would provide deeper insights.") # Feature 4
                events = extract_financial_events(news_item_for_event_analysis)
                sentiment_for_event = analyze_sentiment(news_item_for_event_analysis)
                impact, event_alerts = assess_impact(events, sentiment_for_event)
                event_signal = generate_signal(impact)

                # Displaying the results of the event analysis
                col1, col2 = st.columns(2)
                with col1: # Feature 4
                    st.metric(label="Identified Events", value=(", ".join([e['type'] for e in events]) if events else "None"))
                    st.metric(label="Sentiment of this News", value=f"{sentiment_for_event:.2f}")
                with col2:
                    st.metric(label="Assessed Short-Term Impact", value=impact.get('short_term', 'N/A'))
                    st.metric(label="Event-based Signal", value=event_signal)
                
                if event_alerts:
                    with st.expander("Detailed Event Alerts from this News Item", expanded=False):
                        for alert in event_alerts:
                            st.write(f"• {alert}") # Use bullet points
            else:
                st.info("No news item available to perform financial event impact analysis for this stock.")

            st.markdown("---") # Visual separator (Feature 9: Clean, Interactive UI)
            # --- KEY ANALYSIS SUMMARY ---
            with st.container(border=True):
                st.subheader(f"KEY ANALYSIS SUMMARY FOR {ticker_symbol_processed}")
                st.metric(label="Final Recommendation", value=enhanced_recommendation)
                st.metric(label="Confidence Level", value=f"{enhanced_confidence}%")
                st.markdown("**Primary Reasons:**") # Use markdown for consistency
                st.write(enhanced_reason) # Write the reason
                if alerts: # Display critical alerts from enhanced analysis again for emphasis
                    st.warning("Important Alerts to Consider (from Enhanced Analysis):")
                    for alert in alerts:
                        st.write(f" - {alert}")

            # --- NEW FEATURE SECTIONS ---

            # 1. Chart Pattern Recognition (Feature 1: Chart Pattern Recognition)
            st.markdown("---")
            st.markdown("<h3 style='color: #4682B4;'>📈 Chart Pattern Insights</h3>", unsafe_allow_html=True)
            with st.expander("View Detected Patterns", expanded=True):
                double_pattern = detect_double_top_bottom(historical_data)
                sma_pattern = detect_sma_crossover_pattern(technical_indicators)
                
                if double_pattern:
                    st.success(f"**Detected:** {double_pattern} pattern.")
                    if double_pattern == "Double Top":
                        st.info("A Double Top is a bearish reversal pattern. Expect resistance near recent highs.")
                    elif double_pattern == "Double Bottom":
                        st.info("A Double Bottom is a bullish reversal pattern. Expect support near recent lows.")
                
                if sma_pattern:
                    st.success(f"**Detected:** {sma_pattern} pattern.")
                    if sma_pattern == "Golden Cross (Bullish)":
                        st.info("A Golden Cross (50-day SMA crosses above 200-day SMA) is a bullish signal.")
                    elif sma_pattern == "Death Cross (Bearish)":
                        st.info("A Death Cross (50-day SMA crosses below 200-day SMA) is a bearish signal.")
                
                if not double_pattern and not sma_pattern:
                    st.info("No significant chart patterns detected (Double Top/Bottom, Golden/Death Cross).")
                st.caption("Note: Pattern detection is simplified and may not capture all complex patterns.")

            # 2. Time-Based Forecasting (Feature 2: Time-Based Forecasting)
            st.markdown("---")
            st.markdown("<h3 style='color: #4682B4;'>⏳ Short-Term Forecast (Next 5 Days)</h3>", unsafe_allow_html=True)
            short_term_forecast = forecast_short_term_trend(historical_data, technical_indicators)
            if short_term_forecast['expected_range_low'] is not None and short_term_forecast['expected_range_high'] is not None:
                st.write(f"**Expected Range:** ${short_term_forecast['expected_range_low']:.2f} – ${short_term_forecast['expected_range_high']:.2f}")
                st.write(f"**Trend Bias:** {short_term_forecast['trend_bias']}")
            else:
                st.info("Could not generate short-term forecast (insufficient data for volatility/trend analysis).")
            st.caption("Forecast is based on recent price action and volatility (e.g., Bollinger Bands or ATR).")

            # 3. Dynamic Support & Resistance Levels (Feature 3: Dynamic Support & Resistance Levels)
            st.markdown("---")
            st.markdown("<h3 style='color: #4682B4;'>📊 Support & Resistance Levels</h3>", unsafe_allow_html=True)
            pivot_points = calculate_pivot_points(historical_data)
            if pivot_points:
                st.write(f"**Pivot Point (PP):** ${pivot_points['PP']:.2f}")
                st.write(f"**Resistance Levels:** R1: ${pivot_points['R1']:.2f} / R2: ${pivot_points['R2']:.2f} / R3: ${pivot_points['R3']:.2f}")
                st.write(f"**Support Levels:** S1: ${pivot_points['S1']:.2f} / S2: ${pivot_points['S2']:.2f} / S3: ${pivot_points['S3']:.2f}")
                # Placeholder for Breakout Probability
                st.write("**Breakout Probability:** N/A (requires advanced modeling)")
            else:
                st.info("Could not calculate Support & Resistance levels (insufficient data).")
            st.caption("Levels are based on Classic Pivot Point calculations from the previous trading day.")

            # 4. Realistic Risk–Reward Calculation (Feature 5: Realistic Risk–Reward Calculation)
            st.markdown("---")
            st.markdown("<h3 style='color: #4682B4;'>⚖️ Risk–Reward Calculator</h3>", unsafe_allow_html=True)
            st.write("Enter your trade parameters to calculate the Risk-Reward Ratio.")
            
            col_rr1, col_rr2, col_rr3 = st.columns(3)
            with col_rr1:
                rr_entry = st.number_input("Entry Price ($)", value=current_price, format="%.2f", key="rr_entry")
            with col_rr2:
                rr_stop_loss = st.number_input("Stop Loss ($)", value=current_price * 0.95, format="%.2f", key="rr_stop_loss")
            with col_rr3:
                rr_target = st.number_input("Target Price ($)", value=current_price * 1.10, format="%.2f", key="rr_target")
            
            if st.button("Calculate Risk-Reward", key="calc_rr_btn"):
                rr_result = calculate_risk_reward(rr_entry, rr_stop_loss, rr_target)
                if rr_result['ratio'] is not None:
                    st.success(f"**Risk–Reward Ratio:** 1:{rr_result['ratio']:.2f} → {rr_result['favorable_status']}")
                    st.write(f"Risk: ${rr_result['risk']:.2f}, Reward: ${rr_result['reward']:.2f}")
                else:
                    st.error(f"Error calculating Risk-Reward: {rr_result['favorable_status']}")
            st.caption("A ratio of 2:1 or higher is generally considered favorable.")

            # 5. Fundamentals Overview (Feature 8: Add Fundamentals Overview)
            st.markdown("---")
            st.markdown("<h3 style='color: #4682B4;'>📚 Fundamentals Overview</h3>", unsafe_allow_html=True)
            if company_fundamentals:
                col_fund1, col_fund2, col_fund3 = st.columns(3)
                with col_fund1:
                    st.metric("P/E Ratio (Trailing)", f"{company_fundamentals.get('trailingPE', 'N/A'):.2f}" if company_fundamentals.get('trailingPE') is not None else "N/A")
                    st.metric("EPS (Trailing)", f"{company_fundamentals.get('trailingEPS', 'N/A'):.2f}" if company_fundamentals.get('trailingEPS') is not None else "N/A")
                with col_fund2:
                    st.metric("YoY Revenue Growth", f"{company_fundamentals.get('revenueGrowth', 'N/A'):.2%}" if company_fundamentals.get('revenueGrowth') is not None else "N/A")
                    st.metric("YoY EPS Growth", f"{company_fundamentals.get('earningsGrowth', 'N/A'):.2%}" if company_fundamentals.get('earningsGrowth') is not None else "N/A")
                with col_fund3:
                    st.metric("Market Cap", f"${company_fundamentals.get('marketCap', 'N/A'):,.0f}" if company_fundamentals.get('marketCap') is not None else "N/A")
                    st.metric("Sector", company_fundamentals.get('sector', 'N/A'))
                
                with st.expander("More Fundamentals Details", expanded=False):
                    st.json(company_fundamentals) # Display full fundamentals JSON for advanced users
            else:
                st.info("Company fundamentals not available.")
            st.caption("Data provided by Yahoo Finance. Analyst Consensus requires external scraping.")

            # 6. Watchlist Feature (Conceptual UI) (Bonus Suggestion)
            st.markdown("---")
            st.markdown("<h3 style='color: #4682B4;'>⭐ Watchlist (Conceptual)</h3>", unsafe_allow_html=True)
            st.info("This is a conceptual demonstration. A real watchlist requires a backend database for persistence.")
            
            watchlist_tickers = ["AAPL", "MSFT", "GOOG", "TSLA"] # Mock watchlist
            st.write("Your Watchlist:")
            for ticker in watchlist_tickers:
                col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns([0.5, 1, 1, 1, 1])
                with col_w1:
                    st.write(f"**{ticker}**")
                with col_w2:
                    st.write("Trend: Bullish") # Mock data
                with col_w3:
                    st.write("RSI: 62") # Mock data
                with col_w4:
                    st.write("News Sentiment: +0.31") # Mock data
                with col_w5:
                    st.write("Confidence: 72% Buy") # Mock data
            st.caption("Watchlist data is mocked. Actual implementation requires user authentication and database integration.")

            # 7. Intelligent Alerts System (Conceptual) (Feature 7: Intelligent Alerts System)
            st.markdown("---")
            st.markdown("<h3 style='color: #4682B4;'>🔔 Intelligent Alerts System (Conceptual)</h3>", unsafe_allow_html=True)
            st.info("This feature requires a backend system (e.g., Django with Celery/Redis) to run background jobs and send notifications.")
            st.write("Example Alert Scenarios:")
            st.write("- Notify me when RSI < 50 + MACD turns bullish")
            st.write("- Breakout above $215")
            st.write("- Volume 30% above 10-day avg")
            st.caption("Users would define alert conditions, and the system would monitor them in the background.")

            # 8. Efficient Backend Architecture (Conceptual) (Feature 8: Efficient Backend Architecture)
            st.markdown("---")
            st.markdown("<h3 style='color: #4682B4;'>⚙️ Efficient Backend Architecture (Conceptual)</h3>", unsafe_allow_html=True)
            st.info("For scalability and speed, a robust backend architecture is crucial.")
            st.write("- **Asynchronous Data Fetching:** Use `aiohttp` or `FastAPI` for non-blocking API calls.")
            st.write("- **Caching:** Implement Redis for caching frequently accessed data (e.g., price charts, sentiment).")
            st.write("- **Scheduled Jobs:** Use `cron jobs` or `Celery Beat` to refresh indicators (SMA, RSI, MACD) daily.")
            st.caption("These are architectural considerations for a production-ready application.")