import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import streamlit as st
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Volatility", page_icon="📈", layout="wide")

st.title("🔍 Stock Volatility Forecaster")
st.markdown("**GARCH(1,1) analysis with robust data pipeline**")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
n_obs = st.sidebar.slider("Observations", 100, 2000, 1000)

if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    if 'results' in st.session_state:
        del st.session_state.results
    st.rerun()

# MAIN PIPELINE
if st.button("🚀 RUN ANALYSIS", type="primary"):
    st.markdown("---")
    
    try:
        # STEP 1: Robust data download
        with st.status("📥 Step 1: Download raw data", expanded=True):
            @st.cache_data
            def get_raw_data(_ticker):
                stock = yf.Ticker(_ticker)
                # Try multiple strategies for yfinance reliability
                strategies = [
                    lambda: stock.history(start="2023-01-01", end=pd.Timestamp.now().strftime('%Y-%m-%d')),
                    lambda: yf.download(_ticker, period="2y", progress=False),
                    lambda: yf.download(_ticker, period="1y", progress=False),
                ]
                
                for i, strategy in enumerate(strategies, 1):
                    try:
                        data = strategy()
                        if not data.empty and len(data) > 100:
                            st.info(f"✅ Strategy {i} succeeded: {len(data)} rows")
                            return data
                    except:
                        continue
                
                st.error("All download strategies failed")
                return pd.DataFrame()
            
            raw_data = get_raw_data(ticker)
            if raw_data.empty:
                st.error("❌ No data from yfinance. Try: `pip install --upgrade yfinance`")
                st.stop()
                
            st.metric("Raw data rows", len(raw_data))
            st.dataframe(raw_data.tail())

        # STEP 2: Clean prices
        with st.status("🔧 Step 2: Clean prices", expanded=True):
            prices = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            prices.columns = ['open', 'high', 'low', 'close', 'volume']
            prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
            prices = prices.dropna()
            st.metric("Clean prices rows", len(prices))
            st.line_chart(prices['close'].tail(100))

        # STEP 3: Calculate returns
        with st.status("📊 Step 3: Calculate returns", expanded=True):
            returns = prices['close'].pct_change() * 100
            returns_clean = returns.dropna()
            returns_finite = returns_clean[np.isfinite(returns_clean)]
            
            if len(returns_finite) < 100:
                st.error(f"❌ Insufficient valid returns: {len(returns_finite)}. Need >100.")
                st.stop()
            
            final_returns = returns_finite.tail(n_obs).values  # numpy array for arch
            st.metric("Valid returns for GARCH", len(final_returns))
            st.line_chart(returns_finite.tail(100))
            st.info(f"Returns stats: mean={returns_finite.mean():.2f}%, std={returns_finite.std():.2f}%")

        # STEP 4: Fit GARCH model
        with st.status("🔄 Step 4: Fit GARCH(1,1)", expanded=True):
            model = arch_model(final_returns, p=1, q=1, rescale=False)
            fitted = model.fit(disp="off", show_warning=False)
            
            # Volatility metrics
            daily_vol = returns_finite.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Daily Vol", f"{daily_vol:.2f}%")
            col2.metric("📊 Annual Vol", f"{annual_vol:.2f}%")
            col3.metric("⚙️ Model AIC", f"{fitted.aic:.1f}")
            
            # Store results safely
            st.session_state.results = {
                'model': fitted, 
                'returns': pd.Series(final_returns, index=returns_finite.tail(n_obs).index),
                'prices': prices,
                'ticker': ticker,
                'vol_metrics': {
                    'daily': daily_vol,
                    'annual': annual_vol
                }
            }
            st.success("✅ GARCH model fitted successfully!")
            st.code(fitted.summary().tables[1], language='text')  # Model params

    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)

# RESULTS VISUALIZATION
if 'results' in st.session_state:
    st.markdown("---")
    st.header("📊 Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Returns")
        fig1 = px.line(
            st.session_state.results['returns'].tail(200), 
            title="Last 200 Returns (%)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Standardized Residuals")
        residuals = st.session_state.results['model'].std_resid.tail(200)
        fig2 = px.line(residuals, title="GARCH Standardized Residuals")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Conditional volatility forecast
    st.subheader("🔮 1-Step Ahead Volatility Forecast")
    try:
        forecast = st.session_state.results['model'].forecast(horizon=1)
        vol_forecast = np.sqrt(forecast.variance.iloc[-1].values[0]) * 100
        st.metric("Next Day Vol Forecast", f"{vol_forecast:.2f}%")
    except:
        st.info("Forecast unavailable")

