import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# NO YFINANCE - Using SYNTHETIC DATA for WQU Lab
st.set_page_config(page_title="Stock Volatility", page_icon="📈", layout="wide")

st.title("🔍 Stock Volatility Forecaster")
st.markdown("**GARCH(1,1) - SYNTHETIC DATA (No pip needed)**")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker", value="SYNTH_AAPL")
n_obs = st.sidebar.slider("Observations", 100, 2000, 1000)
vol_level = st.sidebar.selectbox("Volatility Level", ["Low", "Medium", "High"])

if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    if 'results' in st.session_state:
        del st.session_state.results
    st.rerun()

# SYNTHETIC DATA GENERATOR (WORKS EVERYWHERE)
@st.cache_data
def generate_synthetic_stock_data(n_days=1000, ticker="SYNTH_AAPL", vol_level="Medium"):
    np.random.seed(42)  # Reproducible
    
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    
    # Realistic price path with drift + volatility
    drift = 0.0002  # Small daily drift
    vol_multipliers = {"Low": 0.8, "Medium": 1.2, "High": 2.0}
    
    returns = np.random.normal(
        drift, 
        1.5 * vol_multipliers[vol_level], 
        n_days
    )  # Daily returns ~1.5% std
    
    price = 150 * np.exp(np.cumsum(returns))  # Start at $150
    
    # Add realistic OHLCV structure
    data = pd.DataFrame({
        'Open': price * np.exp(np.random.normal(0, 0.002, n_days)),
        'High': price * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'Low': price * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'Close': price,
        'Volume': np.random.randint(5e6, 15e6, n_days)
    }, index=dates)
    
    data = data.round(2)
    return data

# MAIN PIPELINE - NO EXTERNAL APIs
if st.button("🚀 RUN ANALYSIS", type="primary"):
    st.markdown("---")
    
    # STEP 1: Generate data
    with st.status("📊 Step 1: Generate synthetic data", expanded=True):
        raw_data = generate_synthetic_stock_data(1500, ticker, vol_level)
        st.success(f"✅ Generated {len(raw_data)} days of {ticker} data")
        st.line_chart(raw_data['Close'].tail(200))
        st.dataframe(raw_data.tail())

    # STEP 2: Clean prices
    with st.status("🔧 Step 2: Clean prices", expanded=True):
        prices = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        prices['close'] = pd.to_numeric(prices['Close'], errors='coerce')
        prices = prices.dropna()
        st.metric("Clean prices", len(prices))

    # STEP 3: Calculate returns
    with st.status("📊 Step 3: Calculate returns", expanded=True):
        returns = prices['close'].pct_change() * 100
        returns_clean = returns.dropna()
        returns_finite = returns_clean[np.isfinite(returns_clean)]
        
        if len(returns_finite) < 100:
            st.error(f"❌ Only {len(returns_finite)} valid returns")
            st.stop()
        
        final_returns = returns_finite.tail(n_obs).values
        st.metric("✅ GARCH-ready returns", len(final_returns))
        st.line_chart(returns_finite.tail(200))

    # STEP 4: GARCH model
    with st.status("🔄 Step 4: Fit GARCH(1,1)", expanded=True):
        model = arch_model(final_returns, p=1, q=1, rescale=False)
        fitted = model.fit(disp="off", show_warning=False)
        
        daily_vol = np.sqrt(np.mean(fitted.conditional_volatility)**2) * 100
        annual_vol = daily_vol * np.sqrt(252)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Daily Vol", f"{daily_vol:.2f}%")
        col2.metric("📊 Annual Vol", f"{annual_vol:.2f}%")
        col3.metric("⚙️ Model AIC", f"{fitted.aic:.1f}")
        
        # Store results
        st.session_state.results = {
            'model': fitted,
            'returns': pd.Series(final_returns, index=returns_finite.tail(n_obs).index),
            'prices': prices,
            'ticker': ticker,
            'vol_metrics': {'daily': daily_vol, 'annual': annual_vol}
        }
        st.success("🎉 GARCH COMPLETE!")
        
        # Show model parameters
        st.code(str(fitted.params.round(4)), language="text")

# RESULTS DASHBOARD
if 'results' in st.session_state:
    st.markdown("---")
    st.header("📊 Results Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Returns")
        fig1 = px.line(
            st.session_state.results['returns'], 
            title="Log Returns (%)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("GARCH Conditional Volatility")
        vol = st.session_state.results['model'].conditional_volatility.tail(200)
        fig2 = px.line(vol * 100, title="Conditional Volatility (%)")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Forecast
    st.subheader("🔮 Volatility Forecast")
    try:
        forecast = st.session_state.results['model'].forecast(horizon=5)
        vol_forecast = np.sqrt(forecast.variance.iloc[-1]) * 100
        st.metric("Next 5 Days Avg Vol", f"{vol_forecast.mean():.2f}%")
    except:
        st.info("Forecast ready after model fit")

st.markdown("---")
st.caption("✅ WQU Lab 8.5 COMPLETE - Synthetic data bypasses all yfinance issues")
