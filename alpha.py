import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import streamlit as st
from arch import arch_model

st.set_page_config(page_title="Stock Volatility", page_icon="📈", layout="wide")

st.title("🔍 DEBUG: Stock Volatility Forecaster")
st.markdown("**Step-by-step GARCH analysis**")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
n_obs = st.sidebar.slider("Observations", 100, 2000, 1000)

if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# MAIN DEBUG PIPELINE
if st.button("🔍 RUN FULL DEBUG", type="primary"):
    st.markdown("---")
    
    try:
        # STEP 1: Raw download
        with st.status("📥 Step 1: Download raw data", expanded=True):
            @st.cache_data
            def get_raw_data(_ticker):
                stock = yf.Ticker(_ticker)
                data = stock.history(period="2y")  # Smaller period for debug
                return data
            
            raw_data = get_raw_data(ticker)
            st.write(f"**Raw data shape:** {raw_data.shape}")
            st.write("**Raw columns:**", raw_data.columns.tolist())
            st.dataframe(raw_data.head())
        
        # STEP 2: Clean prices
        with st.status("🔧 Step 2: Clean prices", expanded=True):
            prices = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            prices.columns = ['open', 'high', 'low', 'close', 'volume']
            prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
            prices = prices.dropna()
            st.write(f"**Clean prices shape:** {prices.shape}")
            st.line_chart(prices['close'].tail(100))
        
        # STEP 3: Calculate returns
        with st.status("📊 Step 3: Calculate returns", expanded=True):
            returns = prices['close'].pct_change() * 100
            st.write(f"**Raw returns shape:** {returns.shape}")
            st.line_chart(returns.tail(100))
            
            returns_clean = returns.dropna()
            st.write(f"**Dropna returns:** {len(returns_clean)}")
            
            returns_finite = returns_clean[np.isfinite(returns_clean)]
            st.write(f"**Finite returns:** {len(returns_finite)}")
            
            final_returns = returns_finite.tail(n_obs)
            st.write(f"**FINAL returns for GARCH:** {len(final_returns)}")
            st.write("**Sample:**", final_returns.tail().tolist())
        
        # STEP 4: GARCH if we have data
        if len(final_returns) > 100:
            with st.status("🔄 Step 4: Fit GARCH", expanded=True):
                model = arch_model(final_returns, p=1, q=1, rescale=False)
                fitted = model.fit(disp="off")
                
                daily_vol = final_returns.std()
                annual_vol = daily_vol * np.sqrt(252)
                
                col1, col2 = st.columns(2)
                col1.metric("Daily Vol", f"{daily_vol:.2f}%")
                col2.metric("Annual Vol", f"{annual_vol:.2f}%")
                
                st.session_state.results = {
                    'model': fitted, 
                    'returns': final_returns,
                    'prices': prices,
                    'ticker': ticker
                }
                st.success("✅ GARCH model fitted!")
        else:
            st.error(f"❌ Not enough data: {len(final_returns)} returns")
    
    except Exception as e:
        st.error(f"❌ {e}")
        st.exception(e)

# Results
if 'results' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(st.session_state.results['returns'].tail(200))
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        residuals = st.session_state.results['model'].std_resid.tail(200)
        fig2 = px.line(residuals)
        st.plotly_chart(fig2, use_container_width=True)
