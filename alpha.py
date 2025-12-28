import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import streamlit as st
from arch import arch_model

st.set_page_config(page_title="Stock Volatility", page_icon="📈", layout="wide")

st.title("📈 Stock Volatility Forecaster")
st.markdown("**GARCH(1,1) analysis - instant results, no database**")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
n_obs = st.sidebar.slider("Observations", 100, 3000, 1000)
p_order = st.sidebar.slider("GARCH p", 0, 3, 1)
q_order = st.sidebar.slider("GARCH q", 0, 3, 1)

# Clear cache
if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# Main analysis
if st.button("🚀 Analyze Volatility", type="primary"):
    with st.spinner("Processing..."):
        try:
            # 1. Download data
            @st.cache_data
            def get_data(_ticker):
                ticker_obj = yf.Ticker(_ticker)
                df = ticker_obj.history(period="5y")
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.dropna()
                return df
            
            prices = get_data(ticker)
            st.success(f"✅ Downloaded {len(prices):,} days of {ticker}")
            
            # 2. Calculate returns
            returns = prices['close'].pct_change() * 100
            returns = returns.dropna()
            returns = returns[np.isfinite(returns)].tail(n_obs)
            
            st.info(f"✅ {len(returns):,} valid returns ready")
            
            # 3. Fit GARCH
            model = arch_model(returns, p=p_order, q=q_order, rescale=False)
            fitted = model.fit(disp="off")
            
            # 4. Metrics
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Daily Volatility", f"{daily_vol:.2f}%")
            col2.metric("Annual Volatility", f"{annual_vol:.2f}%")
            col3.metric("AIC", f"{fitted.aic:.1f}")
            col4.metric("BIC", f"{fitted.bic:.1f}")
            
            # Store for plots
            st.session_state.results = {
                'model': fitted, 
                'returns': returns, 
                'prices': prices,
                'ticker': ticker
            }
            
        except Exception as e:
            st.error(f"❌ {e}")
            st.info("Try AAPL, MSFT, TSLA")

# Plots tab
if 'results' in st.session_state:
    tab1, tab2 = st.tabs(["📊 Charts", "⚙️ Model"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(
                st.session_state.results['returns'].tail(200), 
                title="Daily Returns (%)"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            residuals = st.session_state.results['model'].std_resid.tail(200)
            fig2 = px.line(residuals, title="GARCH Residuals")
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("GARCH Model Summary")
        st.text(st.session_state.results['model'].summary().as_text())

# Instructions
with st.expander("ℹ️ What this does (for non-experts)"):
    st.markdown("""
    **Simple explanation:**
    - Stock prices go **up and down daily**
    - **Volatility** = how much they jump around
    - **GARCH model** learns patterns in these jumps
    - **Daily volatility** = typical 1-day move
    - **Annual volatility** = expected yearly swings
    
    **Example:** 2% daily volatility = stock typically moves ±2% per day
    """)
