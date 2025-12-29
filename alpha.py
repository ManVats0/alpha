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
st.markdown("**GARCH(1,1) analysis with yfinance FIX**")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
n_obs = st.sidebar.slider("Observations", 100, 2000, 1000)

if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    if 'results' in st.session_state:
        del st.session_state.results
    st.rerun()

# YFINANCE EMERGENCY FIX
st.sidebar.markdown("---")
st.sidebar.code("""
# If still failing, run this in terminal:
pip install yfinance==0.2.40 --force-reinstall
""")

# MAIN PIPELINE with ULTIMATE yfinance fix
if st.button("🚀 RUN ANALYSIS", type="primary"):
    st.markdown("---")
    
    # STEP 0: Test yfinance FIRST
    with st.status("🧪 Step 0: Test yfinance connection", expanded=True):
        try:
            # Direct download test (bypasses Ticker class issues)
            test_data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            st.write(f"✅ yfinance test: {len(test_data)} rows")
            st.dataframe(test_data.tail())
            
            if test_data.empty:
                st.error("❌ yfinance completely broken. Try:")
                st.code("""
pip uninstall yfinance -y
pip install yfinance==0.2.40 --no-cache-dir
                """)
                st.stop()
                
        except Exception as e:
            st.error(f"yfinance test failed: {e}")
            st.stop()
    
    try:
        # STEP 1: ULTIMATE data download
        with st.status("📥 Step 1: Robust data download", expanded=True):
            @st.cache_data(ttl=300)  # 5min cache
            def get_robust_data(_ticker):
                # Method 1: Direct download with explicit params
                data1 = yf.download(
                    _ticker, 
                    start="2024-01-01", 
                    end=pd.Timestamp.now().strftime('%Y-%m-%d'),
                    progress=False, 
                    auto_adjust=True,
                    prepost=False,
                    actions=False
                )
                
                if not data1.empty and len(data1) > 200:
                    return data1
                
                # Method 2: Ticker with interval
                stock = yf.Ticker(_ticker)
                data2 = stock.history(period="2y", interval="1d")
                if not data2.empty and len(data2) > 200:
                    return data2
                
                # Method 3: Max period
                data3 = yf.download(_ticker, period="max", progress=False)
                return data3.tail(800)  # Last 800 days max
            
            raw_data = get_robust_data(ticker)
            
            if raw_data.empty or len(raw_data) < 50:
                st.error(f"❌ Still no data for {ticker}. Market closed? Try SPY or ^GSPC")
                st.stop()
            
            st.success(f"✅ Got {len(raw_data)} rows!")
            st.line_chart(raw_data['Close'].tail(100))
            st.dataframe(raw_data[['Open','High','Low','Close','Volume']].tail())

        # STEP 2: Clean prices
        with st.status("🔧 Step 2: Clean prices", expanded=True):
            if 'Close' in raw_data.columns:
                prices = raw_data[['Open','High','Low','Close','Volume']].copy()
            else:
                prices = raw_data.copy()
                prices = prices.rename(columns={'Adj Close': 'Close'})
            
            prices['close'] = pd.to_numeric(prices['Close'], errors='coerce')
            prices = prices.dropna(subset=['close'])
            st.metric("Clean prices", len(prices))

        # STEP 3: Returns pipeline
        with st.status("📊 Step 3: Calculate returns", expanded=True):
            returns = prices['close'].pct_change() * 100
            returns_clean = returns.dropna()
            returns_finite = returns_clean[returns_clean.isfinite()]
            
            if len(returns_finite) < 100:
                st.error(f"❌ Only {len(returns_finite)} valid returns. Need 100+")
                st.stop()
            
            final_returns = returns_finite.tail(n_obs).values
            st.metric("✅ GARCH-ready returns", len(final_returns))
            st.line_chart(returns_finite.tail(100))

        # STEP 4: GARCH model
        with st.status("🔄 Step 4: GARCH(1,1)", expanded=True):
            model = arch_model(final_returns, p=1, q=1, rescale=False)
            fitted = model.fit(disp="off", show_warning=False)
            
            daily_vol = np.sqrt(np.mean(fitted.conditional_volatility)**2) * 100
            annual_vol = daily_vol * np.sqrt(252)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Daily Vol", f"{daily_vol:.2f}%")
            col2.metric("Annual Vol", f"{annual_vol:.2f}%")
            col3.metric("Model AIC", f"{fitted.aic:.1f}")
            
            st.session_state.results = {
                'model': fitted,
                'returns': pd.Series(final_returns, index=returns_finite.tail(n_obs).index),
                'prices': prices,
                'ticker': ticker
            }
            st.success("🎉 GARCH COMPLETE!")
            st.text(fitted.params)

    except Exception as e:
        st.error(f"❌ {str(e)}")
        st.exception(e)

# RESULTS
if 'results' in st.session_state:
    st.markdown("---")
    st.header("📊 Results Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(st.session_state.results['returns'], title="Returns")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        try:
            residuals = st.session_state.results['model'].std_resid
            fig2 = px.line(residuals.tail(200), title="Std Residuals")
            st.plotly_chart(fig2, use_container_width=True)
        except:
            st.info("Residuals unavailable")

st.markdown("---")
st.caption("🔧 Fixed: Triple yfinance fallback + direct download [WQU Lab 8.5]")
