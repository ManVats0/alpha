import os
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from arch import arch_model

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DB_NAME = "mtn_volatility.db"
DEFAULT_TICKER = "MTNOY"

st.set_page_config(
    page_title="MTN Volatility Forecaster",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------------------
# Database functions
# -------------------------------------------------------------------
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def insert_table(connection, table_name, df):
    df.to_sql(table_name, connection, if_exists="replace", index=True)
    connection.commit()

def read_table(connection, table_name):
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, connection, parse_dates=["Date"], index_col="Date")
    return df

# -------------------------------------------------------------------
# FREE Yahoo Finance (no API key needed!)
# -------------------------------------------------------------------
@st.cache_data
def download_stock_data_cached(_ticker):
    """Download from Yahoo Finance - works everywhere"""
    ticker = yf.Ticker(_ticker)
    df = ticker.history(period="5y", interval="1d")  # 5 years data
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index.name = 'date'
    return df

@st.cache_data
def wrangle_returns_cached(_prices, _n_obs):
    df = _prices.sort_index()
    df["return"] = df["close"].pct_change() * 100
    y = df["return"].dropna()
    return y.iloc[-_n_obs:]

# -------------------------------------------------------------------
# GARCH model
# -------------------------------------------------------------------
def fit_garch_model(y, p=1, q=1):
    model = arch_model(y, p=p, q=q, rescale=False)
    fitted = model.fit(disp="off")
    return fitted

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
st.title("📈 Stock Volatility Forecaster")
st.markdown("**How jumpy is any stock's price? Uses GARCH model on Yahoo Finance data (no API key needed).**")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Stock ticker", value=DEFAULT_TICKER)
use_new_data = st.sidebar.checkbox("Download fresh data", value=True)
n_obs = st.sidebar.slider("Observations", 500, 3000, 2500)
p_order = st.sidebar.slider("GARCH p", 0, 3, 1)
q_order = st.sidebar.slider("GARCH q", 0, 3, 1)

if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Analysis", "⚙️ Model Details"])

with tab1:
    st.markdown("""
    ### What this app does:
    - **Downloads FREE stock data** from Yahoo Finance (works for ANY ticker)
    - **Calculates daily % changes** (returns) 
    - **Fits GARCH model** to measure changing volatility
    - **Shows daily + annual risk** estimates
    """)
    
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing volatility..."):
            conn = get_connection()
            try:
                if use_new_data:
                    with st.status("📥 Downloading from Yahoo...", expanded=False):
                        prices = download_stock_data_cached(ticker)
                        insert_table(conn, ticker, prices)
                        st.success(f"✅ Got {len(prices):,} days of {ticker}")
                
                with st.status("📊 Processing...", expanded=False):
                    prices = read_table(conn, ticker)
                
                with st.status("🔄 Fitting GARCH...", expanded=False):
                    returns = wrangle_returns_cached(prices, n_obs)
                    model = fit_garch_model(returns, p_order, q_order)
                    
                    daily_vol = returns.std()
                    annual_vol = daily_vol * np.sqrt(252)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Daily Volatility", f"{daily_vol:.2f}%")
                    with col2:
                        st.metric("Annual Volatility", f"{annual_vol:.2f}%")
                    with col3:
                        st.metric("AIC", f"{model.aic:.1f}")
                    with col4:
                        st.metric("BIC", f"{model.bic:.1f}")
                    
                    st.session_state.model = model
                    st.session_state.returns = returns
                    st.session_state.prices = prices
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try AAPL, MSFT, or TSLA - Yahoo tickers work best")
            finally:
                conn.close()

with tab2:
    if 'model' in st.session_state:
        st.subheader("Recent Returns")
        fig1 = px.line(st.session_state.returns.tail(200), 
                      title="Daily Returns (%)")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("GARCH Residuals")
        residuals = st.session_state.model.std_resid
        fig2 = px.line(residuals.tail(200), title="Model Residuals")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if 'model' in st.session_state:
        st.subheader("GARCH Model Summary")
        st.text(st.session_state.model.summary().as_text())

st.markdown("---")
st.markdown("*WQU Lab 8.5 → Yahoo Finance + GARCH*")
