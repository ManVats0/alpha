import os
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import streamlit as st
from arch import arch_model

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DB_NAME = "mtn_volatility.db"
DEFAULT_TICKER = "AAPL"

st.set_page_config(
    page_title="Stock Volatility Forecaster",
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
    df = pd.read_sql(query, connection, parse_dates=["date"], index_col="date")
    return df

# -------------------------------------------------------------------
# Yahoo Finance + CLEANING (FIXED)
# -------------------------------------------------------------------
@st.cache_data
def download_stock_data_cached(_ticker):
    ticker = yf.Ticker(_ticker)
    df = ticker.history(period="5y", interval="1d")
    
    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open', 
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df = df.set_index('date')
    
    # CRITICAL: Convert to numeric + drop NaNs
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[numeric_cols].dropna()
    return df

@st.cache_data
def wrangle_returns_cached(_prices, _n_obs):
    df = _prices.sort_index().copy()
    
    # CRITICAL CLEANING: Ensure close is numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    
    # Calculate returns
    df["return"] = df["close"].pct_change() * 100
    
    # Remove NaNs and infinities
    y = df["return"].dropna()
    y = y[np.isfinite(y)]  # Remove inf, -inf, NaN
    
    return y.iloc[-_n_obs:].astype(float)

# -------------------------------------------------------------------
# GARCH model (with validation)
# -------------------------------------------------------------------
def fit_garch_model(y, p=1, q=1):
    # Final validation
    y = y.dropna().astype(float)
    y = y[np.isfinite(y)]
    
    if len(y) < 100:
        raise ValueError(f"Need 100+ observations. Got {len(y)}")
    
    model = arch_model(y, p=p, q=q, rescale=False)
    fitted = model.fit(disp="off")
    return fitted

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
st.title("📈 Stock Volatility Forecaster")
st.markdown("**GARCH(1,1) volatility analysis - Yahoo Finance data**")

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
    ### How it works:
    1. Downloads **5y stock data** from Yahoo Finance
    2. Computes **daily returns** (% price changes) 
    3. Cleans **NaNs/infinities**
    4. Fits **GARCH(p,q)** model
    5. Shows **volatility metrics**
    """)
    
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing volatility..."):
            conn = get_connection()
            try:
                if use_new_data:
                    with st.status("📥 Downloading data...", expanded=False):
                        prices = download_stock_data_cached(ticker)
                        insert_table(conn, ticker, prices)
                        st.success(f"✅ {len(prices):,} clean days of {ticker}")
                
                with st.status("📊 Computing returns...", expanded=False):
                    prices = read_table(conn, ticker)
                
                with st.status("🔄 Fitting GARCH model...", expanded=False):
                    returns = wrangle_returns_cached(prices, n_obs)
                    st.info(f"✅ Using {len(returns):,} returns for modeling")
                    
                    model = fit_garch_model(returns, p_order, q_order)
                    
                    daily_vol = returns.std()
                    annual_vol = daily_vol * np.sqrt(252)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Daily Volatility", f"{daily_vol:.2f}%")
                    with col2:
                        st.metric("Annual Volatility", f"{annual_vol:.2f}%")
                    with col3:
                        st.metric("Model AIC", f"{model.aic:.1f}")
                    with col4:
                        st.metric("Model BIC", f"{model.bic:.1f}")
                    
                    st.session_state.model = model
                    st.session_state.returns = returns
                    st.session_state.prices = prices
                    st.session_state.ticker = ticker
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try AAPL, MSFT, TSLA")
            finally:
                conn.close()

with tab2:
    if 'model' in st.session_state:
        st.subheader(f"Returns - {st.session_state.ticker}")
        fig1 = px.line(st.session_state.returns.tail(200), 
                      title="Daily Returns (%)")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("GARCH Residuals")
        residuals = st.session_state.model.std_resid.tail(200)
        fig2 = px.line(residuals, title="Standardized Residuals")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if 'model' in st.session_state:
        st.subheader("GARCH Model Summary")
        st.text(st.session_state.model.summary().as_text())

st.markdown("---")
st.markdown("*WQU Lab 8.5 → Production GARCH App* [file:1]")
