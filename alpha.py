import os
import sqlite3
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from arch import arch_model

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
ALPHA_API_KEY = "4ER5EIUSHD54K72I"
DB_NAME = "mtn_volatility.db"
DEFAULT_TICKER = "MTNOY"

# Page config
st.set_page_config(
    page_title="MTN Volatility Forecaster",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------------------
# Database functions
# -------------------------------------------------------------------
@st.cache_data
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
# Data functions (same as notebook)
# -------------------------------------------------------------------
def download_stock_data(ticker, output_size="full"):
    url = (
        f"https://learn-api.wqu.edu/1/data-services/alpha-vantage/query?"
        f"function=TIME_SERIES_DAILY&"
        f"symbol={ticker}&"
        f"outputsize={output_size}&"
        f"datatype=json&"
        f"apikey={ALPHA_API_KEY}"
    )
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(data, orient="index", dtype=float)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.columns = [c.split(". ")[1] for c in df.columns]
    return df

def wrangle_returns(df, n_observations=2500):
    df = df.sort_index()
    df["return"] = df["close"].pct_change() * 100
    y = df["return"].dropna()
    return y.iloc[-n_observations:]

# -------------------------------------------------------------------
# GARCH model function
# -------------------------------------------------------------------
def fit_garch_model(y, p=1, q=1):
    model = arch_model(y, p=p, q=q, rescale=False)
    fitted = model.fit(disp="off")
    return fitted

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
st.title("📈 MTN Stock Volatility Forecaster")
st.markdown("**How jumpy is MTN Group's stock price? This app analyzes historical prices to estimate daily and yearly risk using a GARCH model.**")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Stock ticker", value=DEFAULT_TICKER)
use_new_data = st.sidebar.checkbox("Download fresh data", value=True)
n_obs = st.sidebar.slider("Observations", 500, 3000, 2500)
p_order = st.sidebar.slider("GARCH p", 0, 3, 1)
q_order = st.sidebar.slider("GARCH q", 0, 3, 1)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Analysis", "⚙️ Model Details"])

with tab1:
    st.markdown("""
    ### What this app does (for non-experts):
    - **Looks at past stock prices** of MTN Group (or any ticker you choose)
    - **Calculates daily price changes** (returns) 
    - **Measures how "jumpy" the price is** using volatility
    - **Uses GARCH model** to understand how risk changes over time
    - **Shows daily + yearly risk estimates**
    
    **Daily volatility** = How much the stock typically moves in 1 day  
    **Annual volatility** = How much it might move over a full year
    """)
    
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing stock volatility..."):
            conn = get_connection()
            
            if use_new_data:
                with st.status("📥 Downloading fresh data...", expanded=False):
                    prices = download_stock_data(ticker)
                    insert_table(conn, ticker, prices)
            
            with st.status("🔄 Fitting GARCH model...", expanded=False):
                prices = read_table(conn, ticker)
                returns = wrangle_returns(prices, n_obs)
                model = fit_garch_model(returns, p_order, q_order)
                
                # Key metrics
                daily_vol = returns.std()
                annual_vol = daily_vol * np.sqrt(252)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Daily Volatility (%)", f"{daily_vol:.2f}%")
                with col2:
                    st.metric("Annual Volatility (%)", f"{annual_vol:.2f}%")
                with col3:
                    st.metric("Model AIC", f"{model.aic:.1f}")
                with col4:
                    st.metric("Model BIC", f"{model.bic:.1f}")

with tab2:
    if 'model' in locals():
        st.subheader("Price Returns")
        fig1 = px.line(returns.tail(200), title="Recent Daily Returns (%)")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Standardized Residuals")
        residuals = model.std_resid
        fig2 = px.line(residuals.tail(200), title="GARCH Model Residuals")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Model Summary")
    st.text(model.summary().as_text())

# Footer
st.markdown("---")
st.markdown("*Built from WQU Applied Data Science Lab assignment 8.5*")
