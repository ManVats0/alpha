import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Volatility", page_icon="📈", layout="wide")

st.title("🔍 Stock Volatility Forecaster")
st.markdown("**GARCH(1,1) - 100% OFFLINE (No yfinance needed)**")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Name", value="SYNTH_AAPL")
n_obs = st.sidebar.slider("Observations", 100, 2000, 1000)
vol_level = st.sidebar.selectbox("Volatility", ["Low", "Medium", "High"])

if st.sidebar.button("🔄 Reset"):
    st.cache_data.clear()
    if 'results' in st.session_state:
        del st.session_state.results
    st.rerun()

# PURE OFFLINE SYNTHETIC DATA GENERATOR
@st.cache_data
def generate_realistic_stock_data(_n_days=1500, _vol="Medium"):
    np.random.seed(42 + hash(_vol))  # Unique seed per volatility
    
    dates = pd.date_range(start="2023-01-01", periods=_n_days, freq="B")
    
    # GARCH-like volatility clustering simulation
    vol_multipliers = {"Low": 0.8, "Medium": 1.2, "High": 1.8}
    vol_factor = vol_multipliers[_vol]
    
    # Generate returns with volatility clustering
    returns = []
    current_vol = 1.2
    for i in range(_n_days):
        # Volatility persistence (GARCH-like)
        current_vol = 0.1 + 0.8 * current_vol + 0.1 * abs(np.random.normal(0, 1))
        ret = np.random.normal(0.0002, current_vol * vol_factor * 0.015)
        returns.append(ret)
    
    log_returns = np.array(returns)
    price = 150 * np.exp(np.cumsum(log_returns))
    
    # Full OHLCV structure
    data = pd.DataFrame(index=dates, data={
        'Open':   price * (1 + np.random.normal(0, 0.002, _n_days)),
        'High':   price * (1 + np.abs(np.random.normal(0.008, 0.005, _n_days))),
        'Low':    price * (1 - np.abs(np.random.normal(0.008, 0.005, _n_days))),
        'Close':  price,
        'Volume': np.random.randint(8_000_000, 20_000_000, _n_days)
    }).round(2)
    
    return data.dropna()

# MAIN PIPELINE - WORKS IN ANY ENVIRONMENT
if st.button("🚀 GENERATE & ANALYZE", type="primary"):
    st.markdown("---")
    
    with st.spinner("Generating realistic stock data..."):
        # STEP 1: Generate data
        raw_data = generate_realistic_stock_data(1500, vol_level)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Days Generated", len(raw_data))
            st.metric("💰 Final Price", f"${raw_data['Close'].iloc[-1]:,.2f}")
        with col2:
            st.metric("📈 Total Return", f"{((raw_data['Close'].iloc[-1]/raw_data['Close'].iloc[0])-1)*100:.1f}%")
            st.metric("📊 Daily Vol", f"{raw_data['Close'].pct_change().std()*100:.2f}%")
        
        st.line_chart(raw_data['Close'], use_container_width=True)
        st.dataframe(raw_data.tail(5), use_container_width=True)

    # STEP 2: Process returns
    with st.status("📊 Processing returns...", expanded=True):
        prices = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        returns = prices['Close'].pct_change() * 100
        returns_clean = returns.dropna()
        returns_final = returns_clean[np.isfinite(returns_clean)].tail(n_obs)
        
        st.metric("✅ Valid Returns", len(returns_final))
        st.line_chart(returns_final.tail(200), use_container_width=True)

    # STEP 3: GARCH Model
    with st.status("🔄 Fitting GARCH(1,1)...", expanded=True):
        model = arch_model(returns_final.values, p=1, q=1, rescale=False)
        fitted_model = model.fit(disp="off", show_warning=False)
        
        # Volatility metrics
        cond_vol = fitted_model.conditional_volatility
        daily_vol = np.sqrt(np.mean(cond_vol**2)) * 100
        annual_vol = daily_vol * np.sqrt(252)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Daily Vol", f"{daily_vol:.2f}%")
        col2.metric("📊 Annual Vol", f"{annual_vol:.2f}%")
        col3.metric("⚙️ Model AIC", f"{fitted_model.aic:.1f}")
        
        # Save results
        st.session_state.results = {
            'model': fitted_model,
            'returns': returns_final,
            'prices': prices,
            'vol': cond_vol
        }
        
        st.success("✅ GARCH model fitted!")
        with st.expander("Model Parameters"):
            st.code(str(fitted_model.params.round(4)))

# RESULTS VISUALIZATION
if 'results' in st.session_state:
    st.markdown("---")
    st.header("📊 GARCH Analysis Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Returns")
        fig1 = px.line(st.session_state.results['returns'].tail(300), 
                      title="Daily Returns (%)")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("GARCH Volatility")
        vol_series = st.session_state.results['vol'].tail(300) * 100
        fig2 = px.line(vol_series, title="Conditional Volatility (%)")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Forecast
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔮 10-Day Forecast")
        try:
            forecast = st.session_state.results['model'].forecast(horizon=10)
            vol_fc = np.sqrt(forecast.variance.values[-1,:]) * 100
            for i, v in enumerate(vol_fc, 1):
                st.metric(f"Day {i}", f"{v:.2f}%")
        except Exception as e:
            st.info(f"Forecast ready: {e}")
