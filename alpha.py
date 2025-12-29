import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Volatility", page_icon="📈", layout="wide")

st.title("🔍 Stock Volatility Forecaster")
st.markdown("**GARCH(1,1) - BITMASK SEED APPROACH**")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Name", value="SYNTH_AAPL")
n_obs = st.sidebar.slider("Observations", 100, 2000, 1000)
vol_level = st.sidebar.selectbox("Volatility", ["Medium", "Low", "High"])

if st.sidebar.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.rerun()

# NEW APPROACH: BITMASKING THE SEED
@st.cache_data
def generate_realistic_stock_data(_n_days=1500, _vol_level="Medium"):
    # NEW APPROACH: Bitwise AND with 0xFFFFFFFF forces the value into 32-bit range
    # This prevents the NumPy ValueError while keeping the seed unique to the string
    stable_seed = hash(_vol_level) & 0xFFFFFFFF 
    np.random.seed(stable_seed)
    
    dates = pd.date_range(start="2023-01-01", periods=_n_days, freq="B")
    
    vol_map = {"Low": 0.8, "Medium": 1.2, "High": 1.8}
    vol_factor = vol_map.get(_vol_level, 1.2)
    
    returns = []
    current_vol = 1.2
    for i in range(_n_days):
        current_vol = 0.1 + 0.7 * current_vol + 0.2 * abs(np.random.normal(0, 1))
        ret = np.random.normal(0.0002, current_vol * vol_factor * 0.015)
        returns.append(ret)
    
    log_returns = np.array(returns)
    price = 150 * np.exp(np.cumsum(log_returns))
    
    n = len(dates)
    data = pd.DataFrame(index=dates, data={
        'Open':   price * (1 + np.random.normal(0, 0.002, n)),
        'High':   price * (1 + np.abs(np.random.normal(0.008, 0.005, n))),
        'Low':    price * (1 - np.abs(np.random.normal(0.008, 0.005, n))),
        'Close':  price,
        'Volume': np.random.randint(8_000_000, 20_000_000, n)
    }).round(2)
    
    return data.dropna()

# MAIN PIPELINE
if st.button("🚀 RUN GARCH ANALYSIS", type="primary"):
    with st.spinner("Analyzing..."):
        try:
            # STEP 1: Generate data
            with st.status("📊 Generating data", expanded=True):
                raw_data = generate_realistic_stock_data(1500, vol_level)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Days", len(raw_data))
                    st.metric("Final Price", f"${raw_data['Close'].iloc[-1]:.2f}")
                with col2:
                    total_return = (raw_data['Close'].iloc[-1] / raw_data['Close'].iloc[0] - 1) * 100
                    st.metric("Total Return", f"{total_return:.1f}%")
                    st.metric("Raw Vol", f"{raw_data['Close'].pct_change().std()*100:.2f}%")
                
                st.line_chart(raw_data['Close'].tail(200))

            # STEP 2: Returns
            with st.status("📈 Calculating returns", expanded=True):
                prices = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                returns = prices['Close'].pct_change() * 100
                returns_clean = returns.dropna()
                returns_final = returns_clean[np.isfinite(returns_clean)].tail(n_obs)
                
                st.metric("Valid Returns", len(returns_final))
                st.line_chart(returns_final.tail(200))

            # STEP 3: GARCH Fitting
            with st.status("🔬 Fitting GARCH(1,1)", expanded=True):
                model = arch_model(returns_final.values, p=1, q=1, rescale=False)
                fitted = model.fit(disp="off", show_warning=False)
                
                cond_vol = fitted.conditional_volatility
                daily_vol = np.sqrt(np.mean(cond_vol**2))
                annual_vol = daily_vol * np.sqrt(252)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Daily Vol", f"{daily_vol:.2f}%")
                col2.metric("Annual Vol", f"{annual_vol:.2f}%")
                col3.metric("AIC", f"{fitted.aic:.1f}")
                
                st.session_state.results = {
                    'model': fitted,
                    'returns': returns_final,
                    'volatility': cond_vol
                }
                
                with st.expander("Model Coefficients"):
                    st.code(str(fitted.params.round(4)))
                
                st.success("✅ Analysis Complete!")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# RESULTS
if 'results' in st.session_state:
    st.markdown("---")
    st.header("📊 Results")
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(st.session_state.results['returns'].tail(300), title="Returns (%)"), use_container_width=True)
    with c2:
        vol_tail = st.session_state.results['volatility'][-300:]
        st.plotly_chart(px.line(y=vol_tail, title="GARCH Volatility (%)"), use_container_width=True)
    
    st.subheader("🔮 5-Day Volatility Forecast")
    try:
        forecast = st.session_state.results['model'].forecast(horizon=5)
        vol_fc = np.sqrt(forecast.variance.iloc[-1].values)
        cols = st.columns(5)
        for i, v in enumerate(vol_fc):
            cols[i].metric(f"Day {i+1}", f"{v:.2f}%")
    except:
        st.info("Fit the model to see forecasts.")

st.markdown("---")
st.caption("🎓 WQU Lab 8.5 - Programmatic Bitmask Seed Approach")
