import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Volatility & AI Advisor", page_icon="📈", layout="wide")

st.title("🔍 Stock Volatility Forecaster")
st.markdown("**GARCH(1,1) Analysis + AI Risk Advisor**")

st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Name", value="SYNTH_AAPL")
n_obs = st.sidebar.slider("Observations", 100, 2000, 1000)
vol_level = st.sidebar.selectbox("Volatility", ["Medium", "Low", "High"])

if st.sidebar.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.rerun()

@st.cache_data
def generate_realistic_stock_data(_n_days=1500, _vol_level="Medium"):
    seed_map = {"Low": 42, "Medium": 43, "High": 44}
    np.random.seed(seed_map.get(_vol_level, 43))
    
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

if st.button("🚀 RUN GARCH ANALYSIS", type="primary"):
    with st.spinner("Running full GARCH pipeline..."):
        try:
            raw_data = generate_realistic_stock_data(1500, vol_level)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Days", len(raw_data))
            with col2:
                st.metric("Final Price", f"${raw_data['Close'].iloc[-1]:,.2f}")
            with col3:
                total_ret = (raw_data['Close'].iloc[-1] / raw_data['Close'].iloc[0] - 1) * 100
                st.metric("Total Return", f"{total_ret:.1f}%")
            
            st.line_chart(raw_data['Close'].tail(200), height=300)

            returns = raw_data['Close'].pct_change() * 100
            returns_clean = returns.dropna()
            returns_final = returns_clean[np.isfinite(returns_clean)].tail(n_obs)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Valid Returns", len(returns_final))
                st.metric("Return Std", f"{returns_final.std():.2f}%")
            with col2:
                st.line_chart(returns_final.tail(200), height=300)

            model = arch_model(returns_final.values, p=1, q=1, rescale=False)
            fitted = model.fit(disp="off", show_warning=False)
            
            cond_vol = fitted.conditional_volatility
            daily_vol = np.sqrt(np.mean(cond_vol**2)) * 100
            annual_vol = daily_vol * np.sqrt(252)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Daily Vol", f"{daily_vol:.2f}%")
            col2.metric("Annual Vol", f"{annual_vol:.2f}%")
            col3.metric("Model AIC", f"{fitted.aic:.1f}")
            
            forecast = fitted.forecast(horizon=5)
            forecast_vol = np.sqrt(forecast.variance.iloc[-1].values) * 100
            
            st.session_state.results = {
                'model': fitted,
                'returns': returns_final,
                'raw_data': raw_data,
                'volatility': cond_vol,  # Keep as numpy array
                'ticker': ticker,
                'daily_vol': daily_vol,
                'annual_vol': annual_vol,
                'forecast_vol': forecast_vol,
                'current_vol': float(cond_vol[-1]),  # Use array indexing
                'avg_vol': float(np.mean(cond_vol))
            }
            
            with st.expander("Model Parameters"):
                st.code(str(fitted.params.round(4)), language="text")
            
            st.success("GARCH Analysis Complete!")
            
        except Exception as e:
            st.error(f"Pipeline Error: {str(e)}")

if 'results' in st.session_state:
    res = st.session_state.results
    st.markdown("---")
    st.header(f"📊 {res['ticker']} - GARCH Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(res['returns'].tail(300), title="Daily Returns (%)")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        vol_pct = res['volatility'][-300:] * 100  # FIXED: Direct numpy array slicing
        fig2 = px.line(x=range(len(vol_pct)), y=vol_pct, title="GARCH Volatility (%)")
        st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Vol", f"{res['current_vol']:.2f}%")
    col2.metric("Avg Vol", f"{res['avg_vol']:.2f}%")
    col3.metric("Daily Vol", f"{res['daily_vol']:.2f}%")
    col4.metric("Annual Vol", f"{res['annual_vol']:.2f}%")
    
    st.subheader("5-Day Volatility Forecast")
    forecast_cols = st.columns(5)
    for i, vol in enumerate(res['forecast_vol']):
        with forecast_cols[i]:
            st.metric(f"Day {i+1}", f"{vol:.2f}%")

st.markdown("---")
st.header("🤖 AI Investment Risk Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about risks, buy/sell timing, or portfolio strategy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if 'results' in st.session_state:
            res = st.session_state.results
            prompt_lower = prompt.lower()
            
            if any(word in prompt_lower for word in ["buy", "purchase", "long"]):
                if res['current_vol'] < res['avg_vol']:
                    trend = "stabilizing" if res['forecast_vol'][-1] < res['forecast_vol'][0] else "rising"
                    response = f"🟢 BUY SIGNAL: Current vol {res['current_vol']:.2f}% < average {res['avg_vol']:.2f}%. Good entry point. 5-day forecast shows {trend} volatility."
                else:
                    response = f"🟡 CAUTION: Current vol {res['current_vol']:.2f}% > average. Wait for volatility to drop below {res['avg_vol']:.2f}%."
            
            elif any(word in prompt_lower for word in ["sell", "short", "exit"]):
                response = f"🔴 SELL if: Volatility spikes above {res['daily_vol']+10:.1f}% or forecast shows sustained increase."
            
            elif "risk" in prompt_lower or "volatility" in prompt_lower:
                response = f"Risk Profile: {vol_level.upper()} volatility. Annual risk: {res['annual_vol']:.1f}%. Current daily risk: {res['current_vol']:.2f}%."
            
            elif "portfolio" in prompt_lower:
                response = f"For {ticker}: Allocate {min(30, 100-int(res['annual_vol']))}% of portfolio. Use stop-loss at 2x current volatility ({res['current_vol']*2:.1f}%)."
            
            else:
                response = f"GARCH shows {res['current_vol']:.2f}% current volatility vs {res['avg_vol']:.2f}% average. Ask me 'Should I buy?' or 'What is the risk?'"
        else:
            response = "Please run the GARCH analysis first!"
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
