import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="Stock Volatility & AI Advisor", page_icon="📈", layout="wide")

st.title("🔍 Stock Volatility Forecaster")
st.markdown("**GARCH(1,1) Analysis + AI Risk Advisor**")

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

# 2. Data Generator
@st.cache_data
def generate_realistic_stock_data(_n_days=1500, _vol_level="Medium"):
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

# 3. Analysis Pipeline
if st.button("🚀 RUN GARCH ANALYSIS", type="primary"):
    with st.spinner("Analyzing..."):
        try:
            with st.status("📊 Processing Data & Model", expanded=True):
                raw_data = generate_realistic_stock_data(1500, vol_level)
                returns = raw_data['Close'].pct_change() * 100
                returns_final = returns.dropna().tail(n_obs)
                
                model = arch_model(returns_final.values, p=1, q=1, rescale=False)
                fitted = model.fit(disp="off", show_warning=False)
                
                cond_vol = fitted.conditional_volatility
                daily_vol = np.sqrt(np.mean(cond_vol**2))
                
                # Store results for the UI and the Chatbot
                st.session_state.results = {
                    'model': fitted,
                    'returns': returns_final,
                    'volatility': cond_vol,
                    'ticker': ticker,
                    'annual_vol': daily_vol * np.sqrt(252),
                    'forecast': np.sqrt(fitted.forecast(horizon=5).variance.iloc[-1].values)
                }
            st.success("Analysis Complete!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# 4. Results Display
if 'results' in st.session_state:
    st.markdown("---")
    st.header(f"📊 Market Insights: {st.session_state.results['ticker']}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(st.session_state.results['returns'].tail(300), title="Recent Returns (%)"), use_container_width=True)
    with c2:
        vol_tail = st.session_state.results['volatility'][-300:]
        st.plotly_chart(px.line(y=vol_tail, title="Conditional Volatility (%)"), use_container_width=True)

    # 5. Chatbot Interface
    st.markdown("---")
    st.header("🤖 AI Investment Risk Advisor")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"I've analyzed {ticker}. Ask me about the risks or if it's a good time to buy!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Should I buy this stock?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Context-Aware Logic for the "LLM" response
            res = st.session_state.results
            current_vol = res['volatility'][-1]
            avg_vol = np.mean(res['volatility'])
            forecast_trend = "increasing" if res['forecast'][-1] > res['forecast'][0] else "stabilizing"
            
            # Simple Logic-based Advisor (Can be replaced with actual OpenAI/Gemini API call)
            if "buy" in prompt.lower() or "good choice" in prompt.lower():
                if current_vol < avg_vol:
                    response = f"Based on GARCH(1,1), current volatility ({current_vol:.2f}%) is below the historical average. This suggests a period of relative calm, which could be a entry point if you have a high risk tolerance. However, the 5-day forecast shows volatility is {forecast_trend}."
                else:
                    response = f"Caution: Current volatility ({current_vol:.2f}%) is higher than average. High volatility often precedes price swings. It might be better to wait for the GARCH model to show signs of stabilizing."
            elif "risk" in prompt.lower():
                response = f"The annual volatility is currently estimated at {res['annual_vol']:.2f}%. Your 'Day 5' projected risk is {res['forecast'][-1]:.2f}%. This is considered a '{vol_level}' risk profile."
            else:
                response = "I can analyze the GARCH results for you. Try asking 'Is it a good time to buy?' or 'What is the risk forecast?'"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer Removal: No st.caption or st.markdown footer added here.
