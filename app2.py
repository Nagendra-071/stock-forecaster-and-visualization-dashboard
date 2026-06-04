import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import altair as alt
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI-Powered Stock Forecasting & Visualization Dashboard")

with st.sidebar:
    st.header("Control Panel")
    stocks = (
        "RELIANCE.NS", "HDB", "IBN", "SBIN.NS",
        "HINDUNILVR.NS", "INFY", "BAJFINANCE.NS", "LICI.NS", "ITC.NS", "LT.NS",
        "MARUTI.NS", "M&M.NS", "HCLTECH.NS", "KOTAKBANK.NS", "SUNPHARMA.NS",
        "ULTRACEMCO.NS", "AXISBANK.BO", "TITAN.NS", "NTPC.NS", "BAJAJFINSV.NS",
        "DMART.NS", "ONGC.NS", "HAL.NS", "ADANIPORTS.NS",
        "BEL.NS", "POWERGRID.NS", "WIT", "ADANIENT.NS", "JSWSTEEL.NS"
        ,"BAJAJ-AUTO.NS", "ASIANPAINT.NS", "COALINDIA.NS",
        "ADANIPOWER.NS", "NESTLEIND.NS", "INDIGO.NS", "TATASTEEL.NS",
        "HYUNDAI.NS", "JIOFIN.NS", "IOC.NS", "TRENT.NS", "GRASIM.NS", "DLF.NS",
        "HINDZINC.NS", "SBILIFE.NS", "EICHERMOT.NS", "VEDL.NS", "VBL.NS",
        "HDFCLIFE.NS", "DIVISLAB.NS", "HINDALCO.NS", "TVSMOTOR.NS", "IRFC.NS",
        "PIDILITIND.NS", "ADANIGREEN.NS", "LTIM.NS", "BAJAJHLDNG.NS",
        "AMBUJACEM.NS", "BRITANNIA.NS", "BPCL.NS", "TECHM.NS", "GODREJCP.NS",
        "PFC.NS", "SOLARINDS.NS", "CIPLA.NS", "TATAPOWER.NS", "BANKBARODA.NS",
        "BOSCHLTD.NS", "TORNTPHARM.NS", "CHOLAFIN.NS", "LODHA.NS", "HDFCAMC.NS",
        "PNB.NS", "GAIL.NS", "CGPOWER.NS", "SIEMENS.NS", "MAXHEALTH.NS",
        "MUTHOOTFIN.NS", "APOLLOHOSP.NS", "INDHOTEL.NS", "ABB.NS", "MAZDOCK.NS",
        "SHRIRAMFIN.NS", "SHREECEM.NS", "TATACONSUM.NS", "POLYCAB.NS",
        "DIXON.NS", "HEROMOTOCO.NS", "CUMMINSIND.NS", "RDY", "MANKIND.NS",
        "JINDALSTEL.NS", "ZYDUSLIFE.NS", "MOTHERSON.NS", "HAVELLS.NS",
        "SWIGGY.NS", "UNIONBANK.NS","GMBREW.NS"
    ) 
    selected_stock = st.selectbox("Select Stock Ticker", stocks)
    n_years = st.slider("Forecast Period (Years):", 1, 5)
    period = n_years * 365
    predict_button = st.button("Generate AI Prediction")

@st.cache_data(ttl=3600)
def load_data(ticker):
    try:
        # FIX: Removed the custom requests.Session wrapper entirely.
        # yfinance will internally manage its required curl_cffi session configuration automatically.
        data = yf.download(ticker, start=START, end=TODAY, auto_adjust=True)
        
        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data = data.dropna(subset=['Close'])
        data = data.reset_index()
        
        target_col = None
        for col in ['Date', 'index', 'Datetime', 'date']:
            if col in data.columns:
                target_col = col
                break
                
        if target_col:
            data.rename(columns={target_col: 'Date'}, inplace=True)
        else:
            data.rename(columns={data.columns[0]: 'Date'}, inplace=True)

        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_prediction(df, days):
    df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return m, forecast

data = load_data(selected_stock)

if data.empty:
    st.error(f"No data found for {selected_stock}. The API might be rate-limited or the network session failed. Try again later.")
else:
    currency_symbol = "₹" if (selected_stock.endswith(".NS") or selected_stock.endswith(".BO")) else "$"
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader(f"Historical Price Action: {selected_stock}")
        chart = alt.Chart(data).mark_line(color='#1f77b4').encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Close:Q', title=f'Price ({currency_symbol})', scale=alt.Scale(zero=False)),
            tooltip=['Date:T', 'Close:Q']
        ).properties(height=400).interactive()
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.subheader("Latest Market Data")
        last_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2]) if len(data) > 1 else last_price
        change = last_price - prev_price
        
        st.metric("Current Price", f"{currency_symbol}{last_price:.2f}", f"{change:.2f}")
        st.write("Recent Logs", data.tail(5))

    if predict_button:
        st.divider()
        with st.spinner("AI is analyzing trends..."):
            model, forecast = get_prediction(data, period)
            st.subheader('🚀 Future Forecast Analysis')
            fig1 = plot_plotly(model, forecast)
            fig1.update_layout(title=f"Forecast for {selected_stock} ({n_years} Years)")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("📥 Export Forecast Data")
            csv_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period)
            csv_data.columns = ['Date', 'Predicted_Price', 'Min_Expected', 'Max_Expected']
            csv = csv_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name=f'{selected_stock}_forecast.csv', mime='text/csv')
            
            with st.expander("Show AI Logic"):
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)