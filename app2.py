import streamlit as st
from datetime import date

import pandas as pd
import yfinance as yf
from prophet import Prophet
import altair as alt


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Visualizer App")

stocks = (
    "RELIANCE.NS", "BANKNIFTY1.BO", "HDB", "IBN", "SBIN.NS",
    "HINDUNILVR.NS", "INFY", "BAJFINANCE.NS", "LICI.NS", "ITC.NS", "LT.NS",
    "MARUTI.NS", "M&M.NS", "HCLTECH.NS", "KOTAKBANK.NS", "SUNPHARMA.NS",
    "ULTRACEMCO.NS", "AXISBANK.BO", "TITAN.NS", "NTPC.NS", "BAJAJFINSV.NS",
    "DMART.NS", "ONGC.NS", "HAL.NS", "ETERNAL.NS", "ADANIPORTS.NS",
    "BEL.NS", "POWERGRID.NS", "WIT", "ADANIENT.NS", "JSWSTEEL.NS",
    "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "ASIANPAINT.NS", "COALINDIA.NS",
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
    "SWIGGY.NS", "UNIONBANK.NS"
)
selected_stocks = st.selectbox("Select dataset for prediction", stocks)


n_years = st.slider("Period from 1 to 5 yrs:", 1, 5)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data... done!")


st.subheader('Raw Data')
st.write(data.tail())


def plot_raw_data():
    
    col1, col2 = st.columns([3, 1])

    with col1:
        base = alt.Chart(data).encode(
            x=alt.X('Date:T', axis=alt.Axis(title='Date'))
        )
    
        open_line = base.mark_line(color='blue').encode(
            y=alt.Y('Open:Q', axis=alt.Axis(title='Stock Price'))
        )
    
        close_line = base.mark_line(color='red').encode(
            y=alt.Y('Close:Q', axis=alt.Axis(title='Stock Price'))
        )
    
        chart = (open_line + close_line).properties(
            title="Time Series Data"
        ).interactive(
            bind_y=False
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown("""
        <style>
        .legend-container {
            font-size: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
        }
        .blue-box {
            height: 10px;
            width: 30px;
            background-color: blue;
            margin-right: 5px;
            border-radius: 2px;
        }
        .red-box {
            height: 10px;
            width: 30px;
            background-color: red;
            margin-right: 5px;
            border-radius: 2px;
        }
        </style>
        <div class="legend-container">
            <div class="legend-item">
                <span class="blue-box"></span> Open
            </div>
            <div class="legend-item">
                <span class="red-box"></span> Close
            </div>
        </div>
        """, unsafe_allow_html=True)


plot_raw_data()












