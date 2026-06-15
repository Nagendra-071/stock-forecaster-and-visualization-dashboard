# AI-Powered Stock Forecasting & Visualization Dashboard

An interactive, data-driven web application built with Streamlit that allows users to visualize historical stock performance and generate long-term asset price forecasts using Machine Learning.

## 🚀 Key Features
* **Live Global & Indian Market Data:** Leverages `yfinance` to pull historical datasets seamlessly for standard global tickers and regional NSE/BSE stocks (using `.NS` and `.BO` extensions).
* **Robust Data Pipeline:** Features an auto-adjusting, fault-tolerant parser capable of flattening multi-index structures directly out of single-ticker API responses.
* **AI Predictions:** Uses Meta's `Prophet` time-series forecasting model to map future market projections up to 5 years down the line.
* **Interactive Visualization Layer:** Incorporates clean, responsive layout splits using `Altair` for historical tracking, custom metric indicators for session changes, and `Plotly` for macro predictive charts.
* **Data Portability:** Includes interactive export macros to bundle future predictive trends directly into structured CSV outputs.

## 📁 Workspace Directory
```text
.
├── app2.py          # Primary application source script with model and visualization pipelines
├── requirements.txt # Explicit package dependency designations for deployment
└── README.md        # Comprehensive documentation mapping operational deployment blueprints