import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Corporate Credit & Valuation Screener", layout="wide")

st.title("AI-Augmented Corporate Credit & Valuation Screener")
st.caption("Nifty Metal & Mining Universe | Monte Carlo DCF + Credit Scorecard + GenAI Macro Stress Testing")

st.markdown("""
This app demonstrates a quantitative screener combining DCF valuation, credit risk
scoring, and macro stress testing across Nifty Metal & Mining stocks. Full methodology
and debugging notes are in the [GitHub repo README](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME).
""")

RELIABLE_TICKERS = ["JSL.NS", "HINDZINC.NS", "SARDAEN.NS", "SAIL.NS", "TATASTEEL.NS"]

SCREENER_DATA = {
    "Ticker": ["JSL.NS", "HINDZINC.NS", "SARDAEN.NS", "SAIL.NS", "TATASTEEL.NS"],
    "Median Intrinsic Value": [353.03, 345.43, 354.92, 35.50, 33.42],
    "Current Price": [698.15, 536.55, 508.00, 169.88, 189.80],
    "Prob Undervalued": [0.03, 0.05, 0.08, 0.00, 0.00],
    "Relative Value Score": [50.0, 0.0, 33.3, 16.7, 0.0],
    "Credit Health": [0.639, 1.000, 0.056, 0.167, 0.111],
    "Deal Score": [34.28, 27.00, 17.05, 10.02, 2.78],
    "Stress-Adjusted Score": [29.31, 28.12, 21.07, 2.65, -1.60],
    "Worst Case Downside %": [-82.29, -69.27, -69.56, -114.55, -111.83]
}

screener_df = pd.DataFrame(SCREENER_DATA)

MACRO_SCENARIOS = [
    {"scenario": "Global Recession", "fcf_growth_shock": -0.05, "wacc_shock": 0.02, "terminal_growth_shock": -0.02},
    {"scenario": "Trade War Escalation", "fcf_growth_shock": -0.04, "wacc_shock": 0.03, "terminal_growth_shock": -0.03},
    {"scenario": "Commodity Price Crash", "fcf_growth_shock": -0.06, "wacc_shock": 0.04, "terminal_growth_shock": -0.04},
    {"scenario": "Currency Devaluation", "fcf_growth_shock": -0.03, "wacc_shock": 0.02, "terminal_growth_shock": -0.01},
    {"scenario": "Interest Rate Hike", "fcf_growth_shock": -0.02, "wacc_shock": 0.03, "terminal_growth_shock": -0.02},
    {"scenario": "Inflation Surge", "fcf_growth_shock": -0.01, "wacc_shock": 0.04, "terminal_growth_shock": -0.03},
    {"scenario": "Supply Chain Disruption", "fcf_growth_shock": -0.05, "wacc_shock": 0.02, "terminal_growth_shock": -0.04},
    {"scenario": "Political Instability", "fcf_growth_shock": -0.04, "wacc_shock": 0.03, "terminal_growth_shock": -0.03},
    {"scenario": "Regulatory Changes", "fcf_growth_shock": -0.03, "wacc_shock": 0.02, "terminal_growth_shock": -0.02},
    {"scenario": "Technological Disruption", "fcf_growth_shock": -0.02, "wacc_shock": 0.01, "terminal_growth_shock": -0.01},
    {"scenario": "Environmental Regulations", "fcf_growth_shock": -0.03, "wacc_shock": 0.02, "terminal_growth_shock": -0.02},
    {"scenario": "Pandemic Recovery", "fcf_growth_shock": 0.01, "wacc_shock": 0.01, "terminal_growth_shock": 0.01},
]

tab1, tab2, tab3 = st.tabs(["Screener", "Live Ticker Lookup", "Macro Stress Scenarios"])

with tab1:
    st.subheader("Final Deal Score Ranking")
    st.dataframe(
        screener_df.sort_values("Deal Score", ascending=False).style.background_gradient(
            subset=["Deal Score", "Stress-Adjusted Score"], cmap="Greens"
        ),
        use_container_width=True
    )

    st.subheader("Deal Score vs Stress-Adjusted Score")
    chart_df = screener_df.set_index("Ticker")[["Deal Score", "Stress-Adjusted Score"]]
    st.bar_chart(chart_df)

    st.info(
        "**Reading this table:** JSL and HINDZINC rank highest on both raw valuation and "
        "resilience to macro stress. SAIL and TATASTEEL show large downside under adverse "
        "scenarios due to higher leverage and thinner margins, despite reasonable base valuations."
    )

with tab2:
    st.subheader("Live Company Snapshot")
    ticker_input = st.selectbox("Select a ticker from the screened universe", RELIABLE_TICKERS)

    if st.button("Fetch Live Data"):
        with st.spinner("Fetching live data from yfinance..."):
            try:
                stock = yf.Ticker(ticker_input)
                info = stock.info
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{info.get('currentPrice', 'N/A')}")
                col2.metric("Market Cap", f"₹{info.get('marketCap', 0):,.0f}")
                col3.metric("Beta (yfinance)", f"{info.get('beta', 'N/A')}")

                row = screener_df[screener_df["Ticker"] == ticker_input].iloc[0]
                st.write("**Model Output (precomputed):**")
                col4, col5, col6 = st.columns(3)
                col4.metric("Deal Score", row["Deal Score"])
                col5.metric("Credit Health", row["Credit Health"])
                col6.metric("Prob Undervalued", row["Prob Undervalued"])
            except Exception as e:
                st.error(f"Could not fetch live data: {e}")

with tab3:
    st.subheader("GenAI-Generated Macro Stress Scenarios")
    st.caption("Generated using Microsoft Phi-3-mini via structured prompting")
    scenarios_df = pd.DataFrame(MACRO_SCENARIOS)
    st.dataframe(scenarios_df, use_container_width=True)

    st.subheader("Worst-Case Downside by Ticker")
    downside_df = screener_df.set_index("Ticker")[["Worst Case Downside %"]]
    st.bar_chart(downside_df)

st.divider()
st.caption(
    "Note: Full pipeline (DCF Monte Carlo simulation, credit model training, "
    "LLM inference) runs in the linked Kaggle notebook. This app surfaces the "
    "final results for interactive exploration. Live ticker data is fetched "
    "in real time via yfinance; screener scores shown are precomputed from the notebook run."
)
