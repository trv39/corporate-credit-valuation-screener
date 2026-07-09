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

BASE_DCF_INPUTS = {
    "JSL.NS": {"fcf_base": 8_500_000_000, "wacc": 0.117, "growth": 0.045, "shares": 356_000_000, "current_price": 698.15},
    "HINDZINC.NS": {"fcf_base": 62_000_000_000, "wacc": 0.098, "growth": 0.038, "shares": 845_000_000, "current_price": 536.55},
    "SARDAEN.NS": {"fcf_base": 3_100_000_000, "wacc": 0.121, "growth": 0.050, "shares": 143_000_000, "current_price": 508.00},
    "SAIL.NS": {"fcf_base": 40_581_900_000, "wacc": 0.125, "growth": 0.022, "shares": 4_130_525_289, "current_price": 169.88},
    "TATASTEEL.NS": {"fcf_base": 100_221_800_000, "wacc": 0.117, "growth": 0.020, "shares": 12_471_847_611, "current_price": 189.80},
}

def run_custom_dcf(ticker, fcf_growth_shock, wacc_shock, terminal_growth_shock, years=5):
    inputs = BASE_DCF_INPUTS[ticker]
    fcf_base = inputs["fcf_base"]
    growth = inputs["growth"] + fcf_growth_shock
    wacc = inputs["wacc"] + wacc_shock
    terminal_growth = 0.04 + terminal_growth_shock
    wacc = max(wacc, terminal_growth + 0.02)

    projected_fcf = [fcf_base * (1 + growth) ** i for i in range(1, years + 1)]
    discounted = [c / (1 + wacc) ** i for i, c in enumerate(projected_fcf, start=1)]
    terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    ev = sum(discounted) + terminal_value / (1 + wacc) ** years
    intrinsic_value = ev / inputs["shares"]
    current_price = inputs["current_price"]
    downside_pct = ((intrinsic_value - current_price) / current_price) * 100
    return intrinsic_value, current_price, downside_pct, wacc

tab1, tab2, tab3, tab4 = st.tabs(["Screener", "Live Ticker Lookup", "Macro Stress Scenarios", "Build Your Own Scenario"])

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

with tab4:
    st.subheader("Design Your Own Macro Scenario")
    st.caption(
    "The 12 scenarios above were generated by Phi-3-mini (an open-source LLM). "
    "Below, adjust the same shock parameters manually to see live DCF impact — "
    "powered by the same valuation engine, without needing to reload the LLM each time."
)

    scenario_name = st.text_input("Name your scenario", value="Custom Scenario")

    col1, col2, col3 = st.columns(3)
    with col1:
        fcf_shock = st.slider(
            "FCF Growth Shock", min_value=-0.10, max_value=0.05, value=-0.02, step=0.005,
            help="Change to base free cash flow growth rate. -0.03 means growth cut by 3 percentage points."
        )
    with col2:
        wacc_shock = st.slider(
            "WACC Shock (Discount Rate)", min_value=-0.02, max_value=0.05, value=0.01, step=0.005,
            help="Change to discount rate. 0.015 means WACC rises by 150 basis points."
        )
    with col3:
        terminal_shock = st.slider(
            "Terminal Growth Shock", min_value=-0.03, max_value=0.02, value=-0.01, step=0.005,
            help="Change to long-run terminal growth assumption."
        )

    st.markdown(f"**Scenario: {scenario_name}**")

    results = []
    for ticker in RELIABLE_TICKERS:
        intrinsic, price, downside, wacc_used = run_custom_dcf(
            ticker, fcf_shock, wacc_shock, terminal_shock
        )
        results.append({
            "Ticker": ticker,
            "Intrinsic Value Under Scenario": round(intrinsic, 2),
            "Current Price": price,
            "Downside %": round(downside, 2),
            "WACC Used": round(wacc_used, 3)
        })

    custom_df = pd.DataFrame(results).sort_values("Downside %", ascending=False)
    st.dataframe(custom_df, use_container_width=True)

    st.bar_chart(custom_df.set_index("Ticker")["Downside %"])

    most_resilient = custom_df.iloc[0]["Ticker"]
    least_resilient = custom_df.iloc[-1]["Ticker"]
    st.success(
        f"Under **{scenario_name}**, **{most_resilient}** is the most resilient and "
        f"**{least_resilient}** takes the largest hit."
    )

st.divider()
st.caption(
    "Note: Full pipeline (DCF Monte Carlo simulation, credit model training, "
    "LLM inference) runs in the linked Kaggle notebook. This app surfaces the "
    "final results for interactive exploration. Live ticker data is fetched "
    "in real time via yfinance; screener scores shown are precomputed from the notebook run."
)
