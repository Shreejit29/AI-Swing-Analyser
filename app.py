import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Swing Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DEMO DATA
# ============================================================

np.random.seed(42)

dates = pd.date_range(
    end=pd.Timestamp.today(),
    periods=180,
    freq="D",
)

price = (
    3200
    + np.cumsum(
        np.random.normal(
            0.5,
            25,
            len(dates),
        )
    )
)

price = np.maximum(price, 2500)

demo_df = pd.DataFrame(
    {
        "Date": dates,
        "Close": price,
    }
)

demo_df["EMA20"] = (
    demo_df["Close"]
    .ewm(span=20)
    .mean()
)

demo_df["EMA50"] = (
    demo_df["Close"]
    .ewm(span=50)
    .mean()
)

demo_df["RSI"] = (
    55
    + np.sin(
        np.arange(len(demo_df)) / 12
    )
    * 15
)

demo_df["Volume"] = (
    1_000_000
    + np.random.normal(
        0,
        150_000,
        len(demo_df),
    )
)


# ============================================================
# HEADER
# ============================================================

st.title("📈 AI Swing Analyser")

st.caption(
    "Research-first AI system for Indian equity swing trading"
)

st.warning(
    "DEMO MODE — Predictions and trading metrics shown "
    "here are illustrative only. No live trading decision "
    "should be based on this demo."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔍 Stock Analysis")

    symbol = st.text_input(
        "NSE Symbol",
        value="TCS",
    ).upper()

    timeframe = st.selectbox(
        "Primary Timeframe",
        [
            "4H",
            "1D",
            "1W",
            "1M",
        ],
        index=1,
    )

    st.subheader("Analysis Horizons")

    horizons = st.multiselect(
        "Prediction Horizons",
        [
            "1D",
            "3D",
            "5D",
            "10D",
            "20D",
        ],
        default=[
            "1D",
            "3D",
            "5D",
            "10D",
            "20D",
        ],
    )

    st.divider()

    st.subheader("Risk")

    risk = st.slider(
        "Maximum Risk / Trade",
        min_value=0.25,
        max_value=2.0,
        value=1.0,
        step=0.25,
        format="%.2f%%",
    )

    capital = st.number_input(
        "Trading Capital",
        min_value=10_000,
        value=500_000,
        step=10_000,
    )

    st.divider()

    analyze = st.button(
        "🚀 Analyse Stock",
        use_container_width=True,
    )


# ============================================================
# TOP MARKET STATUS
# ============================================================

st.subheader("🌐 Indian Market Context")

market1, market2, market3, market4 = st.columns(4)

with market1:
    st.metric(
        "NIFTY 50",
        "25,480",
        "+0.72%",
    )

with market2:
    st.metric(
        "SENSEX",
        "83,420",
        "+0.61%",
    )

with market3:
    st.metric(
        "NIFTY BANK",
        "57,850",
        "+0.88%",
    )

with market4:
    st.metric(
        "India VIX",
        "13.42",
        "-4.10%",
    )


# ============================================================
# STOCK SNAPSHOT
# ============================================================

st.subheader(
    f"📊 {symbol} — AI Market Snapshot"
)

c1, c2, c3, c4, c5 = st.columns(5)

current_price = float(
    demo_df["Close"].iloc[-1]
)

with c1:
    st.metric(
        "Current Price",
        f"₹{current_price:,.2f}",
        "+1.34%",
    )

with c2:
    st.metric(
        "RSI (14)",
        "58.4",
        "Neutral-Bullish",
    )

with c3:
    st.metric(
        "ADX",
        "27.8",
        "Trending",
    )

with c4:
    st.metric(
        "Relative Strength",
        "+4.8%",
        "vs NIFTY",
    )

with c5:
    st.metric(
        "Market Regime",
        "BULLISH",
        "Stable",
    )


# ============================================================
# PRICE CHART
# ============================================================

st.subheader("📈 Price & Trend Structure")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=demo_df["Date"],
        y=demo_df["Close"],
        name="Price",
        line=dict(width=2),
    )
)

fig.add_trace(
    go.Scatter(
        x=demo_df["Date"],
        y=demo_df["EMA20"],
        name="EMA 20",
        line=dict(width=1.5),
    )
)

fig.add_trace(
    go.Scatter(
        x=demo_df["Date"],
        y=demo_df["EMA50"],
        name="EMA 50",
        line=dict(width=1.5),
    )
)

fig.update_layout(
    height=450,
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# MULTI-TIMEFRAME ANALYSIS
# ============================================================

st.subheader("🕐 Multi-Timeframe Analysis")

mtf = pd.DataFrame(
    {
        "Timeframe": [
            "4H",
            "1D",
            "1W",
            "1M",
        ],
        "Trend": [
            "Bullish",
            "Bullish",
            "Bullish",
            "Neutral-Bullish",
        ],
        "Momentum": [
            "Positive",
            "Positive",
            "Strong",
            "Moderate",
        ],
        "Structure": [
            "Higher High",
            "Higher High",
            "Higher High",
            "Range Breakout",
        ],
        "Alignment": [
            0.78,
            0.86,
            0.91,
            0.67,
        ],
    }
)

st.dataframe(
    mtf,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# AI PREDICTION
# ============================================================

st.subheader("🤖 AI Direction Prediction")

p1, p2, p3 = st.columns(3)

with p1:
    st.metric(
        "UP Probability",
        "78.4%",
    )

with p2:
    st.metric(
        "DOWN Probability",
        "21.6%",
    )

with p3:
    st.metric(
        "Model Agreement",
        "84.7%",
    )


progress = 0.784

st.progress(
    progress,
    text="AI probability of upward movement: 78.4%",
)


# ============================================================
# MULTI-HORIZON PREDICTIONS
# ============================================================

st.subheader("🎯 Multi-Horizon Forecast")

forecast = pd.DataFrame(
    {
        "Horizon": [
            "1D",
            "3D",
            "5D",
            "10D",
            "20D",
        ],
        "Direction": [
            "UP",
            "UP",
            "UP",
            "UP",
            "NEUTRAL",
        ],
        "UP Probability": [
            "68.2%",
            "73.5%",
            "78.4%",
            "75.1%",
            "61.3%",
        ],
        "Expected Return": [
            "+0.8%",
            "+1.9%",
            "+3.7%",
            "+5.2%",
            "+6.1%",
        ],
        "Confidence": [
            "Medium",
            "High",
            "High",
            "High",
            "Medium",
        ],
    }
)

st.dataframe(
    forecast,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TARGET PRICE RANGE
# ============================================================

st.subheader("🎯 AI Target Price Range")

range_col1, range_col2, range_col3 = st.columns(3)

with range_col1:
    st.metric(
        "Lower Range",
        "₹3,180",
    )

with range_col2:
    st.metric(
        "Expected Target",
        "₹3,410",
        "+5.2%",
    )

with range_col3:
    st.metric(
        "Upper Range",
        "₹3,570",
        "+10.1%",
    )


st.info(
    "The target range represents a probabilistic price interval. "
    "It is not a guaranteed target."
)


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

st.subheader("📐 Price Structure")

sr1, sr2, sr3, sr4 = st.columns(4)

with sr1:
    st.metric(
        "Support",
        "₹3,180",
    )

with sr2:
    st.metric(
        "Strong Support",
        "₹3,090",
    )

with sr3:
    st.metric(
        "Resistance",
        "₹3,410",
    )

with sr4:
    st.metric(
        "Breakout Level",
        "₹3,435",
    )


# ============================================================
# SWING SETUP
# ============================================================

st.subheader("💡 AI Swing Setup")

setup_col1, setup_col2 = st.columns(2)

with setup_col1:

    st.success(
        "### 🟢 BUY SETUP"
    )

    st.write(
        "**Entry Zone:** ₹3,250 – ₹3,300"
    )

    st.write(
        "**Stop Loss:** ₹3,150"
    )

    st.write(
        "**Target 1:** ₹3,410"
    )

    st.write(
        "**Target 2:** ₹3,570"
    )

    st.write(
        "**Risk / Reward:** 1 : 2.7"
    )

with setup_col2:

    st.write("### Position Sizing")

    position_value = capital * 0.25

    risk_amount = (
        capital
        * risk
        / 100
    )

    shares = int(
        risk_amount
        / (3250 - 3150)
    )

    sizing = pd.DataFrame(
        {
            "Parameter": [
                "Capital",
                "Maximum Risk",
                "Risk Amount",
                "Maximum Capital Allocation",
                "Suggested Quantity",
            ],
            "Value": [
                f"₹{capital:,.0f}",
                f"{risk:.2f}%",
                f"₹{risk_amount:,.0f}",
                f"₹{position_value:,.0f}",
                f"{shares} shares",
            ],
        }
    )

    st.dataframe(
        sizing,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# DECISION GATES
# ============================================================

st.subheader("🛡️ AI Safety Gates")

gates = pd.DataFrame(
    {
        "Gate": [
            "Walk-Forward Validation",
            "Final Holdout",
            "Probability Calibration",
            "Target Range Validation",
            "Regime Stability",
            "Backtest",
            "Robustness",
            "Model Agreement",
            "MTF Alignment",
            "Risk / Reward",
        ],
        "Status": [
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
        ],
    }
)

st.dataframe(
    gates,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FINAL DECISION
# ============================================================

st.subheader("🚦 Final AI Decision")

st.success(
    "## BUY — HIGH CONFIDENCE SETUP"
)

st.write(
    "The demo model shows bullish multi-timeframe alignment, "
    "acceptable model agreement, a validated target range, "
    "and sufficient risk/reward."
)

st.caption(
    "In the real application this decision will only be enabled "
    "after the complete research and production approval gates pass."
)


# ============================================================
# RESEARCH STATUS
# ============================================================

with st.expander(
    "🔬 Research / Model Status"
):

    status = pd.DataFrame(
        {
            "Component": [
                "Historical Data",
                "Technical Features",
                "Price Action",
                "Volume Analysis",
                "Market Context",
                "Multi-Timeframe",
                "Direction Model",
                "Return Model",
                "Range Model",
                "Calibration",
                "Walk-Forward",
                "Backtest",
                "Robustness",
                "Final Holdout",
            ],
            "Status": [
                "READY",
                "READY",
                "READY",
                "READY",
                "READY",
                "READY",
                "RESEARCH",
                "RESEARCH",
                "RESEARCH",
                "RESEARCH",
                "RESEARCH",
                "RESEARCH",
                "RESEARCH",
                "RESEARCH",
            ],
        }
    )

    st.dataframe(
        status,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Swing Analyser • Research-first • "
    "No guaranteed returns • No forced trades • "
    "WAIT is a valid decision"
)
