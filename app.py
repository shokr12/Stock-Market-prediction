import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from egx_predictor import EgxPredictor

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EGX Stock Prediction & AI Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLES ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .card-buy {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(6,78,59,0.4));
        border: 1px solid #10b981; padding: 20px; border-radius: 12px;
        color: #ffffff; text-align: center;
    }
    .card-sell {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(127,29,29,0.4));
        border: 1px solid #ef4444; padding: 20px; border-radius: 12px;
        color: #ffffff; text-align: center;
    }
    .card-hold {
        background: linear-gradient(135deg, rgba(234,179,8,0.15), rgba(120,80,0,0.3));
        border: 1px solid #eab308; padding: 20px; border-radius: 12px;
        color: #ffffff; text-align: center;
    }
    .rec-badge-buy   { background:#10b981; color:#fff; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .rec-badge-sell  { background:#ef4444; color:#fff; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .rec-badge-hold  { background:#eab308; color:#fff; padding:4px 12px; border-radius:20px; font-weight:bold; }
    </style>
""", unsafe_allow_html=True)


# ─── FULL EGX TICKER LIST ────────────────────────────────────────────────────
EGX_TICKERS = [
    "COMI","TMGH","ISPH","RAYA","SWDY","HRHO","ESRS","FWRY",
    "EKHW","PHDC","ABUK","EAST","AMER","CLHO","MNHD","EFID",
    "CIEB","OINV","AMOC","ETEL","OCDI","GTHE","HELI","BSSC",
    "SPMD","MOSK","PALM","OREG","GIZA","MFPC","MCQE","MCDR",
    "UEGC","DSCW","EKHO","BTFN","AUTO","ACGC","IRON","IDEA",
]

# ─── HELPER: RECOMMENDATION LOGIC ────────────────────────────────────────────
def get_recommendation(prediction: int, confidence: float, sentiment_label: str) -> dict:
    """
    Combines model prediction, confidence, and news sentiment into a
    Buy / Hold / Sell recommendation with an explanation.
    """
    pred_up = prediction == 1
    high_conf = confidence >= 58.0
    bullish_news = sentiment_label == "BULLISH"
    bearish_news = sentiment_label == "BEARISH"

    if pred_up and high_conf:
        action = "BUY"
        reason = "Model predicts UP with high confidence"
        if bullish_news:
            reason += " · News sentiment is BULLISH — strong buy signal."
        elif bearish_news:
            reason += " · But news sentiment is BEARISH — proceed with caution."
        else:
            reason += " · News sentiment is neutral."
    elif not pred_up and high_conf:
        action = "SELL"
        reason = "Model predicts DOWN/FLAT with high confidence"
        if bearish_news:
            reason += " · News sentiment is BEARISH — strong sell signal."
        elif bullish_news:
            reason += " · But news sentiment is BULLISH — consider waiting."
        else:
            reason += " · News sentiment is neutral."
    else:
        action = "HOLD"
        reason = f"Model confidence is below threshold ({confidence:.1f}% < 58%). "
        reason += "Signals are mixed — no strong directional bias detected. Wait for a clearer signal."

    colors = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#eab308"}
    icons  = {"BUY": "&#9650; BUY", "SELL": "&#9660; SELL", "HOLD": "&#9632; HOLD"}
    cards  = {"BUY": "card-buy", "SELL": "card-sell", "HOLD": "card-hold"}
    badges = {"BUY": "rec-badge-buy", "SELL": "rec-badge-sell", "HOLD": "rec-badge-hold"}

    return {
        "action": action,
        "reason": reason,
        "color": colors[action],
        "icon": icons[action],
        "card_class": cards[action],
        "badge_class": badges[action],
    }


# ─── CACHED: SINGLE STOCK PREDICTION ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_prediction(ticker: str, horizon: int, period: str):
    predictor = EgxPredictor(symbol=ticker, target_horizon=horizon)
    predictor.fetch_data(period=period)
    predictor.add_technical_indicators()
    predictor.train_model()
    result = predictor.predict_tomorrow()
    df = predictor.df.copy()
    feature_names = predictor.get_feature_names()
    importances = predictor.model.feature_importances_.tolist()
    return result, df, feature_names, importances


# ─── CACHED: MARKET SCREENER ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=1800)   # refresh every 30 minutes
def run_market_screener(tickers: tuple, horizon: int):
    rows = []
    for ticker in tickers:
        try:
            predictor = EgxPredictor(symbol=ticker, target_horizon=horizon)
            predictor.fetch_data(period="2y")
            predictor.add_technical_indicators()
            predictor.train_model()
            r = predictor.predict_tomorrow()

            rec = get_recommendation(r['prediction'], r['confidence'], r['sentiment']['label'])
            latest = predictor.df['Close'].iloc[-1]
            prev    = predictor.df['Close'].iloc[-2]
            change  = (latest - prev) / prev * 100

            rows.append({
                "Ticker": r['symbol'],
                "Close (EGP)": round(latest, 2),
                "Day Change %": round(change, 2),
                "RSI": round(predictor.df['RSI'].iloc[-1], 1),
                "Signal": r['signal'],
                "Confidence %": r['confidence'],
                "Sentiment": r['sentiment']['label'],
                "Recommendation": rec['action'],
                "_badge": rec['badge_class'],
            })
        except Exception:
            pass   # silently skip delisted / unavailable tickers
        time.sleep(0.3)   # polite delay for Yahoo Finance API
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#   SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📈 EGX AI Predictor")
st.sidebar.markdown("Hybrid ML · Technical Indicators · News Sentiment")
st.sidebar.markdown("---")

PAGE = st.sidebar.radio("Navigation", ["🔍 Stock Analyser", "🌍 Market Screener"])

st.sidebar.markdown("---")

if PAGE == "🔍 Stock Analyser":
    popular_stocks = {
        "COMI — Commercial Int. Bank": "COMI",
        "TMGH — Talaat Moustafa Group": "TMGH",
        "ISPH — Ibnsina Pharma": "ISPH",
        "RAYA — Raya Holding": "RAYA",
        "SWDY — El Sewedy Electric": "SWDY",
        "HRHO — EFG Hermes": "HRHO",
        "ESRS — Ezz Steel": "ESRS",
        "FWRY — Fawry": "FWRY",
        "Custom Ticker": "CUSTOM"
    }
    selected_option = st.sidebar.selectbox("🏦 Select Stock", list(popular_stocks.keys()))
    ticker_input = (
        st.sidebar.text_input("Enter EGX Ticker (e.g. ABUK)", value="ABUK").strip().upper()
        if popular_stocks[selected_option] == "CUSTOM"
        else popular_stocks[selected_option]
    )
    horizon_label = st.sidebar.radio(
        "📅 Prediction Horizon",
        ["1 Trading Day", "3 Trading Days", "5 Trading Days"],
        index=0
    )
    horizon_map = {"1 Trading Day": 1, "3 Trading Days": 3, "5 Trading Days": 5}
    target_horizon = horizon_map[horizon_label]
    history_period = st.sidebar.selectbox("📂 Training Data Range", ["1y","2y","5y","max"], index=2)

else:  # Market Screener
    screener_horizon_label = st.sidebar.radio(
        "📅 Screener Horizon",
        ["1 Trading Day", "3 Trading Days", "5 Trading Days"],
        index=0
    )
    screener_horizon = {"1 Trading Day": 1, "3 Trading Days": 3, "5 Trading Days": 5}[screener_horizon_label]

st.sidebar.markdown("---")
st.sidebar.info("Data updates automatically when you change controls.")


# ─────────────────────────────────────────────────────────────────────────────
#   PAGE 1: STOCK ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
if PAGE == "🔍 Stock Analyser":
    st.title("🔍 Single Stock Analysis")
    st.caption(f"Analysing **{ticker_input}** · {target_horizon}-day horizon · {history_period} training data")
    st.markdown("---")

    with st.spinner(f"Loading {ticker_input}..."):
        try:
            result, df, feature_names, importances = run_prediction(ticker_input, target_horizon, history_period)
        except Exception as e:
            st.error(f"**Could not load `{ticker_input}`**: {e}")
            st.stop()

    symbol    = result['symbol']
    sentiment = result['sentiment']
    signal    = result['signal']
    confidence = result['confidence']
    rec = get_recommendation(result['prediction'], confidence, sentiment['label'])

    # KPI row
    latest_close = df['Close'].iloc[-1]
    prev_close   = df['Close'].iloc[-2]
    price_change = latest_close - prev_close
    pct_change   = price_change / prev_close * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"📌 {symbol} Close", f"{latest_close:.2f} EGP", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    c2.metric("📰 Sentiment", sentiment['label'], f"Score {sentiment['score']:.2f}")
    c3.metric(f"🎯 Signal ({target_horizon}d)", signal, f"Conf. {confidence:.1f}%")
    c4.metric("📈 RSI (14)", f"{df['RSI'].iloc[-1]:.1f}", "Overbought>70 | Oversold<30")
    c5.metric("💡 Recommendation", rec['action'])

    st.markdown("---")

    # ── RECOMMENDATION CARD ──
    st.markdown(f"""
        <div class="{rec['card_class']}">
            <h2>{rec['icon']}</h2>
            <h4>Recommendation for {symbol} · {target_horizon}-Day Horizon</h4>
            <p>{rec['reason']}</p>
            <p style="font-size:12px;color:#94a3b8;">
                This is an AI-generated signal based on XGBoost technical analysis and news sentiment.
                Always do your own due diligence before investing.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("###")

    # ── CHARTS ──
    tab1, tab2, tab3 = st.tabs(["📉 Price & Indicators", "📰 News & Sentiment", "🤖 Feature Importance"])

    with tab1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=[f"{symbol} · Candlestick · MA · Bollinger Bands",
                                            "RSI (14)", "Volume"])

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#facc15', width=1.2), name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='#fb923c', width=1.2), name="MA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(148,163,184,0.4)', dash='dot'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(148,163,184,0.4)', dash='dot'),
                                 fill='tonexty', fillcolor='rgba(148,163,184,0.07)', name="BB Band"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#22d3ee', width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.7)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.7)",  row=2, col=1)
        bar_colors = ['#22c55e' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef4444' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=bar_colors, name="Volume"), row=3, col=1)

        fig.update_layout(height=720, template="plotly_dark", xaxis_rangeslider_visible=False,
                          margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')

    with tab2:
        color = "#10b981" if sentiment['label'] == "BULLISH" else "#ef4444" if sentiment['label'] == "BEARISH" else "#94a3b8"
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:16px;margin-bottom:16px;">
                <span style="font-size:18px;font-weight:bold;color:{color};">{sentiment['label']}</span>
                &nbsp;&nbsp;<span style="color:#94a3b8;">Polarity Score: {sentiment['score']:.3f}</span>
            </div>
        """, unsafe_allow_html=True)
        for idx, h in enumerate(sentiment['headlines'], 1):
            st.markdown(f"**{idx}.** {h}")

    with tab3:
        feature_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance")
        fig_imp = go.Figure(go.Bar(x=feature_df['Importance'], y=feature_df['Feature'], orientation='h',
                                   marker=dict(color=feature_df['Importance'], colorscale='Teal')))
        fig_imp.update_layout(height=520, template="plotly_dark", margin=dict(l=20,r=20,t=20,b=20),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
#   PAGE 2: MARKET SCREENER
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.title("🌍 EGX Market Screener")
    st.caption(f"Scanning {len(EGX_TICKERS)} EGX stocks · {screener_horizon}-Day Horizon · Results cached for 30 minutes")
    st.markdown("---")

    with st.spinner(f"Scanning {len(EGX_TICKERS)} stocks... this may take a minute the first time."):
        screen_df = run_market_screener(tuple(EGX_TICKERS), screener_horizon)

    if screen_df.empty:
        st.warning("Could not retrieve data for any tickers. Please try again later.")
        st.stop()

    # ── Summary Counts ──
    buy_df  = screen_df[screen_df['Recommendation'] == 'BUY'].sort_values('Confidence %', ascending=False)
    sell_df = screen_df[screen_df['Recommendation'] == 'SELL'].sort_values('Confidence %', ascending=False)
    hold_df = screen_df[screen_df['Recommendation'] == 'HOLD'].sort_values('Confidence %', ascending=False)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("📊 Total Scanned", len(screen_df))
    s2.metric("🟢 BUY Signals",   len(buy_df))
    s3.metric("🔴 SELL Signals",  len(sell_df))
    s4.metric("🟡 HOLD Signals",  len(hold_df))

    st.markdown("---")

    # ── BUY RECOMMENDATIONS ──
    st.subheader("🟢 Top BUY Candidates")
    if buy_df.empty:
        st.info("No strong BUY signals found in this scan.")
    else:
        cols = st.columns(min(len(buy_df), 4))
        for i, (_, row) in enumerate(buy_df.head(4).iterrows()):
            with cols[i]:
                st.markdown(f"""
                    <div style="background:rgba(16,185,129,0.12);border:1px solid #10b981;
                                border-radius:10px;padding:14px;text-align:center;">
                        <div style="font-size:20px;font-weight:bold;color:#10b981;">{row['Ticker']}</div>
                        <div style="color:#fff;font-size:15px;">{row['Close (EGP)']} EGP</div>
                        <div style="color:#94a3b8;font-size:13px;">Day: {row['Day Change %']:+.2f}%</div>
                        <div style="color:#22d3ee;font-size:13px;">RSI: {row['RSI']}</div>
                        <div style="margin-top:8px;">
                            <span class="rec-badge-buy">BUY · {row['Confidence %']}% conf.</span>
                        </div>
                        <div style="color:#94a3b8;font-size:11px;margin-top:4px;">{row['Sentiment']}</div>
                    </div>
                """, unsafe_allow_html=True)
        if len(buy_df) > 4:
            st.markdown("##### All BUY signals")
            st.dataframe(buy_df[['Ticker','Close (EGP)','Day Change %','RSI','Confidence %','Sentiment']].reset_index(drop=True),
                         use_container_width=True)

    st.markdown("---")

    # ── SELL RECOMMENDATIONS ──
    st.subheader("🔴 Top SELL Candidates")
    if sell_df.empty:
        st.info("No strong SELL signals found in this scan.")
    else:
        cols = st.columns(min(len(sell_df), 4))
        for i, (_, row) in enumerate(sell_df.head(4).iterrows()):
            with cols[i]:
                st.markdown(f"""
                    <div style="background:rgba(239,68,68,0.12);border:1px solid #ef4444;
                                border-radius:10px;padding:14px;text-align:center;">
                        <div style="font-size:20px;font-weight:bold;color:#ef4444;">{row['Ticker']}</div>
                        <div style="color:#fff;font-size:15px;">{row['Close (EGP)']} EGP</div>
                        <div style="color:#94a3b8;font-size:13px;">Day: {row['Day Change %']:+.2f}%</div>
                        <div style="color:#22d3ee;font-size:13px;">RSI: {row['RSI']}</div>
                        <div style="margin-top:8px;">
                            <span class="rec-badge-sell">SELL · {row['Confidence %']}% conf.</span>
                        </div>
                        <div style="color:#94a3b8;font-size:11px;margin-top:4px;">{row['Sentiment']}</div>
                    </div>
                """, unsafe_allow_html=True)
        if len(sell_df) > 4:
            st.markdown("##### All SELL signals")
            st.dataframe(sell_df[['Ticker','Close (EGP)','Day Change %','RSI','Confidence %','Sentiment']].reset_index(drop=True),
                         use_container_width=True)

    st.markdown("---")

    # ── FULL TABLE ──
    st.subheader("📋 Full Market Table")
    def color_rec(val):
        colors = {'BUY': 'color:#10b981;font-weight:bold',
                  'SELL': 'color:#ef4444;font-weight:bold',
                  'HOLD': 'color:#eab308;font-weight:bold'}
        return colors.get(val, '')
    display_cols = ['Ticker','Close (EGP)','Day Change %','RSI','Signal','Confidence %','Sentiment','Recommendation']
    styled = screen_df[display_cols].reset_index(drop=True).style.map(color_rec, subset=['Recommendation'])
    st.dataframe(styled, use_container_width=True)

    st.caption("⚠️ AI-generated signals. Not financial advice. Always do your own research before investing.")
