import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

st.set_page_config(
    page_title="Crypto Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "SHIB/USD"]

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
.signal-card {
    border: 1px solid rgba(128,128,128,.28);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 12px;
}
.big-score {font-size: 2.2rem; font-weight: 800; line-height: 1;}
.action {font-size: 1.15rem; font-weight: 750;}
.small {opacity: .75; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Indicators ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    x = df.copy()
    x["ema9"] = ema(x["close"], 9)
    x["ema21"] = ema(x["close"], 21)
    x["ema50"] = ema(x["close"], 50)
    x["rsi14"] = rsi(x["close"], 14)
    macd = ema(x["close"], 12) - ema(x["close"], 26)
    x["macd_hist"] = macd - ema(macd, 9)
    x["atr14"] = atr(x, 14)
    x["vol_sma20"] = x["volume"].rolling(20).mean()
    x["volume_ratio"] = x["volume"] / x["vol_sma20"].replace(0, np.nan)
    x["prev_high20"] = x["high"].shift(1).rolling(20).max()
    x["breakout_pct"] = ((x["close"] / x["prev_high20"]) - 1) * 100
    x["ret3"] = x["close"].pct_change(3) * 100
    x["ret12"] = x["close"].pct_change(12) * 100
    return x

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def score_row(row, prev_macd):
    score = 0.0

    if row["ema9"] > row["ema21"]:
        score += 12
    if row["ema21"] > row["ema50"]:
        score += 12
    if row["close"] > row["ema9"]:
        score += 6

    rv = row["rsi14"]
    if 52 <= rv <= 68:
        score += 15
    elif 48 <= rv < 52 or 68 < rv <= 74:
        score += 8
    elif rv > 80:
        score -= 8

    if row["macd_hist"] > 0:
        score += 10
        if row["macd_hist"] > prev_macd:
            score += 5

    vr = row["volume_ratio"] if pd.notna(row["volume_ratio"]) else 0
    score += clamp((vr - .8) / 1.2 * 15, 0, 15)

    bp = row["breakout_pct"] if pd.notna(row["breakout_pct"]) else -99
    if bp >= 0:
        score += clamp(8 + bp * 5, 0, 15)
    elif bp > -.8:
        score += 4

    r3 = row["ret3"] if pd.notna(row["ret3"]) else 0
    r12 = row["ret12"] if pd.notna(row["ret12"]) else 0
    if r3 > 0: score += 4
    if r12 > 0: score += 4
    if r3 > 0 and r12 > 0 and r3 < 5: score += 2
    if r3 > 8: score -= 10
    if row["atr14"] / row["close"] > .08: score -= 10

    return round(clamp(score, 0, 100), 1)

@st.cache_resource
def client():
    # Alpaca crypto historical data can be requested without credentials.
    return CryptoHistoricalDataClient()

@st.cache_data(ttl=45, show_spinner=False)
def fetch(symbol, minutes, lookback=300):
    start = datetime.now(timezone.utc) - timedelta(minutes=minutes * lookback * 3)
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        start=start,
        limit=lookback,
    )
    df = client().get_crypto_bars(req).df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)

    return df[["open", "high", "low", "close", "volume"]].dropna().sort_index()

def analyze(symbol, minutes, threshold):
    df = fetch(symbol, minutes)
    if len(df) < 60:
        raise ValueError("Not enough market data.")

    x = add_indicators(df)
    r = x.iloc[-1]
    prev_macd = x["macd_hist"].iloc[-2]
    score = score_row(r, prev_macd)

    price = float(r["close"])
    a = float(r["atr14"])
    risk = max(a * 1.8, price * .004)

    return {
        "symbol": symbol,
        "price": price,
        "score": score,
        "action": "BUY SETUP" if score >= threshold else "WAIT",
        "stop": price - risk,
        "tp1": price + risk,
        "tp2": price + risk * 2,
        "tp3": price + risk * 3,
        "rsi": float(r["rsi14"]),
        "volume_ratio": float(r["volume_ratio"]) if pd.notna(r["volume_ratio"]) else 0,
        "breakout": float(r["breakout_pct"]) if pd.notna(r["breakout_pct"]) else -99,
        "chart": x.tail(80),
    }

def money(v):
    if v >= 1000: return f"${v:,.2f}"
    if v >= 1: return f"${v:,.4f}"
    if v >= .01: return f"${v:,.5f}"
    return f"${v:,.8f}"

# ---------- UI ----------
st.title("📈 Crypto Signal Dashboard")
st.caption("Market scanner — setup scores are not guaranteed win probabilities.")

with st.expander("⚙️ Scanner settings"):
    symbols_text = st.text_input(
        "Crypto pairs",
        value=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated Alpaca crypto pairs."
    )
    c1, c2 = st.columns(2)
    with c1:
        bar_minutes = st.selectbox("Candle size", [1, 5, 15, 30, 60], index=1)
    with c2:
        entry_threshold = st.slider("BUY threshold", 50, 90, 72)

symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()][:10]

if st.button("🔄 Refresh market now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.caption(f"Updated: {datetime.now().astimezone().strftime('%I:%M:%S %p')} • Data refresh cache: ~45 sec")

results = []
errors = []

with st.spinner("Scanning market..."):
    for symbol in symbols:
        try:
            results.append(analyze(symbol, bar_minutes, entry_threshold))
        except Exception as e:
            errors.append(f"{symbol}: {e}")

results.sort(key=lambda r: r["score"], reverse=True)

if errors:
    with st.expander("Some symbols could not be loaded"):
        for e in errors:
            st.write(e)

if not results:
    st.error("No market data loaded. Try refreshing or changing the symbols.")
    st.stop()

best = results[0]
st.subheader("Best setup right now")
a, b, c = st.columns(3)
a.metric(best["symbol"], money(best["price"]))
b.metric("Signal score", f'{best["score"]}/100')
c.metric("Action", best["action"])

st.divider()
st.subheader("Scanner")

for r in results:
    icon = "🟢" if r["action"] == "BUY SETUP" else "⚪"
    st.markdown(
        f"""
        <div class="signal-card">
          <div class="small">{icon} {r["action"]}</div>
          <div style="display:flex;justify-content:space-between;align-items:end;">
            <div>
              <div style="font-size:1.35rem;font-weight:800;">{r["symbol"]}</div>
              <div style="font-size:1.15rem;">{money(r["price"])}</div>
            </div>
            <div style="text-align:right;">
              <div class="big-score">{r["score"]}</div>
              <div class="small">score / 100</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(4)
    cols[0].metric("Stop", money(r["stop"]))
    cols[1].metric("TP1", money(r["tp1"]))
    cols[2].metric("TP2", money(r["tp2"]))
    cols[3].metric("TP3", money(r["tp3"]))

    stats = st.columns(3)
    stats[0].metric("RSI", f'{r["rsi"]:.1f}')
    stats[1].metric("Volume", f'{r["volume_ratio"]:.2f}×')
    stats[2].metric("Breakout", f'{r["breakout"]:.2f}%')

    with st.expander(f"Chart — {r['symbol']}"):
        chart_df = r["chart"][["close", "ema9", "ema21", "ema50"]].rename(
            columns={"close":"Price", "ema9":"EMA 9", "ema21":"EMA 21", "ema50":"EMA 50"}
        )
        st.line_chart(chart_df, use_container_width=True)

st.divider()
st.caption(
    "Risk warning: Crypto is highly volatile. This dashboard is for analysis/testing and "
    "does not predict future prices with certainty. Test any strategy before risking money."
)
