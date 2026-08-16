from datetime import datetime, timezone
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Simple Market Scanner",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {max-width: 760px; padding-top: 1rem; padding-bottom: 3rem;}
.asset-card {
  border:1px solid rgba(128,128,128,.25);
  border-radius:18px;
  padding:16px;
  margin:10px 0;
}
.asset-title {font-size:1.25rem;font-weight:800;}
.asset-sub {opacity:.72;font-size:.88rem;}
.score {font-size:2rem;font-weight:850;}
.address {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:.78rem;
  overflow-wrap:anywhere;
  opacity:.85;
}
</style>
""", unsafe_allow_html=True)

# Curated CoinGecko IDs for common crypto symbols.
# Add more here if needed.
CRYPTO_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "UNI": "uniswap",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOT": "polkadot",
    "PEPE": "pepe",
    "BONK": "bonk",
    "WIF": "dogwifcoin",
    "TRUMP": "official-trump",
}

def age_text(dt):
    if not dt:
        return "Unknown"
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = max((now - dt).days, 0)
    years = days // 365
    months = (days % 365) // 30
    if years:
        return f"{years}y {months}mo"
    if months:
        return f"{months}mo {days % 30}d"
    return f"{days}d"

def simple_score(prices):
    s = pd.Series(prices).dropna()
    if len(s) < 30:
        return 50.0, "WAIT"

    ema9 = s.ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = s.ewm(span=21, adjust=False).mean().iloc[-1]
    ema50 = s.ewm(span=50, adjust=False).mean().iloc[-1] if len(s) >= 50 else s.mean()

    delta = s.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    au = up.ewm(alpha=1/14, adjust=False).mean()
    ad = dn.ewm(alpha=1/14, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    rsi = float((100 - 100/(1+rs)).fillna(50).iloc[-1])

    score = 50
    if ema9 > ema21: score += 12
    else: score -= 8
    if ema21 > ema50: score += 12
    else: score -= 8
    if s.iloc[-1] > ema9: score += 6
    if 52 <= rsi <= 68: score += 12
    elif rsi > 78: score -= 8
    elif rsi < 38: score -= 5

    ret5 = (s.iloc[-1] / s.iloc[-6] - 1) * 100 if len(s) > 6 else 0
    if ret5 > 0: score += 6
    if ret5 > 8: score -= 6

    score = float(max(0, min(100, round(score, 1))))
    if score >= 72:
        action = "BUY SETUP"
    elif score >= 62:
        action = "WATCH"
    elif score <= 35:
        action = "AVOID"
    else:
        action = "WAIT"
    return score, action

@st.cache_data(ttl=60, show_spinner=False)
def get_crypto(symbol):
    symbol = symbol.upper().strip()
    coin_id = CRYPTO_MAP.get(symbol)
    if not coin_id:
        raise ValueError(f"{symbol} is not in the crypto map yet.")

    detail_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    market_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    headers = {"accept": "application/json", "user-agent": "simple-market-scanner"}

    d = requests.get(detail_url, headers=headers, timeout=15)
    d.raise_for_status()
    detail = d.json()

    m = requests.get(
        market_url,
        headers=headers,
        params={"vs_currency": "usd", "days": "30", "interval": "hourly"},
        timeout=15
    )
    m.raise_for_status()
    market = m.json()

    prices = [p[1] for p in market.get("prices", [])]
    score, action = simple_score(prices)

    current = detail.get("market_data", {}).get("current_price", {}).get("usd")
    name = detail.get("name", symbol)

    # Prefer a non-empty contract address. Native coins may not have one.
    platforms = detail.get("platforms") or {}
    contracts = [(network, addr) for network, addr in platforms.items() if addr]
    network = contracts[0][0] if contracts else None
    contract = contracts[0][1] if contracts else None

    genesis = detail.get("genesis_date")
    launch_dt = None
    if genesis:
        try:
            launch_dt = datetime.fromisoformat(genesis).replace(tzinfo=timezone.utc)
        except Exception:
            pass

    # If genesis date missing, approximate age from earliest returned historical point.
    if launch_dt is None and market.get("prices"):
        ts = market["prices"][0][0] / 1000
        # This is only the beginning of our 30-day lookback, so don't label it as launch.
        launch_label = "Launch date unavailable"
        age = "Unknown"
    else:
        launch_label = genesis or "Unknown"
        age = age_text(launch_dt)

    return {
        "type": "Crypto",
        "name": name,
        "symbol": symbol,
        "price": current,
        "score": score,
        "action": action,
        "network": network,
        "address": contract,
        "age": age,
        "since": launch_label,
    }

@st.cache_data(ttl=120, show_spinner=False)
def get_stock(symbol):
    symbol = symbol.upper().strip()
    t = yf.Ticker(symbol)

    hist = t.history(period="max", interval="1d", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No stock data found for {symbol}.")

    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    closes = hist["Close"].tail(90).dropna()
    score, action = simple_score(closes)

    price = float(closes.iloc[-1])
    first_dt = hist.index[0].to_pydatetime()
    if first_dt.tzinfo is None:
        first_dt = first_dt.replace(tzinfo=timezone.utc)

    return {
        "type": "Stock",
        "name": info.get("shortName") or info.get("longName") or symbol,
        "symbol": symbol,
        "price": price,
        "score": score,
        "action": action,
        "network": info.get("exchange") or info.get("fullExchangeName") or "Exchange unavailable",
        "address": None,
        "age": age_text(first_dt),
        "since": first_dt.date().isoformat(),
    }

def money(v):
    if v is None:
        return "N/A"
    if v >= 1000:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:,.4f}"
    if v >= .01:
        return f"${v:,.5f}"
    return f"${v:,.8f}"

def show_card(a):
    icon = "🟢" if a["action"] == "BUY SETUP" else "🟡" if a["action"] == "WATCH" else "⚪"
    st.markdown(
        f"""
        <div class="asset-card">
          <div class="asset-title">{icon} {a['name']} ({a['symbol']})</div>
          <div class="asset-sub">{a['type']}</div>
          <div style="display:flex;justify-content:space-between;align-items:end;margin-top:10px;">
            <div>
              <div style="font-size:1.35rem;font-weight:750;">{money(a['price'])}</div>
              <div class="asset-sub">{a['action']}</div>
            </div>
            <div style="text-align:right;">
              <div class="score">{a['score']}</div>
              <div class="asset-sub">score / 100</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    c1.metric("How long it's been out", a["age"])
    c2.metric("First/launch date", a["since"])

    if a["type"] == "Crypto":
        st.markdown("**Network / contract address**")
        if a["address"]:
            st.write(a["network"] or "Network")
            st.code(a["address"], language=None)
        else:
            st.info("Native coin — no token contract address is listed for this asset.")
    else:
        st.markdown("**Stock identifier / exchange**")
        st.code(f"{a['symbol']} • {a['network']}", language=None)
        st.caption("Regular stocks do not have blockchain contract addresses.")

st.title("📈 Simple Market Scanner")
st.caption("Simple cards: price, signal, score, age, and address/identifier.")

mode = st.segmented_control("Asset type", ["Crypto", "Stocks"], default="Crypto")

if mode == "Crypto":
    default = "BTC,ETH,SOL,DOGE,SHIB,PEPE,BONK,WIF"
    symbols_text = st.text_input("Coins", value=default)
else:
    default = "AAPL,TSLA,NVDA,AMD,MSFT,AMZN"
    symbols_text = st.text_input("Stocks", value=default)

symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()][:12]

if st.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

assets = []
errors = []

with st.spinner("Loading market data..."):
    for s in symbols:
        try:
            assets.append(get_crypto(s) if mode == "Crypto" else get_stock(s))
        except Exception as e:
            errors.append(str(e))

assets.sort(key=lambda x: x["score"], reverse=True)

if errors:
    with st.expander("Some assets could not be loaded"):
        for e in errors:
            st.write("•", e)

for a in assets:
    show_card(a)

st.caption(
    "Age for stocks uses the earliest daily trading history available from the data provider. "
    "Crypto launch age uses the provider's genesis date when available. Contract addresses must "
    "always be verified before trading because symbols can be copied by scam tokens."
)
