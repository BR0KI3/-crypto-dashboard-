from __future__ import annotations
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Market Radar Pro",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {max-width:860px;padding-top:.8rem;padding-bottom:3rem}
.card{border:1px solid rgba(128,128,128,.22);border-radius:18px;padding:15px;margin:10px 0}
.name{font-size:1.15rem;font-weight:800}
.muted{opacity:.68;font-size:.84rem}
.big{font-size:1.9rem;font-weight:850}
.good{font-weight:800}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.13);padding:8px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

DEX = "https://api.dexscreener.com"
UA = {"accept":"application/json","user-agent":"market-radar-pro/5.0"}

MAJOR_CRYPTO = {
    "BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","XRP":"XRP-USD",
    "DOGE":"DOGE-USD","ADA":"ADA-USD","AVAX":"AVAX-USD","LINK":"LINK-USD",
    "LTC":"LTC-USD","BCH":"BCH-USD","DOT":"DOT-USD","SHIB":"SHIB-USD",
}
STOCKS = "NVDA,TSLA,AAPL,AMD,MSFT,AMZN,META,GOOGL,COIN,MSTR,PLTR,SOFI"

def num(v, default=0.0):
    try:
        if v is None: return default
        return float(v)
    except: return default

def money(v):
    v=num(v, float("nan"))
    if math.isnan(v): return "N/A"
    if abs(v)>=1000: return f"${v:,.2f}"
    if abs(v)>=1: return f"${v:,.4f}"
    if abs(v)>=.01: return f"${v:,.6f}"
    return f"${v:,.10f}"

def compact(v):
    v=num(v)
    if abs(v)>=1_000_000_000:return f"${v/1_000_000_000:.2f}B"
    if abs(v)>=1_000_000:return f"${v/1_000_000:.2f}M"
    if abs(v)>=1_000:return f"${v/1_000:.1f}K"
    return f"${v:.0f}"

def age_from_ms(ms):
    if not ms: return ("Unknown", None)
    dt=datetime.fromtimestamp(num(ms)/1000,tz=timezone.utc)
    seconds=max((datetime.now(timezone.utc)-dt).total_seconds(),0)
    if seconds<3600: return (f"{int(seconds//60)} min", dt)
    if seconds<86400: return (f"{seconds/3600:.1f} hr", dt)
    if seconds<86400*30: return (f"{seconds/86400:.1f} d", dt)
    return (f"{seconds/(86400*30):.1f} mo", dt)

def safe_get(url, params=None, timeout=12):
    r=requests.get(url,params=params,headers=UA,timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------------- TECHNICAL ENGINE: STOCKS + MAJOR CRYPTO ----------------

def calc_rsi(s, n=14):
    d=s.diff()
    up=d.clip(lower=0)
    dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False).mean()
    ad=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return (100-(100/(1+rs))).fillna(50)

def technical_read(df):
    if df is None or df.empty or len(df)<35:
        return None

    close=df["Close"].dropna()
    volume=df["Volume"].reindex(close.index).fillna(0) if "Volume" in df else pd.Series(0,index=close.index)
    if len(close)<35:return None

    e9=close.ewm(span=9,adjust=False).mean()
    e21=close.ewm(span=21,adjust=False).mean()
    e50=close.ewm(span=50,adjust=False).mean()
    rsi=calc_rsi(close)
    macd=close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean()
    mh=macd-macd.ewm(span=9,adjust=False).mean()

    price=float(close.iloc[-1])
    score=50.0
    score += 12 if e9.iloc[-1]>e21.iloc[-1] else -10
    score += 12 if e21.iloc[-1]>e50.iloc[-1] else -8
    score += 6 if price>e9.iloc[-1] else -4
    rv=float(rsi.iloc[-1])
    if 52<=rv<=68: score+=10
    elif rv>78: score-=10
    elif rv<38: score-=7
    if mh.iloc[-1]>0: score+=8
    if mh.iloc[-1]>mh.iloc[-2]: score+=5

    r5=(price/float(close.iloc[-6])-1)*100 if len(close)>6 else 0
    r20=(price/float(close.iloc[-21])-1)*100 if len(close)>21 else 0
    if r5>0:score+=5
    if r20>0:score+=4
    if r5>12:score-=8  # avoid chasing

    score=round(max(0,min(100,score)),1)

    # Recent ATR-like risk estimate.
    hi=df["High"].reindex(close.index).tail(15)
    lo=df["Low"].reindex(close.index).tail(15)
    atr=float((hi-lo).mean()) if len(hi) else price*.02
    atr=max(atr,price*.005)

    if score>=76:
        call="UP BIAS"
        action="BUY / ADD ONLY IF ENTRY FITS"
    elif score>=62:
        call="UP / HOLD"
        action="HOLD or WATCH FOR ENTRY"
    elif score<=36:
        call="DOWN BIAS"
        action="AVOID / REDUCE"
    else:
        call="SIDEWAYS / UNCLEAR"
        action="WAIT"

    # Entry zones avoid asking user to buy an extended candle.
    if r5>8:
        entry_lo=price-1.6*atr
        entry_hi=price-.7*atr
        entry_note="Wait for a pullback; price is extended."
    elif score>=62:
        entry_lo=price-.5*atr
        entry_hi=price+.15*atr
        entry_note="Near-current entry zone if trend stays valid."
    else:
        entry_lo=price-1.0*atr
        entry_hi=price-.4*atr
        entry_note="No urgent entry. Wait for confirmation."

    stop=max(price-2.0*atr,0)
    tp1=price+1.5*atr
    tp2=price+3.0*atr

    return {
        "price":price,"score":score,"call":call,"action":action,"rsi":rv,
        "r5":r5,"r20":r20,"entry_lo":max(entry_lo,0),"entry_hi":max(entry_hi,0),
        "entry_note":entry_note,"stop":stop,"tp1":tp1,"tp2":tp2,
    }

@st.cache_data(ttl=45,show_spinner=False)
def load_yf(symbols, period="6mo", interval="1d"):
    symbols=list(dict.fromkeys(symbols))
    if not symbols:return {}
    data=yf.download(
        tickers=" ".join(symbols),period=period,interval=interval,
        auto_adjust=True,progress=False,threads=True,group_by="ticker"
    )
    out={}
    if len(symbols)==1:
        out[symbols[0]]=data
    else:
        for s in symbols:
            try:
                d=data[s].dropna(how="all")
                if not d.empty:out[s]=d
            except: pass
    return out

# ---------------- EARLY MEME ENGINE: DEX SCREENER ----------------

@st.cache_data(ttl=20,show_spinner=False)
def newest_candidates():
    # Latest profiles = newly surfaced/profiled on-chain tokens.
    profiles=safe_get(f"{DEX}/token-profiles/latest/v1")
    boosts=safe_get(f"{DEX}/token-boosts/latest/v1")

    combined=[]
    seen=set()
    for x in (profiles or []) + (boosts or []):
        chain=x.get("chainId")
        addr=x.get("tokenAddress")
        if chain and addr and (chain,addr) not in seen:
            seen.add((chain,addr))
            combined.append({"chainId":chain,"tokenAddress":addr})
    return combined[:80]

def chunks(items,n=30):
    for i in range(0,len(items),n):
        yield items[i:i+n]

@st.cache_data(ttl=20,show_spinner=False)
def fetch_pairs_for_candidates():
    candidates=newest_candidates()
    by_chain={}
    for c in candidates:
        by_chain.setdefault(c["chainId"],[]).append(c["tokenAddress"])

    pairs=[]
    # Batch token lookups by chain; DEX Screener allows multiple token addresses per request.
    for chain, addresses in by_chain.items():
        for group in chunks(addresses,30):
            try:
                joined=",".join(group)
                got=safe_get(f"{DEX}/tokens/v1/{chain}/{joined}")
                if isinstance(got,list):pairs.extend(got)
            except Exception:
                continue

    # Keep best-liquidity pair per base token.
    best={}
    for p in pairs:
        addr=(p.get("baseToken") or {}).get("address")
        chain=p.get("chainId")
        if not addr or not chain:continue
        key=(chain,addr)
        liq=num((p.get("liquidity") or {}).get("usd"))
        if key not in best or liq>num((best[key].get("liquidity") or {}).get("usd")):
            best[key]=p
    return list(best.values())

def meme_score(p):
    liq=num((p.get("liquidity") or {}).get("usd"))
    vol5=num((p.get("volume") or {}).get("m5"))
    vol1h=num((p.get("volume") or {}).get("h1"))
    pc5=num((p.get("priceChange") or {}).get("m5"))
    pc1h=num((p.get("priceChange") or {}).get("h1"))
    tx5=(p.get("txns") or {}).get("m5") or {}
    tx1=(p.get("txns") or {}).get("h1") or {}
    buys5=num(tx5.get("buys")); sells5=num(tx5.get("sells"))
    buys1=num(tx1.get("buys")); sells1=num(tx1.get("sells"))
    age, dt=age_from_ms(p.get("pairCreatedAt"))
    hours=((datetime.now(timezone.utc)-dt).total_seconds()/3600) if dt else 9999

    s=35.0
    # Liquidity: enough to trade, but not necessarily already huge.
    if liq>=100_000:s+=18
    elif liq>=50_000:s+=14
    elif liq>=20_000:s+=9
    elif liq<8_000:s-=20

    # Real activity.
    if vol1h>=100_000:s+=15
    elif vol1h>=30_000:s+=10
    elif vol1h>=10_000:s+=5

    total5=buys5+sells5
    ratio5=(buys5+1)/(sells5+1)
    if total5>=20 and ratio5>=1.5:s+=14
    elif total5>=8 and ratio5>=1.15:s+=7
    elif sells5>buys5*1.5:s-=12

    # Positive movement, but penalize already-blown-up entries.
    if 1<=pc5<=12:s+=8
    elif pc5<-8:s-=10
    if 3<=pc1h<=35:s+=10
    elif pc1h>80:s-=22
    elif pc1h>45:s-=12
    elif pc1h<-20:s-=12

    # Earlier gets a modest bonus, not a blind buy.
    if hours<=1:s+=8
    elif hours<=6:s+=6
    elif hours<=24:s+=3

    # Volume relative to liquidity can reveal real attention.
    if liq>0 and vol1h/liq>=.5:s+=5

    s=round(max(0,min(100,s)),1)

    if liq<8000:
        conclusion="AVOID"
        reason="Liquidity is too thin."
    elif pc1h>80:
        conclusion="WAIT — ALREADY PUMPED"
        reason="The 1-hour move is too extended; chasing is high risk."
    elif s>=76:
        conclusion="EARLY UP BIAS"
        reason="Strong early liquidity, activity and buyer pressure."
    elif s>=62:
        conclusion="WATCH / POSSIBLE UP"
        reason="Promising, but not enough confirmation for the strongest grade."
    elif s<=38:
        conclusion="DOWN / AVOID"
        reason="Weak liquidity, activity, momentum or buyer pressure."
    else:
        conclusion="WAIT"
        reason="Mixed early signals."

    price=num(p.get("priceUsd"))
    # Suggested entry is deliberately conservative when extended.
    if pc5>12 or pc1h>35:
        entry_lo=price*.88
        entry_hi=price*.95
        entry_note="Wait for pullback — do not chase this candle."
    elif s>=62:
        entry_lo=price*.97
        entry_hi=price*1.01
        entry_note="Entry zone only while buyer pressure and liquidity hold."
    else:
        entry_lo=price*.92
        entry_hi=price*.97
        entry_note="No immediate entry; wait for better confirmation."

    return {
        "score":s,"conclusion":conclusion,"reason":reason,"age":age,"hours":hours,
        "liq":liq,"vol5":vol5,"vol1h":vol1h,"pc5":pc5,"pc1h":pc1h,
        "buys5":buys5,"sells5":sells5,"buys1":buys1,"sells1":sells1,
        "entry_lo":entry_lo,"entry_hi":entry_hi,"entry_note":entry_note,
    }

def render_meme(p):
    base=p.get("baseToken") or {}
    q=p.get("quoteToken") or {}
    name=base.get("name") or "Unknown"
    sym=base.get("symbol") or "?"
    addr=base.get("address") or ""
    chain=p.get("chainId") or "?"
    dex=p.get("dexId") or "?"
    result=meme_score(p)
    price=num(p.get("priceUsd"))

    st.markdown(f"""
    <div class="card">
      <div class="name">{name} ({sym})</div>
      <div class="muted">{chain} • {dex} • paired with {q.get("symbol","?")} • age {result["age"]}</div>
      <div style="display:flex;justify-content:space-between;align-items:end;margin-top:8px">
        <div><div class="big">{money(price)}</div><div class="muted">{result["conclusion"]}</div></div>
        <div style="text-align:right"><div class="big">{result["score"]}</div><div class="muted">early score / 100</div></div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    st.write(result["reason"])
    a,b,c=st.columns(3)
    a.metric("Liquidity",compact(result["liq"]))
    b.metric("1h volume",compact(result["vol1h"]))
    c.metric("1h change",f'{result["pc1h"]:+.1f}%')
    a,b,c=st.columns(3)
    a.metric("5m buys",int(result["buys5"]))
    b.metric("5m sells",int(result["sells5"]))
    c.metric("5m change",f'{result["pc5"]:+.1f}%')

    st.markdown("**Suggested entry zone**")
    st.write(f'{money(result["entry_lo"])} – {money(result["entry_hi"])}')
    st.caption(result["entry_note"])

    st.caption(f"Contract • {chain}")
    st.code(addr,language=None)

    if p.get("url"):
        st.link_button("Open on DEX Screener",p["url"],use_container_width=True)

def render_technical(name,ticker,read,kind):
    if not read:return
    st.markdown(f"""
    <div class="card">
      <div class="name">{name}</div>
      <div class="muted">{kind} • {ticker}</div>
      <div style="display:flex;justify-content:space-between;align-items:end;margin-top:8px">
        <div><div class="big">{money(read["price"])}</div><div class="muted">{read["call"]}</div></div>
        <div style="text-align:right"><div class="big">{read["score"]}</div><div class="muted">trend score / 100</div></div>
      </div>
    </div>
    """,unsafe_allow_html=True)
    st.markdown(f'**Conclusion:** {read["action"]}')
    c1,c2,c3=st.columns(3)
    c1.metric("RSI",f'{read["rsi"]:.1f}')
    c2.metric("5-day",f'{read["r5"]:+.1f}%')
    c3.metric("20-day",f'{read["r20"]:+.1f}%')
    st.markdown("**Entry zone**")
    st.write(f'{money(read["entry_lo"])} – {money(read["entry_hi"])}')
    st.caption(read["entry_note"])
    s1,s2,s3=st.columns(3)
    s1.metric("Risk stop",money(read["stop"]))
    s2.metric("Target 1",money(read["tp1"]))
    s3.metric("Target 2",money(read["tp2"]))

# ---------------- UI ----------------

st.title("📡 Market Radar Pro")
st.caption("Early meme coins + crypto + stocks, with anti-chase logic and simple conclusions.")

mode=st.segmented_control(
    "Scanner",
    ["🔥 New Meme Coins","₿ Crypto","📈 Stocks","⭐ Watchlist"],
    default="🔥 New Meme Coins"
)

with st.expander("How to read the calls"):
    st.write(
        "UP BIAS means the data currently leans bullish. HOLD means an existing position still has "
        "support from the trend. WAIT means the setup is unclear or extended. DOWN / AVOID means the "
        "current data is unfavorable. None of these are guarantees of profit."
    )

# Independent auto-refresh only while this function is visible/session active.
@st.fragment(run_every="30s")
def live_meme_section():
    st.caption(f'Live refresh • {datetime.now().astimezone().strftime("%I:%M:%S %p")}')
    try:
        pairs=fetch_pairs_for_candidates()
    except Exception as e:
        st.error(f"DEX data temporarily unavailable: {e}")
        return

    ranked=[]
    for p in pairs:
        r=meme_score(p)
        # Useful filters: remove obvious dust / dead pairs.
        if r["liq"]>=8_000 and (r["vol1h"]>=2_000 or r["buys1"]+r["sells1"]>=15):
            ranked.append((r["score"],r["hours"],p))
    ranked.sort(key=lambda x:(-x[0],x[1]))

    if not ranked:
        st.info("No fresh candidates passed the minimum liquidity/activity filter on this refresh.")
        return

    st.subheader("Fresh candidates")
    st.caption("Ranked for early quality, not just biggest price pump. Refreshes about every 30 seconds while open.")
    for _,__,p in ranked[:15]:
        render_meme(p)

if mode=="🔥 New Meme Coins":
    live_meme_section()

elif mode=="₿ Crypto":
    raw=st.text_input("Crypto tickers",",".join(MAJOR_CRYPTO.keys()))
    syms=[s.strip().upper() for s in raw.split(",") if s.strip()][:15]
    yf_syms=[MAJOR_CRYPTO.get(s,f"{s}-USD") for s in syms]
    with st.spinner("Loading crypto..."):
        data=load_yf(yf_syms)
    reads=[]
    for s,y in zip(syms,yf_syms):
        rd=technical_read(data.get(y))
        if rd:reads.append((rd["score"],s,y,rd))
    reads.sort(reverse=True,key=lambda x:x[0])
    for _,s,y,rd in reads:
        render_technical(s,y,rd,"Crypto")

elif mode=="📈 Stocks":
    raw=st.text_input("Stock tickers",STOCKS)
    syms=[s.strip().upper() for s in raw.split(",") if s.strip()][:20]
    with st.spinner("Loading stocks..."):
        data=load_yf(syms)
    reads=[]
    for s in syms:
        rd=technical_read(data.get(s))
        if rd:reads.append((rd["score"],s,rd))
    reads.sort(reverse=True,key=lambda x:x[0])
    for _,s,rd in reads:
        render_technical(s,s,rd,"Stock")

else:
    st.write("Paste stock or crypto tickers separated by commas.")
    raw=st.text_input("Watchlist","NVDA,TSLA,BTC,ETH,SOL,DOGE")
    items=[x.strip().upper() for x in raw.split(",") if x.strip()][:20]
    mapped=[MAJOR_CRYPTO.get(x,x) for x in items]
    # If ticker is a common crypto symbol, map to -USD. Stocks stay unchanged.
    mapped=[MAJOR_CRYPTO.get(x, x) for x in items]
    with st.spinner("Loading watchlist..."):
        data=load_yf(mapped)
    reads=[]
    for original,y in zip(items,mapped):
        rd=technical_read(data.get(y))
        if rd:reads.append((rd["score"],original,y,rd))
    reads.sort(reverse=True,key=lambda x:x[0])
    for _,original,y,rd in reads:
        kind="Crypto" if y.endswith("-USD") else "Stock"
        render_technical(original,y,rd,kind)

st.divider()
st.caption(
    "No scanner can know in advance whether a trade will make money. New meme coins are especially risky: "
    "liquidity can disappear, contracts can be malicious, and price can collapse before a signal refreshes. "
    "Use the score as a filter, verify the contract, and size risk small."
)
