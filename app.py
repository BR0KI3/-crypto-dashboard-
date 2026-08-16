from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ============================================================
# MARKET INTELLIGENCE PRO v6
# ============================================================
# This app does NOT promise profit. It ranks current setups and
# makes risk-aware decisions from available market/on-chain data.
# ============================================================

st.set_page_config(
    page_title="Market Intelligence Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {max-width:1100px;padding-top:.8rem;padding-bottom:4rem}
.hero {
    border:1px solid rgba(128,128,128,.22);
    border-radius:22px;padding:18px;margin-bottom:12px;
}
.card {
    border:1px solid rgba(128,128,128,.20);
    border-radius:18px;padding:15px;margin:10px 0;
}
.asset {font-size:1.15rem;font-weight:800}
.muted {opacity:.68;font-size:.84rem}
.big {font-size:1.85rem;font-weight:850;line-height:1.05}
.verdict {font-size:1.05rem;font-weight:800}
.address {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem}
div[data-testid="stMetric"] {
    border:1px solid rgba(128,128,128,.12);
    border-radius:12px;padding:8px;
}
</style>
""", unsafe_allow_html=True)

DEX = "https://api.dexscreener.com"
CG = "https://api.coingecko.com/api/v3"
HEADERS = {"accept":"application/json","user-agent":"market-intelligence-pro/9.0"}

MAJOR_CRYPTO = {
    "BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","XRP":"XRP-USD",
    "DOGE":"DOGE-USD","ADA":"ADA-USD","AVAX":"AVAX-USD","LINK":"LINK-USD",
    "LTC":"LTC-USD","BCH":"BCH-USD","DOT":"DOT-USD","SHIB":"SHIB-USD",
    "SUI":"SUI-USD","PEPE":"PEPE24478-USD"
}
DEFAULT_STOCKS = "NVDA,TSLA,AAPL,AMD,MSFT,AMZN,META,GOOGL,COIN,MSTR,PLTR,SOFI"

# ---------------------------- BASIC HELPERS ----------------------------

def fnum(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def money(v):
    try:
        v=float(v)
    except Exception:
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    av=abs(v)
    if av >= 1000: return f"${v:,.2f}"
    if av >= 1: return f"${v:,.4f}"
    if av >= .01: return f"${v:,.6f}"
    return f"${v:,.10f}"

def compact(v):
    v=fnum(v)
    av=abs(v)
    if av>=1_000_000_000:return f"${v/1_000_000_000:.2f}B"
    if av>=1_000_000:return f"${v/1_000_000:.2f}M"
    if av>=1_000:return f"${v/1_000:.1f}K"
    return f"${v:.0f}"

def safe_json(url, params=None, timeout=12):
    r=requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def age_info(ms):
    if not ms:
        return {"label":"Unknown","hours":999999,"created":None}
    dt=datetime.fromtimestamp(fnum(ms)/1000,tz=timezone.utc)
    seconds=max((datetime.now(timezone.utc)-dt).total_seconds(),0)
    hours=seconds/3600
    if seconds < 3600:
        label=f"{int(seconds//60)} min"
    elif seconds < 86400:
        label=f"{hours:.1f} hr"
    elif seconds < 86400*30:
        label=f"{seconds/86400:.1f} d"
    else:
        label=f"{seconds/(86400*30):.1f} mo"
    return {"label":label,"hours":hours,"created":dt}

def badge_for_score(score):
    if score>=80:return "🟢"
    if score>=65:return "🟡"
    if score<=40:return "🔴"
    return "⚪"

def network_name(raw):
    m={
        "solana":"Solana","ethereum":"Ethereum","base":"Base","bsc":"BNB Chain",
        "arbitrum":"Arbitrum","polygon":"Polygon","avalanche":"Avalanche",
        "sui":"Sui","aptos":"Aptos","pulsechain":"PulseChain"
    }
    return m.get((raw or "").lower(), raw or "Unknown")

# ---------------------------- TECHNICAL ENGINE ----------------------------

def rsi(series, n=14):
    d=series.diff()
    gains=d.clip(lower=0)
    losses=-d.clip(upper=0)
    ag=gains.ewm(alpha=1/n,adjust=False).mean()
    al=losses.ewm(alpha=1/n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return (100-(100/(1+rs))).fillna(50)

def technical_engine(df):
    if df is None or df.empty:
        return None
    df=df.dropna(subset=["Close"]).copy()
    if len(df)<55:
        return None

    c=df["Close"].astype(float)
    h=df["High"].astype(float)
    l=df["Low"].astype(float)
    v=df["Volume"].astype(float).fillna(0)

    e9=c.ewm(span=9,adjust=False).mean()
    e21=c.ewm(span=21,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean()
    e200=c.ewm(span=min(200,len(c)),adjust=False).mean()
    rr=rsi(c)
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    macd_sig=macd.ewm(span=9,adjust=False).mean()
    mh=macd-macd_sig

    prev=c.shift(1)
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False).mean()

    vol_avg=v.rolling(20).mean().replace(0,np.nan)
    rel_vol=(v/vol_avg).fillna(0)

    price=float(c.iloc[-1])
    rv=float(rr.iloc[-1])
    atr_now=max(float(atr.iloc[-1]),price*.004)
    vol_ratio=float(rel_vol.iloc[-1])

    r1=(price/float(c.iloc[-2])-1)*100
    r5=(price/float(c.iloc[-6])-1)*100
    r20=(price/float(c.iloc[-21])-1)*100
    r60=(price/float(c.iloc[-min(61,len(c))])-1)*100

    high20=float(h.shift(1).rolling(20).max().iloc[-1])
    low20=float(l.shift(1).rolling(20).min().iloc[-1])
    breakout=(price/high20-1)*100 if high20 else 0

    score=50.0
    reasons=[]
    risks=[]

    if e9.iloc[-1]>e21.iloc[-1]:
        score+=10; reasons.append("short trend bullish")
    else:
        score-=8; risks.append("short trend bearish")

    if e21.iloc[-1]>e50.iloc[-1]:
        score+=10; reasons.append("medium trend bullish")
    else:
        score-=7

    if e50.iloc[-1]>e200.iloc[-1]:
        score+=7; reasons.append("longer trend supportive")
    else:
        score-=5

    if price>e9.iloc[-1]:
        score+=5
    else:
        score-=3

    if 52<=rv<=68:
        score+=9; reasons.append("healthy RSI momentum")
    elif rv>78:
        score-=10; risks.append("overbought / chase risk")
    elif rv<35:
        score-=7; risks.append("weak RSI")

    if mh.iloc[-1]>0:
        score+=7; reasons.append("MACD positive")
    else:
        score-=5
    if mh.iloc[-1]>mh.iloc[-2]:
        score+=5

    if vol_ratio>=1.5:
        score+=6; reasons.append("volume expansion")
    elif vol_ratio<.65:
        score-=3

    if breakout>=0 and breakout<=4:
        score+=8; reasons.append("fresh breakout")
    elif breakout>7:
        score-=7; risks.append("extended above resistance")

    if 0<r5<=8:
        score+=6
    elif r5>15:
        score-=12; risks.append("already moved sharply")
    elif r5<-8:
        score-=7

    if r20>0:
        score+=4
    else:
        score-=3

    score=round(clamp(score),1)

    # Direction/conclusion
    if score>=80:
        direction="UP"
        confidence="High"
        action="BUY / HOLD"
    elif score>=67:
        direction="UP"
        confidence="Medium"
        action="HOLD / WATCH ENTRY"
    elif score>=55:
        direction="SIDEWAYS → UP"
        confidence="Low"
        action="WAIT / HOLD SMALL"
    elif score<=35:
        direction="DOWN"
        confidence="High"
        action="AVOID / REDUCE"
    elif score<=45:
        direction="DOWN"
        confidence="Medium"
        action="WAIT / REDUCE"
    else:
        direction="SIDEWAYS"
        confidence="Low"
        action="WAIT"

    # Avoid late entries.
    extended=(r5>8 or rv>73 or breakout>5)
    if score>=67 and not extended:
        entry_lo=price-.45*atr_now
        entry_hi=price+.10*atr_now
        entry_grade="GOOD NOW"
        entry_reason="Trend is supportive and price is not heavily extended."
    elif score>=67 and extended:
        entry_lo=price-1.4*atr_now
        entry_hi=price-.55*atr_now
        entry_grade="WAIT FOR PULLBACK"
        entry_reason="Setup is bullish, but entering now risks chasing."
    else:
        entry_lo=price-1.0*atr_now
        entry_hi=price-.35*atr_now
        entry_grade="NO RUSH"
        entry_reason="Wait for stronger confirmation before committing capital."

    stop=max(price-2.0*atr_now,0)
    tp1=price+1.5*atr_now
    tp2=price+3.0*atr_now
    tp3=price+4.5*atr_now

    return {
        "price":price,"score":score,"direction":direction,"confidence":confidence,
        "action":action,"entry_lo":max(0,entry_lo),"entry_hi":max(0,entry_hi),
        "entry_grade":entry_grade,"entry_reason":entry_reason,"stop":stop,
        "tp1":tp1,"tp2":tp2,"tp3":tp3,"rsi":rv,"rel_vol":vol_ratio,
        "r1":r1,"r5":r5,"r20":r20,"r60":r60,"breakout":breakout,
        "support":low20,"resistance":high20,"reasons":reasons[:5],"risks":risks[:5],
    }

@st.cache_data(ttl=60,show_spinner=False)
def yf_batch(tickers, period="1y"):
    tickers=list(dict.fromkeys(tickers))
    if not tickers:
        return {}
    raw=yf.download(
        " ".join(tickers),period=period,interval="1d",auto_adjust=True,
        progress=False,threads=True,group_by="ticker"
    )
    out={}
    if len(tickers)==1:
        out[tickers[0]]=raw
    else:
        for t in tickers:
            try:
                d=raw[t].dropna(how="all")
                if not d.empty:
                    out[t]=d
            except Exception:
                pass
    return out

# ---------------------------- DEX SCREENER DATA ----------------------------

@st.cache_data(ttl=20,show_spinner=False)
def dex_discovery_sources():
    endpoints={
        "profiles":f"{DEX}/token-profiles/latest/v1",
        "boost_latest":f"{DEX}/token-boosts/latest/v1",
        "boost_top":f"{DEX}/token-boosts/top/v1",
        "community":f"{DEX}/community-takeovers/latest/v1",
        "ads":f"{DEX}/ads/latest/v1",
    }
    out={}
    def load(k,u):
        try:return k,safe_json(u)
        except Exception:return k,[]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs=[ex.submit(load,k,u) for k,u in endpoints.items()]
        for f in as_completed(futs):
            k,v=f.result(); out[k]=v or []
    return out

def unique_tokens(records):
    seen=set(); out=[]
    for r in records:
        chain=r.get("chainId"); addr=r.get("tokenAddress")
        if chain and addr and (chain,addr) not in seen:
            seen.add((chain,addr)); out.append((chain,addr))
    return out

def chunked(xs,n=30):
    for i in range(0,len(xs),n):
        yield xs[i:i+n]

@st.cache_data(ttl=20,show_spinner=False)
def dex_pairs_all():
    sources=dex_discovery_sources()
    records=[]
    for vals in sources.values():
        records.extend(vals)
    toks=unique_tokens(records)[:180]

    by_chain={}
    for chain,addr in toks:
        by_chain.setdefault(chain,[]).append(addr)

    pairs=[]
    for chain,addrs in by_chain.items():
        for group in chunked(addrs,30):
            try:
                data=safe_json(f"{DEX}/tokens/v1/{chain}/{','.join(group)}")
                if isinstance(data,list):pairs.extend(data)
            except Exception:
                continue

    # best-liquidity pool per token
    best={}
    for p in pairs:
        b=p.get("baseToken") or {}
        key=(p.get("chainId"),b.get("address"))
        if not key[0] or not key[1]:continue
        liq=fnum((p.get("liquidity") or {}).get("usd"))
        if key not in best or liq>fnum((best[key].get("liquidity") or {}).get("usd")):
            best[key]=p
    return list(best.values()),sources

def token_attention_meta(chain,addr,sources):
    key=(chain,addr)
    meta={"boost":0,"boost_total":0,"community":False,"ad":False,"social_links":0,"link_types":[]}

    for row in sources.get("boost_latest",[])+sources.get("boost_top",[]):
        if (row.get("chainId"),row.get("tokenAddress"))==key:
            meta["boost"]=max(meta["boost"],fnum(row.get("amount")))
            meta["boost_total"]=max(meta["boost_total"],fnum(row.get("totalAmount")))
            for link in row.get("links") or []:
                label=(link.get("label") or "").lower()
                url=(link.get("url") or "").lower()
                meta["link_types"].append(label or url)

    for row in sources.get("community",[]):
        if (row.get("chainId"),row.get("tokenAddress"))==key:
            meta["community"]=True
            for link in row.get("links") or []:
                meta["link_types"].append((link.get("label") or link.get("url") or "").lower())

    for row in sources.get("ads",[]):
        if (row.get("chainId"),row.get("tokenAddress"))==key:
            meta["ad"]=True

    for row in sources.get("profiles",[]):
        if (row.get("chainId"),row.get("tokenAddress"))==key:
            for link in row.get("links") or []:
                meta["link_types"].append((link.get("label") or link.get("url") or "").lower())

    # pair info may also contain social links; this gets completed later.
    meta["social_links"]=len(set([x for x in meta["link_types"] if x]))
    return meta

def dex_metrics(p,sources):
    liq=fnum((p.get("liquidity") or {}).get("usd"))
    fdv=fnum(p.get("fdv"))
    mcap=fnum(p.get("marketCap"))
    price=fnum(p.get("priceUsd"))
    vol=p.get("volume") or {}
    changes=p.get("priceChange") or {}
    tx=p.get("txns") or {}
    age=age_info(p.get("pairCreatedAt"))

    vol5=fnum(vol.get("m5")); vol1=fnum(vol.get("h1")); vol6=fnum(vol.get("h6")); vol24=fnum(vol.get("h24"))
    pc5=fnum(changes.get("m5")); pc1=fnum(changes.get("h1")); pc6=fnum(changes.get("h6")); pc24=fnum(changes.get("h24"))

    t5=tx.get("m5") or {}; t1=tx.get("h1") or {}; t6=tx.get("h6") or {}
    b5=fnum(t5.get("buys")); s5=fnum(t5.get("sells"))
    b1=fnum(t1.get("buys")); s1=fnum(t1.get("sells"))
    b6=fnum(t6.get("buys")); s6=fnum(t6.get("sells"))

    base=p.get("baseToken") or {}
    chain=p.get("chainId")
    addr=base.get("address")
    meta=token_attention_meta(chain,addr,sources)

    info=p.get("info") or {}
    social_rows=info.get("socials") or []
    web_rows=info.get("websites") or []
    social_types=[(x.get("type") or x.get("label") or "").lower() for x in social_rows]
    meta["social_links"] += len(social_rows)+len(web_rows)
    meta["link_types"] += social_types

    ratio5=(b5+1)/(s5+1)
    ratio1=(b1+1)/(s1+1)
    activity5=b5+s5
    activity1=b1+s1

    return {
        "price":price,"liq":liq,"fdv":fdv,"mcap":mcap,
        "vol5":vol5,"vol1":vol1,"vol6":vol6,"vol24":vol24,
        "pc5":pc5,"pc1":pc1,"pc6":pc6,"pc24":pc24,
        "b5":b5,"s5":s5,"b1":b1,"s1":s1,"b6":b6,"s6":s6,
        "ratio5":ratio5,"ratio1":ratio1,"activity5":activity5,"activity1":activity1,
        "age":age,"meta":meta
    }

def early_quality_score(m):
    s=25.0; good=[]; risk=[]

    # liquidity
    if m["liq"]>=250_000:s+=20;good.append("strong liquidity")
    elif m["liq"]>=100_000:s+=17;good.append("healthy liquidity")
    elif m["liq"]>=40_000:s+=12
    elif m["liq"]>=15_000:s+=5
    else:s-=25;risk.append("very thin liquidity")

    # age bonus: early, but not blind-new
    hrs=m["age"]["hours"]
    if .25<=hrs<=6:s+=12;good.append("very early")
    elif 6<hrs<=48:s+=9;good.append("still early")
    elif 48<hrs<=168:s+=4
    elif hrs<.25:s-=4;risk.append("extremely new")

    # buyer pressure
    if m["activity5"]>=20 and m["ratio5"]>=1.5:s+=13;good.append("5m buyers leading")
    elif m["activity5"]>=8 and m["ratio5"]>=1.15:s+=7
    elif m["s5"]>m["b5"]*1.6:s-=12;risk.append("5m sellers leading")

    if m["activity1"]>=80 and m["ratio1"]>=1.25:s+=8
    elif m["s1"]>m["b1"]*1.5:s-=8

    # volume / liquidity quality
    if m["liq"]>0:
        turn=m["vol1"]/m["liq"]
        if .25<=turn<=2.5:s+=10;good.append("real trading activity")
        elif turn>5:s-=4;risk.append("extreme turnover")

    # momentum without chase
    if 1<=m["pc5"]<=12:s+=7
    elif m["pc5"]<-10:s-=9
    if 3<=m["pc1"]<=35:s+=9;good.append("positive early momentum")
    elif m["pc1"]>100:s-=25;risk.append("already exploded")
    elif m["pc1"]>60:s-=15;risk.append("late-entry risk")
    elif m["pc1"]<-25:s-=12

    # basic project footprint, but don't overrate marketing
    if m["meta"]["social_links"]>=2:s+=3
    if m["meta"]["community"]:s+=2

    s=round(clamp(s),1)

    if m["liq"]<10_000:
        verdict="AVOID"
        direction="DOWN / EXTREME RISK"
        conf="High"
    elif m["pc1"]>100:
        verdict="DO NOT CHASE"
        direction="UP MOVE ALREADY EXTENDED"
        conf="High"
    elif s>=80:
        verdict="BEST EARLY SETUP"
        direction="UP BIAS"
        conf="High"
    elif s>=67:
        verdict="GOOD EARLY WATCH"
        direction="UP BIAS"
        conf="Medium"
    elif s<=40:
        verdict="AVOID"
        direction="DOWN / WEAK"
        conf="Medium"
    else:
        verdict="WAIT"
        direction="UNCLEAR"
        conf="Low"

    return s,verdict,direction,conf,good[:4],risk[:4]

def attention_score(m):
    s=15.0; drivers=[]; warnings=[]
    meta=m["meta"]

    # On-chain attention
    if m["activity5"]>=30:s+=15;drivers.append("high 5m transaction activity")
    elif m["activity5"]>=12:s+=8
    if m["vol5"]>=25_000:s+=12;drivers.append("fast 5m volume")
    elif m["vol5"]>=5_000:s+=6
    if m["vol1"]>=150_000:s+=10
    elif m["vol1"]>=30_000:s+=5

    # acceleration
    avg5_from_1h=m["vol1"]/12 if m["vol1"]>0 else 0
    if avg5_from_1h>0 and m["vol5"]>=avg5_from_1h*1.8:
        s+=12;drivers.append("volume accelerating")

    if m["ratio5"]>=1.5 and m["activity5"]>=10:
        s+=8;drivers.append("buyers accelerating")

    # Promotional/community attention
    if meta["boost_total"]>0:
        s+=min(15,5+math.log10(meta["boost_total"]+1)*3)
        drivers.append("DEX Screener boost activity")
        warnings.append("boosts are paid promotion")
    if meta["community"]:
        s+=8;drivers.append("community takeover activity")
    if meta["ad"]:
        s+=5;drivers.append("DEX ad activity")
        warnings.append("advertising is paid promotion")
    if meta["social_links"]>=3:
        s+=7;drivers.append("multiple social/community links")
    elif meta["social_links"]>=1:
        s+=3

    # Freshness
    if m["age"]["hours"]<=24:s+=7

    # Avoid mistaking a dead dump for attention.
    if m["pc5"]<-20:s-=10
    if m["liq"]<8_000:s-=20
    if m["pc1"]>150:
        warnings.append("already extremely extended")

    return round(clamp(s),1),drivers[:5],warnings[:4]

# ---------------------------- COINGECKO TRENDING ----------------------------

@st.cache_data(ttl=600,show_spinner=False)
def coingecko_trending():
    try:
        j=safe_json(f"{CG}/search/trending")
        out=[]
        for row in j.get("coins",[]):
            item=row.get("item") or {}
            data=item.get("data") or {}
            out.append({
                "name":item.get("name"),
                "symbol":item.get("symbol"),
                "rank":item.get("market_cap_rank"),
                "score":item.get("score"),
                "price":fnum(data.get("price")),
                "change24":fnum((data.get("price_change_percentage_24h") or {}).get("usd")),
            })
        return out
    except Exception:
        return []

# ---------------------------- STOCK NEWS ATTENTION ----------------------------

@st.cache_data(ttl=300,show_spinner=False)
def ticker_news_count(ticker):
    try:
        news=yf.Ticker(ticker).news or []
        now=datetime.now(timezone.utc).timestamp()
        recent=0
        headlines=[]
        for n in news[:20]:
            # yfinance structures vary; handle both common shapes.
            ts=n.get("providerPublishTime")
            title=n.get("title")
            if not title and isinstance(n.get("content"),dict):
                content=n["content"]
                title=content.get("title")
                pub=content.get("pubDate")
                if pub:
                    try:
                        ts=datetime.fromisoformat(pub.replace("Z","+00:00")).timestamp()
                    except Exception:
                        ts=None
            if title:
                headlines.append(title)
            if ts and now-fnum(ts)<86400:
                recent+=1
        return recent,headlines[:3]
    except Exception:
        return 0,[]


# ---------------------------- PUBLIC INSIDER ACTIVITY ----------------------------

@st.cache_data(ttl=300, show_spinner=False)
def public_insider_data(ticker):
    """
    Reads publicly reported insider transactions exposed by yfinance.
    These are public filings/aggregated filing data, not private information.
    """
    t = yf.Ticker(ticker)
    tx = None
    purchases = None
    roster = None

    try:
        tx = t.insider_transactions
    except Exception:
        tx = None

    try:
        purchases = t.insider_purchases
    except Exception:
        purchases = None

    try:
        roster = t.insider_roster_holders
    except Exception:
        roster = None

    return tx, purchases, roster

def _find_col(df, names):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    for c in df.columns:
        lc = str(c).lower()
        for name in names:
            if name.lower() in lc:
                return c
    return None

def insider_engine(ticker):
    tx, purchases, roster = public_insider_data(ticker)

    result = {
        "ticker": ticker,
        "score": 50.0,
        "signal": "MIXED / NO EDGE",
        "confidence": "Low",
        "open_buy_value": 0.0,
        "open_sell_value": 0.0,
        "buy_count": 0,
        "sell_count": 0,
        "cluster_buyers": 0,
        "recent_rows": [],
        "notes": [],
        "roster": roster,
        "raw": tx,
    }

    if tx is None or not isinstance(tx, pd.DataFrame) or tx.empty:
        result["notes"].append("No insider transaction table was returned by the data provider.")
        return result

    df = tx.copy()

    date_col = _find_col(df, ["Start Date", "Date", "Transaction Date"])
    insider_col = _find_col(df, ["Insider", "Insider Name"])
    position_col = _find_col(df, ["Position", "Title"])
    trans_col = _find_col(df, ["Transaction", "Text"])
    shares_col = _find_col(df, ["Shares", "Shares Traded"])
    value_col = _find_col(df, ["Value", "Transaction Value"])
    ownership_col = _find_col(df, ["Ownership"])

    if date_col:
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    else:
        df["_date"] = pd.NaT

    now = pd.Timestamp.now(tz="UTC")
    recent = df[df["_date"].isna() | (df["_date"] >= now - pd.Timedelta(days=120))].copy()
    if recent.empty:
        recent = df.head(30).copy()

    buy_people = set()

    for _, row in recent.head(60).iterrows():
        text = str(row.get(trans_col, "") if trans_col else "").lower()
        insider = str(row.get(insider_col, "") if insider_col else "")
        position = str(row.get(position_col, "") if position_col else "")
        shares = fnum(row.get(shares_col, 0) if shares_col else 0)
        value = abs(fnum(row.get(value_col, 0) if value_col else 0))
        ownership = str(row.get(ownership_col, "") if ownership_col else "")

        # Prioritize clearly voluntary open-market purchases/sales.
        is_buy = any(k in text for k in [
            "purchase", "buy", "bought", "open market purchase",
            "acquisition", "p-purchase"
        ])
        is_sell = any(k in text for k in [
            "sale", "sell", "sold", "open market sale", "s-sale"
        ])

        # Avoid treating option exercises, grants, gifts, awards as bullish purchases.
        non_open = any(k in text for k in [
            "option", "exercise", "grant", "award", "gift", "conversion",
            "vesting", "tax", "withholding"
        ])
        if non_open and "open market" not in text:
            is_buy = False
            is_sell = False

        if is_buy:
            result["buy_count"] += 1
            result["open_buy_value"] += value
            if insider:
                buy_people.add(insider)
        elif is_sell:
            result["sell_count"] += 1
            result["open_sell_value"] += value

        if is_buy or is_sell:
            result["recent_rows"].append({
                "Date": row.get(date_col, "") if date_col else "",
                "Insider": insider or "Unknown",
                "Role": position or "Unknown",
                "Action": "BUY" if is_buy else "SELL",
                "Shares": shares,
                "Value": value,
                "Ownership": ownership,
                "Raw": str(row.get(trans_col, "") if trans_col else ""),
            })

    result["cluster_buyers"] = len(buy_people)

    score = 50.0
    buyv = result["open_buy_value"]
    sellv = result["open_sell_value"]
    bc = result["buy_count"]
    sc = result["sell_count"]
    cluster = result["cluster_buyers"]

    if buyv >= 5_000_000:
        score += 25
        result["notes"].append("Large recent open-market insider buying.")
    elif buyv >= 1_000_000:
        score += 19
        result["notes"].append("Meaningful recent insider buying.")
    elif buyv >= 250_000:
        score += 12
    elif buyv > 0:
        score += 6

    if cluster >= 3:
        score += 18
        result["notes"].append("Cluster buying: several different insiders bought.")
    elif cluster == 2:
        score += 10
        result["notes"].append("More than one insider bought.")
    elif cluster == 1 and bc > 0:
        score += 4

    if bc >= 4:
        score += 8
    elif bc >= 2:
        score += 4

    if sellv > 0:
        # Insider sales can happen for many non-bearish reasons, so penalize more gently.
        if sellv >= max(5_000_000, buyv * 4):
            score -= 18
            result["notes"].append("Large insider selling relative to buying.")
        elif sellv >= max(1_000_000, buyv * 2):
            score -= 10
        else:
            score -= 3

    if sc >= 5 and bc == 0:
        score -= 8

    score = round(clamp(score), 1)
    result["score"] = score

    if score >= 82 and bc > 0:
        result["signal"] = "STRONG INSIDER BUYING"
        result["confidence"] = "High"
    elif score >= 68 and bc > 0:
        result["signal"] = "INSIDER BUYING"
        result["confidence"] = "Medium"
    elif score <= 35 and sc > 0:
        result["signal"] = "HEAVY INSIDER SELLING"
        result["confidence"] = "Medium"
    elif score <= 44 and sc > 0:
        result["signal"] = "INSIDER SELLING"
        result["confidence"] = "Low"
    else:
        result["signal"] = "MIXED / NO EDGE"
        result["confidence"] = "Low"

    if bc == 0 and sc == 0:
        result["notes"].append(
            "No clearly classified voluntary open-market buys/sells were found in the recent table."
        )

    return result

def combined_stock_conviction(technical, insider):
    if not technical:
        return {"score": None, "label": "NO TECHNICAL DATA", "action": "WAIT"}

    # Technicals remain dominant; public insider activity is confirmation, not a substitute.
    combined = technical["score"] * 0.75 + insider["score"] * 0.25

    if technical["direction"] == "DOWN" and insider["score"] >= 80:
        label = "INSIDERS BULLISH, CHART NOT READY"
        action = "WATCH — WAIT FOR TECHNICAL TURN"
    elif combined >= 80 and technical["score"] >= 67:
        label = "HIGH-CONVICTION BULLISH"
        action = "BUY / HOLD WITH RISK CONTROL"
    elif combined >= 68:
        label = "BULLISH"
        action = "HOLD / WATCH ENTRY"
    elif combined <= 38:
        label = "BEARISH"
        action = "AVOID / REDUCE"
    else:
        label = "MIXED"
        action = "WAIT"

    return {"score": round(combined, 1), "label": label, "action": action}

def render_insider_card(ticker, technical=None):
    try:
        alert_insider_asset(ticker,technical)
    except Exception:
        pass
    ins = insider_engine(ticker)
    combo = combined_stock_conviction(technical, ins) if technical else None
    icon = badge_for_score(ins["score"])

    st.markdown(f"""
    <div class="card">
      <div class="asset">{icon} {ticker} — Public Insider Activity</div>
      <div class="muted">Publicly reported corporate-insider transactions</div>
      <div style="display:flex;justify-content:space-between;align-items:end;margin-top:9px">
        <div>
          <div class="verdict">{ins["signal"]}</div>
          <div class="muted">Confidence: {ins["confidence"]}</div>
        </div>
        <div style="text-align:right">
          <div class="big">{ins["score"]}</div>
          <div class="muted">insider score / 100</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Open-market buys", ins["buy_count"])
    c2.metric("Buy value", compact(ins["open_buy_value"]))
    c3.metric("Open-market sells", ins["sell_count"])
    c4.metric("Sell value", compact(ins["open_sell_value"]))

    st.metric("Different insiders buying", ins["cluster_buyers"])

    if combo:
        st.markdown("#### Insider + Chart Combined")
        c1,c2,c3=st.columns(3)
        c1.metric("Combined conviction", f'{combo["score"]}/100')
        c2.metric("Conclusion", combo["label"])
        c3.metric("Action", combo["action"])

    if ins["notes"]:
        st.write("**What the bot sees:** " + " • ".join(ins["notes"]))

    rows=ins["recent_rows"][:20]
    if rows:
        display=pd.DataFrame(rows)
        display["Value"]=display["Value"].map(lambda x: compact(x))
        display["Shares"]=display["Shares"].map(lambda x: f"{x:,.0f}")
        st.dataframe(
            display[["Date","Insider","Role","Action","Shares","Value","Ownership"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No clearly classified recent open-market insider transactions were found.")

    st.link_button(
        "Open SEC EDGAR insider filings",
        f"https://www.sec.gov/edgar/search/#/q={ticker}&filter_forms=4",
        use_container_width=True
    )
    st.caption(
        "Insider sales can happen for taxes, diversification, estate planning, or compensation reasons. "
        "Open-market purchases and cluster buying generally carry more signal than routine sales."
    )



# ---------------------------- UNIVERSAL INSIDER / SMART-MONEY LAYER ----------------------------

def classify_stock_insider_bot(insider_result):
    return {
        "bot_type": "Public Corporate Insider Bot",
        "source_type": "Public insider filings / aggregated filing data",
        "score": insider_result.get("score", 50),
        "signal": insider_result.get("signal", "MIXED / NO EDGE"),
        "confidence": insider_result.get("confidence", "Low"),
        "explanation": (
            "Tracks publicly reported corporate-insider transactions. "
            "Open-market purchases and cluster buying carry more weight than routine sales."
        ),
        "is_true_insider": True,
    }

def crypto_smart_money_proxy(rd):
    if not rd:
        return {
            "bot_type":"Crypto Smart-Money Bot",
            "source_type":"Market-flow proxy",
            "score":50.0,
            "signal":"NO DATA",
            "confidence":"Low",
            "explanation":"Uses market-flow and technical behavior as a proxy. This is not private insider information.",
            "is_true_insider":False,
        }

    score=50.0
    reasons=[]
    risks=[]

    # Relative volume / momentum / breakout alignment as smart-money-style proxy.
    if rd["rel_vol"] >= 1.8:
        score += 18
        reasons.append("strong relative volume")
    elif rd["rel_vol"] >= 1.2:
        score += 10
        reasons.append("above-average volume")
    elif rd["rel_vol"] < .6:
        score -= 8
        risks.append("weak participation")

    if rd["r5"] > 0 and rd["r20"] > 0:
        score += 14
        reasons.append("multi-timeframe momentum aligned")
    elif rd["r5"] < 0 and rd["r20"] < 0:
        score -= 14
        risks.append("momentum aligned down")

    if 52 <= rd["rsi"] <= 70:
        score += 8
    elif rd["rsi"] > 78:
        score -= 10
        risks.append("overheated")

    if 0 <= rd["breakout"] <= 4:
        score += 10
        reasons.append("fresh breakout behavior")
    elif rd["breakout"] > 8:
        score -= 8
        risks.append("late breakout / chase risk")

    score=round(clamp(score),1)
    if score>=80:
        signal="STRONG SMART-MONEY PROXY"
        confidence="High"
    elif score>=67:
        signal="POSITIVE FLOW PROXY"
        confidence="Medium"
    elif score<=35:
        signal="NEGATIVE FLOW PROXY"
        confidence="Medium"
    else:
        signal="MIXED FLOW"
        confidence="Low"

    return {
        "bot_type":"Crypto Smart-Money Bot",
        "source_type":"Market-flow proxy",
        "score":score,
        "signal":signal,
        "confidence":confidence,
        "explanation":"Uses volume, momentum, breakout behavior and participation as a public smart-money proxy. It is not private insider information.",
        "is_true_insider":False,
        "reasons":reasons[:4],
        "risks":risks[:4],
    }

def meme_insider_risk_proxy(m, sources=None):
    score=50.0
    reasons=[]
    risks=[]

    # Early buyer dominance.
    if m["activity5"] >= 20 and m["ratio5"] >= 1.6:
        score += 16
        reasons.append("early buyer dominance")
    elif m["s5"] > m["b5"]*1.5:
        score -= 16
        risks.append("early seller dominance")

    # Liquidity depth.
    if m["liq"] >= 150_000:
        score += 14
        reasons.append("healthy liquidity depth")
    elif m["liq"] >= 40_000:
        score += 7
    elif m["liq"] < 10_000:
        score -= 25
        risks.append("thin liquidity / exit risk")

    # Volume acceleration.
    avg5 = m["vol1"]/12 if m["vol1"] > 0 else 0
    if avg5 > 0 and m["vol5"] >= avg5*2:
        score += 12
        reasons.append("5m volume accelerating")

    # Anti-chase / insider-dump style warning.
    if m["pc1"] > 100:
        score -= 22
        risks.append("already exploded / dump risk")
    elif m["pc1"] > 60:
        score -= 12
        risks.append("late-entry risk")

    # Very new + low liquidity is especially dangerous.
    if m["age"]["hours"] < .5 and m["liq"] < 30_000:
        score -= 12
        risks.append("extremely new with weak liquidity")

    # Marketing signals don't count as bullish by themselves.
    meta=m["meta"]
    if meta.get("boost_total",0) > 0:
        risks.append("paid boost detected")
    if meta.get("ad"):
        risks.append("paid ad detected")

    score=round(clamp(score),1)

    if score>=80:
        signal="HEALTHY EARLY FLOW"
        confidence="High"
    elif score>=67:
        signal="PROMISING EARLY FLOW"
        confidence="Medium"
    elif score<=35:
        signal="HIGH INSIDER/RUG RISK PROXY"
        confidence="High"
    else:
        signal="MIXED EARLY FLOW"
        confidence="Low"

    return {
        "bot_type":"Meme Insider-Risk Bot",
        "source_type":"Public on-chain / liquidity proxy",
        "score":score,
        "signal":signal,
        "confidence":confidence,
        "explanation":(
            "This is a risk proxy, not access to private developer or insider wallets. "
            "It looks for early buyer/seller imbalance, liquidity depth, volume acceleration, "
            "extreme pumps and paid-promotion risk."
        ),
        "is_true_insider":False,
        "reasons":reasons[:4],
        "risks":risks[:5],
    }

def meme_attention_whale_proxy(m):
    score=50.0
    reasons=[]
    risks=[]
    meta=m["meta"]

    if m["activity5"] >= 35:
        score += 15
        reasons.append("high 5m transaction activity")
    elif m["activity5"] >= 15:
        score += 8

    if m["ratio5"] >= 1.5 and m["activity5"] >= 10:
        score += 12
        reasons.append("buyers dominating recent flow")
    elif m["s5"] > m["b5"]*1.5:
        score -= 12
        risks.append("seller-heavy flow")

    avg5 = m["vol1"]/12 if m["vol1"] > 0 else 0
    if avg5 > 0 and m["vol5"] >= avg5*1.8:
        score += 12
        reasons.append("attention accelerating")

    if meta.get("boost_total",0) > 0:
        risks.append("paid DEX boost present")
    if meta.get("ad"):
        risks.append("paid ad present")
    if meta.get("community"):
        reasons.append("community takeover signal")

    if m["pc1"] > 120:
        score -= 18
        risks.append("attention arrived after a huge pump")

    score=round(clamp(score),1)
    if score>=80:
        signal="STRONG ORGANIC-STYLE ATTENTION"
        confidence="High"
    elif score>=67:
        signal="ATTENTION BUILDING"
        confidence="Medium"
    elif score<=35:
        signal="WEAK / PROMOTION-HEAVY"
        confidence="Medium"
    else:
        signal="MIXED ATTENTION"
        confidence="Low"

    return {
        "bot_type":"Meme Attention / Whale Proxy",
        "source_type":"Trading acceleration + public promotion signals",
        "score":score,
        "signal":signal,
        "confidence":confidence,
        "explanation":(
            "Separates trading acceleration from paid promotion. "
            "This is not private insider information or direct access to every social platform."
        ),
        "is_true_insider":False,
        "reasons":reasons[:4],
        "risks":risks[:5],
    }

def render_insider_proxy_panel(bot):
    st.markdown("### 🧭 Insider / Smart-Money Signal")
    st.caption(f'**Bot type:** {bot["bot_type"]} • **Signal source:** {bot["source_type"]}')
    c1,c2,c3=st.columns(3)
    c1.metric("Bot score",f'{bot["score"]}/100')
    c2.metric("Signal",bot["signal"])
    c3.metric("Confidence",bot["confidence"])
    st.write(bot["explanation"])
    if bot.get("reasons"):
        st.write("**Positive evidence:** " + " • ".join(bot["reasons"]))
    if bot.get("risks"):
        st.write("**Risk flags:** " + " • ".join(bot["risks"]))
    if not bot.get("is_true_insider",False):
        st.info("This section is a public-data proxy. It is not access to material non-public information.")


# ---------------------------- RENDERERS ----------------------------

def render_dex_card(p,sources,mode="early"):
    try:
        alert_meme_asset(p,sources,mode)
    except Exception:
        pass
    base=p.get("baseToken") or {}
    quote=p.get("quoteToken") or {}
    m=dex_metrics(p,sources)
    eq,verdict,direction,confidence,good,risk=early_quality_score(m)
    attn,drivers,warnings=attention_score(m)

    name=base.get("name") or "Unknown"
    symbol=base.get("symbol") or "?"
    chain=network_name(p.get("chainId"))
    dex=p.get("dexId") or "DEX"
    addr=base.get("address") or ""

    main_score=eq if mode=="early" else attn
    main_label="early quality" if mode=="early" else "attention"
    icon=badge_for_score(main_score)

    if mode=="attention":
        if attn>=80:
            verdict2="ATTENTION SURGING"
            direction2="WATCH NOW"
        elif attn>=65:
            verdict2="GETTING ATTENTION"
            direction2="WATCH"
        elif attn>=50:
            verdict2="EARLY ATTENTION"
            direction2="MONITOR"
        else:
            verdict2="LOW ATTENTION"
            direction2="WAIT"
        verdict_show=verdict2
        direction_show=direction2
    else:
        verdict_show=verdict
        direction_show=direction

    st.markdown(f"""
    <div class="card">
      <div class="asset">{icon} {name} ({symbol})</div>
      <div class="muted">{chain} • {dex} • {quote.get("symbol","?")} pair • age {m["age"]["label"]}</div>
      <div style="display:flex;justify-content:space-between;align-items:end;margin-top:9px">
        <div>
          <div class="big">{money(m["price"])}</div>
          <div class="verdict">{verdict_show}</div>
          <div class="muted">{direction_show}</div>
        </div>
        <div style="text-align:right">
          <div class="big">{main_score}</div>
          <div class="muted">{main_label} / 100</div>
        </div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    if mode=="early":
        insider_proxy = meme_insider_risk_proxy(m, sources)
        render_insider_proxy_panel(insider_proxy)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Liquidity",compact(m["liq"]))
        c2.metric("1h volume",compact(m["vol1"]))
        c3.metric("5m buys",int(m["b5"]))
        c4.metric("5m sells",int(m["s5"]))

        c1,c2,c3,c4=st.columns(4)
        c1.metric("5m move",f'{m["pc5"]:+.1f}%')
        c2.metric("1h move",f'{m["pc1"]:+.1f}%')
        c3.metric("6h move",f'{m["pc6"]:+.1f}%')
        c4.metric("Confidence",confidence)

        st.markdown("**Entry timing**")
        price=m["price"]
        if verdict=="BEST EARLY SETUP" and m["pc5"]<=10:
            lo=price*.97; hi=price*1.01
            st.success(f"Entry zone: {money(lo)} – {money(hi)} • only while liquidity and buyer pressure stay healthy.")
        elif eq>=67:
            lo=price*.90 if m["pc1"]>35 else price*.95
            hi=price*.96 if m["pc1"]>35 else price*.99
            st.warning(f"Better pullback zone: {money(lo)} – {money(hi)}. Avoid chasing.")
        else:
            st.info("No strong entry yet. Wait for better liquidity, buyer pressure, and confirmation.")

        if good:
            st.write("**What looks good:** " + " • ".join(good))
        if risk:
            st.write("**Risks:** " + " • ".join(risk))
    else:
        attention_proxy = meme_attention_whale_proxy(m)
        render_insider_proxy_panel(attention_proxy)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Attention score",f"{attn}/100")
        c2.metric("5m volume",compact(m["vol5"]))
        c3.metric("5m transactions",int(m["activity5"]))
        c4.metric("Boosts",int(m["meta"]["boost_total"]))

        if drivers:
            st.write("**Why it is showing up:** " + " • ".join(drivers))
        if warnings:
            st.warning(" • ".join(warnings))
        st.caption("Attention ≠ good investment. Use the Early Quality score before considering an entry.")
        st.write(f"**Early Quality score:** {eq}/100 • {verdict}")

    with st.expander("Contract, platforms & details"):
        st.write(f"**Network:** {chain}")
        st.code(addr,language=None)
        st.write(f"**FDV:** {compact(m['fdv'])} • **Market cap:** {compact(m['mcap'])}")
        if m["meta"]["link_types"]:
            st.write("**Detected links/socials:** " + ", ".join(sorted(set(m["meta"]["link_types"]))[:8]))
        if p.get("url"):
            st.link_button("Open on DEX Screener",p["url"],use_container_width=True)

def render_technical_card(label,ticker,rd,kind,news_count=0,headlines=None):
    try:
        alert_technical_asset(label,ticker,rd,kind)
    except Exception:
        pass
    icon=badge_for_score(rd["score"])
    st.markdown(f"""
    <div class="card">
      <div class="asset">{icon} {label}</div>
      <div class="muted">{kind} • {ticker}</div>
      <div style="display:flex;justify-content:space-between;align-items:end;margin-top:9px">
        <div>
          <div class="big">{money(rd["price"])}</div>
          <div class="verdict">{rd["action"]}</div>
          <div class="muted">Expected direction: {rd["direction"]} • confidence {rd["confidence"]}</div>
        </div>
        <div style="text-align:right">
          <div class="big">{rd["score"]}</div>
          <div class="muted">setup score / 100</div>
        </div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("#### Entry / Hold / Exit")
    c1,c2,c3=st.columns(3)
    c1.metric("Entry zone",f'{money(rd["entry_lo"])} – {money(rd["entry_hi"])}')
    c2.metric("Risk stop",money(rd["stop"]))
    c3.metric("Entry timing",rd["entry_grade"])
    st.caption(rd["entry_reason"])

    c1,c2,c3=st.columns(3)
    c1.metric("Target 1",money(rd["tp1"]))
    c2.metric("Target 2",money(rd["tp2"]))
    c3.metric("Target 3",money(rd["tp3"]))

    st.markdown("#### Technical picture")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("RSI",f'{rd["rsi"]:.1f}')
    c2.metric("Relative volume",f'{rd["rel_vol"]:.2f}×')
    c3.metric("5-day",f'{rd["r5"]:+.1f}%')
    c4.metric("20-day",f'{rd["r20"]:+.1f}%')

    c1,c2=st.columns(2)
    c1.metric("Support",money(rd["support"]))
    c2.metric("Resistance",money(rd["resistance"]))

    if rd["reasons"]:
        st.write("**Bullish evidence:** " + " • ".join(rd["reasons"]))
    if rd["risks"]:
        st.write("**Risk flags:** " + " • ".join(rd["risks"]))

    if kind=="Crypto":
        smart = crypto_smart_money_proxy(rd)
        render_insider_proxy_panel(smart)

    if kind=="Stock":
        st.markdown("#### Public Insider / Smart-Money Signal")
        try:
            ins = insider_engine(ticker)
            bot = classify_stock_insider_bot(ins)
            render_insider_proxy_panel(bot)
        except Exception:
            st.info("Public insider data could not be loaded for this stock.")

        st.markdown("#### News / attention")
        st.write(f"Recent news items detected: **{news_count}**")
        if headlines:
            for h in headlines[:3]:
                st.caption("• "+h)


# ---------------------------- NOTIFICATION ENGINE ----------------------------

def secret_value(section, key, default=""):
    try:
        if section in st.secrets and key in st.secrets[section]:
            return str(st.secrets[section][key])
    except Exception:
        pass
    return default

def init_notification_state():
    defaults = {
        "notify_enabled": True,
        "notify_telegram": True,
        "notify_pushover": False,
        "notify_early_meme": True,
        "notify_attention": True,
        "notify_crypto": True,
        "notify_stocks": True,
        "notify_insiders": True,
        "notify_watchlist": True,
        "notify_min_score": 80,
        "notify_attention_score": 80,
        "notify_insider_score": 82,
        "notify_cooldown_minutes": 30,
        "_alert_last_sent": {},
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def telegram_ready():
    return bool(secret_value("telegram","bot_token") and secret_value("telegram","chat_id"))

def pushover_ready():
    return bool(secret_value("pushover","api_token") and secret_value("pushover","user_key"))

def send_telegram(title, message):
    token=secret_value("telegram","bot_token")
    chat_id=secret_value("telegram","chat_id")
    if not token or not chat_id:
        return False, "Telegram secrets missing."
    try:
        r=requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id":chat_id,"text":f"{title}\n\n{message}"},
            timeout=12
        )
        r.raise_for_status()
        payload=r.json()
        if not payload.get("ok",False):
            return False, str(payload)
        return True, "Telegram sent."
    except Exception as e:
        return False, str(e)

def send_pushover(title, message, priority=0):
    token=secret_value("pushover","api_token")
    user=secret_value("pushover","user_key")
    if not token or not user:
        return False, "Pushover secrets missing."
    try:
        r=requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token":token,
                "user":user,
                "title":title[:250],
                "message":message[:1024],
                "priority":priority,
            },
            timeout=12
        )
        r.raise_for_status()
        return True, "Pushover sent."
    except Exception as e:
        return False, str(e)

def _alert_due(key):
    now=datetime.now(timezone.utc).timestamp()
    last=st.session_state["_alert_last_sent"].get(key,0)
    cooldown=float(st.session_state["notify_cooldown_minutes"])*60
    return (now-last)>=cooldown

def send_market_alert(key, title, message, priority=0):
    if not st.session_state.get("notify_enabled",False):
        return
    if not _alert_due(key):
        return

    sent=False
    if st.session_state.get("notify_telegram",False) and telegram_ready():
        ok,_=send_telegram(title,message)
        sent=sent or ok
    if st.session_state.get("notify_pushover",False) and pushover_ready():
        ok,_=send_pushover(title,message,priority)
        sent=sent or ok

    if sent:
        st.session_state["_alert_last_sent"][key]=datetime.now(timezone.utc).timestamp()

def alert_technical_asset(label, ticker, rd, kind, source="section"):
    if not rd:
        return
    enabled = st.session_state.get("notify_crypto",True) if kind=="Crypto" else st.session_state.get("notify_stocks",True)
    if not enabled:
        return
    threshold=float(st.session_state.get("notify_min_score",80))
    if rd["score"] < threshold:
        return

    title=f"🚨 {kind} setup: {label}"
    msg=(
        f"{rd['action']} | {rd['direction']} | score {rd['score']}/100\n"
        f"Price: {money(rd['price'])}\n"
        f"Entry: {money(rd['entry_lo'])} – {money(rd['entry_hi'])}\n"
        f"Stop: {money(rd['stop'])}\n"
        f"TP1: {money(rd['tp1'])} | TP2: {money(rd['tp2'])} | TP3: {money(rd['tp3'])}\n"
        f"Confidence: {rd['confidence']}"
    )
    send_market_alert(f"{source}:{kind}:{ticker}:{rd['action']}",title,msg,0)

def alert_meme_asset(p, sources, mode):
    m=dex_metrics(p,sources)
    base=p.get("baseToken") or {}
    sym=base.get("symbol") or "?"
    name=base.get("name") or sym
    chain=network_name(p.get("chainId"))
    addr=base.get("address") or ""

    if mode=="early":
        if not st.session_state.get("notify_early_meme",True):
            return
        score,verdict,direction,confidence,good,risk=early_quality_score(m)
        if score < float(st.session_state.get("notify_min_score",80)):
            return
        title=f"🌱 Early meme alert: {name} ({sym})"
        msg=(
            f"{verdict} | {direction} | {score}/100\n"
            f"Age: {m['age']['label']} | {chain}\n"
            f"Price: {money(m['price'])}\n"
            f"Liquidity: {compact(m['liq'])} | 1h vol: {compact(m['vol1'])}\n"
            f"5m buys/sells: {int(m['b5'])}/{int(m['s5'])}\n"
            f"1h move: {m['pc1']:+.1f}%\n"
            f"Contract: {addr}"
        )
        send_market_alert(f"meme-early:{chain}:{addr}:{verdict}",title,msg,1 if score>=90 else 0)
    else:
        if not st.session_state.get("notify_attention",True):
            return
        score,drivers,warnings=attention_score(m)
        if score < float(st.session_state.get("notify_attention_score",80)):
            return
        title=f"🔥 Meme attention: {name} ({sym})"
        msg=(
            f"Attention score {score}/100\n"
            f"Age: {m['age']['label']} | {chain}\n"
            f"5m volume: {compact(m['vol5'])} | 1h volume: {compact(m['vol1'])}\n"
            f"5m buys/sells: {int(m['b5'])}/{int(m['s5'])}\n"
            f"Boost total: {int(m['meta']['boost_total'])}\n"
            f"Contract: {addr}"
        )
        send_market_alert(f"meme-attn:{chain}:{addr}",title,msg,0)

def alert_insider_asset(ticker, tech=None):
    if not st.session_state.get("notify_insiders",True):
        return
    ins=insider_engine(ticker)
    threshold=float(st.session_state.get("notify_insider_score",82))
    if ins["score"] < threshold or ins["buy_count"] <= 0:
        return
    combo=combined_stock_conviction(tech,ins) if tech else None
    title=f"🕵️ Insider buying: {ticker}"
    msg=(
        f"{ins['signal']} | insider score {ins['score']}/100\n"
        f"Open-market buys: {ins['buy_count']} | value: {compact(ins['open_buy_value'])}\n"
        f"Different insiders buying: {ins['cluster_buyers']}\n"
        + (f"Combined conviction: {combo['score']}/100 | {combo['label']}\n" if combo else "")
    )
    send_market_alert(f"insider:{ticker}:{ins['signal']}",title,msg,1 if ins["score"]>=90 else 0)

init_notification_state()


# ---------------------------- MAIN UI ----------------------------

st.markdown("""
<div class="hero">
  <div style="font-size:1.7rem;font-weight:900">🧠 Market Intelligence Pro v9</div>
  <div class="muted">Early meme discovery • attention radar • crypto • stocks • insider/smart-money layer • Telegram/Pushover alerts • entries • hold/exit logic • risk controls</div>
</div>
""",unsafe_allow_html=True)

st.warning(
    "No model or scanner can guarantee 99.99999% profits. This app is designed to improve filtering and timing, "
    "not promise wealth. New meme coins can still rug, honeypot, lose liquidity, or collapse between refreshes."
)

tabs=st.tabs([
    "🌱 Meme: Early Quality",
    "🔥 Meme: Attention Radar",
    "₿ Crypto",
    "📈 Stocks",
    "🕵️ Public Insider Bot",
    "⭐ Watchlist",
    "🔔 Notifications",
    "🛡️ Risk Calculator"
])

# ---------------- TAB 1: EARLY QUALITY ----------------
with tabs[0]:
    st.subheader("🌱 Somewhat-New Meme Coins — Quality First")
    st.caption(
        "Looks for newer on-chain tokens with liquidity, transactions, buyer pressure and momentum, "
        "while penalizing coins that already exploded."
    )

    @st.fragment(run_every="30s")
    def early_fragment():
        st.caption("Auto-refresh: ~30 seconds while this page/session is active.")
        try:
            pairs,sources=dex_pairs_all()
        except Exception as e:
            st.error(f"DEX feed temporarily unavailable: {e}")
            return

        rows=[]
        for p in pairs:
            m=dex_metrics(p,sources)
            eq,*_=early_quality_score(m)
            # somewhat new: max 14 days; enough activity to be meaningful
            if m["age"]["hours"]<=24*14 and m["liq"]>=10_000 and (m["activity1"]>=10 or m["vol1"]>=2_000):
                rows.append((eq,m["age"]["hours"],p))
        rows.sort(key=lambda x:(-x[0],x[1]))

        if not rows:
            st.info("No candidates passed the current quality filters.")
            return
        for _,__,p in rows[:20]:
            render_dex_card(p,sources,"early")
    early_fragment()

# ---------------- TAB 2: ATTENTION ----------------
with tabs[1]:
    st.subheader("🔥 Meme Coin Attention Radar")
    st.caption(
        "Finds coins receiving unusual attention from on-chain activity plus DEX Screener promotion/community signals. "
        "It also shows CoinGecko search trends. Paid boosts/ads are labeled and are never treated as proof of quality."
    )

    trending=coingecko_trending()
    if trending:
        with st.expander("🌐 CoinGecko search trends",expanded=True):
            cols=st.columns(3)
            for i,x in enumerate(trending[:15]):
                with cols[i%3]:
                    st.markdown(f"**{i+1}. {x['name']} ({x['symbol']})**")
                    st.caption(f"24h: {x['change24']:+.1f}% • market-cap rank: {x['rank'] or 'N/A'}")

    @st.fragment(run_every="30s")
    def attention_fragment():
        try:
            pairs,sources=dex_pairs_all()
        except Exception as e:
            st.error(f"Attention feed temporarily unavailable: {e}")
            return
        rows=[]
        for p in pairs:
            m=dex_metrics(p,sources)
            attn,_,_=attention_score(m)
            if m["liq"]>=8_000 and (attn>=45 or m["meta"]["boost_total"]>0 or m["meta"]["community"]):
                rows.append((attn,p))
        rows.sort(key=lambda x:-x[0])
        for _,p in rows[:20]:
            render_dex_card(p,sources,"attention")
    attention_fragment()

# ---------------- TAB 3: CRYPTO ----------------
with tabs[2]:
    st.subheader("₿ Established Crypto")
    st.caption("Trend + momentum + volume + anti-chase entry logic.")
    crypto_input=st.text_input("Crypto symbols",",".join(MAJOR_CRYPTO.keys()),key="crypto_input")
    symbols=[x.strip().upper() for x in crypto_input.split(",") if x.strip()][:20]
    tickers=[MAJOR_CRYPTO.get(s,f"{s}-USD") for s in symbols]
    with st.spinner("Scanning crypto..."):
        data=yf_batch(tickers)
    ranked=[]
    for s,t in zip(symbols,tickers):
        rd=technical_engine(data.get(t))
        if rd:ranked.append((rd["score"],s,t,rd))
    ranked.sort(reverse=True,key=lambda x:x[0])
    for _,s,t,rd in ranked:
        render_technical_card(s,t,rd,"Crypto")

# ---------------- TAB 4: STOCKS ----------------
with tabs[3]:
    st.subheader("📈 Stocks")
    st.caption("Ranks trend setups and adds recent-news attention without letting news override risk.")
    stock_input=st.text_input("Stock tickers",DEFAULT_STOCKS,key="stock_input")
    stocks=[x.strip().upper() for x in stock_input.split(",") if x.strip()][:20]
    with st.spinner("Scanning stocks..."):
        data=yf_batch(stocks)
    ranked=[]
    for s in stocks:
        rd=technical_engine(data.get(s))
        if rd:
            ranked.append((rd["score"],s,rd))
    ranked.sort(reverse=True,key=lambda x:x[0])
    for _,s,rd in ranked:
        nc,heads=ticker_news_count(s)
        render_technical_card(s,s,rd,"Stock",nc,heads)


# ---------------- TAB 5: PUBLIC INSIDER BOT ----------------
with tabs[4]:
    st.subheader("🕵️ Public Insider Bot")
    st.caption(
        "Tracks publicly reported corporate-insider activity. It gives more weight to voluntary "
        "open-market purchases and cluster buying than to routine insider sales."
    )
    st.info(
        "This is public filing analysis — not private or non-public insider information. "
        "SEC Section 16 insiders generally include officers, directors, and >10% beneficial owners."
    )

    insider_input=st.text_input(
        "Stock tickers for insider scan",
        "NVDA,TSLA,AAPL,AMD,MSFT,AMZN,META,COIN,MSTR,PLTR",
        key="insider_input"
    )
    insider_tickers=[x.strip().upper() for x in insider_input.split(",") if x.strip()][:15]

    with st.spinner("Reading public insider activity and technical confirmation..."):
        insider_market=yf_batch(insider_tickers)

    ranked_insiders=[]
    for symbol in insider_tickers:
        tech=technical_engine(insider_market.get(symbol))
        ins=insider_engine(symbol)
        combo=combined_stock_conviction(tech,ins) if tech else {"score":ins["score"]}
        ranked_insiders.append((combo.get("score") or ins["score"],symbol,tech,ins))

    ranked_insiders.sort(reverse=True,key=lambda x:x[0])

    st.markdown("### Strongest combined signals")
    summary_rows=[]
    for _,symbol,tech,ins in ranked_insiders:
        combo=combined_stock_conviction(tech,ins) if tech else None
        summary_rows.append({
            "Ticker":symbol,
            "Insider":ins["signal"],
            "Insider Score":ins["score"],
            "Technical":tech["direction"] if tech else "N/A",
            "Technical Score":tech["score"] if tech else "N/A",
            "Combined":combo["score"] if combo else "N/A",
            "Conclusion":combo["label"] if combo else ins["signal"],
        })
    st.dataframe(pd.DataFrame(summary_rows),use_container_width=True,hide_index=True)

    for _,symbol,tech,ins in ranked_insiders:
        with st.expander(
            f'{badge_for_score(ins["score"])} {symbol} — {ins["signal"]} — {ins["score"]}/100',
            expanded=False
        ):
            render_insider_card(symbol,tech)


# ---------------- TAB 5: WATCHLIST ----------------
with tabs[5]:
    st.subheader("⭐ Mixed Watchlist")
    st.caption("Track stocks and common crypto together.")
    raw=st.text_input("Examples: NVDA, TSLA, BTC, ETH, SOL","NVDA,TSLA,BTC,ETH,SOL,DOGE",key="watch")
    originals=[x.strip().upper() for x in raw.split(",") if x.strip()][:25]
    mapped=[MAJOR_CRYPTO.get(x,x) for x in originals]
    with st.spinner("Loading watchlist..."):
        data=yf_batch(mapped)
    ranked=[]
    for original,t in zip(originals,mapped):
        rd=technical_engine(data.get(t))
        if rd:ranked.append((rd["score"],original,t,rd))
    ranked.sort(reverse=True,key=lambda x:x[0])
    for _,orig,t,rd in ranked:
        kind="Crypto" if t.endswith("-USD") else "Stock"
        render_technical_card(orig,t,rd,kind)


# ---------------- TAB 7: NOTIFICATIONS ----------------
with tabs[6]:
    st.subheader("🔔 Notifications")
    st.caption("Send high-quality setup alerts to Telegram and/or Pushover on your iPhone.")

    st.warning(
        "In-app alerts run while the Streamlit app/server is active. Community Cloud can sleep, "
        "so this alone is not guaranteed to be 24/7 background monitoring."
    )

    st.markdown("### Provider status")
    c1,c2=st.columns(2)
    with c1:
        st.metric("Telegram", "READY" if telegram_ready() else "NOT CONFIGURED")
    with c2:
        st.metric("Pushover", "READY" if pushover_ready() else "NOT CONFIGURED")

    st.markdown("### Alert switches")
    st.session_state["notify_enabled"]=st.toggle(
        "Master notifications",value=st.session_state["notify_enabled"]
    )
    c1,c2=st.columns(2)
    with c1:
        st.session_state["notify_telegram"]=st.toggle(
            "Telegram alerts",value=st.session_state["notify_telegram"]
        )
    with c2:
        st.session_state["notify_pushover"]=st.toggle(
            "Pushover iPhone alerts",value=st.session_state["notify_pushover"]
        )

    c1,c2=st.columns(2)
    with c1:
        st.session_state["notify_early_meme"]=st.toggle(
            "New meme quality alerts",value=st.session_state["notify_early_meme"]
        )
        st.session_state["notify_crypto"]=st.toggle(
            "Crypto alerts",value=st.session_state["notify_crypto"]
        )
        st.session_state["notify_insiders"]=st.toggle(
            "Public insider-buying alerts",value=st.session_state["notify_insiders"]
        )
    with c2:
        st.session_state["notify_attention"]=st.toggle(
            "Meme attention alerts",value=st.session_state["notify_attention"]
        )
        st.session_state["notify_stocks"]=st.toggle(
            "Stock alerts",value=st.session_state["notify_stocks"]
        )
        st.session_state["notify_watchlist"]=st.toggle(
            "Watchlist alerts",value=st.session_state["notify_watchlist"]
        )

    st.markdown("### Trigger strength")
    st.session_state["notify_min_score"]=st.slider(
        "Minimum setup score",60,95,int(st.session_state["notify_min_score"])
    )
    st.session_state["notify_attention_score"]=st.slider(
        "Minimum meme attention score",60,95,int(st.session_state["notify_attention_score"])
    )
    st.session_state["notify_insider_score"]=st.slider(
        "Minimum insider score",60,95,int(st.session_state["notify_insider_score"])
    )
    st.session_state["notify_cooldown_minutes"]=st.slider(
        "Don't repeat the same alert for",5,240,int(st.session_state["notify_cooldown_minutes"]),5,
        help="Cooldown is per signal/asset in the current running app session."
    )

    st.markdown("### Send a test")
    c1,c2=st.columns(2)
    with c1:
        if st.button("Test Telegram",use_container_width=True):
            ok,msg=send_telegram(
                "✅ Market Intelligence Pro v9",
                "Telegram notifications are connected correctly."
            )
            st.success(msg) if ok else st.error(msg)
    with c2:
        if st.button("Test Pushover",use_container_width=True):
            ok,msg=send_pushover(
                "Market Intelligence Pro v9",
                "Pushover notifications are connected correctly."
            )
            st.success(msg) if ok else st.error(msg)

    st.markdown("### Streamlit Secrets setup")
    st.write(
        "Put the credentials in Streamlit **Settings → Secrets**. Do not paste API tokens into "
        "your public `app.py` file."
    )
    secrets_example = """[telegram]
bot_token = "YOUR_TELEGRAM_BOT_TOKEN"
chat_id = "YOUR_TELEGRAM_CHAT_ID"

[pushover]
api_token = "YOUR_PUSHOVER_APP_TOKEN"
user_key = "YOUR_PUSHOVER_USER_KEY"
"""
    st.code(secrets_example,language="toml")

    with st.expander("How the alerts trigger"):
        st.write(
            "Early Meme alerts trigger when the Early Quality score crosses your threshold. "
            "Attention alerts use the Attention score. Crypto and stocks use their technical setup score. "
            "The Insider Bot only alerts on strong public insider-buying signals. A cooldown prevents the "
            "same signal from being sent every refresh."
        )


# ---------------- TAB 6: RISK ----------------
with tabs[7]:
    st.subheader("🛡️ Position Size & Risk Calculator")
    st.caption("The part that helps keep one bad trade from destroying the account.")

    account=st.number_input("Account size ($)",min_value=10.0,value=1000.0,step=100.0)
    risk_pct=st.slider("Maximum account risk per trade",0.25,5.0,1.0,0.25)
    entry=st.number_input("Planned entry price",min_value=0.00000001,value=1.0,format="%.8f")
    stop=st.number_input("Stop price",min_value=0.0,value=.90,format="%.8f")

    risk_dollars=account*(risk_pct/100)
    distance=max(entry-stop,0)
    if distance>0 and entry>0:
        units=risk_dollars/distance
        position_value=units*entry
        cap=account*.20
        capped_value=min(position_value,cap)
        capped_units=capped_value/entry
        st.metric("Maximum dollars at risk",money(risk_dollars))
        st.metric("Suggested max position",money(capped_value))
        st.metric("Approximate units/tokens",f"{capped_units:,.6f}")
        if position_value>cap:
            st.info("Position was capped at 20% of account size even though the stop-based calculation allowed more.")
    else:
        st.info("Stop must be below the planned entry for a long trade.")

    st.markdown("#### Simple risk rules")
    st.write(
        "For speculative meme coins, smaller size is safer because stops may not fill where expected. "
        "A strong score does not justify risking a large percentage of the account."
    )

st.divider()
st.caption(
    "Data sources can be delayed, incomplete, rate-limited or wrong. Social-attention signals here include "
    "CoinGecko search trends, DEX Screener boosts/community/ads/social links, and on-chain trading acceleration; "
    "this is not direct full-firehose access to every post on X, TikTok, Reddit, Telegram, Discord, YouTube, or Instagram. The Insider Bot uses publicly reported/aggregated filing data only and is not access to non-public information."
)
