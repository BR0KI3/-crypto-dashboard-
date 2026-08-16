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
HEADERS = {"accept":"application/json","user-agent":"market-intelligence-pro/6.0"}

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

# ---------------------------- RENDERERS ----------------------------

def render_dex_card(p,sources,mode="early"):
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

    if kind=="Stock":
        st.markdown("#### News / attention")
        st.write(f"Recent news items detected: **{news_count}**")
        if headlines:
            for h in headlines[:3]:
                st.caption("• "+h)

# ---------------------------- MAIN UI ----------------------------

st.markdown("""
<div class="hero">
  <div style="font-size:1.7rem;font-weight:900">🧠 Market Intelligence Pro v6</div>
  <div class="muted">Early meme discovery • attention radar • crypto • stocks • entries • hold/exit logic • risk controls</div>
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
    "⭐ Watchlist",
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

# ---------------- TAB 5: WATCHLIST ----------------
with tabs[4]:
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

# ---------------- TAB 6: RISK ----------------
with tabs[5]:
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
    "this is not direct full-firehose access to every post on X, TikTok, Reddit, Telegram, Discord, YouTube, or Instagram."
)
