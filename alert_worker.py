from __future__ import annotations

import os
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DEX = "https://api.dexscreener.com"
HEADERS = {"accept":"application/json","user-agent":"market-intelligence-pro-worker/10.0"}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","")
PUSHOVER_API_TOKEN = os.getenv("PUSHOVER_API_TOKEN","")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY","")

MIN_MEME_SCORE = float(os.getenv("MIN_MEME_SCORE","82"))
MIN_ATTN_SCORE = float(os.getenv("MIN_ATTN_SCORE","84"))
MIN_MARKET_SCORE = float(os.getenv("MIN_MARKET_SCORE","82"))

CRYPTO = {
    "BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","XRP":"XRP-USD",
    "DOGE":"DOGE-USD","ADA":"ADA-USD","AVAX":"AVAX-USD","LINK":"LINK-USD",
}
STOCKS = [s.strip().upper() for s in os.getenv(
    "STOCK_WATCHLIST","NVDA,TSLA,AAPL,AMD,MSFT,AMZN,META,COIN,MSTR,PLTR"
).split(",") if s.strip()]

def fnum(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except:
        return default

def clamp(v, lo=0, hi=100):
    return max(lo,min(hi,v))

def money(v):
    v=fnum(v,float("nan"))
    if not np.isfinite(v): return "N/A"
    if abs(v)>=1000:return f"${v:,.2f}"
    if abs(v)>=1:return f"${v:,.4f}"
    if abs(v)>=.01:return f"${v:,.6f}"
    return f"${v:,.10f}"

def compact(v):
    v=fnum(v)
    if abs(v)>=1e9:return f"${v/1e9:.2f}B"
    if abs(v)>=1e6:return f"${v/1e6:.2f}M"
    if abs(v)>=1e3:return f"${v/1e3:.1f}K"
    return f"${v:.0f}"

def get_json(url, timeout=12):
    r=requests.get(url,headers=HEADERS,timeout=timeout)
    r.raise_for_status()
    return r.json()

def send_telegram(title,msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    r=requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id":TELEGRAM_CHAT_ID,"text":f"{title}\n\n{msg}"},
        timeout=12
    )
    r.raise_for_status()
    return True

def send_pushover(title,msg):
    if not PUSHOVER_API_TOKEN or not PUSHOVER_USER_KEY:
        return False
    r=requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token":PUSHOVER_API_TOKEN,"user":PUSHOVER_USER_KEY,
              "title":title[:250],"message":msg[:1024]},
        timeout=12
    )
    r.raise_for_status()
    return True

def notify(title,msg):
    sent=False
    try: sent = send_telegram(title,msg) or sent
    except Exception as e: print("Telegram error:",e)
    try: sent = send_pushover(title,msg) or sent
    except Exception as e: print("Pushover error:",e)
    print(("SENT" if sent else "NOT SENT"),title)

def age_hours(ms):
    if not ms:return 999999
    dt=datetime.fromtimestamp(fnum(ms)/1000,tz=timezone.utc)
    return max((datetime.now(timezone.utc)-dt).total_seconds()/3600,0)

def discovery_pairs():
    profiles=get_json(f"{DEX}/token-profiles/latest/v1")
    boosts=get_json(f"{DEX}/token-boosts/latest/v1")
    boosts_top=get_json(f"{DEX}/token-boosts/top/v1")
    records=(profiles or [])+(boosts or [])+(boosts_top or [])
    seen=set(); by_chain={}
    for x in records:
        c=x.get("chainId"); a=x.get("tokenAddress")
        if c and a and (c,a) not in seen:
            seen.add((c,a)); by_chain.setdefault(c,[]).append(a)

    pairs=[]
    for chain,addrs in by_chain.items():
        for i in range(0,len(addrs),30):
            group=",".join(addrs[i:i+30])
            try:
                data=get_json(f"{DEX}/tokens/v1/{chain}/{group}")
                if isinstance(data,list):pairs.extend(data)
            except Exception as e:
                print("DEX batch failed",chain,e)

    best={}
    for p in pairs:
        b=p.get("baseToken") or {}
        key=(p.get("chainId"),b.get("address"))
        liq=fnum((p.get("liquidity") or {}).get("usd"))
        if key[0] and key[1] and (key not in best or liq>fnum((best[key].get("liquidity") or {}).get("usd"))):
            best[key]=p
    return list(best.values()), boosts or [], boosts_top or []

def meme_scores(p, boosts, boosts_top):
    liq=fnum((p.get("liquidity") or {}).get("usd"))
    vol=p.get("volume") or {}
    ch=p.get("priceChange") or {}
    tx=p.get("txns") or {}
    t5=tx.get("m5") or {}; t1=tx.get("h1") or {}
    b5=fnum(t5.get("buys")); s5=fnum(t5.get("sells"))
    b1=fnum(t1.get("buys")); s1=fnum(t1.get("sells"))
    v5=fnum(vol.get("m5")); v1=fnum(vol.get("h1"))
    pc5=fnum(ch.get("m5")); pc1=fnum(ch.get("h1"))
    hrs=age_hours(p.get("pairCreatedAt"))
    ratio=(b5+1)/(s5+1)
    activity=b5+s5

    quality=25.0
    if liq>=250000:quality+=20
    elif liq>=100000:quality+=17
    elif liq>=40000:quality+=12
    elif liq<10000:quality-=25

    if .25<=hrs<=6:quality+=12
    elif hrs<=48:quality+=9
    elif hrs<=168:quality+=4

    if activity>=20 and ratio>=1.5:quality+=13
    elif activity>=8 and ratio>=1.15:quality+=7
    elif s5>b5*1.6:quality-=12

    if liq>0:
        turn=v1/liq
        if .25<=turn<=2.5:quality+=10

    if 1<=pc5<=12:quality+=7
    elif pc5<-10:quality-=9
    if 3<=pc1<=35:quality+=9
    elif pc1>100:quality-=25
    elif pc1>60:quality-=15
    elif pc1<-25:quality-=12
    quality=round(clamp(quality),1)

    b=(p.get("baseToken") or {})
    chain=p.get("chainId"); addr=b.get("address")
    boost_total=0
    for row in boosts+boosts_top:
        if row.get("chainId")==chain and row.get("tokenAddress")==addr:
            boost_total=max(boost_total,fnum(row.get("totalAmount")))

    attention=15.0
    if activity>=30:attention+=15
    elif activity>=12:attention+=8
    if v5>=25000:attention+=12
    elif v5>=5000:attention+=6
    if v1>=150000:attention+=10
    elif v1>=30000:attention+=5
    avg5=v1/12 if v1 else 0
    if avg5 and v5>=avg5*1.8:attention+=12
    if ratio>=1.5 and activity>=10:attention+=8
    if boost_total>0:attention+=min(15,5+math.log10(boost_total+1)*3)
    if hrs<=24:attention+=7
    if liq<8000:attention-=20
    if pc1>150:attention-=12
    attention=round(clamp(attention),1)

    return {
        "quality":quality,"attention":attention,"liq":liq,"v5":v5,"v1":v1,
        "pc5":pc5,"pc1":pc1,"b5":b5,"s5":s5,"hrs":hrs,"boost":boost_total
    }

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False).mean()
    ad=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return (100-100/(1+rs)).fillna(50)

def tech(df):
    if df is None or df.empty:return None
    df=df.dropna(subset=["Close"])
    if len(df)<55:return None
    c=df["Close"].astype(float); h=df["High"].astype(float); l=df["Low"].astype(float)
    e9=c.ewm(span=9,adjust=False).mean()
    e21=c.ewm(span=21,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean()
    rr=rsi(c)
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    mh=macd-macd.ewm(span=9,adjust=False).mean()
    price=float(c.iloc[-1]); score=50
    score+=10 if e9.iloc[-1]>e21.iloc[-1] else -8
    score+=10 if e21.iloc[-1]>e50.iloc[-1] else -7
    score+=5 if price>e9.iloc[-1] else -3
    rv=float(rr.iloc[-1])
    if 52<=rv<=68:score+=9
    elif rv>78:score-=10
    if mh.iloc[-1]>0:score+=7
    if mh.iloc[-1]>mh.iloc[-2]:score+=5
    r5=(price/float(c.iloc[-6])-1)*100
    r20=(price/float(c.iloc[-21])-1)*100
    if 0<r5<=8:score+=6
    elif r5>15:score-=12
    if r20>0:score+=4
    score=round(clamp(score),1)
    return {"score":score,"price":price,"r5":r5,"r20":r20,"rsi":rv}

def scan_memes():
    pairs,boosts,boosts_top=discovery_pairs()
    # Alert only during a narrow age-time bucket to avoid repeat alerts every 5 min:
    # early quality: first qualifying pass in each 30-min block, attention: first 5 min of each hour.
    minute=datetime.now(timezone.utc).minute
    for p in pairs:
        m=meme_scores(p,boosts,boosts_top)
        b=p.get("baseToken") or {}
        name=b.get("name") or "Unknown"; sym=b.get("symbol") or "?"
        addr=b.get("address") or ""; chain=p.get("chainId") or "?"
        price=fnum(p.get("priceUsd"))

        if m["quality"]>=MIN_MEME_SCORE and m["liq"]>=10000 and m["hrs"]<=24*14:
            # alert only in 0-5 or 30-35 minute windows
            if minute < 5 or 30 <= minute < 35:
                notify(
                    f"🌱 Early meme: {name} ({sym})",
                    f"Quality {m['quality']}/100\nAge {m['hrs']:.1f}h | {chain}\n"
                    f"Price {money(price)}\nLiquidity {compact(m['liq'])}\n"
                    f"1h volume {compact(m['v1'])}\n5m buys/sells {int(m['b5'])}/{int(m['s5'])}\n"
                    f"1h move {m['pc1']:+.1f}%\nContract {addr}"
                )

        if m["attention"]>=MIN_ATTN_SCORE and m["liq"]>=8000:
            if minute < 5:
                notify(
                    f"🔥 Meme attention: {name} ({sym})",
                    f"Attention {m['attention']}/100\nAge {m['hrs']:.1f}h | {chain}\n"
                    f"Price {money(price)}\n5m volume {compact(m['v5'])}\n"
                    f"5m buys/sells {int(m['b5'])}/{int(m['s5'])}\n"
                    f"Boost {int(m['boost'])}\nContract {addr}"
                )

def scan_markets():
    # Run market alerts only around every 6th hour to reduce repeats.
    hour=datetime.now(timezone.utc).hour
    minute=datetime.now(timezone.utc).minute
    if minute>=5 or hour%6!=0:
        return

    symbols=list(CRYPTO.values())+STOCKS
    raw=yf.download(" ".join(symbols),period="1y",interval="1d",auto_adjust=True,
                    progress=False,threads=True,group_by="ticker")
    for label,ticker in list(CRYPTO.items())+[(s,s) for s in STOCKS]:
        try:
            df=raw[ticker] if len(symbols)>1 else raw
            rd=tech(df)
            if rd and rd["score"]>=MIN_MARKET_SCORE:
                kind="Crypto" if ticker.endswith("-USD") else "Stock"
                notify(
                    f"🚨 {kind} setup: {label}",
                    f"Score {rd['score']}/100\nPrice {money(rd['price'])}\n"
                    f"5-day {rd['r5']:+.1f}% | 20-day {rd['r20']:+.1f}% | RSI {rd['rsi']:.1f}"
                )
        except Exception as e:
            print("market error",ticker,e)

def main():
    if not TELEGRAM_BOT_TOKEN and not PUSHOVER_API_TOKEN:
        raise RuntimeError("No notification provider secrets configured.")
    print("Starting scheduled scan",datetime.now(timezone.utc).isoformat())
    scan_memes()
    scan_markets()

if __name__=="__main__":
    main()
