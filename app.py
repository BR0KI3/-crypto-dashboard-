from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import streamlit as st
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

st.set_page_config(page_title="Crypto Market Scanner Pro", page_icon="📊", layout="wide")
DEFAULT_SYMBOLS = ["BTC/USD","ETH/USD","SOL/USD","DOGE/USD","SHIB/USD","AVAX/USD","LINK/USD","LTC/USD","BCH/USD","UNI/USD"]

def ema(s,n): return s.ewm(span=n, adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False).mean()
    ad=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return (100-(100/(1+rs))).fillna(50)

def atr(df,n=14):
    pc=df["close"].shift(1)
    tr=pd.concat([df["high"]-df["low"],(df["high"]-pc).abs(),(df["low"]-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def add_indicators(df):
    x=df.copy()
    x["ema9"]=ema(x["close"],9); x["ema21"]=ema(x["close"],21); x["ema50"]=ema(x["close"],50)
    x["rsi14"]=rsi(x["close"],14)
    macd=ema(x["close"],12)-ema(x["close"],26); sig=ema(macd,9)
    x["macd_hist"]=macd-sig
    x["atr14"]=atr(x,14); x["atr_pct"]=x["atr14"]/x["close"]*100
    x["vol_sma20"]=x["volume"].rolling(20).mean(); x["volume_ratio"]=x["volume"]/x["vol_sma20"].replace(0,np.nan)
    x["prev_high20"]=x["high"].shift(1).rolling(20).max(); x["prev_low20"]=x["low"].shift(1).rolling(20).min()
    x["breakout_pct"]=(x["close"]/x["prev_high20"]-1)*100
    x["ret3"]=x["close"].pct_change(3)*100; x["ret12"]=x["close"].pct_change(12)*100
    return x

def clamp(v,a,b): return max(a,min(b,v))

def score_row(r,p):
    s=0.0
    if r["ema9"]>r["ema21"]: s+=12
    if r["ema21"]>r["ema50"]: s+=12
    if r["close"]>r["ema9"]: s+=6
    rv=r["rsi14"]
    if 52<=rv<=68: s+=15
    elif 48<=rv<52 or 68<rv<=74: s+=8
    elif rv>80: s-=8
    if r["macd_hist"]>0:
        s+=10
        if r["macd_hist"]>p["macd_hist"]: s+=5
    vr=r["volume_ratio"] if pd.notna(r["volume_ratio"]) else 0
    s+=clamp((vr-.8)/1.2*15,0,15)
    bp=r["breakout_pct"] if pd.notna(r["breakout_pct"]) else -99
    if bp>=0: s+=clamp(8+bp*5,0,15)
    elif bp>-.8: s+=4
    r3=r["ret3"] if pd.notna(r["ret3"]) else 0
    r12=r["ret12"] if pd.notna(r["ret12"]) else 0
    if r3>0: s+=4
    if r12>0: s+=4
    if r3>0 and r12>0 and r3<5: s+=2
    if r3>8: s-=10
    if r["atr_pct"]>8: s-=10
    return round(clamp(s,0,100),1)

@st.cache_resource
def client(): return CryptoHistoricalDataClient()

@st.cache_data(ttl=45, show_spinner=False)
def fetch(symbol,minutes,lookback=320):
    start=datetime.now(timezone.utc)-timedelta(minutes=minutes*lookback*3)
    req=CryptoBarsRequest(symbol_or_symbols=[symbol],timeframe=TimeFrame(minutes,TimeFrameUnit.Minute),start=start,limit=lookback)
    df=client().get_crypto_bars(req).df
    if isinstance(df.index,pd.MultiIndex): df=df.xs(symbol,level=0)
    return df[["open","high","low","close","volume"]].dropna().sort_index()

def money(v):
    if v>=1000: return f"${v:,.2f}"
    if v>=1: return f"${v:,.4f}"
    if v>=.01: return f"${v:,.5f}"
    return f"${v:,.8f}"

def analyze(symbol,minutes,threshold):
    x=add_indicators(fetch(symbol,minutes))
    if len(x)<70: raise ValueError("Not enough market data")
    r=x.iloc[-1]; p=x.iloc[-2]
    score=score_row(r,p); price=float(r["close"]); a=float(r["atr14"]); risk=max(a*1.8,price*.004)
    if score>=threshold: action="BUY SETUP"
    elif score>=threshold-10: action="WATCH"
    elif score<=35: action="AVOID"
    else: action="WAIT"
    return dict(symbol=symbol,score=score,action=action,price=price,stop=price-risk,tp1=price+risk,tp2=price+2*risk,tp3=price+3*risk,row=r,prev=p,chart=x.tail(100))

st.title("📊 Crypto Market Scanner Pro")
st.caption("Ranks strongest setups first and explains every coin section by section.")

with st.expander("⚙️ Scanner settings"):
    symbols_text=st.text_area("Crypto pairs",",".join(DEFAULT_SYMBOLS))
    c1,c2=st.columns(2)
    with c1: bar_minutes=st.selectbox("Candle size",[1,5,15,30,60],index=1)
    with c2: threshold=st.slider("BUY threshold",55,90,72)

symbols=[s.strip().upper() for s in symbols_text.split(",") if s.strip()][:20]
if st.button("🔄 Scan market now",use_container_width=True):
    st.cache_data.clear(); st.rerun()

results=[]; errors=[]
with st.spinner("Reading each market section..."):
    for sym in symbols:
        try: results.append(analyze(sym,bar_minutes,threshold))
        except Exception as e: errors.append(f"{sym}: {e}")
results.sort(key=lambda x:x["score"],reverse=True)

if errors:
    with st.expander("⚠️ Symbols that could not be loaded"):
        for e in errors: st.write(e)
if not results:
    st.error("No market data loaded."); st.stop()

st.subheader("🏆 Strongest setups")
ranked=pd.DataFrame([{
    "Rank":i+1,"Coin":r["symbol"],"Action":r["action"],"Score":r["score"],
    "Price":money(r["price"]),"Stop":money(r["stop"]),"TP1":money(r["tp1"]),"TP2":money(r["tp2"]),"TP3":money(r["tp3"])
} for i,r in enumerate(results)])
st.dataframe(ranked,use_container_width=True,hide_index=True)

st.divider()
st.subheader("🔎 Full section-by-section analysis")

for r in results:
    row=r["row"]; prev=r["prev"]
    icon="🟢" if r["action"]=="BUY SETUP" else "🟡" if r["action"]=="WATCH" else "⚪"
    with st.expander(f"{icon} {r['symbol']} — {r['action']} — {r['score']}/100", expanded=(r is results[0])):
        a,b,c=st.columns(3)
        a.metric("Current price",money(r["price"])); b.metric("Final score",f'{r["score"]}/100'); c.metric("Decision",r["action"])

        st.markdown("### 1. Trend")
        if row["ema9"]>row["ema21"]>row["ema50"]:
            st.success("Strong bullish trend — EMA 9 > EMA 21 > EMA 50.")
        elif row["ema9"]<row["ema21"]<row["ema50"]:
            st.error("Bearish trend — EMA 9 < EMA 21 < EMA 50.")
        else:
            st.info("Mixed trend — EMA structure is not fully aligned.")
        t1,t2,t3=st.columns(3)
        t1.metric("EMA 9",money(float(row["ema9"]))); t2.metric("EMA 21",money(float(row["ema21"]))); t3.metric("EMA 50",money(float(row["ema50"])))

        st.markdown("### 2. RSI")
        rv=float(row["rsi14"]); st.metric("RSI 14",f"{rv:.1f}")
        if 52<=rv<=68: st.write("Healthy bullish momentum without being extremely overbought.")
        elif rv>75: st.write("Overbought/chase risk is elevated.")
        elif rv<35: st.write("Oversold; weak momentum, but oversold alone is not a buy signal.")
        else: st.write("RSI is neutral or mixed.")

        st.markdown("### 3. MACD")
        mh=float(row["macd_hist"]); pmh=float(prev["macd_hist"])
        st.metric("MACD histogram",f"{mh:.6g}")
        if mh>0 and mh>pmh: st.write("Bullish and strengthening.")
        elif mh>0: st.write("Bullish but momentum is slowing.")
        elif mh<pmh: st.write("Bearish and weakening.")
        else: st.write("Bearish but improving.")

        st.markdown("### 4. Volume")
        vr=float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else 0
        st.metric("Relative volume",f"{vr:.2f}×")
        st.write("Very strong volume." if vr>=2 else "Above-average volume." if vr>=1.3 else "Weak volume." if vr<.7 else "Normal volume.")

        st.markdown("### 5. Momentum")
        r3=float(row["ret3"]) if pd.notna(row["ret3"]) else 0
        r12=float(row["ret12"]) if pd.notna(row["ret12"]) else 0
        m1,m2=st.columns(2); m1.metric("3-bar move",f"{r3:+.2f}%"); m2.metric("12-bar move",f"{r12:+.2f}%")

        st.markdown("### 6. Breakout / Support / Resistance")
        bp=float(row["breakout_pct"]) if pd.notna(row["breakout_pct"]) else -99
        st.metric("Vs 20-bar resistance",f"{bp:+.2f}%")
        s1,s2=st.columns(2)
        s1.metric("20-bar support",money(float(row["prev_low20"])) if pd.notna(row["prev_low20"]) else "N/A")
        s2.metric("20-bar resistance",money(float(row["prev_high20"])) if pd.notna(row["prev_high20"]) else "N/A")

        st.markdown("### 7. Volatility / ATR")
        ap=float(row["atr_pct"]); st.metric("ATR as % of price",f"{ap:.2f}%")
        st.write("High volatility." if ap>4 else "Moderate volatility." if ap>=1.5 else "Low volatility.")

        st.markdown("### 8. Trade levels")
        l1,l2,l3,l4=st.columns(4)
        l1.metric("Stop",money(r["stop"])); l2.metric("TP1",money(r["tp1"])); l3.metric("TP2",money(r["tp2"])); l4.metric("TP3",money(r["tp3"]))

        st.markdown("### 9. Final decision")
        st.markdown(f"## {r['action']} — {r['score']}/100")
        if r["action"]=="BUY SETUP":
            st.success("Multiple indicators are aligned bullishly. Strongest category in this scanner, but not a guaranteed winning trade.")
        elif r["action"]=="WATCH":
            st.warning("Close to qualifying. Wait for stronger confirmation.")
        elif r["action"]=="AVOID":
            st.error("Current technical setup is weak for a long trade.")
        else:
            st.info("Not enough confirmation yet.")

        st.markdown("### 10. Chart")
        chart=r["chart"][["close","ema9","ema21","ema50"]].rename(columns={"close":"Price","ema9":"EMA 9","ema21":"EMA 21","ema50":"EMA 50"})
        st.line_chart(chart,use_container_width=True)

st.caption("Technical scanner only. A high score is not a guaranteed probability of profit.")
