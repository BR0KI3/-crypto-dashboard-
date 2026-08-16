from datetime import datetime, timezone
from urllib.parse import quote
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Simple Crypto Scanner", page_icon="📈", layout="centered")

st.markdown("""
<style>
.block-container {max-width:760px;padding-top:1rem;padding-bottom:3rem}
.card{border:1px solid rgba(128,128,128,.25);border-radius:18px;padding:16px;margin:12px 0}
.title{font-size:1.25rem;font-weight:800}
.sub{opacity:.72;font-size:.88rem}
.score{font-size:2rem;font-weight:850}
.stLinkButton a {width:100%;text-align:center}
</style>
""", unsafe_allow_html=True)

COINS = {
 "BTC":"bitcoin","ETH":"ethereum","SOL":"solana","DOGE":"dogecoin","SHIB":"shiba-inu",
 "AVAX":"avalanche-2","LINK":"chainlink","LTC":"litecoin","BCH":"bitcoin-cash",
 "UNI":"uniswap","XRP":"ripple","ADA":"cardano","PEPE":"pepe","BONK":"bonk",
 "WIF":"dogwifcoin","TRUMP":"official-trump"
}

def age_text(date_string):
    if not date_string: return "Unknown"
    try:
        dt=datetime.fromisoformat(date_string).replace(tzinfo=timezone.utc)
        days=max((datetime.now(timezone.utc)-dt).days,0)
        y=days//365; m=(days%365)//30
        return f"{y}y {m}mo" if y else f"{m}mo {days%30}d" if m else f"{days}d"
    except: return "Unknown"

def score(prices):
    s=pd.Series(prices).dropna()
    if len(s)<30:return 50,"WAIT"
    e9=s.ewm(span=9,adjust=False).mean().iloc[-1]
    e21=s.ewm(span=21,adjust=False).mean().iloc[-1]
    e50=s.ewm(span=50,adjust=False).mean().iloc[-1]
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.ewm(alpha=1/14,adjust=False).mean()/dn.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
    rsi=float((100-100/(1+rs)).fillna(50).iloc[-1])
    v=50
    v += 12 if e9>e21 else -8
    v += 12 if e21>e50 else -8
    v += 6 if s.iloc[-1]>e9 else 0
    if 52<=rsi<=68:v+=12
    elif rsi>78:v-=8
    v=max(0,min(100,round(v,1)))
    return v, "BUY SETUP" if v>=72 else "WATCH" if v>=62 else "AVOID" if v<=35 else "WAIT"

@st.cache_data(ttl=60,show_spinner=False)
def load_coin(sym):
    cid=COINS.get(sym)
    if not cid: raise ValueError(f"{sym} isn't in the built-in list yet.")
    h={"accept":"application/json","user-agent":"simple-crypto-scanner"}
    d=requests.get(f"https://api.coingecko.com/api/v3/coins/{cid}",headers=h,timeout=15); d.raise_for_status()
    j=d.json()
    m=requests.get(f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart",
                   headers=h,params={"vs_currency":"usd","days":"30","interval":"hourly"},timeout=15); m.raise_for_status()
    prices=[x[1] for x in m.json().get("prices",[])]
    sc,action=score(prices)
    platforms=j.get("platforms") or {}
    contracts=[(n,a) for n,a in platforms.items() if a]
    network,address=(contracts[0] if contracts else (None,None))
    return {
      "name":j.get("name",sym),"symbol":sym,
      "price":j.get("market_data",{}).get("current_price",{}).get("usd"),
      "score":sc,"action":action,"network":network,"address":address,
      "genesis":j.get("genesis_date"),"age":age_text(j.get("genesis_date"))
    }

def money(v):
    if v is None:return "N/A"
    if v>=1000:return f"${v:,.2f}"
    if v>=1:return f"${v:,.4f}"
    if v>=.01:return f"${v:,.5f}"
    return f"${v:,.8f}"

st.title("📈 Simple Crypto Scanner")
st.caption("Price • signal • age • one-tap Fomo search")

symbols=st.text_input("Coins","BTC,ETH,SOL,DOGE,SHIB,PEPE,BONK,WIF")
symbols=[x.strip().upper() for x in symbols.split(",") if x.strip()][:12]

if st.button("🔄 Refresh",use_container_width=True):
    st.cache_data.clear(); st.rerun()

items=[]; errors=[]
with st.spinner("Scanning..."):
    for sym in symbols:
        try:items.append(load_coin(sym))
        except Exception as e:errors.append(str(e))
items.sort(key=lambda x:x["score"],reverse=True)

if errors:
    with st.expander("Could not load"):
        for e in errors:st.write(e)

for a in items:
    icon="🟢" if a["action"]=="BUY SETUP" else "🟡" if a["action"]=="WATCH" else "⚪"
    st.markdown(f"""
    <div class="card">
      <div class="title">{icon} {a['name']} ({a['symbol']})</div>
      <div style="display:flex;justify-content:space-between;align-items:end;margin-top:10px">
        <div><b>{money(a['price'])}</b><div class="sub">{a['action']}</div></div>
        <div style="text-align:right"><div class="score">{a['score']}</div><div class="sub">score / 100</div></div>
      </div>
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    c1.metric("Age",a["age"])
    c2.metric("Launch",a["genesis"] or "Unknown")

    if a["address"]:
        st.caption(f"Contract • {a['network'] or 'network'}")
        st.code(a["address"],language=None)
        # Fomo link: send the exact contract address as a search query.
        fomo_url="https://fomo.biz/search?q="+quote(a["address"],safe="")
        st.link_button("🚀 Open this contract on Fomo",fomo_url,use_container_width=True)
    else:
        st.info("Native coin — no token contract address.")
        # For native assets, search by symbol instead.
        fomo_url="https://fomo.biz/search?q="+quote(a["symbol"],safe="")
        st.link_button("🚀 Search this coin on Fomo",fomo_url,use_container_width=True)

st.caption("Always verify the contract shown by the destination before buying. Symbols and names can be copied by unrelated tokens.")
