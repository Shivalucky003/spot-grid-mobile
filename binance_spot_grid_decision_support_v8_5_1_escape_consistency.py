import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import math

EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
TICKER_24H_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

STABLECOIN_BLACKLIST = {
    "USDC","FDUSD","TUSD","BUSD","DAI","USDP","EUR","AEUR","USDT",
    "PAX","USD1","RLUSD","PYUSD","USDE","USDS","USDD","GUSD","LUSD",
    "FRAX","USDJ","USDB","DEUSD","SUSD","EUSD","CUSD","EURS","TRY",
    "BRL","BIDR","U"
}

@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "SpotGridDecisionSupport/8.4.1"})
    return s

@st.cache_data(ttl=600)
def fetch_universe():
    try:
        r = get_session().get(EXCHANGE_INFO_URL, timeout=15)
        r.raise_for_status()
        symbols, filters = [], {}
        for s in r.json().get("symbols", []):
            base = str(s.get("baseAsset", "")).upper()
            quote = str(s.get("quoteAsset", "")).upper()
            stable = base in STABLECOIN_BLACKLIST or "USD" in base or "EUR" in base
            if s.get("status") == "TRADING" and quote == "USDT" and s.get("isSpotTradingAllowed", True) and not stable:
                sym = s["symbol"]
                symbols.append(sym)
                f = {"tickSize":0.0,"stepSize":0.0,"minQty":0.0,"minNotional":0.0}
                for x in s.get("filters", []):
                    typ = x.get("filterType")
                    if typ == "PRICE_FILTER":
                        f["tickSize"] = float(x.get("tickSize",0) or 0)
                    elif typ == "LOT_SIZE":
                        f["stepSize"] = float(x.get("stepSize",0) or 0)
                        f["minQty"] = float(x.get("minQty",0) or 0)
                    elif typ in ("MIN_NOTIONAL","NOTIONAL"):
                        f["minNotional"] = float(x.get("minNotional",0) or 0)
                filters[sym] = f
        return symbols, filters
    except Exception:
        return [], {}

@st.cache_data(ttl=300)
def fetch_24h():
    try:
        r = get_session().get(TICKER_24H_URL, timeout=15)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        for c in ["lastPrice","highPrice","lowPrice","volume","quoteVolume","priceChangePercent"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["symbol","lastPrice","highPrice","lowPrice","volume","quoteVolume","priceChangePercent"]]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_klines(symbol, interval="1h", limit=1000):
    try:
        r = get_session().get(KLINES_URL, params={"symbol":symbol,"interval":interval,"limit":min(int(limit),1000)}, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        cols = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVolume","Trades","TBB","TBQ","Ignore"]
        df = pd.DataFrame(r.json(), columns=cols)
        for c in ["Open","High","Low","Close","Volume","QuoteVolume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
        return add_indicators(df)
    except Exception:
        return pd.DataFrame()

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def add_indicators(df):
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    for n in [9,21,50,200]:
        df[f"EMA{n}"] = ema(close,n)

    m12, m26 = ema(close,12), ema(close,26)
    df["MACD"] = m12-m26
    df["MACDSignal"] = ema(df["MACD"],9)
    df["MACDHist"] = df["MACD"]-df["MACDSignal"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ag/al.replace(0,np.nan)
    df["RSI"] = (100-100/(1+rs)).fillna(50)

    prev = close.shift(1)
    tr = pd.concat([(high-low),(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    df["TR"] = tr
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up>down)&(up>0),up,0.0), index=df.index)
    minus_dm = pd.Series(np.where((down>up)&(down>0),down,0.0), index=df.index)
    atr = df["ATR"].replace(0,np.nan)
    plus_di = 100*plus_dm.ewm(alpha=1/14,adjust=False).mean()/atr
    minus_di = 100*minus_dm.ewm(alpha=1/14,adjust=False).mean()/atr
    dx = 100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    df["ADX"] = dx.ewm(alpha=1/14,adjust=False).mean().fillna(0)

    mid = close.rolling(20,min_periods=20).mean()
    std = close.rolling(20,min_periods=20).std(ddof=0)
    df["BBMid"] = mid
    df["BBUpper"] = mid+2*std
    df["BBLower"] = mid-2*std
    df["BBWidthPct"] = (df["BBUpper"]-df["BBLower"])/mid.replace(0,np.nan)*100

    vm = vol.rolling(20,min_periods=20).mean()
    df["VolumeRatio"] = vol/vm.replace(0,np.nan)

    df["Regime"] = df["ADX"].apply(lambda x: "Ranging" if np.isfinite(x) and x<20 else ("Transition" if np.isfinite(x) and x<25 else ("Trending" if np.isfinite(x) else "Unknown")))
    return df

def floor_step(x, step):
    return float(x) if not step or step<=0 else math.floor((x+1e-12)/step)*step

def build_grid(lower, upper, n, mode, tick=0):
    """Build a Binance-valid grid and reject degenerate tick-rounded ranges."""
    try:
        lower=float(lower); upper=float(upper); n=max(1,int(n)); tick=float(tick or 0)
    except (TypeError, ValueError):
        return []

    if not np.isfinite(lower) or not np.isfinite(upper) or lower<=0 or upper<=lower:
        return []

    if mode=="Geometric":
        levels=lower*(upper/lower)**(np.arange(n+1,dtype=float)/n)
    else:
        levels=np.linspace(lower,upper,n+1,dtype=float)

    if tick>0 and np.isfinite(tick):
        # Binance requires every order price to be a tick multiple.
        # Flooring can turn a very small lower bound into zero, so never
        # allow zero/negative grid levels.
        rounded=np.array([floor_step(float(x),tick) for x in levels],dtype=float)
        rounded=rounded[np.isfinite(rounded) & (rounded>0)]
        levels=np.unique(rounded)
    else:
        levels=np.unique(levels[np.isfinite(levels) & (levels>0)])

    if len(levels)<2 or levels[-1] <= levels[0]:
        return []

    return levels.tolist()

def make_range(train, atr_mult):
    hi, lo = float(train["High"].max()), float(train["Low"].min())
    atr=float(train["ATR"].iloc[-1])
    if not np.isfinite(atr) or atr<=0:
        atr=float(train["Close"].pct_change().abs().median()*train["Close"].iloc[-1])
    if not np.isfinite(atr) or atr<=0 or hi<=lo:
        return None
    return max(1e-12,lo-atr_mult*atr), hi+atr_mult*atr

def simulate(test, grid, capital, fee_pct, filters, inventory_pct, low_first):
    if test.empty or capital<=0 or len(grid)<2:
        return None
    step_size=filters.get("stepSize",0.0)
    min_qty=filters.get("minQty",0.0)
    min_notional=filters.get("minNotional",0.0)
    intervals=len(grid)-1
    order_quote=capital/intervals

    base_value=capital*max(0,min(1,inventory_pct))
    quote=capital-base_value
    base=base_value/float(test["Close"].iloc[0])
    inventory={}
    realized=0.0; trades=buys=sells=skipped=cycles=escapes=0
    escape_severity=0.0; peak=capital; max_dd=0.0

    for _,row in test.iterrows():
        o,h,l,c=[float(row[x]) for x in ["Open","High","Low","Close"]]
        # Range-escape metrics are measured ONCE PER CANDLE so the ranking
        # uses the same definition as the Active Grid Audit.
        # Escape Rate = % of test candles whose High/Low breached the range.
        # Escape Severity = average maximum breach beyond either boundary,
        # expressed as % of the breached boundary.
        down_pct=max(0.0,(grid[0]-l)/grid[0]*100) if grid[0]>0 else 0.0
        up_pct=max(0.0,(h-grid[-1])/grid[-1]*100) if grid[-1]>0 else 0.0
        if down_pct>0 or up_pct>0:
            escapes+=1
            escape_severity += max(down_pct,up_pct)

        path=[o,l,h,c] if low_first else [o,h,l,c]
        for a,bp in zip(path[:-1],path[1:]):
            if bp>a:
                crossed=[i for i in range(1,len(grid)) if a<grid[i]<=bp]
                for i in crossed:
                    if i-1 not in inventory:
                        continue
                    price=grid[i]
                    qty=inventory[i-1]["qty"]
                    notional=qty*price
                    if (min_notional and notional<min_notional):
                        skipped+=1; continue
                    fee=notional*fee_pct/100
                    quote+=notional-fee
                    base=max(0,base-qty)
                    cost=inventory.pop(i-1)["cost"]
                    realized+=notional-fee-cost
                    trades+=1; sells+=1; cycles+=1
            elif bp<a:
                crossed=[i for i in range(len(grid)-1) if bp<=grid[i]<a]
                for i in reversed(crossed):
                    price=grid[i]
                    qty=floor_step(order_quote/price,step_size)
                    if qty<=0 or (min_qty and qty<min_qty) or (min_notional and qty*price<min_notional):
                        skipped+=1; continue
                    notional=qty*price
                    fee=notional*fee_pct/100
                    total=notional+fee
                    if total>quote:
                        skipped+=1; continue
                    quote-=total; base+=qty
                    inventory[i]={"qty":qty,"cost":total}
                    trades+=1; buys+=1

        equity=quote+base*c
        peak=max(peak,equity)
        max_dd=max(max_dd,(peak-equity)/peak*100 if peak>0 else 0)

    final_price=float(test["Close"].iloc[-1])
    final_equity=quote+base*final_price
    total_return=(final_equity/capital-1)*100
    buy_hold=(final_price/float(test["Close"].iloc[0])-1)*100
    realized_roi=realized/capital*100
    loss=max(-realized,0)
    profit_factor=(max(realized,0)/loss) if loss>0 else (float("inf") if realized>0 else 0)
    return {
        "final_equity":final_equity,"total_return":total_return,"buy_hold":buy_hold,
        "alpha":total_return-buy_hold,"realized_roi":realized_roi,"realized":realized,
        "max_dd":max_dd,"profit_factor":profit_factor,"trades":trades,"buys":buys,"sells":sells,"cycles":cycles,
        "skipped":skipped,"skip_rate":skipped/max(trades+skipped,1)*100,
        "escapes":escapes,"escape_rate":escapes/max(len(test),1)*100,
        "escape_severity":escape_severity/max(escapes,1)
    }

def backtest(test,lower,upper,n,capital,fee_pct,mode,filters,inventory_pct):
    grid=build_grid(lower,upper,n,mode,filters.get("tickSize",0))
    if len(grid)<2 or grid[0]<=0 or grid[-1]<=grid[0]:
        return None
    a=simulate(test,grid,capital,fee_pct,filters,inventory_pct,True)
    b=simulate(test,grid,capital,fee_pct,filters,inventory_pct,False)
    if not a or not b: return None
    c=min([a,b],key=lambda x:(x["total_return"],-x["max_dd"]))
    return {"conservative":c,"low_first":a,"high_first":b,"path_spread":abs(a["total_return"]-b["total_return"]),"grid":grid}

def score_row(r, feasible):
    """Decision score for a grid candidate.

    The unseen-test comparison with buy-and-hold is deliberately the
    strongest signal. A positive training result cannot rescue a grid
    that fails the unseen test or is not executable with the wallet.
    """
    s=50.0
    adx=float(r.get("ADX",50))
    alpha=float(r.get("Test Alpha %",0))
    ret=float(r.get("Test Return %",0))
    dd=float(r.get("Max DD %",100))
    skipped=float(r.get("Skipped %",100))
    escapes=float(r.get("Escape Rate %",100))
    cycles=int(r.get("Cycles",0))

    # Market suitability.
    s += 12 if adx<20 else (5 if adx<25 else -10)

    # Out-of-sample alpha is the core signal.
    s += 28 if alpha>=3 else (18 if alpha>=1 else (5 if alpha>=0 else (-12 if alpha>-3 else -28)))

    # Absolute test return is secondary.
    s += 8 if ret>5 else (4 if ret>0 else -8)

    # Risk/execution quality.
    s += 10 if dd<5 else (4 if dd<10 else (-8 if dd<15 else -15))
    s += 6 if skipped<=5 else (-6 if skipped>15 else 0)
    s += 6 if escapes<=5 else (-6 if escapes>15 else 0)
    s += 5 if cycles>=10 else (-8 if cycles<3 else 0)
    s += 7 if feasible else -20
    s -= 4 if r.get("Step %",0) < r.get("Target Step %",1)*0.75 else 0
    s=max(0,min(100,s))

    # Hard gates: a high score must never hide a failed unseen test.
    if not feasible:
        decision="🔴 NOT FEASIBLE"
    elif alpha < -3:
        decision="🔴 AVOID"
    elif cycles==0 or skipped>60:
        decision="🔴 AVOID"
    elif alpha < 0:
        decision="🟡 WEAK"
    elif alpha>=2 and dd<8 and cycles>=5 and escapes<=15:
        decision="🟢 CONSIDER"
    else:
        decision="🟡 REVIEW"
    return s,decision

def render_finder():
    st.title("⚡ Binance Spot Grid Decision Support V8.5.1")
    st.caption("Read-only tool. It does not place Binance orders.")

    with st.sidebar:
        st.header("Controls")
        capital=st.number_input("Available Wallet (USDT)",min_value=10.0,value=160.0,step=10.0)
        fee_choice=st.selectbox("Round-trip fee",["Standard 0.20%","BNB Discount 0.15%","Zero 0.00%"])
        fee_pct=0.20 if fee_choice.startswith("Standard") else (0.15 if fee_choice.startswith("BNB") else 0)
        mode=st.selectbox("Grid Mode",["Arithmetic","Geometric"])
        preselect_count=st.number_input("Candidates for Full Backtest",10,50,25,5)
        min_quote_volume=st.number_input("Minimum 24h Quote Volume (USDT)",0.0,1_000_000_000.0,5_000_000.0,1_000_000.0)
        train_frac=st.slider("Training Fraction",0.60,0.85,0.70,0.05)
        atr_mult=st.slider("ATR Range Buffer",0.5,3.0,1.5,0.1)
        st.caption("24h ticker is used only for liquidity screening. Candidate quality is determined with 5m, 1h and 4h data.")
        min_grid=st.number_input("Minimum USDT per Grid Interval",5.0,100.0,8.0,1.0)
        inventory_pct=st.slider("Starting Base-Coin Inventory %",0,50,0,5)
        tight=st.number_input("Tight 3D target step %",0.2,5.0,0.8,0.1)
        moderate=st.number_input("Moderate 7D target step %",0.2,5.0,1.2,0.1)
        wide=st.number_input("Wide 14D target step %",0.2,5.0,1.8,0.1)
        only_ranging=st.checkbox("Filter out trending markets",False)

    if "candidates" not in st.session_state:
        st.session_state.candidates=[]
    if "filters" not in st.session_state:
        st.session_state.filters={}
    if "results" not in st.session_state:
        st.session_state.results=None

    st.subheader("1. Multi-Timeframe Market Scanner")
    st.caption("The scanner does not simply take the highest-volume coins. It uses a diversified liquidity pool, then evaluates 1h/4h grid suitability and 5m execution conditions.")

    if st.button("🔎 Scan Binance + Multi-Timeframe Screen",type="primary"):
        with st.spinner("Loading Binance universe and 24h liquidity data..."):
            symbols,filters=fetch_universe()
            ticker=fetch_24h()

        if ticker.empty or not symbols:
            st.error("Could not load Binance market data.")
            return

        valid=ticker[ticker["symbol"].isin(symbols)].copy()
        valid=valid.dropna(subset=["quoteVolume","lastPrice"])
        valid=valid[valid["quoteVolume"]>=min_quote_volume]

        # ------------------------------------------------------------
        # V8.4.1 CANDIDATE DISCOVERY
        # Do NOT use "top N by volume" as the candidate universe.
        #
        # Stage A: create a diversified liquidity pool from several
        # liquidity bands.
        # Stage B: use 1h + 4h data as the main pre-screen.
        # Stage C: use 5m data for the best Stage-B markets.
        #
        # Missing 5m data does NOT automatically eliminate a market.
        # ------------------------------------------------------------
        valid = valid[valid["quoteVolume"] >= min_quote_volume].copy()

        if valid.empty:
            st.error(
                "No Binance USDT spot markets meet the current liquidity "
                "threshold. Reduce the minimum 24h quote volume."
            )
            return

        valid = valid.sort_values("quoteVolume", ascending=False).reset_index(drop=True)

        target_scan = max(120, int(preselect_count) * 6)
        target_scan = min(target_scan, len(valid))

        # Split the eligible universe into liquidity bands. This prevents
        # BTC/ETH/large-cap pairs from consuming the entire discovery pool.
        # Use explicit index slices instead of np.array_split(DataFrame).
        # This avoids pandas/numpy compatibility differences where a band
        # can be returned as an ndarray instead of a DataFrame.
        n_bands = min(4, len(valid))
        index_bands = np.array_split(valid.index.to_numpy(), n_bands)
        pool_parts = []
        per_band = max(1, math.ceil(target_scan / n_bands))

        for band_index in index_bands:
            if len(band_index) == 0:
                continue
            band_df = valid.loc[band_index]
            pool_parts.append(band_df.head(per_band))

        pool = pd.concat(pool_parts, ignore_index=True).drop_duplicates("symbol")
        pool = pool.head(target_scan).reset_index(drop=True)

        st.info(
            f"Eligible markets: {len(valid)} | "
            f"Diversified discovery pool: {len(pool)} | "
            f"Target detailed candidates: {int(preselect_count)}"
        )

        # ------------------------------------------------------------
        # Stage A/B: 1h + 4h first.
        # These timeframes are much more important to grid suitability
        # than raw 24h volume.
        # ------------------------------------------------------------
        rows = []
        progress = st.progress(0)

        for n, (_, trow) in enumerate(pool.iterrows()):
            symbol = trow["symbol"]

            df1 = fetch_klines(symbol, "1h", 336)
            df4 = fetch_klines(symbol, "4h", 168)

            if len(df1) < 120 or len(df4) < 80:
                progress.progress((n + 1) / len(pool))
                continue

            x1 = df1.iloc[-1]
            x4 = df4.iloc[-1]

            adx1 = float(x1["ADX"]) if np.isfinite(x1["ADX"]) else 50.0
            atr1 = (
                float(x1["ATR"] / x1["Close"] * 100)
                if np.isfinite(x1["ATR"]) and x1["Close"] > 0
                else 99.0
            )
            bb1 = (
                float(x1["BBWidthPct"])
                if np.isfinite(x1["BBWidthPct"])
                else 99.0
            )

            adx4 = float(x4["ADX"]) if np.isfinite(x4["ADX"]) else 50.0
            ema50 = float(x4["EMA50"]) if np.isfinite(x4["EMA50"]) else np.nan
            ema200 = float(x4["EMA200"]) if np.isfinite(x4["EMA200"]) else np.nan
            close4 = float(x4["Close"])

            score = 50.0

            # 1h primary grid regime.
            score += 18 if adx1 < 20 else (8 if adx1 < 25 else -12)
            score += 8 if 0.20 <= atr1 <= 2.50 else (-8 if atr1 > 4 else 0)
            score += 5 if 0.5 <= bb1 <= 12 else (-5 if bb1 > 18 else 0)

            # 4h trend / breakout risk.
            if adx4 < 20:
                score += 12
            elif adx4 < 25:
                score += 5
            else:
                score -= 12

            if np.isfinite(ema50) and np.isfinite(ema200):
                if abs(close4 / ema200 - 1) < 0.08:
                    score += 4
                elif close4 > ema50 > ema200 or close4 < ema50 < ema200:
                    score -= 6

            qv = float(trow["quoteVolume"])
            # Liquidity contributes modestly; it does not dominate.
            if qv >= 50_000_000:
                score += 6
            elif qv >= 10_000_000:
                score += 3
            else:
                score += 0

            rows.append({
                "Coin": symbol,
                "Pre-Score": float(score),
                "24h Quote Volume": qv,
                "5m ADX": np.nan,
                "5m ATR %": np.nan,
                "5m BB Width %": np.nan,
                "1h ADX": round(adx1, 1),
                "1h ATR %": round(atr1, 2),
                "1h BB Width %": round(bb1, 2),
                "4h ADX": round(adx4, 1),
                "4h Regime": x4["Regime"],
                "24h Change %": float(trow["priceChangePercent"]),
                "_df5": None
            })

            progress.progress((n + 1) / len(pool))

        if not rows:
            st.error(
                "No markets produced enough 1h/4h historical data. "
                "This is usually a temporary Binance API/data availability issue. "
                "Try scanning again."
            )
            return

        # ------------------------------------------------------------
        # Stage C: 5m analysis only for a broad set of the best 1h/4h
        # markets. Missing 5m data is tolerated rather than eliminating
        # the market.
        # ------------------------------------------------------------
        pre_stage = (
            pd.DataFrame(rows)
            .sort_values(
                ["Pre-Score", "1h ADX", "24h Quote Volume"],
                ascending=[False, True, False]
            )
            .reset_index(drop=True)
        )

        five_min_budget = min(
            len(pre_stage),
            max(int(preselect_count) * 4, 60)
        )

        five_min_symbols = set(pre_stage.head(five_min_budget)["Coin"].tolist())

        for n, row in pre_stage.iterrows():
            symbol = row["Coin"]
            if symbol not in five_min_symbols:
                continue

            df5 = fetch_klines(symbol, "5m", 288)
            if len(df5) < 100:
                # Keep the candidate. We simply don't award/penalize it
                # for unavailable 5m data.
                continue

            x5 = df5.iloc[-1]

            adx5 = float(x5["ADX"]) if np.isfinite(x5["ADX"]) else np.nan
            atr5 = (
                float(x5["ATR"] / x5["Close"] * 100)
                if np.isfinite(x5["ATR"]) and x5["Close"] > 0
                else np.nan
            )
            bb5 = (
                float(x5["BBWidthPct"])
                if np.isfinite(x5["BBWidthPct"])
                else np.nan
            )

            score = float(row["Pre-Score"])

            if np.isfinite(adx5):
                score += 5 if adx5 < 30 else -4
            if np.isfinite(atr5) and atr5 > 5:
                score -= 8
            if np.isfinite(bb5) and bb5 > 25:
                score -= 5

            pre_stage.at[n, "Pre-Score"] = max(0.0, min(100.0, score))
            pre_stage.at[n, "5m ADX"] = round(adx5, 1) if np.isfinite(adx5) else np.nan
            pre_stage.at[n, "5m ATR %"] = round(atr5, 2) if np.isfinite(atr5) else np.nan
            pre_stage.at[n, "5m BB Width %"] = round(bb5, 2) if np.isfinite(bb5) else np.nan

        pre_df = pre_stage.drop(columns=["_df5"], errors="ignore").sort_values(
            ["Pre-Score", "1h ADX", "24h Quote Volume"],
            ascending=[False, True, False]
        ).reset_index(drop=True)

        selected = pre_df.head(int(preselect_count))

        st.session_state.candidates = selected["Coin"].tolist()
        st.session_state.filters = filters
        st.session_state["pre_df"] = pre_df

        if len(selected) < int(preselect_count):
            st.warning(
                f"Only {len(selected)} markets currently have sufficient historical "
                f"data for the detailed stage. The scanner did NOT force this down "
                f"to one coin."
            )
        else:
            st.success(
                f"Scanned {len(valid)} eligible USDT pairs → "
                f"{len(pool)} diversified discovery markets → "
                f"{len(pre_df)} analyzed → "
                f"{len(selected)} selected for detailed backtesting."
            )

        # User-facing table: show useful selection information, not raw
        # indicator calculations.
        display_pre = selected.copy()
        display_pre["Liquidity"] = display_pre["24h Quote Volume"].map(
            lambda x: f"{x:,.0f} USDT"
        )
        display_pre["5m"] = np.where(
            display_pre["5m ADX"].notna(), "✓", "—"
        )
        display_pre["4h"] = display_pre["4h Regime"]
        display_pre["Market View"] = np.select(
            [
                display_pre["Pre-Score"] >= 75,
                display_pre["Pre-Score"] >= 60
            ],
            [
                "🟢 Good grid conditions",
                "🟡 Review"
            ],
            default="🔴 Weak grid conditions"
        )

        st.dataframe(
            display_pre[
                ["Coin", "Market View", "Pre-Score", "Liquidity",
                 "5m", "4h", "24h Change %"]
            ],
            use_container_width=True,
            hide_index=True
        )

    if not st.session_state.candidates:
        st.warning("Scan Binance Markets first.")
        return

    st.subheader("2. Detailed Train/Test Grid Backtest")
    if st.button("🚀 Run V8.5.1 Train/Test Decision Backtest"):
        results=[]; progress=st.progress(0)
        strategies=[("Tight (3D)",72,tight),("Moderate (7D)",168,moderate),("Wide (14D)",336,wide)]
        total=len(st.session_state.candidates)
        for idx,symbol in enumerate(st.session_state.candidates):
            df=fetch_klines(symbol,"1h",336)
            if len(df)<120:
                progress.progress((idx+1)/total); continue
            latest=df.iloc[-1]
            if only_ranging and latest["Regime"]=="Trending":
                progress.progress((idx+1)/total); continue

            for name,candles,target in strategies:
                sub=df.tail(candles).copy()
                split=int(len(sub)*train_frac)
                train,test=sub.iloc[:split],sub.iloc[split:]
                if len(train)<40 or len(test)<20: continue
                rng=make_range(train,atr_mult)
                if not rng: continue
                lower,upper=rng
                width=(upper-lower)/lower*100
                grids=max(5,min(100,int(width/max(target,0.01))))
                filt=st.session_state.filters.get(symbol,{})
                # Cap grid count so each interval receives at least the chosen capital allocation.
                grids=min(grids,max(5,int(capital/max(min_grid,1))))
                bt=backtest(test,lower,upper,grids,capital,fee_pct,mode,filt,inventory_pct/100)
                if not bt: continue
                c=bt["conservative"]; actual_grids=len(bt["grid"])-1
                usdt_grid=capital/max(actual_grids,1)
                first_grid=float(bt["grid"][0]) if bt.get("grid") else 0.0
                second_grid=float(bt["grid"][1]) if len(bt.get("grid",[]))>1 else 0.0
                grid_span=float(bt["grid"][-1]-bt["grid"][0]) if len(bt.get("grid",[]))>1 else 0.0
                step_pct=((second_grid-first_grid)/first_grid*100) if first_grid>0 else 0.0
                feasible=(usdt_grid>=min_grid and first_grid>0 and grid_span>0)
                current=float(latest["Close"])
                range_span=float(upper-lower)
                position=max(0,min(100,(current-lower)/range_span*100)) if range_span>0 else 50.0
                row={
                    "Coin":symbol,"Strategy":name,"Decision":"","Score":0,
                    "Current":current,"Lower":lower,"Upper":upper,"Price Position %":position,
                    "Grids":actual_grids,"USDT/Grid":usdt_grid,"Step %":step_pct,"Target Step %":target,
                    "Test Return %":c["total_return"],"Buy-Hold %":c.get("buy_hold", 0),"Test Alpha %":c.get("alpha", 0),
                    "Realized ROI %":c["realized_roi"],"Final Equity":c["final_equity"],"Max DD %":c.get("max_dd", 0),
                    "Profit Factor":c.get("profit_factor", 0.0),"Trades":c.get("trades", 0),"Cycles":c.get("cycles", 0),
                    "Skipped":c.get("skipped", 0),"Skipped %":c["skip_rate"],"Escape Bars":c["escapes"],
                    "Escape Rate %":c["escape_rate"],"Escape Severity %":c["escape_severity"],
                    "Path Spread %":bt["path_spread"],"ADX":float(latest["ADX"]),"RSI":float(latest["RSI"]),
                    "Regime":latest["Regime"],"EMA9":float(latest["EMA9"]) if pd.notna(latest["EMA9"]) else np.nan,
                    "EMA21":float(latest["EMA21"]) if pd.notna(latest["EMA21"]) else np.nan,
                    "EMA50":float(latest["EMA50"]) if pd.notna(latest["EMA50"]) else np.nan,
                    "EMA200":float(latest["EMA200"]) if pd.notna(latest["EMA200"]) else np.nan,
                    "MACD Hist":float(latest["MACDHist"]) if pd.notna(latest["MACDHist"]) else np.nan,
                    "ATR":float(latest["ATR"]) if pd.notna(latest["ATR"]) else np.nan,
                    "BB Width %":float(latest["BBWidthPct"]) if pd.notna(latest["BBWidthPct"]) else np.nan,
                    "Volume Ratio":float(latest["VolumeRatio"]) if pd.notna(latest["VolumeRatio"]) else np.nan
                }
                row["Score"],row["Decision"]=score_row(row,feasible)
                results.append(row)
            progress.progress((idx+1)/total)

        if results:
            st.session_state.results=pd.DataFrame(results).sort_values(["Score","Test Alpha %","Test Return %"],ascending=[False,False,False]).reset_index(drop=True)
        else:
            st.error("No valid configurations were produced.")

    if st.session_state.results is None:
        return

    final=st.session_state.results
    st.subheader("3. Decision Summary")
    strong=final[final["Decision"]=="🟢 CONSIDER"]
    if strong.empty:
        st.warning("⚠️ No grid currently passes the strong out-of-sample decision gate. Do not deploy a grid just because it has the highest score.")
    else:
        best=strong.iloc[0]
        st.success(f"🏆 **BEST CURRENT GRID: {best['Coin']} — {best['Strategy']}** | **{best['Decision']}** | Score **{best['Score']:.0f}/100**")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Grid Test Return",f"{best['Test Return %']:.2f}%")
        c2.metric("Buy & Hold",f"{best['Buy-Hold %']:.2f}%")
        c3.metric("Grid Advantage",f"{best['Test Alpha %']:+.2f}%")
        c4.metric("Range Escape Rate",f"{best['Escape Rate %']:.1f}%")
        e1,e2,e3=st.columns(3)
        e1.metric("Max Drawdown",f"{best['Max DD %']:.2f}%")
        e2.metric("Escape Severity",f"{best['Escape Severity %']:.2f}%")
        e3.metric("Skipped Orders",f"{best['Skipped %']:.1f}%")
        st.info("The recommendation is based primarily on unseen-test performance. The engine does not treat a positive grid return as sufficient if buy-and-hold did better.")

    st.markdown("### Quick ranking")
    quick_cols=["Coin","Strategy","Decision","Score","Test Return %","Buy-Hold %","Test Alpha %","Max DD %","Cycles","Skipped %","Escape Rate %","Escape Severity %"]
    st.dataframe(final[quick_cols],use_container_width=True,height=420,hide_index=True)

    with st.expander("🔬 Show all calculations"):
        st.dataframe(final,use_container_width=True,height=600)

    labels=[f"{i} | {r.Coin} | {r.Strategy} | {r.Decision} | Score {r.Score:.0f}" for i,r in final.iterrows()]
    selected=st.selectbox("Select candidate",labels)
    i=int(selected.split(" | ")[0]); r=final.loc[i]

    a,b,c,d=st.columns(4)
    a.metric("Decision",r["Decision"]); b.metric("Score",f'{r["Score"]:.0f}/100')
    c.metric("Test Return",f'{r["Test Return %"]:.2f}%'); d.metric("Max DD",f'{r["Max DD %"]:.2f}%')

    st.subheader("4. Decision for Selected Candidate")
    if r["Decision"]=="🟢 CONSIDER":
        st.success(f"🟢 **CONSIDER DEPLOYING** — {r['Coin']} {r['Strategy']}")
    elif r["Decision"] in ("🟡 REVIEW","🟡 WEAK"):
        st.warning(f"🟡 **DO NOT DEPLOY BLINDLY** — {r['Coin']} {r['Strategy']}")
    else:
        st.error(f"🔴 **AVOID FOR NOW** — {r['Coin']} {r['Strategy']}")
    st.write(f"**Grid:** {r['Lower']:.10g} → {r['Upper']:.10g} | **{int(r['Grids'])} intervals** | **{r['USDT/Grid']:.2f} USDT/interval**")
    st.write(f"**Test advantage:** {r['Test Alpha %']:+.2f}% versus buy-and-hold | **Cycles:** {int(r['Cycles'])} | **Skipped:** {r['Skipped %']:.1f}% | **Escape rate:** {r['Escape Rate %']:.1f}% | **Escape severity:** {r['Escape Severity %']:.2f}%")

    st.subheader("5. Risk interpretation")
    positives=[]; warnings=[]
    if r["Regime"]=="Ranging": positives.append("ADX indicates a ranging environment.")
    elif r["Regime"]=="Trending": warnings.append("ADX indicates a trending environment.")
    else: warnings.append("ADX is in the transition zone.")
    if r["Test Alpha %"]>0: positives.append("Grid outperformed buy-and-hold in the unseen test period.")
    else: warnings.append("Grid did not outperform buy-and-hold in the unseen test period.")
    if r["Max DD %"]<5: positives.append("Historical maximum drawdown was below 5%.")
    elif r["Max DD %"]>=10: warnings.append("Historical maximum drawdown reached 10% or more.")
    if r["Skipped %"]<=5: positives.append("Skipped-order rate was low.")
    elif r["Skipped %"]>15: warnings.append("Skipped-order rate was high.")
    if r["Escape Rate %"]<=5: positives.append("Range escapes were infrequent.")
    elif r["Escape Rate %"]>15: warnings.append("The test frequently moved outside the proposed grid.")
    if r["Path Spread %"]>2: warnings.append("Results are sensitive to the unknown intrabar high/low order.")
    if r["Cycles"]<3: warnings.append("Too few completed cycles; historical evidence is weak.")
    for x in positives: st.success("✓ "+x)
    for x in warnings: st.warning("⚠ "+x)

    st.subheader("6. Manual Binance Grid Parameters")
    atr=float(r["ATR"]) if np.isfinite(r["ATR"]) else 0
    sl=max(1e-12,float(r["Lower"])-atr)
    tp=float(r["Upper"])+atr
    st.table(pd.DataFrame({
        "Parameter":["Pair","Grid Mode","Lower Price","Upper Price","Grid Intervals","Grid Levels","Approx. USDT/Interval","Reference Stop-Loss","Reference Take-Profit"],
        "Value":[r["Coin"],mode,f'{r["Lower"]:.10g}',f'{r["Upper"]:.10g}',int(r["Grids"]),int(r["Grids"])+1,f'{r["USDT/Grid"]:.2f}',f'{sl:.10g}',f'{tp:.10g}']
    }))
    st.info("Stop-loss and take-profit are reference boundaries, not validated optimal exits. Verify Binance's actual Spot Grid rules, minimums and displayed investment before placing an order.")

    st.subheader("7. Proposed Grid Levels")
    filt=st.session_state.filters.get(r["Coin"],{})
    levels=build_grid(float(r["Lower"]),float(r["Upper"]),int(r["Grids"]),mode,filt.get("tickSize",0))
    st.dataframe(pd.DataFrame({"Level":range(1,len(levels)+1),"Price":[f"{x:.10g}" for x in levels]}),use_container_width=True,height=350)

    st.subheader("8. Indicators")
    st.dataframe(pd.DataFrame({
        "Indicator":["EMA 9","EMA 21","EMA 50","EMA 200","MACD Histogram","RSI 14","ADX 14","ATR 14","Bollinger Width %","Volume Ratio","Regime"],
        "Value":[r["EMA9"],r["EMA21"],r["EMA50"],r["EMA200"],r["MACD Hist"],r["RSI"],r["ADX"],r["ATR"],r["BB Width %"],r["Volume Ratio"],r["Regime"]]
    }),use_container_width=True)

    st.subheader("9. Backtest Risk Metrics")
    st.dataframe(pd.DataFrame({
        "Metric":["Test Return %","Buy-and-Hold %","Test Alpha %","Max Drawdown %","Profit Factor","Trades","Completed Cycles","Skipped Orders","Skipped %","Range Escape Bars","Range Escape Rate %","Escape Severity %","Intrabar Path Spread %"],
        "Value":[r["Test Return %"],r["Buy-Hold %"],r["Test Alpha %"],r["Max DD %"],r["Profit Factor"],r["Trades"],r["Cycles"],r["Skipped"],r["Skipped %"],r["Escape Bars"],r["Escape Rate %"],r["Escape Severity %"],r["Path Spread %"]]
    }),use_container_width=True)

    st.caption("Educational/decision-support use only. Historical backtests do not predict future profitability. OHLC candles cannot reconstruct exact intrabar order.")

def render_active_audit():
    st.header("🤖 Active Running Grid Audit")
    st.caption(
        "Enter the grid you are currently running on Binance. This module does not place, "
        "cancel, or modify orders; it audits the live market against your configured range."
    )

    c1, c2 = st.columns(2)
    with c1:
        active_coin = st.text_input(
            "Running Coin Symbol", value="SOLUSDT", key="audit_coin"
        ).upper().strip()
        active_lower = st.number_input(
            "Running Lower Price", min_value=0.00000001, value=170.0,
            step=1.0, key="audit_lower"
        )
        active_upper = st.number_input(
            "Running Upper Price", min_value=0.00000002, value=200.0,
            step=1.0, key="audit_upper"
        )
        active_grids = st.number_input(
            "Grid Count", min_value=2, max_value=150, value=20,
            step=1, key="audit_grids"
        )
        active_mode = st.selectbox(
            "Grid Mode", ["Arithmetic", "Geometric"], key="audit_mode"
        )
    with c2:
        active_trailing = st.selectbox(
            "Trailing Up Enabled?", ["No", "Yes"], key="audit_trailing"
        )
        active_sl = st.number_input(
            "Configured Stop Loss (0 = none)", min_value=0.0, value=160.0,
            step=1.0, key="audit_sl"
        )
        active_tp = st.number_input(
            "Configured Take Profit (0 = none)", min_value=0.0, value=210.0,
            step=1.0, key="audit_tp"
        )
        active_capital = st.number_input(
            "Allocated Capital (USDT)", min_value=10.0, value=200.0,
            step=10.0, key="audit_capital"
        )

    if st.button("🚀 Audit Active Running Grid", type="primary"):
        with st.spinner(f"Fetching live market data for {active_coin}..."):
            df1 = fetch_klines(active_coin, "1h", 336)
            df5 = fetch_klines(active_coin, "5m", 288)
            df4 = fetch_klines(active_coin, "4h", 168)
            ticker = fetch_24h()
            ticker_row = ticker[ticker["symbol"] == active_coin]

        if df1.empty:
            st.error(
                f"Could not retrieve Binance candle data for {active_coin}. "
                "Check the symbol and make sure it is a Binance Spot USDT pair."
            )
            return

        current = float(df1["Close"].iloc[-1])
        lower = float(active_lower)
        upper = float(active_upper)

        if upper <= lower:
            st.error("Upper price must be greater than lower price.")
            return

        position_pct = ((current - lower) / (upper - lower)) * 100.0
        range_width_pct = ((upper - lower) / lower) * 100.0

        latest1 = df1.iloc[-1]
        latest5 = df5.iloc[-1] if not df5.empty else None
        latest4 = df4.iloc[-1] if not df4.empty else None

        adx1 = float(latest1["ADX"]) if pd.notna(latest1["ADX"]) else np.nan
        rsi1 = float(latest1["RSI"]) if pd.notna(latest1["RSI"]) else np.nan
        atr1 = float(latest1["ATR"]) if pd.notna(latest1["ATR"]) else np.nan

        adx5 = float(latest5["ADX"]) if latest5 is not None and pd.notna(latest5["ADX"]) else np.nan
        rsi5 = float(latest5["RSI"]) if latest5 is not None and pd.notna(latest5["RSI"]) else np.nan
        adx4 = float(latest4["ADX"]) if latest4 is not None and pd.notna(latest4["ADX"]) else np.nan
        rsi4 = float(latest4["RSI"]) if latest4 is not None and pd.notna(latest4["RSI"]) else np.nan

        # Recent range escape audit.
        recent = df1.tail(168)
        below = recent["Low"] < lower
        above = recent["High"] > upper
        escape_bars = int((below | above).sum())
        below_bars = int(below.sum())
        above_bars = int(above.sum())

        severity_values = []
        for _, row in recent.loc[below | above].iterrows():
            down_pct = ((lower - float(row["Low"])) / lower * 100.0) if lower > 0 else 0.0
            up_pct = ((float(row["High"]) - upper) / upper * 100.0) if upper > 0 else 0.0
            # One severity value per candle, matching the ranking calculation.
            severity_values.append(max(0.0, down_pct, up_pct))
        escape_severity = float(np.mean(severity_values)) if severity_values else 0.0
        escape_rate = escape_bars / max(len(recent), 1) * 100.0

        # Approximate grid interval and capital allocation.
        grid_step_pct = 0.0
        if active_grids > 0:
            grid_step_pct = ((upper - lower) / active_grids) / lower * 100.0
        usdt_per_grid = active_capital / max(int(active_grids), 1)

        # Current boundary distances.
        distance_lower = (current - lower) / lower * 100.0
        distance_upper = (upper - current) / upper * 100.0

        if position_pct < 0:
            status = "🚨 BELOW RANGE"
            status_detail = "Price is below the configured lower boundary."
        elif position_pct > 100:
            status = "🚨 ABOVE RANGE"
            status_detail = "Price is above the configured upper boundary."
        elif position_pct <= 15 or position_pct >= 85:
            status = "🟡 RANGE EDGE"
            status_detail = "Price is close to a grid boundary."
        else:
            status = "🟢 INSIDE RANGE"
            status_detail = "Price is inside the configured grid range."

        # Multi-timeframe risk assessment.
        warnings = []
        positives = []

        if np.isfinite(adx1):
            if adx1 >= 30:
                warnings.append("1h ADX indicates a strong directional market.")
            elif adx1 < 20:
                positives.append("1h ADX is low, which is generally more compatible with ranging behavior.")

        if np.isfinite(adx4):
            if adx4 >= 30:
                warnings.append("4h ADX indicates a strong higher-timeframe trend.")
            elif adx4 < 20:
                positives.append("4h ADX is low; higher-timeframe trend pressure is relatively weak.")

        if escape_rate > 15:
            warnings.append("The configured range was breached frequently in the recent 1h history.")
        elif escape_rate <= 5:
            positives.append("Recent range breaches were infrequent.")

        if escape_severity > 3:
            warnings.append(f"Average range-escape severity was about {escape_severity:.2f}%.")

        if position_pct < 0:
            warnings.append("Current price is below the grid; downside exposure/range failure requires attention.")
        elif position_pct > 100:
            if active_trailing == "Yes":
                positives.append("Trailing Up is enabled, so an upside breakout may be handled by the bot's trailing mechanism.")
            else:
                warnings.append("Current price is above the grid and Trailing Up is disabled.")

        if active_sl > 0:
            if current <= active_sl:
                warnings.append("Current price is at or below the configured Stop Loss.")
            else:
                positives.append("Current price is above the configured Stop Loss.")
        else:
            warnings.append("No Stop Loss is configured.")

        if active_tp > 0:
            if current >= active_tp:
                positives.append("Current price has reached or exceeded the configured Take Profit.")
        else:
            warnings.append("No Take Profit is configured.")

        st.markdown("---")
        st.subheader(f"📊 Active Bot Health Report — {active_coin}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"{current:.10g}")
        m2.metric("Grid Position", f"{position_pct:.1f}%")
        m3.metric("1h Regime", str(latest1["Regime"]))
        m4.metric("1h ADX", f"{adx1:.1f}" if np.isfinite(adx1) else "N/A")

        if status == "🟢 INSIDE RANGE":
            st.success(f"{status} — {status_detail}")
        elif status == "🟡 RANGE EDGE":
            st.warning(f"{status} — {status_detail}")
        else:
            st.error(f"{status} — {status_detail}")

        st.subheader("📐 Running Grid Configuration")
        st.dataframe(pd.DataFrame({
            "Parameter": [
                "Pair", "Grid Mode", "Lower", "Upper", "Grid Count", "Approx. Grid Step %",
                "Approx. USDT/Grid", "Trailing Up", "Stop Loss", "Take Profit",
                "Allocated Capital"
            ],
            "Value": [
                active_coin, active_mode, lower, upper, int(active_grids), round(grid_step_pct, 3),
                round(usdt_per_grid, 2), active_trailing,
                active_sl if active_sl > 0 else "None",
                active_tp if active_tp > 0 else "None",
                active_capital
            ]
        }), use_container_width=True, hide_index=True)

        st.subheader("⏱️ Multi-Timeframe Live Condition")
        mt_rows = [
            ["5m", adx5, rsi5, "Short-term volatility / execution"],
            ["1h", adx1, rsi1, "Primary grid regime"],
            ["4h", adx4, rsi4, "Higher-timeframe trend risk"],
        ]
        st.dataframe(pd.DataFrame(mt_rows, columns=["Timeframe","ADX","RSI","Role"]),
                     use_container_width=True, hide_index=True)

        st.subheader("🚨 Recent Range Escape Audit")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Escape Bars (168h)", escape_bars)
        e2.metric("Below Lower", below_bars)
        e3.metric("Above Upper", above_bars)
        e4.metric("Escape Rate", f"{escape_rate:.1f}%")
        st.metric("Average Escape Severity", f"{escape_severity:.2f}%")

        if ticker_row.empty:
            st.info("24h ticker data was unavailable for this symbol.")
        else:
            tr = ticker_row.iloc[0]
            qv = float(tr["quoteVolume"])
            ch = float(tr["priceChangePercent"])
            st.subheader("📊 Current 24h Market Snapshot")
            q1, q2 = st.columns(2)
            q1.metric("24h Quote Volume", f"{qv:,.0f} USDT")
            q2.metric("24h Change", f"{ch:.2f}%")

        st.subheader("🧭 Audit Interpretation")
        for x in positives:
            st.success("✓ " + x)
        for x in warnings:
            st.warning("⚠ " + x)

        st.info(
            "This audit is a decision-support check. It does not know your bot's exact "
            "live inventory, filled-order history, unrealized P&L, or Binance's internal "
            "grid state. For those values, compare with the Binance bot screen."
        )

        st.subheader("🔢 Proposed Grid Levels")
        filt = st.session_state.filters.get(active_coin, {})
        tick = filt.get("tickSize", 0)
        levels = build_grid(lower, upper, int(active_grids), active_mode, tick)
        st.dataframe(
            pd.DataFrame({
                "Level": range(1, len(levels) + 1),
                "Price": [f"{x:.10g}" for x in levels]
            }),
            use_container_width=True,
            height=350
        )


def main():
    st.set_page_config(
        page_title="Binance Spot Grid Decision Support V8.5.1",
        layout="wide"
    )
    st.title("⚡ Binance Spot Grid Decision Support V8.5.1")
    st.info(
        f"**Data Freshness:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | "
        "Multi-timeframe decision support + active grid audit"
    )

    tab1, tab2 = st.tabs([
        "🔍 New Grid Finder & Ranking",
        "🤖 Active Running Grid Audit"
    ])

    with tab1:
        render_finder()

    with tab2:
        render_active_audit()

if __name__=="__main__":
    main()
