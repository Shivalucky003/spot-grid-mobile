import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import math
from google import genai
from google.genai import types

# ============================================================
# BINANCE SPOT GRID DECISION SUPPORT - V7.0 (DUAL MODE + NEWS)
# ============================================================
st.set_page_config(
    page_title="Binance Grid Master V7.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
TICKER_24H_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

STABLECOIN_BLACKLIST = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP",
    "EUR", "AEUR", "USDT", "PAX", "USD1", "RLUSD",
    "PYUSD", "USDE", "USDS", "USDD", "GUSD", "LUSD",
    "FRAX", "USDJ", "USDB", "DEUSD", "SUSD", "EUSD",
    "CUSD", "EURS", "TRY", "BRL", "BIDR", "U"
}

INTERVAL = "1h"
KLINE_LIMIT = 1000

# ============================================================
# SESSION STATE & HTTP SESSION
# ============================================================
defaults = {
    "candidates": [],
    "filters": {},
    "market_data": pd.DataFrame(),
    "scan_done": False,
    "results": pd.DataFrame(),
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "BinanceGridMaster/7.0"})
    return session

# ============================================================
# PRECISION & FORMATTING HELPERS
# ============================================================
def get_decimals(step):
    if not step or step <= 0:
        return 8
    step_str = f"{step:.10f}".rstrip('0')
    if '.' in step_str:
        return len(step_str.split('.')[1])
    return 0

def format_precision(value, step):
    decimals = get_decimals(step)
    if decimals == 0:
        return f"{int(value)}"
    return f"{value:.{decimals}f}"

def floor_to_step(value, step):
    if not step or step <= 0:
        return float(value)
    return math.floor(value / step + 1e-12) * step

# ============================================================
# BINANCE DATA FETCHING
# ============================================================
@st.cache_data(ttl=600)
def fetch_spot_universe():
    try:
        response = get_http_session().get(EXCHANGE_INFO_URL, timeout=20)
        response.raise_for_status()
        data = response.json()
        symbols = []
        filters = {}

        for item in data.get("symbols", []):
            base = str(item.get("baseAsset", "")).upper()
            quote = str(item.get("quoteAsset", "")).upper()
            is_stablecoin = base in STABLECOIN_BLACKLIST or "USD" in base or "EUR" in base

            if (item.get("status") == "TRADING" and quote == "USDT" 
                and item.get("isSpotTradingAllowed", True) and not is_stablecoin):
                symbol = item["symbol"]
                symbols.append(symbol)

                f = {"tickSize": 0.0, "stepSize": 0.0, "minQty": 0.0, "minNotional": 10.0}
                for rule in item.get("filters", []):
                    ft = rule.get("filterType")
                    if ft == "PRICE_FILTER":
                        f["tickSize"] = float(rule.get("tickSize", 0))
                    elif ft == "LOT_SIZE":
                        f["stepSize"] = float(rule.get("stepSize", 0))
                        f["minQty"] = float(rule.get("minQty", 0))
                    elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                        f["minNotional"] = float(rule.get("minNotional", 10.0))
                filters[symbol] = f
        return symbols, filters
    except Exception:
        return [], {}

@st.cache_data(ttl=300)
def fetch_24h_data():
    try:
        response = get_http_session().get(TICKER_24H_URL, timeout=20)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        needed = ["symbol", "lastPrice", "priceChangePercent", "quoteVolume"]
        if any(col not in df.columns for col in needed):
            return pd.DataFrame()
        for col in needed[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[needed].dropna().copy()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_klines(symbol, interval="1h", limit=1000):
    try:
        response = get_http_session().get(
            KLINES_URL, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20
        )
        if response.status_code != 200:
            return pd.DataFrame()
        rows = response.json()
        if not rows:
            return pd.DataFrame()
        columns = ["OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume", "Trades", "TBB", "TBQ", "Ignore"]
        df = pd.DataFrame(rows, columns=columns)
        for col in ["Open", "High", "Low", "Close", "Volume", "QuoteVolume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
        return calculate_indicators(df.dropna(subset=["Open", "High", "Low", "Close"]))
    except Exception:
        return pd.DataFrame()

# ============================================================
# GEMINI NEWS FINDER & PROFIT ADVISOR
# ============================================================
@st.cache_data(ttl=1800)
def fetch_gemini_news_and_advice(api_key, top_coins):
    if not api_key:
        return "No API key provided.", "NEUTRAL", "Provide a Gemini API key to enable live news & AI profit strategies."
    try:
        client = genai.Client(api_key=api_key)
        coins_str = ", ".join(top_coins[:5])
        prompt = (
            f"Perform a live web search for recent crypto news affecting the market and specifically these coins: {coins_str}.\n"
            f"Provide your output in 3 sections:\n"
            f"1. **NEWS SUMMARY**: List 2-3 top news stories from the past 24-48 hours.\n"
            f"2. **MACRO SENTIMENT**: State one word: BULLISH, BEARISH, or NEUTRAL.\n"
            f"3. **PROFIT STRATEGY**: Provide specific advice on how a Spot Grid trader should adjust their lower/upper bounds or grid spacing to maximize profit based on this news."
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2
            )
        )
        text = response.text
        sentiment = "NEUTRAL"
        if "BULLISH" in text.upper(): sentiment = "BULLISH"
        elif "BEARISH" in text.upper(): sentiment = "BEARISH"
        return text, sentiment
    except Exception as e:
        return f"Error fetching news: {e}", "NEUTRAL"

@st.cache_data(ttl=900)
def audit_running_bot_with_gemini(api_key, coin, lower, upper, grids, trailing, sl, tp, current_price, regime, adx, rsi):
    if not api_key:
        return "Paste your Gemini API key in settings to unlock AI active bot recommendations."
    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"You are an expert crypto grid trading risk auditor. Audit this active Binance Spot Grid bot:\n"
            f"- Pair: {coin}\n"
            f"- Current Market Price: {current_price}\n"
            f"- Grid Lower Bound: {lower} | Upper Bound: {upper}\n"
            f"- Grid Count: {grids}\n"
            f"- Trailing Up Enabled: {trailing}\n"
            f"- Stop Loss: {sl if sl > 0 else 'None'} | Take Profit: {tp if tp > 0 else 'None'}\n"
            f"- Technical Indicators: Market Regime = {regime}, ADX = {adx}, RSI = {rsi}\n\n"
            f"Search for recent breaking news on {coin}. Then give clear recommendations:\n"
            f"1. **HEALTH RATING**: Excellent / Caution / High Risk / Dangerous\n"
            f"2. **ACTION**: KEEP RUNNING / ADJUST BOUNDS / ACTIVATE STOP LOSS / CLOSE BOT\n"
            f"3. **REASONING**: Explain why based on current price level, range bounds, indicators, and live news."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2
            )
        )
        return response.text
    except Exception as e:
        return f"Could not complete AI audit: {e}"

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def calculate_indicators(df):
    df = df.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["TR"] = tr
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = (100 - (100 / (1 + rs))).fillna(50)

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_di = df["ATR14"].replace(0, np.nan)

    df["PlusDI"] = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_di
    df["MinusDI"] = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_di
    di_sum = (df["PlusDI"] + df["MinusDI"]).replace(0, np.nan)
    df["ADX14"] = (100 * (df["PlusDI"] - df["MinusDI"]).abs() / di_sum).ewm(alpha=1/14, adjust=False).mean().fillna(0)

    df["Regime"] = np.select(
        [df["ADX14"] < 20, (df["ADX14"] >= 20) & (df["ADX14"] < 25), df["ADX14"] >= 25],
        ["Ranging 🟢", "Transition 🟡", "Trending 🔴"],
        default="Unknown"
    )
    return df

# ============================================================
# GRID RANGE & BACKTEST ENGINE
# ============================================================
def calculate_grid_range(train, atr_multiplier, sentiment="NEUTRAL"):
    high = float(train["High"].max())
    low = float(train["Low"].min())
    atr = float(train["ATR14"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        atr = max(low * 0.005, 1e-12)

    if sentiment == "BULLISH":
        high += (atr * 0.5)
        low += (atr * 0.5)
    elif sentiment == "BEARISH":
        high -= (atr * 0.5)
        low -= (atr * 0.5)

    upper = high + (atr * atr_multiplier)
    lower = max(1e-12, low - (atr * atr_multiplier))
    return lower, upper, atr

def backtest_grid(df, lower, upper, grid_count, capital, fee_pct, grid_type, exchange_filter):
    if df.empty or capital <= 0: return None

    tick_size = exchange_filter.get("tickSize", 0)
    step_size = exchange_filter.get("stepSize", 0)
    min_notional = exchange_filter.get("minNotional", 10.0)

    if grid_type == "Arithmetic":
        step = (upper - lower) / grid_count
        raw = [lower + step * i for i in range(grid_count + 1)]
    else:
        ratio = (upper / lower) ** (1 / grid_count)
        raw = [lower * (ratio ** i) for i in range(grid_count + 1)]

    if tick_size > 0: raw = [floor_to_step(p, tick_size) for p in raw]
    grid = sorted(set(round(p, 10) for p in raw if p > 0))
    if len(grid) < 2: return None

    first_price = float(df["Close"].iloc[0])
    quote_balance = capital * 0.50
    base_balance = (capital * 0.50) / first_price
    initial_equity = capital

    order_quote = capital / max(len(grid) - 1, 1)
    realized_profit = 0.0
    completed_cycles = 0
    peak_equity = capital
    max_drawdown = 0.0
    inventory = {}

    for _, row in df.iterrows():
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        path = [o, l, h, c] if c >= o else [o, h, l, c]

        for start, end in zip(path[:-1], path[1:]):
            if end > start:  # SELL
                crossed = [i for i in range(1, len(grid)) if start < grid[i] <= end]
                for i in crossed:
                    price = grid[i]
                    buy_lvl = i - 1
                    if buy_lvl in inventory:
                        qty = inventory[buy_lvl]["qty"]
                        notional = qty * price
                        if notional < min_notional: continue
                        fee = notional * fee_pct / 100
                        quote_balance += (notional - fee)
                        buy_cost = inventory.pop(buy_lvl)["cost"]
                        realized_profit += (notional - fee - buy_cost)
                        completed_cycles += 1
                        base_balance -= qty

            elif end < start:  # BUY
                crossed = [i for i in range(len(grid) - 1) if end <= grid[i] < start]
                for i in reversed(crossed):
                    price = grid[i]
                    qty = floor_to_step(order_quote / price, step_size)
                    notional = qty * price
                    if notional < min_notional: continue
                    fee = notional * fee_pct / 100
                    total_cost = notional + fee
                    if total_cost > quote_balance: continue
                    quote_balance -= total_cost
                    base_balance += qty
                    inventory[i] = {"qty": qty, "cost": total_cost}

        equity = quote_balance + (base_balance * c)
        peak_equity = max(peak_equity, equity)
        drawdown = ((peak_equity - equity) / peak_equity) * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

    final_price = float(df["Close"].iloc[-1])
    final_equity = quote_balance + (base_balance * final_price)
    total_return = ((final_equity / initial_equity) - 1) * 100
    grid_step_pct = ((grid[1] - grid[0]) / grid[0]) * 100 if len(grid) > 1 else 0

    return {
        "Total Return %": total_return,
        "Realized Profit": realized_profit,
        "Max Drawdown %": max_drawdown,
        "Completed Cycles": completed_cycles,
        "Grid Step %": grid_step_pct,
        "Grid": grid,
    }

# ============================================================
# EVALUATE NEW BOT CANDIDATES
# ============================================================
def evaluate_coin(symbol, df, wallet, fee_pct, grid_type, atr_multiplier, targets, exchange_filter, train_ratio, sentiment):
    if len(df) < 240: return []
    results = []
    tick_size = exchange_filter.get("tickSize", 0.0001)
    step_size = exchange_filter.get("stepSize", 0.001)
    min_notional = max(exchange_filter.get("minNotional", 10.0), 10.0)

    windows = [
        ("Tight (3D)", 72, targets["tight"]),
        ("Moderate (7D)", 168, targets["moderate"]),
        ("Wide (14D)", 336, targets["wide"]),
    ]

    for strategy, window, target in windows:
        data = df.tail(min(window, len(df))).copy()
        if len(data) < 72: continue

        split = max(48, min(int(len(data) * train_ratio), len(data) - 24))
        train = data.iloc[:split].copy()
        test = data.iloc[split:].copy()
        if len(test) < 24: continue

        lower, upper, atr = calculate_grid_range(train, atr_multiplier, sentiment)
        if lower <= 0 or upper <= lower: continue

        range_pct = ((upper - lower) / lower) * 100
        grid_count = max(5, min(int(range_pct / target), 100))

        min_capital_required = grid_count * (min_notional + 0.5)
        is_capital_feasible = wallet >= min_capital_required

        backtest = backtest_grid(test, lower, upper, grid_count, wallet, fee_pct, grid_type, exchange_filter)
        if not backtest: continue

        current_price = float(data["Close"].iloc[-1])
        levels_above = max(0, sum(1 for p in backtest["Grid"] if p > current_price))
        base_ratio = levels_above / max(len(backtest["Grid"]) - 1, 1)
        
        initial_base_usdt = wallet * base_ratio
        initial_base_qty = floor_to_step(initial_base_usdt / current_price, step_size) if current_price > 0 else 0
        initial_quote_usdt = wallet - (initial_base_qty * current_price)

        stop_loss = max(0.0, lower - (1.0 * atr))
        take_profit = upper + (1.0 * atr)
        adx = float(data["ADX14"].iloc[-1])
        rsi = float(data["RSI14"].iloc[-1])
        regime = str(data["Regime"].iloc[-1])

        score = 50.0
        if adx < 20: score += 20
        elif adx < 25: score += 10
        else: score -= 10

        if backtest["Total Return %"] > 2.0: score += 15
        elif backtest["Total Return %"] > 0: score += 5
        else: score -= 15

        if backtest["Max Drawdown %"] < 5.0: score += 15
        elif backtest["Max Drawdown %"] < 10.0: score += 5
        else: score -= 10

        if is_capital_feasible: score += 10
        else: score -= 20

        score = max(0.0, min(100.0, score))
        decision = "🟢 CONSIDER" if score >= 70 else ("🟡 CAUTION" if score >= 50 else "🔴 AVOID")

        results.append({
            "Coin": symbol, "Strategy": strategy, "Decision": decision, "Score": round(score, 1),
            "Regime": regime, "ADX": round(adx, 1), "RSI": round(rsi, 1),
            "Current Price Raw": current_price, "Current Price": format_precision(current_price, tick_size),
            "Lower": format_precision(lower, tick_size), "Upper": format_precision(upper, tick_size),
            "Stop Loss": format_precision(stop_loss, tick_size), "Take Profit": format_precision(take_profit, tick_size),
            "Grids": grid_count, "Grid Step %": round(backtest["Grid Step %"], 2),
            "Capital Feasible": "✅ Yes" if is_capital_feasible else f"❌ Need ~{int(min_capital_required)} USDT",
            "Initial USDT": round(initial_quote_usdt, 2), "Initial Base Coin": format_precision(initial_base_qty, step_size),
            "Return %": round(backtest["Total Return %"], 2), "Max DD %": round(backtest["Max Drawdown %"], 2),
            "Cycles": backtest["Completed Cycles"], "Tick Size": tick_size, "Step Size": step_size,
        })
    return results

# ============================================================
# MAIN APPLICATION SETUP WITH TABS
# ============================================================
def main():
    st.title("⚡ Binance Spot Grid Master V7.0")
    st.info("Data Refresh Time: " + dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    # GLOBAL SIDEBAR FOR API KEY
    st.sidebar.header("🔑 Gemini AI Integration")
    api_key = st.sidebar.text_input("Paste Gemini API Key", type="password")
    if api_key:
        st.sidebar.success("Gemini API Connected!")
    else:
        st.sidebar.warning("Paste Key for Live News Grounding & AI Active Bot Audits.")

    # TOP TAB SELECTION
    tab1, tab2 = st.tabs(["🔍 Mode 1: New Bot Finder & Rank Decision", "🤖 Mode 2: Audit Active Running Bot"])

    # =========================================================
    # TAB 1: NEW BOT FINDER
    # =========================================================
    with tab1:
        st.header("1. New Strategy Parameters")
        with st.expander("⚙️ Configure Rules & Wallet", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                wallet = st.number_input("Available Wallet (USDT)", min_value=20.0, value=160.0, step=10.0)
                fee_choice = st.selectbox("Trading Fee Tier", ["Standard 0.20%", "BNB discount 0.15%", "Zero 0.00%"])
                fee_pct = 0.20 if "0.20" in fee_choice else (0.15 if "0.15" in fee_choice else 0.0)
            with c2:
                grid_type = st.selectbox("Grid Type", ["Arithmetic", "Geometric"])
                atr_multiplier = st.slider("ATR Range Buffer", 0.5, 4.0, 1.5, 0.1)
            with c3:
                top_n = st.slider("Coins to Analyze", 5, 30, 10)
                train_ratio = st.slider("Training Fraction", 0.60, 0.85, 0.70, 0.05)

            st.markdown("---")
            st.subheader("Target Grid Step Profit (%)")
            c1, c2, c3 = st.columns(3)
            with c1: tight_target = st.number_input("Tight (3D) %", min_value=0.1, max_value=10.0, value=0.6, step=0.1)
            with c2: moderate_target = st.number_input("Moderate (7D) %", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            with c3: wide_target = st.number_input("Wide (14D) %", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

        st.header("2. Market Scan")
        if st.button("🔎 Scan Binance Markets", type="primary"):
            with st.spinner("Fetching Binance USDT Spot Pairs..."):
                symbols, filters = fetch_spot_universe()
                market = fetch_24h_data()
                if not symbols or market.empty:
                    st.error("Failed to fetch market data.")
                else:
                    valid = market[market["symbol"].isin(symbols) & (market["quoteVolume"] > 0)].copy()
                    valid = valid.sort_values("quoteVolume", ascending=False).head(top_n)
                    st.session_state.candidates = valid["symbol"].tolist()
                    st.session_state.filters = filters
                    st.session_state.market_data = valid
                    st.session_state.scan_done = True
                    st.session_state.results = pd.DataFrame()

        if st.session_state.scan_done:
            st.success(f"{len(st.session_state.candidates)} liquid USDT pairs validated.")
            
            st.header("3. Run Strategy Analysis & News Grounding")
            if st.button("🚀 Evaluate Grid Configurations", type="primary"):
                sentiment = "NEUTRAL"
                if api_key:
                    with st.spinner("Gemini is searching live news & preparing profit strategies..."):
                        news_report, sentiment = fetch_gemini_news_and_advice(api_key, st.session_state.candidates)
                        st.markdown("### 📰 Gemini Live News Finder & Profit Advice")
                        st.info(news_report)
                            
                all_results = []
                progress = st.progress(0)
                candidates = st.session_state.candidates
                filters = st.session_state.filters
                targets = {"tight": tight_target, "moderate": moderate_target, "wide": wide_target}

                for i, symbol in enumerate(candidates):
                    df = fetch_klines(symbol, INTERVAL, KLINE_LIMIT)
                    if not df.empty:
                        res = evaluate_coin(
                            symbol, df, wallet, fee_pct, grid_type, 
                            atr_multiplier, targets, filters.get(symbol, {}), train_ratio, sentiment
                        )
                        all_results.extend(res)
                    progress.progress((i + 1) / len(candidates))

                if all_results:
                    result_df = pd.DataFrame(all_results).sort_values(by=["Score", "Return %"], ascending=False).reset_index(drop=True)
                    st.session_state.results = result_df

            result_df = st.session_state.results
            if not result_df.empty:
                st.header("4. Final Candidate Ranking")
                cols_to_show = ["Coin", "Strategy", "Decision", "Score", "Regime", "Lower", "Upper", "Grids", "Grid Step %", "Capital Feasible", "Return %", "Max DD %"]
                st.dataframe(result_df[cols_to_show], use_container_width=True, hide_index=True)

                best = result_df.iloc[0]
                st.header(f"📋 Manual Binance Parameter Card — {best['Coin']}")
                st.success(f"Selected Strategy: **{best['Strategy']}** | Score: **{best['Score']}/100**")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🛠️ Input to Binance Grid Bot UI")
                    st.code(
                        f"Pair:            {best['Coin']}\n"
                        f"Grid Mode:       {grid_type}\n"
                        f"Lower Price:     {best['Lower']}\n"
                        f"Upper Price:     {best['Upper']}\n"
                        f"Grids Count:     {best['Grids']}\n"
                        f"Stop Loss:       {best['Stop Loss']}\n"
                        f"Take Profit:     {best['Take Profit']}",
                        language="yaml"
                    )
                with col2:
                    st.markdown("### 💰 Required Wallet Preparation")
                    st.code(
                        f"Total Investment:   {wallet} USDT\n"
                        f"Feasibility Status: {best['Capital Feasible']}\n"
                        f"-----------------------------------------\n"
                        f"Keep in USDT:       {best['Initial USDT']} USDT\n"
                        f"Pre-Buy Base Coin:  {best['Initial Base Coin']} {best['Coin'].replace('USDT', '')}\n"
                        f"(Or let Binance auto-buy base coin on bot startup)",
                        language="yaml"
                    )

    # =========================================================
    # TAB 2: ACTIVE RUNNING BOT AUDITOR
    # =========================================================
    with tab2:
        st.header("🤖 Active Running Bot Auditor")
        st.caption("Input your currently active Binance grid parameters to perform a real-time risk, range, and news health check.")

        symbols, filters = fetch_spot_universe()
        default_coin = symbols[0] if symbols else "SOLUSDT"

        col1, col2 = st.columns(2)
        with col1:
            active_coin = st.text_input("Running Coin Symbol (e.g. BTCUSDT, SOLUSDT)", value="SOLUSDT").upper().strip()
            active_lower = st.number_input("Running Lower Price", min_value=0.00000001, value=170.0, step=1.0)
            active_upper = st.number_input("Running Upper Price", min_value=0.00000002, value=200.0, step=1.0)
            active_grids = st.number_input("Grid Count", min_value=2, max_value=150, value=20, step=1)

        with col2:
            active_trailing = st.selectbox("Is Trailing Up Enabled?", ["No", "Yes"])
            active_sl = st.number_input("Configured Stop Loss (0 if none)", min_value=0.0, value=160.0, step=1.0)
            active_tp = st.number_input("Configured Take Profit (0 if none)", min_value=0.0, value=210.0, step=1.0)
            active_capital = st.number_input("Allocated Capital (USDT)", min_value=10.0, value=200.0, step=10.0)

        if st.button("🚀 Audit Active Running Bot", type="primary"):
            with st.spinner(f"Fetching real-time market data & klines for {active_coin}..."):
                df_active = fetch_klines(active_coin, INTERVAL, KLINE_LIMIT)

            if df_active.empty:
                st.error(f"Could not retrieve price data for symbol '{active_coin}'. Please verify the coin symbol.")
            else:
                current_p = float(df_active["Close"].iloc[-1])
                adx_v = round(float(df_active["ADX14"].iloc[-1]), 1)
                rsi_v = round(float(df_active["RSI14"].iloc[-1]), 1)
                regime_v = str(df_active["Regime"].iloc[-1])

                # Range Position Calculation
                if active_upper > active_lower:
                    pos_pct = ((current_p - active_lower) / (active_upper - active_lower)) * 100
                else:
                    pos_pct = 50.0

                st.markdown("---")
                st.subheader(f"📊 Active Bot Health Report — {active_coin}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Market Price", f"{current_p:.4f}")
                c2.metric("Position in Grid Range", f"{pos_pct:.1f}%")
                c3.metric("Market Regime", regime_v)
                c4.metric("ADX Trend Strength", adx_v)

                # Diagnostic Banner
                if pos_pct < 0:
                    st.error("🚨 DANGER: Price dropped BELOW your Lower Bound! You are holding 100% base coin and taking full downside loss.")
                elif pos_pct > 100:
                    if active_trailing == "Yes":
                        st.success("📈 TRAILING ACTIVE: Price exceeded Upper Bound, but Trailing Up is shifting your grid higher.")
                    else:
                        st.warning("⚠️ OUT OF RANGE: Price exceeded Upper Bound! You are holding 100% USDT and missing further profits.")
                elif 20 <= pos_pct <= 80:
                    st.success("🟢 HEALTHY ZONE: Price is comfortably within your grid range. Capturing continuous grid steps.")
                else:
                    st.warning("🟡 EDGE WARNING: Price is approaching a grid boundary. Prepare for potential range breakout.")

                # Gemini Active Bot Audit Recommendation
                st.markdown("### 🤖 Gemini AI Active Bot Recommendation")
                if api_key:
                    with st.spinner("Gemini AI is analyzing live news & technicals for your active bot..."):
                        ai_audit = audit_running_bot_with_gemini(
                            api_key, active_coin, active_lower, active_upper, 
                            active_grids, active_trailing, active_sl, active_tp, 
                            current_p, regime_v, adx_v, rsi_v
                        )
                        st.markdown(ai_audit)
                else:
                    st.info("Paste your Gemini API key in the left sidebar to unlock live news auditing and AI action recommendations for this running bot.")

if __name__ == "__main__":
    main()