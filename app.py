import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import math

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(page_title="Binance Spot Grid Assistant V3.5", layout="wide")

EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
TICKER_24H_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

STABLECOIN_BLACKLIST = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", "USDT", "PAX", "USD1", "RLUSD", "PYUSD", "USDE", "USDS", "USDD", "GUSD", "LUSD", "FRAX", "USDJ", "USDB", "DEUSD", "SUSD", "EUSD", "CUSD", "EURS", "TRY", "BRL", "BIDR", "U"}

# ============================================================
# SESSION STATE
# ============================================================
if "candidates" not in st.session_state:
    st.session_state["candidates"] = []
if "market_data" not in st.session_state:
    st.session_state["market_data"] = pd.DataFrame()
if "filters" not in st.session_state:
    st.session_state["filters"] = {}

@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "SpotGridAssistant/3.5"})
    return session

@st.cache_data(ttl=600)
def fetch_spot_universe():
    session = get_http_session()
    try:
        response = session.get(EXCHANGE_INFO_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        symbols = []
        filters = {}
        for s in data.get("symbols", []):
            base_asset = s.get("baseAsset", "").upper()
            quote_asset = s.get("quoteAsset", "").upper()
            is_stablecoin = base_asset in STABLECOIN_BLACKLIST or "USD" in base_asset or "EUR" in base_asset
            if s.get("status") == "TRADING" and quote_asset == "USDT" and s.get("isSpotTradingAllowed", True) and not is_stablecoin:
                symbols.append(s["symbol"])
                sym_filters = {"tickSize": None, "stepSize": None, "minQty": None, "minNotional": None}
                for f in s.get("filters", []):
                    ftype = f.get("filterType")
                    if ftype == "PRICE_FILTER":
                        sym_filters["tickSize"] = float(f.get("tickSize", 0))
                    elif ftype == "LOT_SIZE":
                        sym_filters["stepSize"] = float(f.get("stepSize", 0))
                        sym_filters["minQty"] = float(f.get("minQty", 0))
                    elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                        sym_filters["minNotional"] = float(f.get("minNotional", 0))
                filters[s["symbol"]] = sym_filters
        return symbols, filters
    except Exception:
        return [], {}

@st.cache_data(ttl=300)
def fetch_24h_data():
    session = get_http_session()
    try:
        response = session.get(TICKER_24H_URL, timeout=15)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        numeric_cols = ["lastPrice", "highPrice", "lowPrice", "volume", "quoteVolume", "priceChangePercent"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["symbol", "lastPrice", "priceChangePercent", "quoteVolume"]]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_klines(symbol, interval="1h", limit=336):
    session = get_http_session()
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = session.get(KLINES_URL, params=params, timeout=15)
        if response.status_code == 200:
            columns = ["OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume", "Trades", "TBB", "TBQ", "Ignore"]
            df = pd.DataFrame(response.json(), columns=columns)
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = calculate_indicators(df)
            return df
    except Exception:
        pass
    return pd.DataFrame()

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def calculate_indicators(df, period=14):
    if len(df) < period + 1:
        df['ATR'] = 0.0
        df['RSI'] = 50.0
        df['ADX'] = 0.0
        df['Regime'] = "Unknown"
        return df

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period).mean().bfill()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, float('nan'))
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(50.0)

    up = df['High'].diff()
    down = -df['Low'].diff()
    pos_dm = [u if (pd.notna(u) and pd.notna(d) and u > d and u > 0) else 0.0 for u, d in zip(up, down)]
    neg_dm = [d if (pd.notna(u) and pd.notna(d) and d > u and d > 0) else 0.0 for u, d in zip(up, down)]
    
    tr_sum = tr.rolling(period).sum()
    pos_di = 100 * (pd.Series(pos_dm, index=df.index).rolling(period).sum() / tr_sum.replace(0, float('nan')))
    neg_di = 100 * (pd.Series(neg_dm, index=df.index).rolling(period).sum() / tr_sum.replace(0, float('nan')))
    
    di_diff = (pos_di - neg_di).abs()
    di_sum = pos_di + neg_di
    dx = 100 * (di_diff / di_sum.replace(0, float('nan')))
    df['ADX'] = dx.rolling(period).mean().fillna(0.0)
    df['Regime'] = df['ADX'].apply(lambda x: "Ranging" if x < 25 else "Trending")
    return df

def calculate_range_atr(df, atr_multiplier=2.0):
    high = float(df["High"].max())
    low = float(df["Low"].min())
    latest_atr = float(df["ATR"].iloc[-1]) if "ATR" in df and not df["ATR"].empty else 0.0
    upper = high + (latest_atr * atr_multiplier)
    lower = max(0.0001, low - (latest_atr * atr_multiplier))
    return lower, upper

# ============================================================
# BACKTEST ENGINE
# ============================================================
def round_to_tick(price, tick_size):
    if not tick_size or tick_size <= 0: return price
    return math.floor(price / tick_size) * tick_size

def round_quantity(quantity, step_size):
    if not step_size or step_size <= 0: return quantity
    return math.floor(quantity / step_size) * step_size

def build_grid(lower, upper, grid_count, grid_type):
    if grid_type == "Arithmetic" or lower <= 0 or upper <= 0:
        step = (upper - lower) / grid_count
        return [lower + (step * i) for i in range(grid_count + 1)]
    ratio = (upper / lower) ** (1 / grid_count)
    return [lower * (ratio ** i) for i in range(grid_count + 1)]

def backtest_grid(df, lower, upper, grid_count, capital, fee_pct, grid_type, filters):
    if df.empty: return None
    current_price = float(df["Close"].iloc[-1])
    if current_price <= 0: return None

    grid = build_grid(lower, upper, grid_count, grid_type)
    tick_size = filters.get("tickSize")
    if tick_size:
        grid = sorted(list(set([round_to_tick(p, tick_size) for p in grid])))
    if len(grid) < 2: return None

    quote_balance = capital * 0.50
    base_balance = (capital * 0.50) / current_price
    starting_equity = quote_balance + (base_balance * current_price)
    order_quote = capital / max(grid_count, 1)

    step_size = filters.get("stepSize")
    min_qty = filters.get("minQty")
    min_notional = filters.get("minNotional")

    crosses = 0
    realized_profit = 0.0
    peak_equity = starting_equity
    max_drawdown = 0.0
    inventory = {}

    for _, row in df.iterrows():
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        path = [o, l, h, c] if c >= o else [o, h, l, c]
        for start_price, end_price in zip(path[:-1], path[1:]):
            
            # --- PRICE MOVING UP ---
            if end_price > start_price:
                crossed_levels = [i for i in range(1, len(grid)) if start_price < grid[i] <= end_price]
                for lvl in crossed_levels:
                    price = grid[lvl]
                    buy_lvl = lvl - 1  # THE FIX: Sell the inventory acquired at the grid level BELOW this one
                    if buy_lvl in inventory:
                        qty = inventory[buy_lvl]["quantity"]
                        notional = qty * price
                        if min_notional and notional < min_notional: continue
                        fee = notional * fee_pct / 100
                        quote_balance += (notional - fee)
                        buy_cost = inventory.pop(buy_lvl)["cost"]
                        realized_profit += (notional - fee - buy_cost)
                        crosses += 1
                        
            # --- PRICE MOVING DOWN ---
            elif end_price < start_price:
                crossed_levels = [i for i in range(len(grid) - 1) if end_price <= grid[i] < start_price]
                for lvl in reversed(crossed_levels):
                    price = grid[lvl]
                    qty = round_quantity(order_quote / price, step_size)
                    if min_qty and qty < min_qty: continue
                    notional = qty * price
                    if min_notional and notional < min_notional: continue
                    fee = notional * fee_pct / 100
                    total_cost = notional + fee
                    if total_cost > quote_balance: continue
                    quote_balance -= total_cost
                    base_balance += qty
                    inventory[lvl] = {"quantity": qty, "cost": total_cost}
                    crosses += 1

        equity = quote_balance + (base_balance * c)
        if equity > peak_equity: peak_equity = equity
        dd = ((peak_equity - equity) / peak_equity) * 100
        if dd > max_drawdown: max_drawdown = dd

    realized_roi = (realized_profit / starting_equity) * 100 if starting_equity > 0 else 0
    grid_spacing = (grid[-1] - grid[0]) / (len(grid) - 1) if len(grid) > 1 else 0

    return {
        "Realized ROI %": realized_roi,
        "Trades": crosses,
        "Max Drawdown %": max_drawdown,
        "Grid Spacing": grid_spacing
    }

def main():
    st.title("⚡ Binance Spot Grid Assistant V3.5")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Data Freshness:** {current_time} UTC | Engine: Operational")

    st.markdown("### 1. Strategy Parameters")
    with st.expander("⚙️ Tap to Edit Trading Rules & Indicator Settings", expanded=False):
        wallet_balance = st.number_input("Available Capital (USDT)", min_value=0.0, value=1000.0, step=50.0)
        fee_choice = st.selectbox("Trading Fee Tier (Round-Trip)", ["Standard (0.20%)", "BNB Discount (0.15%)", "Zero Fee (0.00%)"])
        fee_pct = 0.20 if "0.20" in fee_choice else (0.15 if "0.15" in fee_choice else 0.00)
        grid_type = st.selectbox("Grid Type", ["Arithmetic", "Geometric"])
        only_ranging = st.checkbox("Filter Out Trending Coins (Only keep ADX < 25)", value=False)
        atr_mult = st.slider("ATR Volatility Range Multiplier", 0.5, 4.0, 1.5, step=0.1)
        st.markdown("**Target Profit per Grid Step (%)**")
        col1, col2, col3 = st.columns(3)
        with col1: tight_pct = st.number_input("Tight (3D)", min_value=0.1, value=0.6, step=0.1)
        with col2: mod_pct = st.number_input("Mod (7D)", min_value=0.1, value=1.0, step=0.1)
        with col3: wide_pct = st.number_input("Wide (14D)", min_value=0.1, value=1.5, step=0.1)

    st.markdown("### 2. Market Universe")
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe..."):
            symbols, filters = fetch_spot_universe()
            market_data = fetch_24h_data()
            if not market_data.empty and symbols:
                valid_markets = market_data[market_data['symbol'].isin(symbols)]
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(5)
                st.success(f"Success: {len(valid_markets)} USDT pairs validated.")
                st.dataframe(top_candidates[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']], use_container_width=True)
                st.session_state['candidates'] = top_candidates['symbol'].tolist()
                st.session_state['filters'] = filters

    if st.session_state.get('candidates'):
        st.markdown("### 3. Backtest Engine (Phases 3-9)")
        if st.button("Run Indicator-Guided Backtest"):
            results = []
            progress_bar = st.progress(0)
            candidates = st.session_state['candidates']
            filters_dict = st.session_state.get('filters', {})
            
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=336)
                if not df_klines.empty:
                    latest_adx = round(df_klines['ADX'].iloc[-1], 1)
                    latest_rsi = round(df_klines['RSI'].iloc[-1], 1)
                    regime = df_klines['Regime'].iloc[-1]
                    
                    if only_ranging and "Trending" in regime:
                        progress_bar.progress((i + 1) / len(candidates))
                        continue

                    strategies = [
                        {"Type": "Tight (3D)", "df": df_klines.tail(72).copy(), "Target_Pct": tight_pct},
                        {"Type": "Mod (7D)", "df": df_klines.tail(168).copy(), "Target_Pct": mod_pct},
                        {"Type": "Wide (14D)", "df": df_klines.tail(336).copy(), "Target_Pct": wide_pct}
                    ]
                    
                    for strat in strategies:
                        lower, upper = calculate_range_atr(strat["df"], atr_multiplier=atr_mult)
                        if lower > 0:
                            ideal_grids = int((((upper - lower) / lower) * 100) / strat["Target_Pct"])
                        else:
                            ideal_grids = 5
                            
                        suggested_grids = max(5, min(ideal_grids, 150))
                        
                        bt_result = backtest_grid(
                            df=strat["df"], lower=lower, upper=upper,
                            grid_count=suggested_grids, capital=wallet_balance,
                            fee_pct=fee_pct, grid_type=grid_type,
                            filters=filters_dict.get(coin, {})
                        )
                        
                        if bt_result:
                            step_pct = (bt_result["Grid Spacing"] / lower) * 100 if lower > 0 else 0
                            results.append({
                                "Coin": coin, "Strategy": strat["Type"], "Regime": regime,
                                "ADX": latest_adx, "RSI": latest_rsi,
                                "Lower": round(lower, 4), "Upper": round(upper, 4),
                                "Grids": suggested_grids, "Step %": round(step_pct, 2),
                                "Crosses": bt_result["Trades"],
                                "Max DD %": round(bt_result["Max Drawdown %"], 2),
                                "ROI %": round(bt_result["Realized ROI %"], 2)
                            })
                progress_bar.progress((i + 1) / len(candidates))
                
            st.markdown("### 🏆 V3.5 Final Indicator Ranking")
            if results:
                final_df = pd.DataFrame(results).sort_values(by="ROI %", ascending=False).reset_index(drop=True)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No candidates matched your criteria.")

if __name__ == "__main__":
    main()
                        
