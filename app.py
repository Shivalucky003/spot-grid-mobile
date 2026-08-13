import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import math

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Binance Spot Grid Assistant V3.5",
    layout="wide"
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

# ============================================================
# SESSION STATE
# ============================================================

if "candidates" not in st.session_state:
    st.session_state["candidates"] = []

if "market_data" not in st.session_state:
    st.session_state["market_data"] = pd.DataFrame()

if "results" not in st.session_state:
    st.session_state["results"] = pd.DataFrame()

# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SpotGridAssistant/3.5"
    })
    return session

# ============================================================
# BINANCE EXCHANGE INFO & UNIVERSE
# ============================================================

@st.cache_data(ttl=600)
def fetch_exchange_info():
    session = get_http_session()
    try:
        response = session.get(EXCHANGE_INFO_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}

@st.cache_data(ttl=600)
def fetch_spot_universe():
    data = fetch_exchange_info()
    symbols = []
    filters = {}

    if not data:
        return [], {}

    for s in data.get("symbols", []):
        symbol = s.get("symbol", "")
        base_asset = s.get("baseAsset", "").upper()
        quote_asset = s.get("quoteAsset", "").upper()

        is_stablecoin = (
            base_asset in STABLECOIN_BLACKLIST
            or "USD" in base_asset
            or "EUR" in base_asset
        )

        if (
            s.get("status") == "TRADING"
            and quote_asset == "USDT"
            and s.get("isSpotTradingAllowed", True)
            and not is_stablecoin
        ):
            symbols.append(symbol)
            symbol_filters = {
                "tickSize": None,
                "stepSize": None,
                "minQty": None,
                "minNotional": None,
                "maxQty": None
            }

            for f in s.get("filters", []):
                filter_type = f.get("filterType")

                if filter_type == "PRICE_FILTER":
                    symbol_filters["tickSize"] = float(f.get("tickSize", 0))
                elif filter_type == "LOT_SIZE":
                    symbol_filters["stepSize"] = float(f.get("stepSize", 0))
                    symbol_filters["minQty"] = float(f.get("minQty", 0))
                    symbol_filters["maxQty"] = float(f.get("maxQty", 0))
                elif filter_type in ("MIN_NOTIONAL", "NOTIONAL"):
                    symbol_filters["minNotional"] = float(f.get("minNotional", 0))

            filters[symbol] = symbol_filters

    return symbols, filters

# ============================================================
# 24H MARKET DATA
# ============================================================

@st.cache_data(ttl=300)
def fetch_24h_data():
    session = get_http_session()
    try:
        response = session.get(TICKER_24H_URL, timeout=15)
        response.raise_for_status()
        df = pd.DataFrame(response.json())

        if df.empty:
            return pd.DataFrame()

        numeric_cols = [
            "lastPrice", "highPrice", "lowPrice", 
            "volume", "quoteVolume", "priceChangePercent"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["symbol", "lastPrice", "priceChangePercent", "quoteVolume"]]
    except Exception:
        return pd.DataFrame()

# ============================================================
# KLINES & TECHNICAL INDICATORS (PHASE 4)
# ============================================================

@st.cache_data(ttl=300)
def fetch_klines(symbol, interval="1h", limit=336):
    session = get_http_session()
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        response = session.get(KLINES_URL, params=params, timeout=15)
        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()
        if not data:
            return pd.DataFrame()

        columns = [
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteVolume", "Trades", "TBB", "TBQ", "Ignore"
        ]

        df = pd.DataFrame(data, columns=columns)

        for col in ["Open", "High", "Low", "Close", "Volume", "QuoteVolume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
        
        # --- CALCULATE INDICATORS ---
        df = calculate_indicators(df)
        
        return df
    except Exception:
        return pd.DataFrame()

def calculate_indicators(df, period=14):
    """Calculates ATR, RSI, and ADX natively using Pandas/Numpy."""
    if len(df) < period + 1:
        df['ATR'] = 0.0
        df['RSI'] = 50.0
        df['ADX'] = 0.0
        df['Regime'] = "Unknown"
        return df

    # 1. True Range (TR) & ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period).mean().bfill()

    # 2. Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(50.0)

    # 3. Average Directional Index (ADX)
    up = df['High'].diff()
    down = -df['Low'].diff()
    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr_sum = tr.rolling(period).sum()
    pos_di = 100 * (pd.Series(pos_dm, index=df.index).rolling(period).sum() / tr_sum.replace(0, np.nan))
    neg_di = 100 * (pd.Series(neg_dm, index=df.index).rolling(period).sum() / tr_sum.replace(0, np.nan))
    
    di_diff = (pos_di - neg_di).abs()
    di_sum = pos_di + neg_di
    dx = 100 * (di_diff / di_sum.replace(0, np.nan))
    df['ADX'] = dx.rolling(period).mean().fillna(0.0)

    # 4. Regime Classification (ADX < 25 indicates a Ranging/Sideways Market)
    df['Regime'] = np.where(df['ADX'] < 25, "Ranging 🟢", "Trending 🔴")

    return df

# ============================================================
# PRICE ROUNDING & GRID BUILDER
# ============================================================

def round_to_tick(price, tick_size):
    if not tick_size or tick_size <= 0:
        return price
    return math.floor(price / tick_size) * tick_size

def round_quantity(quantity, step_size):
    if not step_size or step_size <= 0:
        return quantity
    return math.floor(quantity / step_size) * step_size

def build_grid(lower, upper, grid_count, grid_type):
    if grid_type == "Arithmetic":
        return np.linspace(lower, upper, grid_count + 1).tolist()

    if lower <= 0 or upper <= 0:
        return np.linspace(lower, upper, grid_count + 1).tolist()

    ratio = (upper / lower) ** (1 / grid_count)
    levels = [lower * (ratio ** i) for i in range(grid_count + 1)]
    return levels

def find_current_grid_index(grid, current_price):
    index = 0
    for i in range(len(grid) - 1):
        if grid[i] <= current_price < grid[i + 1]:
            index = i
            break
        if current_price >= grid[-1]:
            index = len(grid) - 2
    return index

# ============================================================
# GRID BACKTEST ENGINE
# ============================================================

def backtest_grid(df, lower, upper, grid_count, capital, fee_pct, grid_type, filters):
    if df.empty:
        return None

    current_price = float(df["Close"].iloc[0])
    if current_price <= 0:
        return None

    grid = build_grid(lower, upper, grid_count, grid_type)

    tick_size = filters.get("tickSize")
    if tick_size:
        grid = [round_to_tick(p, tick_size) for p in grid]
        grid = sorted(list(set(grid)))

    if len(grid) < 2:
        return None

    quote_balance = capital * 0.50
    base_balance = (capital * 0.50) / current_price
    starting_equity = quote_balance + (base_balance * current_price)

    order_quote = capital / max(grid_count, 1)
    if order_quote <= 0:
        return None

    step_size = filters.get("stepSize")
    min_qty = filters.get("minQty")
    min_notional = filters.get("minNotional")

    trades = []
    completed_cycles = 0
    gross_profit = 0.0
    total_fees = 0.0
    realized_profit = 0.0
    peak_equity = starting_equity
    max_drawdown = 0.0
    inventory = {}

    for _, row in df.iterrows():
        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])

        path = [o, l, h, c] if c >= o else [o, h, l, c]

        for start_price, end_price in zip(path[:-1], path[1:]):
            if end_price > start_price:
                crossed_levels = [
                    i for i in range(1, len(grid))
                    if start_price < grid[i] <= end_price
                ]
                for level_index in crossed_levels:
                    price = grid[level_index]
                    if level_index in inventory:
                        quantity = inventory[level_index]["quantity"]
                        if quantity <= 0:
                            continue

                        notional = quantity * price
                        if min_notional and notional < min_notional:
                            continue

                        fee = notional * fee_pct / 100
                        quote_balance += (notional - fee)

                        buy_info = inventory.pop(level_index)
                        buy_cost = buy_info["cost"]

                        profit = notional - fee - buy_cost
                        realized_profit += profit
                        gross_profit += (notional - buy_cost)
                        total_fees += fee
                        completed_cycles += 1

                        trades.append({
                            "Time": row["OpenTime"],
                            "Side": "SELL",
                            "Grid": level_index,
                            "Price": price,
                            "Quantity": quantity,
                            "Notional": notional,
                            "Fee": fee,
                            "Profit": profit
                        })

            elif end_price < start_price:
                crossed_levels = [
                    i for i in range(len(grid) - 1)
                    if end_price <= grid[i] < start_price
                ]
                for level_index in reversed(crossed_levels):
                    price = grid[level_index]
                    quantity = order_quote / price
                    quantity = round_quantity(quantity, step_size)

                    if quantity <= 0:
                        continue
                    if min_qty and quantity < min_qty:
                        continue

                    notional = quantity * price
                    if min_notional and notional < min_notional:
                        continue

                    fee = notional * fee_pct / 100
                    total_cost = notional + fee

                    if total_cost > quote_balance:
                        continue

                    quote_balance -= total_cost
                    base_balance += quantity
                    inventory[level_index] = {
                        "quantity": quantity,
                        "price": price,
                        "cost": total_cost
                    }

                    trades.append({
                        "Time": row["OpenTime"],
                        "Side": "BUY",
                        "Grid": level_index,
                        "Price": price,
                        "Quantity": quantity,
                        "Notional": notional,
                        "Fee": fee,
                        "Profit": 0.0
                    })

        equity = quote_balance + (base_balance * c)
        if equity > peak_equity:
            peak_equity = equity

        drawdown = ((peak_equity - equity) / peak_equity) * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    final_price = float(df["Close"].iloc[-1])
    final_equity = quote_balance + (base_balance * final_price)
    total_return = final_equity - starting_equity
    roi_pct = (total_return / starting_equity) * 100
    realized_roi = (realized_profit / starting_equity) * 100

    return {
        "Final Equity": final_equity,
        "Total Return": total_return,
        "ROI %": roi_pct,
        "Realized Grid Profit": realized_profit,
        "Realized ROI %": realized_roi,
        "Trades": len(trades),
        "Max Drawdown %": max_drawdown,
        "Grid Spacing": np.mean(np.diff(grid)) if len(grid) > 1 else 0
    }

# ============================================================
# ATR DYNAMIC RANGE CALCULATION (PHASE 5)
# ============================================================

def calculate_range_atr(df, atr_multiplier=2.0):
    """Calculates Grid Range using ATR Volatility instead of static buffers."""
    high = float(df["High"].max())
    low = float(df["Low"].min())
    latest_atr = float(df["ATR"].iloc[-1]) if "ATR" in df and not df["ATR"].empty else 0.0

    # Expand boundaries using ATR volatility multiplier
    upper = high + (latest_atr * atr_multiplier)
    lower = max(0.0001, low - (latest_atr * atr_multiplier))
    
    return lower, upper

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.title("⚡ Binance Spot Grid Assistant V3.5")
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Data Freshness:** {current_time} UTC | Engine: Operational")

    # --- INTERACTIVE SETTINGS MENU ---
    st.markdown("### 1. Strategy Parameters")
    with st.expander("⚙️ Tap to Edit Trading Rules & Indicator Settings", expanded=False):
        wallet_balance = st.number_input("Available Capital (USDT)", min_value=0.0, value=1000.0, step=50.0)
        
        fee_choice = st.selectbox("Trading Fee Tier (Round-Trip)", ["Standard (0.20%)", "BNB Discount (0.15%)", "Zero Fee (0.00%)"])
        fee_pct = 0.20 if "0.20" in fee_choice else (0.15 if "0.15" in fee_choice else 0.00)
            
        grid_type = st.selectbox("Grid Type", ["Arithmetic", "Geometric"])
        
        # Indicator Controls
        only_ranging = st.checkbox("Filter Out Trending Coins (Only keep ADX < 25)", value=False)
        atr_mult = st.slider("ATR Volatility Range Multiplier", 0.5, 4.0, 1.5, step=0.1)
            
        st.markdown("**Target Profit per Grid Step (%)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            tight_pct = st.number_input("Tight (3D)", min_value=0.1, value=0.6, step=0.1)
        with col2:
            mod_pct = st.number_input("Mod (7D)", min_value=0.1, value=1.0, step=0.1)
        with col3:
            wide_pct = st.number_input("Wide (14D)", min_value=0.1, value=1.5, step=0.1)

    st.markdown("### 2. Market Universe")
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe & Exchange Filters..."):
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
                    
                    # Apply ADX Filter if enabled by user
                    if only_ranging and "Trending" in regime:
                        progress_bar.progress((i + 1) / len(candidates))
                        continue

                    df_3d = df_klines.tail(72).copy()
                    df_7d = df_klines.tail(168).copy()
                    df_14d = df_klines.tail(336).copy()
                    
                    coin_filter = filters_dict.get(coin, {})
                    
                    strategies = [
                        {"Type": "Tight (3D)", "df": df_3d, "Target_Pct": tight_pct},
                        {"Type": "Mod (7D)", "df": df_7d, "Target_Pct": mod_pct},
                        {"Type": "Wide (14D)", "df": df_14d, "Target_Pct": wide_pct}
                    ]
                    
                    for strat in strategies:
                        # Dynamic Range using ATR
                        lower, upper = calculate_range_atr(strat["df"], atr_multiplier=atr_mult)
                        
                        if lower > 0:
                            range_width_pct = ((upper - lower) / lower) * 100
                            ideal_grids = int(range_width_pct / strat["Target_Pct"])
                        else:
                            ideal_grids = 5
                            
                        suggested_grids = max(5, min(ideal_grids, 150))
                        
                        bt_result = backtest_grid(
                            df=strat["df"],
                            lower=lower,
                            upper=upper,
                            grid_count=suggested_grids,
                            capital=wallet_balance,
                            fee_pct=fee_pct,
                            grid_type=grid_type,
                            filters=coin_filter
                        )
                        
                        if bt_result:
                            step_pct = (bt_result["Grid Spacing"] / lower) * 100 if lower > 0 else 0
  
