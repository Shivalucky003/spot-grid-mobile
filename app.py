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
    page_title="Binance Spot Grid Assistant V3",
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

if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = None

# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SpotGridAssistant/3.0"
    })
    return session

# ============================================================
# BINANCE EXCHANGE INFO
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

# ============================================================
# BUILD SPOT UNIVERSE + FILTERS
# ============================================================

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
# KLINES
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
        return df
    except Exception:
        return pd.DataFrame()

# ============================================================
# PRICE ROUNDING
# ============================================================

def round_to_tick(price, tick_size):
    if not tick_size or tick_size <= 0:
        return price
    return math.floor(price / tick_size) * tick_size

def round_quantity(quantity, step_size):
    if not step_size or step_size <= 0:
        return quantity
    return math.floor(quantity / step_size) * step_size

# ============================================================
# BUILD GRID
# ============================================================

def build_grid(lower, upper, grid_count, grid_type):
    if grid_type == "Arithmetic":
        return np.linspace(lower, upper, grid_count + 1).tolist()

    if lower <= 0 or upper <= 0:
        return np.linspace(lower, upper, grid_count + 1).tolist()

    ratio = (upper / lower) ** (1 / grid_count)
    levels = [lower * (ratio ** i) for i in range(grid_count + 1)]
    return levels

# ============================================================
# INITIAL GRID POSITION
# ============================================================

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
# GRID BACKTEST
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

    current_index = find_current_grid_index(grid, current_price)

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
    range_width_pct = ((upper - lower) / lower) * 100

    result = {
        "Final Equity": final_equity,
        "Total Return": total_return,
        "ROI %": roi_pct,
        "Realized Grid Profit": realized_profit,
        "Realized ROI %": realized_roi,
        "Gross Profit": gross_profit,
        "Fees": total_fees,
        "Completed Cycles": completed_cycles,
        "Trades": len(trades),
        "Max Drawdown %": max_drawdown,
        "Remaining USDT": quote_balance,
        "Remaining Coin": base_balance,
        "Range %": range_width_pct,
        "Grid Levels": len(grid),
        "Grid Spacing": np.mean(np.diff(grid)) if len(grid) > 1 else 0,
        "Grid": grid,
        "Trades Data": pd.DataFrame(trades)
    }

    return result

# ============================================================
# STRATEGY RANGE
# ============================================================

def calculate_range(df, buffer_pct):
    high = float(df["High"].max())
    low = float(df["Low"].min())
    upper = high * (1 + buffer_pct / 100)
    lower = low * (1 - buffer_pct / 100)
    return lower, upper

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.title("⚡ Binance Spot Grid Assistant V3")
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Data Freshness:** {current_time} UTC | Engine: Operational")

    # --- INTERACTIVE SETTINGS MENU ---
    st.markdown("### 1. Strategy Parameters")
    with st.expander("⚙️ Tap to Edit Trading Rules & Capital", expanded=False):
        wallet_balance = st.number_input("Available Capital (USDT)", min_value=0.0, value=1000.0, step=50.0)
        
        fee_choice = st.selectbox("Trading Fee Tier (Round-Trip)", ["Standard (0.20%)", "BNB Discount (0.15%)", "Zero Fee (0.00%)"])
        if "0.20" in fee_choice:
            fee_pct = 0.20
        elif "0.15" in fee_choice:
            fee_pct = 0.15
        else:
            fee_pct = 0.00
            
        grid_type = st.selectbox("Grid Type", ["Arithmetic", "Geometric"])
            
        st.markdown("**Grid Density per Strategy**")
        col1, col2, col3 = st.columns(3)
        with col1:
            tight_grids = st.number_input("Tight (3D) Grids", min_value=5, value=20, step=1)
        with col2:
            mod_grids = st.number_input("Mod (7D) Grids", min_value=5, value=15, step=1)
        with col3:
            wide_grids = st.number_input("Wide (14D) Grids", min_value=5, value=10, step=1)
            
        buffer_pct = st.number_input("Range Safety Buffer (%)", min_value=0.0, value=1.0, step=0.1)

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
        if st.button("Run Deep Backtest"):
            results = []
            progress_bar = st.progress(0)
            candidates = st.session_state['candidates']
            filters_dict = st.session_state.get('filters', {})
            
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=336)
                
                if not df_klines.empty:
                    df_3d = df_klines.tail(72).copy()
                    df_7d = df_klines.tail(168).copy()
                    df_14d = df_klines.tail(336).copy()
                    
                    coin_filter = filters_dict.get(coin, {})
                    
                    strategies = [
                        {"Type": "Tight (3D)", "df": df_3d, "grids": int(tight_grids)},
                        {"Type": "Mod (7D)", "df": df_7d, "grids": int(mod_grids)},
                        {"Type": "Wide (14D)", "df": df_14d, "grids": int(wide_grids)}
                    ]
                    
                    for strat in strategies:
                        lower, upper = calculate_range(strat["df"], buffer_pct)
                        
                        bt_result = backtest_grid(
                            df=strat["df"],
                            lower=lower,
                            upper=upper,
                            grid_count=strat["grids"],
                            capital=wallet_balance,
                            fee_pct=fee_pct,
                            grid_type=grid_type,
                            filters=coin_filter
                        )
                        
                        if bt_result:
                            results.append({
                                "Coin": coin, 
                                "Strategy": strat["Type"], 
                                "Lower": round(lower, 4), 
                                "Upper": round(upper, 4), 
                                "Grids": strat["grids"], 
                                "Trades": bt_result["Trades"],
                                "Max Drawdown %": round(bt_result["Max Drawdown %"], 2),
                                "Realized ROI %": round(bt_result["Realized ROI %"], 2)
                            })
                            
                progress_bar.progress((i + 1) / len(candidates))
                
            st.markdown("### 🏆 V3 Final Backtest Ranking")
            if results:
                final_df = pd.DataFrame(results).sort_values(by="Realized ROI %", ascending=False).reset_index(drop=True)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No valid data could be calculated.")

if __name__ == "__main__":
    main()
    
