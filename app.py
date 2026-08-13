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
        response = session.get(
            EXCHANGE_INFO_URL,
            timeout=15
        )

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
                    symbol_filters["tickSize"] = float(
                        f.get("tickSize", 0)
                    )

                elif filter_type == "LOT_SIZE":
                    symbol_filters["stepSize"] = float(
                        f.get("stepSize", 0)
                    )
                    symbol_filters["minQty"] = float(
                        f.get("minQty", 0)
                    )
                    symbol_filters["maxQty"] = float(
                        f.get("maxQty", 0)
                    )

                elif filter_type in (
                    "MIN_NOTIONAL",
                    "NOTIONAL"
                ):
                    symbol_filters["minNotional"] = float(
                        f.get("minNotional", 0)
                    )

            filters[symbol] = symbol_filters

    return symbols, filters


# ============================================================
# 24H MARKET DATA
# ============================================================

@st.cache_data(ttl=300)
def fetch_24h_data():

    session = get_http_session()

    try:

        response = session.get(
            TICKER_24H_URL,
            timeout=15
        )

        response.raise_for_status()

        df = pd.DataFrame(response.json())

        if df.empty:
            return pd.DataFrame()

        numeric_cols = [
            "lastPrice",
            "highPrice",
            "lowPrice",
            "volume",
            "quoteVolume",
            "priceChangePercent"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return df[
            [
                "symbol",
                "lastPrice",
                "priceChangePercent",
                "quoteVolume"
            ]
        ]

    except Exception:
        return pd.DataFrame()


# ============================================================
# KLINES
# ============================================================

@st.cache_data(ttl=300)
def fetch_klines(
    symbol,
    interval="1h",
    limit=336
):

    session = get_http_session()

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:

        response = session.get(
            KLINES_URL,
            params=params,
            timeout=15
        )

        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()

        if not data:
            return pd.DataFrame()

        columns = [
            "OpenTime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "CloseTime",
            "QuoteVolume",
            "Trades",
            "TBB",
            "TBQ",
            "Ignore"
        ]

        df = pd.DataFrame(
            data,
            columns=columns
        )

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "QuoteVolume"
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df["OpenTime"] = pd.to_datetime(
            df["OpenTime"],
            unit="ms"
        )

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# PRICE ROUNDING
# ============================================================

def round_to_tick(price, tick_size):

    if not tick_size or tick_size <= 0:
        return price

    return math.floor(
        price / tick_size
    ) * tick_size


def round_quantity(quantity, step_size):

    if not step_size or step_size <= 0:
        return quantity

    return math.floor(
        quantity / step_size
    ) * step_size


# ============================================================
# BUILD GRID
# ============================================================

def build_grid(
    lower,
    upper,
    grid_count,
    grid_type
):

    if grid_type == "Arithmetic":

        return np.linspace(
            lower,
            upper,
            grid_count + 1
        ).tolist()

    # Geometric / percentage-style grid

    if lower <= 0 or upper <= 0:

        return np.linspace(
            lower,
            upper,
            grid_count + 1
        ).tolist()

    ratio = (
        upper / lower
    ) ** (1 / grid_count)

    levels = [
        lower * (ratio ** i)
        for i in range(grid_count + 1)
    ]

    return levels


# ============================================================
# INITIAL GRID POSITION
# ============================================================

def find_current_grid_index(
    grid,
    current_price
):

    index = 0

    for i in range(len(grid) - 1):

        if (
            grid[i]
            <= current_price
            < grid[i + 1]
        ):
            index = i
            break

        if current_price >= grid[-1]:
            index = len(grid) - 2

    return index


# ============================================================
# GRID BACKTEST
# ============================================================

def backtest_grid(
    df,
    lower,
    upper,
    grid_count,
    capital,
    fee_pct,
    grid_type,
    filters
):

    if df.empty:
        return None

    current_price = float(
        df["Close"].iloc[0]
    )

    if current_price <= 0:
        return None

    grid = build_grid(
        lower,
        upper,
        grid_count,
        grid_type
    )

    # --------------------------------------------------------
    # BINANCE PRICE FILTER
    # --------------------------------------------------------

    tick_size = filters.get(
        "tickSize"
    )

    if tick_size:

        grid = [
            round_to_tick(
                p,
                tick_size
            )
            for p in grid
        ]

        grid = sorted(
            list(set(grid))
        )

    if len(grid) < 2:
        return None

    # --------------------------------------------------------
    # DETERMINE GRID POSITION
    # --------------------------------------------------------

    current_index = find_current_grid_index(
        grid,
        current_price
    )

    # --------------------------------------------------------
    # CAPITAL ALLOCATION
    #
    # Half quote / half base.
    #
    # This is a simplified but much more realistic starting
    # inventory model than the original code.
    # --------------------------------------------------------

    quote_balance = capital * 0.50
    base_balance = (
        capital * 0.50
    ) / current_price

    starting_equity = (
        quote_balance
        + base_balance * current_price
    )

    # Each grid trade gets an equal quote value.
    order_quote = capital / max(
        grid_count,
        1
    )

    if order_quote <= 0:
        return None

    step_size = filters.get(
        "stepSize"
    )

    min_qty = filters.get(
        "minQty"
    )

    min_notional = filters.get(
        "minNotional"
    )

    trades = []

    completed_cycles = 0

    gross_profit = 0.0

    total_fees = 0.0

    realized_profit = 0.0

    peak_equity = starting_equity

    max_drawdown = 0.0

    # Track bought quantity for each grid level.
    inventory = {}

    # --------------------------------------------------------
    # CANDLE PATH
    #
    # If candle closes above open:
    #
    # Open -> Low -> High -> Close
    #
    # If candle closes below open:
    #
    # Open -> High -> Low -> Close
    #
    # This gives deterministic treatment when both directions
    # occur in the same candle.
    # --------------------------------------------------------

    for _, row in df.iterrows():

        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])

        if c >= o:

            path = [
                o,
                l,
                h,
                c
            ]

        else:

            path = [
                o,
                h,
                l,
                c
            ]

        for start_price, end_price in zip(
            path[:-1],
            path[1:]
        ):

            # ----------------------------------------------
            # PRICE MOVING UP
            # ----------------------------------------------

            if end_price > start_price:

                crossed_levels = [
                    i
                    for i in range(
                        1,
                        len(grid)
                    )
                    if (
                        start_price
                        < grid[i]
                        <= end_price
                    )
                ]

                for level_index in crossed_levels:

                    price = grid[level_index]

                    # SELL if we have inventory
                    if level_index in inventory:

                        quantity = inventory[
                            level_index
                        ]

                        if quantity <= 0:
                            continue

                        notional = (
                            quantity * price
                        )

                        if (
                            min_notional
                            and notional < min_notional
                        ):
                            continue

                        fee = (
                            notional
                            * fee_pct
                            / 100
                        )

                        quote_balance += (
                            notional - fee
                        )

                        buy_info = inventory.pop(
                            level_index
                        )

                        buy_price = (
                            buy_info["price"]
                        )

                        buy_cost = (
                            buy_info["cost"]
                        )

                        profit = (
                            notional
                            - fee
                            - buy_cost
                        )

                        realized_profit += profit

                        gross_profit += (
                            notional
                            - buy_cost
                        )

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

                    # If we don't have inventory at this
                    # level, this upward movement can simply
                    # pass through.

            # ----------------------------------------------
            # PRICE MOVING DOWN
            # ----------------------------------------------

            elif end_price < start_price:

                crossed_levels = [
                    i
                    for i in range(
                        len(grid) - 1
                    )
                    if (
                        end_price
                        <= grid[i]
                        < start_price
                    )
                ]

                for level_index in reversed(
                    crossed_levels
                ):

                    price = grid[level_index]

                    # BUY at this level

                    quantity = (
                        order_quote / price
                    )

                    quantity = round_quantity(
                        quantity,
                        step_size
                    )

                    if quantity <= 0:
                        continue

                    if (
                        min_qty
                        and quantity < min_qty
                    ):
                        continue

                    notional = (
                        quantity * price
                    )

                    if (
                        min_notional
                        and notional < min_notional
                    ):
                        continue

                    fee = (
                        notional
                        * fee_pct
                        / 100
                    )

                    total_cost = (
                        notional + fee
                    )

                    if (
                        total_cost
                        > quote_balance
                    ):
                        continue

                    quote_balance -= (
                        total_cost
                    )

                    base_balance += quantity

                    inventory[
                        level_index
                    ] = {
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

        # ----------------------------------------------------
        # MARK-TO-MARKET EQUITY
        # ----------------------------------------------------

        equity = (
            quote_balance
            + base_balance * c
        )

        if equity > peak_equity:
            peak_equity = equity

        drawdown = (
            (peak_equity - equity)
            / peak_equity
            * 100
        )

        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # --------------------------------------------------------
    # FINAL EQUITY
    # --------------------------------------------------------

    final_price = float(
        df["Close"].iloc[-1]
    )

    final_equity = (
        quote_balance
        + base_balance * final_price
    )

    total_return = (
        final_equity
        - starting_equity
    )

    roi_pct = (
        total_return
        / starting_equity
        * 100
    )

    # Realized grid profit / starting capital
    realized_roi = (
        realized_profit
        / starting_equity
        * 100
    )

    # --------------------------------------------------------
    # RANGE POSITION
    # --------------------------------------------------------

    range_width_pct = (
        (upper - lower)
        / lower
        * 100
    )

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
        "Grid Spacing": (
            np.mean(
                np.diff(grid)
            )
            if len(grid) > 1
            else 0
        ),
        "Grid": grid,
        "Trades Data": pd.DataFrame(
            trades
        )
    }

    return result


# ============================================================
# STRATEGY RANGE
# ============================================================

def calculate_range(
    df,
    buffer_pct
):

    high = float(
        df["High"].max()
    )

    low = float(
        df["Low"].min()
    )

    upper = (
        high
        * (1 + buffer_pct / 100)
    )

    lower = (
        low
        * (1 - buffer_pct / 100)
    )

    return lower, upper


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.title(
        "⚡ Binance Spot Grid Assistant V3"
    )

    
