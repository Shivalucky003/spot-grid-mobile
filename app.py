import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import math

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Binance Spot Grid Assistant V4",
    layout="wide"
)

EXCHANGE_INFO_URL = (
    "https://data-api.binance.vision/api/v3/exchangeInfo"
)

TICKER_24H_URL = (
    "https://data-api.binance.vision/api/v3/ticker/24hr"
)

KLINES_URL = (
    "https://data-api.binance.vision/api/v3/klines"
)

STABLECOIN_BLACKLIST = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI",
    "USDP", "EUR", "AEUR", "USDT", "PAX",
    "USD1", "RLUSD", "PYUSD", "USDE", "USDS",
    "USDD", "GUSD", "LUSD", "FRAX", "USDJ",
    "USDB", "DEUSD", "SUSD", "EUSD", "CUSD",
    "EURS", "TRY", "BRL", "BIDR", "U"
}


# ============================================================
# SESSION STATE
# ============================================================

if "candidates" not in st.session_state:
    st.session_state.candidates = []

if "filters" not in st.session_state:
    st.session_state.filters = {}

if "market_data" not in st.session_state:
    st.session_state.market_data = pd.DataFrame()

if "scan_done" not in st.session_state:
    st.session_state.scan_done = False


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_session():

    session = requests.Session()

    session.headers.update({
        "User-Agent": "SpotGridAssistant/4.0"
    })

    return session


# ============================================================
# BINANCE SPOT UNIVERSE
# ============================================================

@st.cache_data(ttl=600)
def fetch_spot_universe():

    try:

        response = get_session().get(
            EXCHANGE_INFO_URL,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        symbols = []
        filters = {}

        for s in data.get("symbols", []):

            base = str(
                s.get("baseAsset", "")
            ).upper()

            quote = str(
                s.get("quoteAsset", "")
            ).upper()

            is_stablecoin = (
                base in STABLECOIN_BLACKLIST
                or "USD" in base
                or "EUR" in base
            )

            if (
                s.get("status") == "TRADING"
                and quote == "USDT"
                and s.get(
                    "isSpotTradingAllowed",
                    True
                )
                and not is_stablecoin
            ):

                symbol = s["symbol"]

                symbols.append(symbol)

                symbol_filters = {
                    "tickSize": 0.0,
                    "stepSize": 0.0,
                    "minQty": 0.0,
                    "minNotional": 0.0
                }

                for item in s.get("filters", []):

                    filter_type = item.get(
                        "filterType"
                    )

                    if filter_type == "PRICE_FILTER":

                        symbol_filters["tickSize"] = float(
                            item.get("tickSize", 0)
                        )

                    elif filter_type == "LOT_SIZE":

                        symbol_filters["stepSize"] = float(
                            item.get("stepSize", 0)
                        )

                        symbol_filters["minQty"] = float(
                            item.get("minQty", 0)
                        )

                    elif filter_type in (
                        "MIN_NOTIONAL",
                        "NOTIONAL"
                    ):

                        symbol_filters["minNotional"] = float(
                            item.get("minNotional", 0)
                        )

                filters[symbol] = symbol_filters

        return symbols, filters

    except Exception:

        return [], {}


# ============================================================
# 24H MARKET DATA
# ============================================================

@st.cache_data(ttl=300)
def fetch_24h_data():

    try:

        response = get_session().get(
            TICKER_24H_URL,
            timeout=20
        )

        response.raise_for_status()

        df = pd.DataFrame(
            response.json()
        )

        required_columns = [
            "symbol",
            "lastPrice",
            "priceChangePercent",
            "quoteVolume"
        ]

        for column in required_columns[1:]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        return df[
            required_columns
        ].dropna()

    except Exception:

        return pd.DataFrame()


# ============================================================
# KLINE DATA
# ============================================================

@st.cache_data(ttl=300)
def fetch_klines(
    symbol,
    interval="1h",
    limit=1000
):

    try:

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = get_session().get(
            KLINES_URL,
            params=params,
            timeout=20
        )

        if response.status_code != 200:
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
            response.json(),
            columns=columns
        )

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "QuoteVolume"
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df["OpenTime"] = pd.to_datetime(
            df["OpenTime"],
            unit="ms",
            utc=True
        )

        df["CloseTime"] = pd.to_datetime(
            df["CloseTime"],
            unit="ms",
            utc=True
        )

        df = add_indicators(df)

        return df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(
    df,
    period=14
):

    df = df.copy()

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"] -
                previous_close
            ).abs(),
            (
                df["Low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .rolling(period)
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["Close"].diff()

    gain = (
        delta
        .clip(lower=0)
        .rolling(period)
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .rolling(period)
        .mean()
    )

    rs = (
        gain /
        loss.replace(0, np.nan)
    )

    df["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    df["RSI"] = df["RSI"].fillna(50)

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    up_move = df["High"].diff()

    down_move = -df["Low"].diff()

    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
        down_move,
        0.0
    )

    tr_sum = (
        true_range
        .rolling(period)
        .sum()
        .replace(0, np.nan)
    )

    plus_di = (
        100 *
        pd.Series(
            plus_dm,
            index=df.index
        )
        .rolling(period)
        .sum()
        /
        tr_sum
    )

    minus_di = (
        100 *
        pd.Series(
            minus_dm,
            index=df.index
        )
        .rolling(period)
        .sum()
        /
        tr_sum
    )

    dx = (
        100 *
        (
            plus_di -
            minus_di
        ).abs()
        /
        (
            plus_di +
            minus_di
        ).replace(0, np.nan)
    )

    df["ADX"] = (
        dx
        .rolling(period)
        .mean()
        .fillna(0)
    )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    df["Regime"] = np.where(
        df["ADX"] < 25,
        "Ranging",
        "Trending"
    )

    return df


# ============================================================
# BINANCE PRECISION HELPERS
# ============================================================

def floor_to_step(
    value,
    step
):

    if not step or step <= 0:
        return float(value)

    return math.floor(
        value / step + 1e-12
    ) * step


def round_price(
    value,
    tick_size
):

    return floor_to_step(
        value,
        tick_size
    )


# ============================================================
# GRID BUILDER
# ============================================================

def build_grid(
    lower,
    upper,
    count,
    grid_type,
    tick_size=0
):

    if (
        lower <= 0
        or upper <= lower
        or count < 2
    ):
        return []

    # --------------------------------------------------------
    # Arithmetic Grid
    # --------------------------------------------------------

    if grid_type == "Arithmetic":

        step = (
            upper - lower
        ) / count

        raw_levels = [
            lower + step * i
            for i in range(count + 1)
        ]

    # --------------------------------------------------------
    # Geometric Grid
    # --------------------------------------------------------

    else:

        ratio = (
            upper / lower
        ) ** (
            1.0 / count
        )

        raw_levels = [
            lower * (
                ratio ** i
            )
            for i in range(count + 1)
        ]

    # --------------------------------------------------------
    # Binance Tick Size
    # --------------------------------------------------------

    if tick_size and tick_size > 0:

        raw_levels = [
            round_price(
                price,
                tick_size
            )
            for price in raw_levels
        ]

    levels = sorted(
        set(
            round(
                float(price),
                12
            )
            for price in raw_levels
        )
    )

    if len(levels) < 2:
        return []

    return levels


# ============================================================
# ATR RANGE
# ============================================================

def range_from_training(
    df,
    atr_multiplier
):

    historical_high = float(
        df["High"].max()
    )

    historical_low = float(
        df["Low"].min()
    )

    atr = float(
        df["ATR"].iloc[-1]
    )

    if (
        not np.isfinite(atr)
        or atr <= 0
    ):

        atr = float(
            (
                df["High"] -
                df["Low"]
            ).mean()
        )

    upper = (
        historical_high +
        atr * atr_multiplier
    )

    lower = max(
        1e-12,
        historical_low -
        atr * atr_multiplier
    )

    return lower, upper, atr


# ============================================================
# GRID BACKTEST ENGINE
# ============================================================

def backtest_grid(
    df,
    lower,
    upper,
    grid_count,
    capital,
    fee_pct,
    grid_type,
    exchange_filter
):

    if (
        df.empty
        or capital <= 0
    ):
        return None

    tick_size = exchange_filter.get(
        "tickSize",
        0
    )

    step_size = exchange_filter.get(
        "stepSize",
        0
    )

    min_qty = exchange_filter.get(
        "minQty",
        0
    )

    min_notional = exchange_filter.get(
        "minNotional",
        0
    )

    # --------------------------------------------------------
    # Build Grid
    # --------------------------------------------------------

    levels = build_grid(
        lower,
        upper,
        grid_count,
        grid_type,
        tick_size
    )

    if len(levels) < 2:
        return None

    # --------------------------------------------------------
    # IMPORTANT:
    # Backtest starts at FIRST candle.
    # --------------------------------------------------------

    start_price = float(
        df["Close"].iloc[0]
    )

    # --------------------------------------------------------
    # 50/50 Starting Portfolio
    # --------------------------------------------------------

    quote_balance = (
        capital * 0.50
    )

    base_balance = (
        capital * 0.50
    ) / start_price

    # --------------------------------------------------------
    # Inventory Lots
    # --------------------------------------------------------

    lots = []

    if base_balance > 0:

        lots.append(
            {
                "qty": base_balance,
                "cost_per_unit": start_price,
                "grid_buy": False
            }
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    initial_equity = capital

    realized_profit = 0.0

    completed_cycles = 0

    buy_orders = 0

    sell_orders = 0

    skipped_orders = 0

    range_escape_bars = 0

    peak_equity = capital

    max_drawdown = 0.0

    # --------------------------------------------------------
    # Sell Inventory Function
    # --------------------------------------------------------

    def sell_inventory(
        quantity,
        price
    ):

        nonlocal realized_profit
        nonlocal completed_cycles

        remaining = quantity

        sold = 0.0

        cost_removed = 0.0

        while (
            remaining > 1e-15
            and lots
        ):

            lot = lots[0]

            take = min(
                remaining,
                lot["qty"]
            )

            cost_removed += (
                take *
                lot["cost_per_unit"]
            )

            if lot["grid_buy"]:

                completed_cycles += 1

            lot["qty"] -= take

            remaining -= take

            sold += take

            if lot["qty"] <= 1e-15:

                lots.pop(0)

        if sold <= 0:
            return 0.0

        gross_proceeds = (
            sold *
            price
        )

        selling_fee = (
            gross_proceeds *
            fee_pct /
            100
        )

        net_proceeds = (
            gross_proceeds -
            selling_fee
        )

        realized_profit += (
            net_proceeds -
            cost_removed
        )

        return sold

    # ========================================================
    # CANDLE LOOP
    # ========================================================

    for _, row in df.iterrows():

        open_price = float(
            row["Open"]
        )

        high_price = float(
            row["High"]
        )

        low_price = float(
            row["Low"]
        )

        close_price = float(
            row["Close"]
        )

        # ----------------------------------------------------
        # Detect range escape
        # ----------------------------------------------------

        if (
            high_price > upper
            or low_price < lower
        ):

            range_escape_bars += 1

        # ----------------------------------------------------
        # OHLC Path Assumption
        #
        # Bullish candle:
        # Open -> Low -> High -> Close
        #
        # Bearish candle:
        # Open -> High -> Low -> Close
        # ----------------------------------------------------

        if close_price >= open_price:

            path = [
                open_price,
                low_price,
                high_price,
                close_price
            ]

        else:

            path = [
                open_price,
                high_price,
                low_price,
                close_price
            ]

        # ----------------------------------------------------
        # Simulate movements between path points
        # ----------------------------------------------------

        for start, end in zip(
            path[:-1],
            path[1:]
        ):

            if end == start:
                continue

            # =================================================
            # PRICE MOVING UP
            # =================================================

            if end > start:

                crossed_levels = [
                    i
                    for i in range(
                        1,
                        len(levels)
                    )
                    if (
                        start < levels[i]
                        <= end
                    )
                ]

                for idx in crossed_levels:

                    price = levels[idx]

                    # -----------------------------------------
                    # Order size
                    # -----------------------------------------

                    order_quote = (
                        capital /
                        max(
                            len(levels) - 1,
                            1
                        )
                    )

                    desired_qty = (
                        order_quote /
                        price
                    )

                    # -----------------------------------------
                    # Cannot sell more than inventory
                    # -----------------------------------------

                    available_qty = sum(
                        lot["qty"]
                        for lot in lots
                    )

                    quantity = min(
                        desired_qty,
                        available_qty
                    )

                    quantity = floor_to_step(
                        quantity,
                        step_size
                    )

                    if quantity <= 0:

                        skipped_orders += 1
                        continue

                    notional = (
                        quantity *
                        price
                    )

                    if (
                        min_qty
                        and quantity < min_qty
                    ):

                        skipped_orders += 1
                        continue

                    if (
                        min_notional
                        and notional < min_notional
                    ):

                        skipped_orders += 1
                        continue

                    sold = sell_inventory(
                        quan
