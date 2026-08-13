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

EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
TICKER_24H_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

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
            "QuoteVolume",
            "Trades",
            "TBB",
            "TBQ",
            "Ignore"
        ]

        # Binance returns 12 fields. Keep compatibility explicit.
        columns = [
            "OpenTime", "Open", "High", "Low", "Close",
            "Volume", "CloseTime", "QuoteVolume",
            "Trades", "TBB", "TBQ", "Ignore"
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

def add_indicators(df, period=14):

    df = df.copy()

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

    df["Regime"] = np.where(
        df["ADX"] < 25,
        "Ranging",
        "Trending"
    )

    return df


# ============================================================
# BINANCE PRECISION HELPERS
# ============================================================

def floor_to_step(value, step):

    if not step or step <= 0:
        return float(value)

    return math.floor(
        value / step + 1e-12
    ) * step


def round_price(value, tick_size):

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

    if grid_type == "Arithmetic":

        step = (
            upper - lower
        ) / count

        raw_levels = [
            lower + step * i
            for i in range(count + 1)
        ]

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

    levels = build_grid(
        lower,
        upper,
        grid_count,
        grid_type,
        tick_size
    )

    if len(levels) < 2:
        return None

    start_price = float(
        df["Close"].iloc[0]
    )

    quote_balance = (
        capital * 0.50
    )

    base_balance = (
        capital * 0.50
    ) / start_price

    initial_equity = capital

    lots = []

    if base_balance > 0:

        lots.append(
            {
                "qty": base_balance,
                "cost_per_unit": start_price,
                "grid_buy": False
            }
        )

    realized_profit = 0.0
    completed_cycles = 0
    buy_orders = 0
    sell_orders = 0
    skipped_orders = 0
    range_escape_bars = 0

    peak_equity = capital
    max_drawdown = 0.0

    def sell_inventory(quantity, price):

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
            sold * price
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

    for _, row in df.iterrows():

        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])

        if (
            high_price > upper
            or low_price < lower
        ):
            range_escape_bars += 1

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

        for start, end in zip(
            path[:-1],
            path[1:]
        ):

            if end == start:
                continue

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
                        quantity * price
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
                        quantity,
                        price
                    )

                    if sold > 0:

                        selling_fee = (
                            sold *
                            price *
                            fee_pct /
                            100
                        )

                        quote_balance += (
                            sold * price
                            -
                            selling_fee
                        )

                        base_balance -= sold
                        sell_orders += 1

            elif end < start:

                crossed_levels = [
                    i
                    for i in range(
                        len(levels) - 1
                    )
                    if (
                        end <= levels[i]
                        < start
                    )
                ]

                for idx in reversed(
                    crossed_levels
                ):

                    price = levels[idx]

                    order_quote = (
                        capital /
                        max(
                            len(levels) - 1,
                            1
                        )
                    )

                    spend = min(
                        order_quote,
                        quote_balance
                    )

                    if spend <= 0:
                        skipped_orders += 1
                        continue

                    fee = (
                        spend *
                        fee_pct /
                        100
                    )

                    net_for_asset = (
                        spend - fee
                    )

                    quantity = (
                        net_for_asset /
                        price
                    )

                    quantity = floor_to_step(
                        quantity,
                        step_size
                    )

                    if quantity <= 0:
                        skipped_orders += 1
                        continue

                    notional = (
                        quantity * price
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

                    actual_spend = (
                        quantity * price
                    )

                    actual_fee = (
                        actual_spend *
                        fee_pct /
                        100
                    )

                    total_cost = (
                        actual_spend +
                        actual_fee
                    )

                    if (
                        total_cost >
                        quote_balance +
                        1e-12
                    ):
                        skipped_orders += 1
                        continue

                    quote_balance -= total_cost
                    base_balance += quantity

                    lots.append(
                        {
                            "qty": quantity,
                            "cost_per_unit": (
                                price +
                                actual_fee /
                                quantity
                            ),
                            "grid_buy": True
                        }
                    )

                    buy_orders += 1

        equity = (
            quote_balance +
            base_balance *
            close_price
        )

        peak_equity = max(
            peak_equity,
            equity
        )

        drawdown = (
            (
                peak_equity -
                equity
            )
            /
            peak_equity
            *
            100
        )

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

    final_price = float(
        df["Close"].iloc[-1]
    )

    final_equity = (
        quote_balance +
        base_balance *
        final_price
    )

    total_return = (
        final_equity /
        initial_equity -
        1
    ) * 100

    realized_roi = (
        realized_profit /
        initial_equity
    ) * 100

    unrealized_pnl = (
        final_equity -
        initial_equity -
        realized_profit
    )

    return {
        "Start Equity": initial_equity,
        "Final Equity": final_equity,
        "Total Return %": total_return,
        "Realized Profit": realized_profit,
        "Realized ROI %": realized_roi,
        "Unrealized PnL": unrealized_pnl,
        "Max Drawdown %": max_drawdown,
        "Buy Orders": buy_orders,
        "Sell Orders": sell_orders,
        "Completed Cycles": completed_cycles,
        "Skipped Orders": skipped_orders,
        "Range Escape Bars": range_escape_bars,
        "Final Base": base_balance,
        "Final Quote": quote_balance,
        "Grid Levels": len(levels),
        "Grid": levels
    }


# ============================================================
# COIN EVALUATION
# ============================================================

def evaluate_coin(
    symbol,
    df,
    wallet,
    fee_pct,
    grid_type,
    atr_mult,
    tight_target,
    mod_target,
    wide_target,
    exchange_filter,
    train_ratio
):

    if len(df) < 100:
        return []

    strategies = [
        ("Tight (3D)", 72, tight_target),
        ("Moderate (7D)", 168, mod_target),
        ("Wide (14D)", 336, wide_target)
    ]

    results = []

    for (
        strategy_name,
        window,
        target
    ) in strategies:

        data = df.tail(
            min(
                window,
                len(df)
            )
        ).copy()

        if len(data) < 72:
            continue

        split = int(
            len(data) *
            train_ratio
        )

        split = max(
            48,
            min(
                split,
                len(data) - 24
            )
        )

        train = data.iloc[:split].copy()
        test = data.iloc[split:].copy()

        if len(test) < 12:
            continue

        lower, upper, atr = (
            range_from_training(
                train,
                atr_mult
            )
        )

        range_pct = (
            (
                upper - lower
            )
            /
            lower
            *
            100
        )

        grid_count = int(
            range_pct /
            target
        )

        grid_count = max(
            5,
            min(
                grid_count,
                100
            )
        )

        backtest = backtest_grid(
            test,
            lower,
            upper,
            grid_count,
            wallet,
            fee_pct,
            grid_type,
            exchange_filter
        )

        if not backtest:
            continue

        current_price = float(
            data["Close"].iloc[-1]
        )

        grid = backtest["Grid"]

        if len(grid) > 1:

            step_pct = (
                (
                    grid[1] -
                    grid[0]
                )
                /
                grid[0]
                *
                100
            )

        else:
            step_pct = 0

        latest = data.iloc[-1]

        score = (
            backtest["Total Return %"]
            -
            backtest["Max Drawdown %"]
            * 0.50
            +
            min(
                backtest["Completed Cycles"],
                50
            )
            * 0.03
            -
            backtest["Range Escape Bars"]
            * 0.05
        )

        results.append(
            {
                "Coin": symbol,
                "Strategy": strategy_name,
                "Current Price": current_price,
                "Regime": latest["Regime"],
                "ADX": round(
                    float(latest["ADX"]),
                    1
                ),
                "RSI": round(
                    float(latest["RSI"]),
                    1
                ),
                "ATR": round(
                    float(atr),
                    6
                ),
                "Train Range %": round(
                    range_pct,
                    2
                ),
                "Lower": lower,
                "Upper": upper,
                "Grids": grid_count,
                "Step %": round(
                    step_pct,
                    3
                ),
                "Test Hours": len(test),
                "Completed Cycles":
                    backtest["Completed Cycles"],
                "Buy Orders":
                    backtest["Buy Orders"],
                "Sell Orders":
                    backtest["Sell Orders"],
                "Total Return %":
                    round(
                        backtest["Total Return %"],
                        3
                    ),
                "Realized ROI %":
                    round(
                        backtest["Realized ROI %"],
                        3
                    ),
                "Realized Profit":
                    round(
                        backtest["Realized Profit"],
                        4
                    ),
                "Unrealized PnL":
                    round(
                        backtest["Unrealized PnL"],
                        4
                    ),
                "Max DD %":
                    round(
                        backtest["Max Drawdown %"],
                        3
                    ),
                "Range Escape":
                    backtest["Range Escape Bars"],
                "Skipped Orders":
                    backtest["Skipped Orders"],
                "Score":
                    round(
                        score,
                        3
                    ),
                "_grid": grid
            }
        )

    return results


# ============================================================
# DISPLAY HELPER
# ============================================================

def format_results(df):

    if df.empty:
        return df

    output = df.copy()

    for column in [
        "Lower",
        "Upper",
        "Current Price"
    ]:

        if column in output.columns:

            output[column] = output[
                column
            ].map(
                lambda x:
                f"{x:.8g}"
            )

    return output


# ============================================================
# MAIN APP
# ============================================================

def main():

    st.title(
        "⚡ Binance Spot Grid Assistant V4"
    )

    st.caption(
        "Analysis/backtesting only — "
        "this application does not place Binance orders."
    )

    st.info(
        "Data checked: "
        +
        dt.datetime.now(
            dt.timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    # ========================================================
    # SETTINGS
    # ========================================================

    st.markdown(
        "## 1. Capital & Strategy"
    )

    with st.expander(
        "⚙️ Strategy Settings",
        expanded=True
    ):

        col1, col2, col3 = st.columns(3)

        with col1:

            wallet = st.number_input(
                "Capital (USDT)",
                min_value=20.0,
                value=160.0,
                step=10.0
            )

            fee_choice = st.selectbox(
                "Round-trip fee assumption",
                [
                    "Standard 0.20%",
                    "BNB discount 0.15%",
                    "Zero 0.00%"
                ]
            )

        with col2:

            grid_type = st.selectbox(
                "Grid type",
                [
                    "Arithmetic",
                    "Geometric"
                ]
            )

            atr_mult = st.slider(
                "ATR range multiplier",
                0.5,
                4.0,
                1.5,
                0.1
            )

        with col3:

            top_n = st.slider(
                "Coins to analyze",
                5,
                30,
                15
            )

            train_ratio = st.slider(
                "Training fraction",
                0.60,
                0.85,
                0.70,
                0.05
            )

        if "0.20" in fee_choice:
            fee_pct = 0.20
        elif "0.15" in fee_choice:
            fee_pct = 0.15
        else:
            fee_pct = 0.00

        st.markdown(
            "**Grid Target Percentages**"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            tight_target = st.number_input(
                "Tight target %",
                0.1,
                10.0,
                0.6,
                0.1
            )

        with col2:
            mod_target = st.number_input(
                "Moderate target %",
                0.1,
                10.0,
                1.0,
                0.1
            )

        with col3:
            wide_target = st.number_input(
                "Wide target %",
                0.1,
                10.0,
                1.5,
                0.1
            )

        only_ranging = st.checkbox(
            "Filter to ADX < 25 (Ranging only)",
            value=False
        )

        max_dd_filter = st.slider(
            "Maximum acceptable test drawdown %",
            1.0,
            50.0,
            15.0,
            0.5
        )

    # ========================================================
    # MARKET SCAN
    # ========================================================

    st.markdown(
        "## 2. Binance Market Scan"
    )

    if st.button(
        "🔎 Scan Binance",
        type="primary"
    ):

        with st.spinner(
            "Loading Binance spot universe "
            "and 24h liquidity..."
        ):

            symbols, filters = (
                fetch_spot_universe()
            )

            market = fetch_24h_data()

            if (
                not symbols
                or market.empty
            ):

                st.error(
                    "Could not retrieve Binance "
                    "market data."
                )

                return

            valid = market[
                market["symbol"].isin(
                    symbols
                )
            ].copy()

            valid = valid[
                (
                    valid["quoteVolume"]
                    > 0
                )
                &
                (
                    valid["lastPrice"]
                    > 0
                )
            ]

            valid = valid.sort_values(
                "quoteVolume",
                ascending=False
            ).head(top_n)

            st.session_state.candidates = (
                valid["symbol"].tolist()
            )

            st.session_state.filters = filters
            st.session_state.market_data = valid
            st.session_state.scan_done = True

    # ========================================================
    # SHOW MARKET SCAN
    # ========================================================

    if st.session_state.scan_done:

        market = st.session_state.market_data

        st.success(
            f"{len(st.session_state.candidates)} "
            "liquid USDT spot markets selected."
        )

        display_market = market.copy()

        display_market["quoteVolume"] = (
            display_market["quoteVolume"].map(
                lambda x: f"{x:,.0f}"
            )
        )

        display_market["lastPrice"] = (
            display_market["lastPrice"].map(
                lambda x: f"{x:.8g}"
            )
        )

        st.dataframe(
            display_market,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # BACKTEST
        # ====================================================

        st.markdown(
            "## 3. Walk-Forward Grid Backtest"
        )

        if st.button(
            "🚀 Run Improved Backtest",
            type="primary"
        ):

            results = []

            progress = st.progress(0)
            status = st.empty()

            candidates = (
                st.session_state.candidates
            )

            filters = (
                st.session_state.filters
            )

            for i, symbol in enumerate(
                candidates
            ):

                status.write(
                    f"Analyzing {symbol} "
                    f"({i + 1}/{len(candidates)})..."
                )

                df = fetch_klines(
                    symbol,
                    "1h",
                    336
                )

                if not df.empty:

                    coin_results = evaluate_coin(
                        symbol,
                        df,
                        wallet,
                        fee_pct,
                        grid_type,
                        atr_mult,
                        tight_target,
                        mod_target,
                        wide_target,
                        filters.get(
                            symbol,
                            {}
                        ),
                        train_ratio
                    )

                    results.extend(
                        coin_results
                    )

                progress.progress(
                    (i + 1) /
                    len(candidates)
                )

            status.empty()

            if not results:

                st.warning(
                    "No valid backtest results. "
                    "Try increasing capital, "
                    "lowering target percentages, "
                    "or analyzing more coins."
                )

                return

            result_df = pd.DataFrame(results)

            if only_ranging:

                result_df = result_df[
                    result_df["Regime"]
                    == "Ranging"
                ]

            if result_df.empty:

                st.warning(
                    "No strategies remain "
                    "after the ADX filter."
                )

                return

            result_df = result_df[
                result_df["Max DD %"]
                <= max_dd_filter
            ]

            if result_df.empty:

                st.warning(
                    "No strategies passed "
                    "the maximum drawdown filter."
                )

                return

            result_df = result_df.sort_values(
                [
                    "Score",
                    "Total Return %"
                ],
                ascending=False
            ).reset_index(
                drop=True
            )

            # =================================================
            # FINAL RANKING
            # =================================================

            st.markdown(
                "## 🏆 Final Ranking"
            )

            shown_columns = [
                "Coin",
                "Strategy",
                "Regime",
                "ADX",
                "RSI",
                "Lower",
                "Upper",
                "Grids",
                "Step %",
                "Completed Cycles",
                "Total Return %",
                "Realized ROI %",
                "Realized Profit",
                "Unrealized PnL",
                "Max DD %",
                "Range Escape",
                "Skipped Orders",
                "Score"
            ]

            st.dataframe(
                format_results(
                    result_df[
                        shown_columns
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # BEST CANDIDATE
            # =================================================

            best = result_df.iloc[0]

            st.markdown(
                "## ⭐ Best Backtested Candidate"
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:
                st.metric(
                    "Coin",
                    best["Coin"]
                )

            with col2:
                st.metric(
                    "Strategy",
                    best["Strategy"]
                )

            with col3:
                st.metric(
                    "Test Return",
                    f'{best["Total Return %"]:.2f}%'
                )

            with col4:
                st.metric(
                    "Max Drawdown",
                    f'{best["Max DD %"]:.2f}%'
                )

            st.write(
                f"""
**Grid range:** `{best["Lower"]:.8g} → {best["Upper"]:.8g}`

**Grid levels:** `{best["Grids"]}`

**Approx. grid step:** `{best["Step %"]:.3f}%`

**Completed grid cycles:** `{best["Completed Cycles"]}`

**Realized grid profit:** `{best["Realized Profit"]:.4f} USDT`

**Unrealized P&L:** `{best["Unrealized PnL"]:.4f} USDT`

**ADX:** `{best["ADX"]}`

**RSI:** `{best["RSI"]}`

**Range escape bars:** `{best["Range Escape"]}`

**Skipped orders:** `{best["Skipped Orders"]}`
"""
            )

            # =================================================
            # GRID LEVELS
            # =================================================

            grid = best["_grid"]

            st.markdown(
                "### 📊 Suggested Grid Levels"
            )

            grid_df = pd.DataFrame(
                {
                    "Level": range(
                        1,
                        len(grid) + 1
                    ),
                    "Price": grid
                }
            )

            st.dataframe(
                grid_df.style.format(
                    {
                        "Price": "{:.8g}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # CSV EXPORT
            # =================================================

            csv = (
                result_df
                .drop(
                    columns=["_grid"]
                )
                .to_csv(
                    index=False
                )
            )

            st.download_button(
                "⬇️ Download Ranking CSV",
                data=csv,
                file_name="binance_grid_backtest_v4.csv",
                mime="text/csv"
            )

            st.caption(
                "Important: historical results are simulations, "
                "not forecasts. OHLC candles cannot reveal the "
                "exact intrabar order of price movements."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
