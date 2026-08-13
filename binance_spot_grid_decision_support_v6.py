import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import math

# ============================================================
# BINANCE SPOT GRID DECISION SUPPORT - V6
# ============================================================
# READ-ONLY ANALYSIS TOOL
# - Does NOT place Binance orders.
# - Uses public Binance market data only.
# - Historical backtests are simulations, not forecasts.
# - OHLC candles do not reveal exact intrabar price sequence.
# ============================================================

st.set_page_config(
    page_title="Binance Spot Grid Decision Support V6",
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
# SESSION STATE
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


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "BinanceSpotGridDecisionSupport/6.0"
    })
    return session


# ============================================================
# BINANCE EXCHANGE INFO
# ============================================================

@st.cache_data(ttl=600)
def fetch_spot_universe():
    try:
        response = get_http_session().get(
            EXCHANGE_INFO_URL,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        symbols = []
        filters = {}

        for item in data.get("symbols", []):
            base = str(item.get("baseAsset", "")).upper()
            quote = str(item.get("quoteAsset", "")).upper()

            is_stablecoin = (
                base in STABLECOIN_BLACKLIST
                or "USD" in base
                or "EUR" in base
            )

            if not (
                item.get("status") == "TRADING"
                and quote == "USDT"
                and item.get("isSpotTradingAllowed", True)
                and not is_stablecoin
            ):
                continue

            symbol = item["symbol"]
            symbols.append(symbol)

            f = {
                "tickSize": 0.0,
                "stepSize": 0.0,
                "minQty": 0.0,
                "minNotional": 0.0,
            }

            for rule in item.get("filters", []):
                ft = rule.get("filterType")

                if ft == "PRICE_FILTER":
                    f["tickSize"] = float(
                        rule.get("tickSize", 0)
                    )

                elif ft == "LOT_SIZE":
                    f["stepSize"] = float(
                        rule.get("stepSize", 0)
                    )
                    f["minQty"] = float(
                        rule.get("minQty", 0)
                    )

                elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                    f["minNotional"] = float(
                        rule.get("minNotional", 0)
                    )

            filters[symbol] = f

        return symbols, filters

    except Exception:
        return [], {}


# ============================================================
# 24H TICKER
# ============================================================

@st.cache_data(ttl=300)
def fetch_24h_data():
    try:
        response = get_http_session().get(
            TICKER_24H_URL,
            timeout=20
        )
        response.raise_for_status()

        df = pd.DataFrame(response.json())

        needed = [
            "symbol",
            "lastPrice",
            "priceChangePercent",
            "quoteVolume",
        ]

        if any(col not in df.columns for col in needed):
            return pd.DataFrame()

        for col in needed[1:]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return (
            df[needed]
            .dropna()
            .copy()
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# KLINES
# ============================================================

@st.cache_data(ttl=300)
def fetch_klines(symbol, interval="1h", limit=1000):
    try:
        response = get_http_session().get(
            KLINES_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
            timeout=20,
        )

        if response.status_code != 200:
            return pd.DataFrame()

        rows = response.json()

        if not rows:
            return pd.DataFrame()

        columns = [
            "OpenTime", "Open", "High", "Low", "Close",
            "Volume", "CloseTime", "QuoteVolume", "Trades",
            "TBB", "TBQ", "Ignore",
        ]

        df = pd.DataFrame(rows, columns=columns)

        numeric_cols = [
            "Open", "High", "Low", "Close",
            "Volume", "QuoteVolume"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col],
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

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        return calculate_indicators(df)

    except Exception:
        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------
    df["EMA9"] = df["Close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA21"] = df["Close"].ewm(
        span=21,
        adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # True Range / ATR(14)
    # --------------------------------------------------------
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    df["TR"] = tr

    # Wilder-style ATR approximation using EWM alpha=1/n.
    df["ATR14"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # RSI(14) - Wilder-style
    # --------------------------------------------------------
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI14"] = (
        100 -
        (100 / (1 + rs))
    )

    df["RSI14"] = df["RSI14"].fillna(50)

    # --------------------------------------------------------
    # ADX(14), +DI, -DI
    # --------------------------------------------------------
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) &
            (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    atr_for_di = df["ATR14"].replace(
        0,
        np.nan
    )

    df["PlusDI"] = (
        100 *
        plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        /
        atr_for_di
    )

    df["MinusDI"] = (
        100 *
        minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        /
        atr_for_di
    )

    di_sum = (
        df["PlusDI"] +
        df["MinusDI"]
    ).replace(0, np.nan)

    df["DX"] = (
        100 *
        (
            df["PlusDI"] -
            df["MinusDI"]
        ).abs()
        /
        di_sum
    )

    df["ADX14"] = df["DX"].ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    df["ADX14"] = df["ADX14"].fillna(0)

    # --------------------------------------------------------
    # MACD 12/26/9
    # --------------------------------------------------------
    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACDSignal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACDHist"] = (
        df["MACD"] -
        df["MACDSignal"]
    )

    # --------------------------------------------------------
    # Bollinger Bands 20 / 2
    # --------------------------------------------------------
    df["BBMiddle"] = df["Close"].rolling(
        20
    ).mean()

    bb_std = df["Close"].rolling(
        20
    ).std(ddof=0)

    df["BBUpper"] = (
        df["BBMiddle"] +
        2 * bb_std
    )

    df["BBLower"] = (
        df["BBMiddle"] -
        2 * bb_std
    )

    df["BBWidthPct"] = np.where(
        df["BBMiddle"] > 0,
        (
            (
                df["BBUpper"] -
                df["BBLower"]
            )
            /
            df["BBMiddle"]
            *
            100
        ),
        0
    )

    # --------------------------------------------------------
    # Volatility and trend helpers
    # --------------------------------------------------------
    df["ATRPct"] = np.where(
        df["Close"] > 0,
        df["ATR14"] /
        df["Close"] *
        100,
        0
    )

    df["EMA21SlopePct"] = np.where(
        df["EMA21"].shift(6) > 0,
        (
            (
                df["EMA21"] -
                df["EMA21"].shift(6)
            )
            /
            df["EMA21"].shift(6)
            *
            100
        ),
        0
    )

    df["VolumeSMA20"] = df["Volume"].rolling(
        20
    ).mean()

    df["VolumeRatio"] = np.where(
        df["VolumeSMA20"] > 0,
        df["Volume"] /
        df["VolumeSMA20"],
        1
    )

    # --------------------------------------------------------
    # EMA structure
    # --------------------------------------------------------
    df["EMAAlignment"] = np.select(
        [
            (
                (df["EMA9"] > df["EMA21"]) &
                (df["EMA21"] > df["EMA50"]) &
                (df["EMA50"] > df["EMA200"])
            ),
            (
                (df["EMA9"] < df["EMA21"]) &
                (df["EMA21"] < df["EMA50"]) &
                (df["EMA50"] < df["EMA200"])
            ),
        ],
        [
            "Bullish",
            "Bearish",
        ],
        default="Mixed"
    )

    # --------------------------------------------------------
    # MACD state
    # --------------------------------------------------------
    df["MACDState"] = np.select(
        [
            (
                (df["MACD"] > 0) &
                (df["MACDHist"] > 0)
            ),
            (
                (df["MACD"] < 0) &
                (df["MACDHist"] < 0)
            ),
        ],
        [
            "Bullish",
            "Bearish",
        ],
        default="Mixed"
    )

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------
    df["Regime"] = np.select(
        [
            df["ADX14"] < 20,
            (
                (df["ADX14"] >= 20) &
                (df["ADX14"] < 25)
            ),
            df["ADX14"] >= 25,
        ],
        [
            "Weak Trend / Range",
            "Transition",
            "Trending",
        ],
        default="Unknown"
    )

    return df


# ============================================================
# PRECISION HELPERS
# ============================================================

def floor_to_step(value, step):
    if not step or step <= 0:
        return float(value)

    return math.floor(
        value / step + 1e-12
    ) * step


def floor_price(value, tick_size):
    return floor_to_step(
        value,
        tick_size
    )


# ============================================================
# GRID
# ============================================================

def build_grid(
    lower,
    upper,
    grid_count,
    grid_type,
    tick_size=0
):
    if (
        lower <= 0
        or upper <= lower
        or grid_count < 2
    ):
        return []

    if grid_type == "Arithmetic":
        step = (
            upper - lower
        ) / grid_count

        raw = [
            lower + step * i
            for i in range(grid_count + 1)
        ]

    else:
        ratio = (
            upper / lower
        ) ** (
            1 / grid_count
        )

        raw = [
            lower * (
                ratio ** i
            )
            for i in range(grid_count + 1)
        ]

    if tick_size and tick_size > 0:
        raw = [
            floor_price(
                p,
                tick_size
            )
            for p in raw
        ]

    raw = [
        float(p)
        for p in raw
        if p > 0
    ]

    levels = sorted(
        set(
            round(p, 14)
            for p in raw
        )
    )

    if len(levels) < 2:
        return []

    return levels


# ============================================================
# RANGE CALCULATION
# ============================================================

def calculate_grid_range(
    train,
    atr_multiplier
):
    high = float(
        train["High"].max()
    )

    low = float(
        train["Low"].min()
    )

    atr = float(
        train["ATR14"].iloc[-1]
    )

    if not np.isfinite(atr) or atr <= 0:
        atr = float(
            (
                train["High"] -
                train["Low"]
            ).mean()
        )

    if not np.isfinite(atr) or atr <= 0:
        atr = max(
            low * 0.005,
            1e-12
        )

    upper = high + (
        atr *
        atr_multiplier
    )

    lower = max(
        1e-12,
        low - (
            atr *
            atr_multiplier
        )
    )

    return lower, upper, atr


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
    exchange_filter,
    path_mode="assumed"
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

    grid = build_grid(
        lower,
        upper,
        grid_count,
        grid_type,
        tick_size
    )

    if len(grid) < 2:
        return None

    first_price = float(
        df["Close"].iloc[0]
    )

    if first_price <= 0:
        return None

    # 50% quote + 50% base starting inventory.
    quote_balance = capital * 0.50
    base_balance = (
        capital * 0.50
    ) / first_price

    initial_equity = capital

    # Inventory lots allow us to calculate approximate
    # realized P&L for each sell.
    lots = [
        {
            "qty": base_balance,
            "cost_per_unit": first_price,
            "grid_buy": False,
        }
    ]

    realized_profit = 0.0
    completed_cycles = 0
    buy_orders = 0
    sell_orders = 0
    skipped_orders = 0
    range_escape_bars = 0
    range_escape_severity_sum = 0.0
    profitable_sales = 0.0
    losing_sales = 0.0

    peak_equity = capital
    max_drawdown = 0.0

    order_quote = (
        capital /
        max(
            len(grid) - 1,
            1
        )
    )

    def available_inventory():
        return sum(
            lot["qty"]
            for lot in lots
        )

    def consume_inventory(
        requested_qty,
        price
    ):
        nonlocal realized_profit
        nonlocal completed_cycles
        nonlocal profitable_sales
        nonlocal losing_sales

        remaining = requested_qty
        sold = 0.0
        removed_cost = 0.0
        completed_from_sale = 0

        while (
            remaining > 1e-15
            and lots
        ):
            lot = lots[0]

            take = min(
                remaining,
                lot["qty"]
            )

            removed_cost += (
                take *
                lot["cost_per_unit"]
            )

            if lot["grid_buy"]:
                completed_from_sale += 1

            lot["qty"] -= take
            remaining -= take
            sold += take

            if lot["qty"] <= 1e-15:
                lots.pop(0)

        if sold <= 0:
            return 0.0, 0.0

        gross = sold * price
        fee = gross * fee_pct / 100
        net = gross - fee

        sale_pnl = net - removed_cost
        realized_profit += sale_pnl
        if sale_pnl > 0:
            profitable_sales += sale_pnl
        elif sale_pnl < 0:
            losing_sales += abs(sale_pnl)

        completed_cycles += (
            completed_from_sale
        )

        return sold, fee

    for _, row in df.iterrows():

        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])

        if (
            h > upper
            or l < lower
        ):
            range_escape_bars += 1
            upside_escape = max(0.0, (h - upper) / upper * 100.0)
            downside_escape = max(0.0, (lower - l) / lower * 100.0)
            range_escape_severity_sum += upside_escape + downside_escape

        # OHLC does not reveal intrabar order. Three modes are supported:
        # assumed = common heuristic; conservative = choose adverse direction;
        # high-first/low-first are explicit scenario paths.

        # This is an approximation because OHLC candles do not
        # reveal the exact sequence of prices inside the candle.
        if path_mode == "low-first":
            path = [o, l, h, c]
        elif path_mode == "high-first":
            path = [o, h, l, c]
        elif path_mode == "conservative":
            # For a bullish candle, high-first tends to realize sells before
            # later buys; for a bearish candle, low-first tends to realize buys
            # before later sells. We use the path that is less favorable to the
            # grid's realized result as a deterministic conservative scenario.
            path = [o, h, l, c] if c >= o else [o, l, h, c]
        else:
            path = [o, l, h, c] if c >= o else [o, h, l, c]

        for start, end in zip(
            path[:-1],
            path[1:]
        ):

            if end == start:
                continue

            # ------------------------------------------------
            # PRICE MOVING UP -> SELL
            # ------------------------------------------------
            if end > start:

                crossed = [
                    i
                    for i in range(
                        1,
                        len(grid)
                    )
                    if (
                        start < grid[i]
                        <= end
                    )
                ]

                for i in crossed:

                    price = grid[i]

                    desired_qty = (
                        order_quote /
                        price
                    )

                    quantity = min(
                        desired_qty,
                        available_inventory()
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
                        min_qty > 0
                        and quantity < min_qty
                    ):
                        skipped_orders += 1
                        continue

                    if (
                        min_notional > 0
                        and notional < min_notional
                    ):
                        skipped_orders += 1
                        continue

                    sold, fee = consume_inventory(
                        quantity,
                        price
                    )

                    if sold > 0:
                        quote_balance += (
                            sold * price -
                            fee
                        )
                        base_balance -= sold
                        sell_orders += 1

            # ------------------------------------------------
            # PRICE MOVING DOWN -> BUY
            # ------------------------------------------------
            elif end < start:

                crossed = [
                    i
                    for i in range(
                        len(grid) - 1
                    )
                    if (
                        end <= grid[i]
                        < start
                    )
                ]

                for i in reversed(crossed):

                    price = grid[i]

                    spend_limit = min(
                        order_quote,
                        quote_balance
                    )

                    if spend_limit <= 0:
                        skipped_orders += 1
                        continue

                    # Quantity is based on quote amount after
                    # accounting for the buy-side fee.
                    quantity = (
                        spend_limit /
                        (
                            price *
                            (
                                1 +
                                fee_pct / 100
                            )
                        )
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

                    fee = (
                        notional *
                        fee_pct /
                        100
                    )

                    total_cost = (
                        notional +
                        fee
                    )

                    if (
                        min_qty > 0
                        and quantity < min_qty
                    ):
                        skipped_orders += 1
                        continue

                    if (
                        min_notional > 0
                        and notional < min_notional
                    ):
                        skipped_orders += 1
                        continue

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
                                fee / quantity
                            ),
                            "grid_buy": True,
                        }
                    )

                    buy_orders += 1

        equity = (
            quote_balance +
            base_balance * c
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

    grid_step_pct = 0.0

    profit_factor = (
        profitable_sales / losing_sales
        if losing_sales > 0 else
        (float("inf") if profitable_sales > 0 else 0.0)
    )
    attempted_orders = buy_orders + sell_orders + skipped_orders
    skipped_order_pct = (
        skipped_orders / attempted_orders * 100.0
        if attempted_orders > 0 else 0.0
    )

    if (
        len(grid) > 1
        and grid[0] > 0
        and grid[1] > grid[0]
    ):
        grid_step_pct = (
            (
                grid[1] -
                grid[0]
            )
            /
            grid[0]
            *
            100
        )

    return {
        "Initial Equity": initial_equity,
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
        "Range Escape Severity %": range_escape_severity_sum,
        "Skipped Order %": skipped_order_pct,
        "Profit Factor": profit_factor,
        "Final Base": base_balance,
        "Final Quote": quote_balance,
        "Grid Levels": len(grid),
        "Grid Step %": grid_step_pct,
        "Grid": grid,
    }


# ============================================================
# INDICATOR SNAPSHOT
# ============================================================

def indicator_snapshot(df):
    row = df.iloc[-1]

    return {
        "Price": float(row["Close"]),
        "EMA9": float(row["EMA9"]),
        "EMA21": float(row["EMA21"]),
        "EMA50": float(row["EMA50"]),
        "EMA200": float(row["EMA200"]),
        "RSI": float(row["RSI14"]),
        "ADX": float(row["ADX14"]),
        "PlusDI": float(row["PlusDI"]),
        "MinusDI": float(row["MinusDI"]),
        "MACD": float(row["MACD"]),
        "MACDSignal": float(row["MACDSignal"]),
        "MACDHist": float(row["MACDHist"]),
        "BBUpper": float(row["BBUpper"]),
        "BBMiddle": float(row["BBMiddle"]),
        "BBLower": float(row["BBLower"]),
        "BBWidthPct": float(row["BBWidthPct"]),
        "ATRPct": float(row["ATRPct"]),
        "VolumeRatio": float(row["VolumeRatio"]),
        "EMAAlignment": str(row["EMAAlignment"]),
        "MACDState": str(row["MACDState"]),
        "Regime": str(row["Regime"]),
    }


# ============================================================
# DECISION SCORE
# ============================================================

def calculate_suitability_score(
    snapshot,
    backtest,
    price_position,
    net_grid_edge,
    liquidity_score,
    test_alpha_pct=0.0,
):
    score = 50.0
    reasons_good = []
    reasons_bad = []
    warnings = []

    adx = snapshot["ADX"]
    rsi = snapshot["RSI"]
    ema_alignment = snapshot["EMAAlignment"]
    macd_state = snapshot["MACDState"]
    bb_width = snapshot["BBWidthPct"]
    atr_pct = snapshot["ATRPct"]

    # --------------------------------------------------------
    # 1. Market regime: 25 points
    # --------------------------------------------------------
    if adx < 20:
        score += 25
        reasons_good.append(
            "ADX is below 20, indicating weak trend strength."
        )
    elif adx < 25:
        score += 16
        reasons_good.append(
            "ADX is below 25, so trend strength is moderate."
        )
    elif adx < 35:
        score += 2
        warnings.append(
            "ADX indicates a meaningful trend."
        )
    else:
        score -= 15
        reasons_bad.append(
            "ADX is high, indicating a strong trend."
        )

    # --------------------------------------------------------
    # 2. EMA structure: 15 points
    # --------------------------------------------------------
    if ema_alignment == "Mixed":
        score += 15
        reasons_good.append(
            "EMA structure is mixed rather than strongly directional."
        )
    elif ema_alignment == "Bullish":
        score -= 8
        reasons_bad.append(
            "EMA 9/21/50/200 are bullishly aligned; "
            "one-directional movement can hurt a grid."
        )
    else:
        score -= 8
        reasons_bad.append(
            "EMA 9/21/50/200 are bearishly aligned; "
            "persistent downside movement can hurt a grid."
        )

    # --------------------------------------------------------
    # 3. MACD: 10 points
    # --------------------------------------------------------
    if macd_state == "Mixed":
        score += 10
        reasons_good.append(
            "MACD is near a mixed/transition state."
        )
    else:
        score -= 3
        warnings.append(
            f"MACD is currently {macd_state.lower()}."
        )

    # --------------------------------------------------------
    # 4. RSI: 5 points
    # --------------------------------------------------------
    if 40 <= rsi <= 60:
        score += 5
        reasons_good.append(
            "RSI is near neutral territory."
        )
    elif 30 <= rsi < 40 or 60 < rsi <= 70:
        score += 1
        warnings.append(
            "RSI shows moderate momentum."
        )
    else:
        score -= 3
        warnings.append(
            "RSI is at an extreme level."
        )

    # --------------------------------------------------------
    # 5. Backtest return: 20 points
    # --------------------------------------------------------
    total_return = backtest["Total Return %"]

    if total_return >= 5:
        score += 20
        reasons_good.append(
            "The historical test produced a strong positive return."
        )
    elif total_return >= 2:
        score += 14
        reasons_good.append(
            "The historical test produced a positive return."
        )
    elif total_return > 0:
        score += 7
        warnings.append(
            "Historical return was positive but small."
        )
    else:
        score -= 15
        reasons_bad.append(
            "The historical test was not profitable."
        )

    # --------------------------------------------------------
    # 6. Drawdown: 15 points
    # --------------------------------------------------------
    dd = backtest["Max Drawdown %"]

    if dd <= 5:
        score += 15
        reasons_good.append(
            "Historical maximum drawdown was low."
        )
    elif dd <= 10:
        score += 10
        reasons_good.append(
            "Historical drawdown was moderate."
        )
    elif dd <= 20:
        score -= 2
        warnings.append(
            "Historical drawdown was significant."
        )
    else:
        score -= 15
        reasons_bad.append(
            "Historical maximum drawdown was high."
        )

    # --------------------------------------------------------
    # 7. Fee-adjusted grid edge: 5 points
    # --------------------------------------------------------
    if net_grid_edge >= 0.50:
        score += 5
        reasons_good.append(
            "Grid spacing provides a reasonable margin over assumed fees."
        )
    elif net_grid_edge >= 0.20:
        score += 2
        warnings.append(
            "Fee-adjusted grid edge is relatively small."
        )
    else:
        score -= 8
        reasons_bad.append(
            "Grid spacing is too close to the assumed fee cost."
        )

    # --------------------------------------------------------
    # 8. Current price position: 5 points
    # --------------------------------------------------------
    if 25 <= price_position <= 75:
        score += 5
        reasons_good.append(
            "Current price is reasonably central within the proposed range."
        )
    elif 10 <= price_position < 25 or 75 < price_position <= 90:
        score += 0
        warnings.append(
            "Current price is getting close to a grid boundary."
        )
    else:
        score -= 8
        reasons_bad.append(
            "Current price is very close to a grid boundary."
        )

    # --------------------------------------------------------
    # 9. Liquidity: 5 points
    # --------------------------------------------------------
    score += (
        liquidity_score -
        2.5
    )

    if liquidity_score >= 4:
        reasons_good.append(
            "24h quote volume is strong relative to the scan."
        )
    elif liquidity_score <= 1:
        reasons_bad.append(
            "Liquidity is weak relative to the scan."
        )

    # --------------------------------------------------------
    # 10. Out-of-sample quality / execution realism: 15 points
    # --------------------------------------------------------
    pf = backtest.get("Profit Factor", 0.0)
    skipped_pct = backtest.get("Skipped Order %", 0.0)
    escapes = backtest.get("Range Escape Bars", 0)

    if test_alpha_pct > 2:
        score += 6
        reasons_good.append("Test-period return beat simple buy-and-hold by a useful margin.")
    elif test_alpha_pct < -2:
        score -= 6
        reasons_bad.append("The grid underperformed simple buy-and-hold during the test period.")

    if pf >= 1.5:
        score += 5
        reasons_good.append("Profit factor was strong in the historical test.")
    elif 1.0 <= pf < 1.5:
        score += 1
        warnings.append("Profit factor was positive but not especially strong.")
    elif pf > 0:
        score -= 5
        reasons_bad.append("Losses outweighed profitable sales in the test.")

    if skipped_pct <= 5:
        score += 2
    elif skipped_pct > 20:
        score -= 4
        warnings.append("A high percentage of attempted grid orders were skipped.")

    if escapes == 0:
        reasons_good.append("The test period stayed inside the proposed grid range.")
    elif escapes >= max(3, int(backtest.get("Test Bars", 100) * 0.10)):
        score -= 5
        reasons_bad.append("The market frequently escaped the proposed grid range.")

    # --------------------------------------------------------
    # 11. Bollinger / volatility warning
    # --------------------------------------------------------
    if bb_width > 15 or atr_pct > 8:
        score -= 5
        warnings.append(
            "Volatility is high; price can escape the grid quickly."
        )

    score = max(
        0,
        min(
            100,
            score
        )
    )

    if score >= 75:
        decision = "🟢 CONSIDER"
    elif score >= 60:
        decision = "🟡 CAUTION"
    elif score >= 45:
        decision = "🟠 WEAK"
    else:
        decision = "🔴 AVOID"

    return (
        score,
        decision,
        reasons_good,
        reasons_bad,
        warnings,
    )


# ============================================================
# EVALUATE ONE COIN
# ============================================================

def evaluate_coin(
    symbol,
    df,
    wallet,
    fee_pct,
    grid_type,
    atr_multiplier,
    targets,
    exchange_filter,
    train_ratio,
    liquidity_score,
):
    if len(df) < 240:
        return []

    results = []

    windows = [
        ("Tight (3D)", 72, targets["tight"]),
        ("Moderate (7D)", 168, targets["moderate"]),
        ("Wide (14D)", 336, targets["wide"]),
    ]

    for strategy, window, target in windows:

        data = df.tail(
            min(window, len(df))
        ).copy()

        if len(data) < 72:
            continue

        # Walk-forward:
        # training data determines the grid;
        # later unseen data tests it.
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

        if len(test) < 24:
            continue

        lower, upper, atr = (
            calculate_grid_range(
                train,
                atr_multiplier
            )
        )

        if lower <= 0 or upper <= lower:
            continue

        range_pct = (
            (
                upper -
                lower
            )
            /
            lower
            *
            100
        )

        if target <= 0:
            continue

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

        # Two explicit intrabar scenarios are tested. OHLC candles do not
        # reveal whether high or low occurred first, so V6 uses the lower
        # return scenario for the decision score rather than silently relying
        # on one arbitrary path.
        low_first = backtest_grid(test, lower, upper, grid_count, wallet, fee_pct, grid_type, exchange_filter, "low-first")
        high_first = backtest_grid(test, lower, upper, grid_count, wallet, fee_pct, grid_type, exchange_filter, "high-first")

        if not low_first or not high_first:
            continue

        backtest = low_first if low_first["Total Return %"] <= high_first["Total Return %"] else high_first
        best_case_return = max(low_first["Total Return %"], high_first["Total Return %"])
        path_spread = abs(low_first["Total Return %"] - high_first["Total Return %"])
        backtest["Best Case Return %"] = best_case_return
        backtest["Path Spread %"] = path_spread
        backtest["Test Bars"] = len(test)

        test_start = float(test["Close"].iloc[0])
        test_end = float(test["Close"].iloc[-1])
        buy_hold_return = ((test_end / test_start) - 1.0) * 100.0 if test_start > 0 else 0.0
        test_alpha = backtest["Total Return %"] - buy_hold_return

        snap = indicator_snapshot(data)

        current_price = snap["Price"]

        price_position = (
            (
                current_price -
                lower
            )
            /
            (
                upper -
                lower
            )
            *
            100
        )

        price_position = max(
            0,
            min(
                100,
                price_position
            )
        )

        grid_step_pct = backtest[
            "Grid Step %"
        ]

        net_grid_edge = (
            grid_step_pct -
            fee_pct
        )

        score, decision, good, bad, warnings = (
            calculate_suitability_score(
                snap,
                backtest,
                price_position,
                net_grid_edge,
                liquidity_score,
                test_alpha_pct=test_alpha,
            )
        )

        # Capital per grid level.
        capital_per_grid = (
            wallet /
            max(
                grid_count,
                1
            )
        )

        # How many grid levels the current price is away
        # from either edge.
        lower_distance_pct = (
            (
                current_price -
                lower
            )
            /
            current_price
            *
            100
        )

        upper_distance_pct = (
            (
                upper -
                current_price
            )
            /
            current_price
            *
            100
        )

        results.append({
            "Coin": symbol,
            "Strategy": strategy,
            "Decision": decision,
            "Score": round(score, 1),
            "Current Price": current_price,
            "Lower": lower,
            "Upper": upper,
            "Price Position %": round(
                price_position,
                1
            ),
            "Grid Levels": grid_count,
            "USDT/Grid": capital_per_grid,
            "Grid Step %": round(
                grid_step_pct,
                3
            ),
            "Net Grid Edge %": round(
                net_grid_edge,
                3
            ),
            "ADX": round(
                snap["ADX"],
                2
            ),
            "RSI": round(
                snap["RSI"],
                2
            ),
            "EMA Structure": snap[
                "EMAAlignment"
            ],
            "MACD State": snap[
                "MACDState"
            ],
            "MACD Histogram": snap[
                "MACDHist"
            ],
            "ATR %": round(
                snap["ATRPct"],
                2
            ),
            "BB Width %": round(
                snap["BBWidthPct"],
                2
            ),
            "Regime": snap[
                "Regime"
            ],
            "Volume Ratio": round(
                snap["VolumeRatio"],
                2
            ),
            "Test Hours": len(test),
            "Train Fraction %": round(train_ratio * 100, 1),
            "Buy Hold Return %": round(buy_hold_return, 3),
            "Test Alpha %": round(test_alpha, 3),
            "Best Case Return %": round(backtest["Best Case Return %"], 3),
            "Path Spread %": round(backtest["Path Spread %"], 3),
            "Profit Factor": round(backtest["Profit Factor"], 3) if math.isfinite(backtest["Profit Factor"]) else 99.0,
            "Skipped Order %": round(backtest["Skipped Order %"], 2),
            "Escape Severity %": round(backtest["Range Escape Severity %"], 3),
            "Completed Cycles": backtest[
                "Completed Cycles"
            ],
            "Buy Orders": backtest[
                "Buy Orders"
            ],
            "Sell Orders": backtest[
                "Sell Orders"
            ],
            "Total Return %": round(
                backtest[
                    "Total Return %"
                ],
                3
            ),
            "Realized Profit": round(
                backtest[
                    "Realized Profit"
                ],
                4
            ),
            "Unrealized PnL": round(
                backtest[
                    "Unrealized PnL"
                ],
                4
            ),
            "Max DD %": round(
                backtest[
                    "Max Drawdown %"
                ],
                3
            ),
            "Range Escape Bars": backtest[
                "Range Escape Bars"
            ],
            "Skipped Orders": backtest[
                "Skipped Orders"
            ],
            "Lower Distance %": round(
                lower_distance_pct,
                2
            ),
            "Upper Distance %": round(
                upper_distance_pct,
                2
            ),
            "_grid": backtest["Grid"],
            "_good": good,
            "_bad": bad,
            "_warnings": warnings,
        })

    return results


# ============================================================
# FORMATTING
# ============================================================

def display_results(df):
    if df.empty:
        return df

    out = df.copy()

    price_cols = [
        "Current Price",
        "Lower",
        "Upper",
    ]

    for col in price_cols:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: f"{x:.8g}"
            )

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    st.title(
        "⚡ Binance Spot Grid Decision Support V6"
    )

    st.caption(
        "Read-only educational analysis. "
        "No Binance orders are placed."
    )

    st.info(
        "Data time: "
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

    st.header(
        "1. Your Capital & Rules"
    )

    with st.expander(
        "⚙️ Configure analysis",
        expanded=True
    ):

        c1, c2, c3 = st.columns(3)

        with c1:
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
                    "Zero 0.00%",
                ]
            )

        with c2:
            grid_type = st.selectbox(
                "Grid type",
                [
                    "Arithmetic",
                    "Geometric",
                ]
            )

            atr_multiplier = st.slider(
                "ATR range buffer",
                0.5,
                4.0,
                1.5,
                0.1
            )

        with c3:
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
            fee_pct = 0.0

        st.subheader(
            "Target grid spacing"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            tight_target = st.number_input(
                "Tight (3D) %",
                min_value=0.1,
                max_value=10.0,
                value=0.6,
                step=0.1
            )

        with c2:
            moderate_target = st.number_input(
                "Moderate (7D) %",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        with c3:
            wide_target = st.number_input(
                "Wide (14D) %",
                min_value=0.1,
                max_value=10.0,
                value=1.5,
                step=0.1
            )

        only_ranging = st.checkbox(
            "Only show ranging/transition markets",
            value=False
        )

        max_dd_filter = st.slider(
            "Maximum acceptable historical drawdown %",
            2.0, 50.0, 15.0, 0.5
        )

        max_skipped_filter = st.slider(
            "Maximum skipped-order rate %",
            0.0, 50.0, 20.0, 1.0
        )

        min_cycles_filter = st.number_input(
            "Minimum completed grid cycles",
            min_value=0, max_value=1000, value=3, step=1
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    with st.expander(
        "📘 What the indicators mean"
    ):
        st.markdown(
            """
**ATR(14)** — measures typical price volatility.

**ADX(14)** — measures trend strength. Lower ADX generally
means less directional trend, which can be more compatible
with grid trading.

**RSI(14)** — measures recent momentum. It is context, not
a standalone buy/sell signal.

**EMA 9 / 21 / 50 / 200** — show short-, medium- and
long-term trend structure.

**MACD 12/26/9** — measures momentum and changes in trend.

**Bollinger Bands 20/2** — show the statistical price range
around a 20-period moving average.

**BB Width** — shows whether volatility is expanding or
contracting.

**Price Position** — tells you where the current price sits
inside the proposed grid range.

**Backtest** — simulates how the grid behaved during an
unseen historical test period.

The final score combines these factors. It is not a prediction
of future profit.
"""
        )

    # ========================================================
    # MARKET SCAN
    # ========================================================

    st.header(
        "2. Market Scan"
    )

    if st.button(
        "🔎 Scan Binance",
        type="primary"
    ):

        with st.spinner(
            "Loading Binance spot markets..."
        ):

            symbols, filters = (
                fetch_spot_universe()
            )

            market = fetch_24h_data()

            if not symbols or market.empty:
                st.error(
                    "Could not retrieve Binance market data."
                )
                return

            valid = market[
                market["symbol"].isin(symbols)
            ].copy()

            valid = valid[
                (valid["quoteVolume"] > 0) &
                (valid["lastPrice"] > 0)
            ]

            valid = valid.sort_values(
                "quoteVolume",
                ascending=False
            ).head(top_n)

            # Liquidity score relative to the selected scan.
            volumes = valid[
                "quoteVolume"
            ].values

            if len(volumes) > 1:
                log_vol = np.log10(
                    np.maximum(
                        volumes,
                        1
                    )
                )

                lo = float(
                    np.min(log_vol)
                )
                hi = float(
                    np.max(log_vol)
                )

                if hi > lo:
                    valid["Liquidity Score"] = (
                        1 +
                        4 *
                        (
                            (
                                log_vol -
                                lo
                            )
                            /
                            (
                                hi -
                                lo
                            )
                        )
                    )
                else:
                    valid["Liquidity Score"] = 3.0
            else:
                valid["Liquidity Score"] = 3.0

            st.session_state.candidates = (
                valid["symbol"].tolist()
            )

            st.session_state.filters = filters
            st.session_state.market_data = valid
            st.session_state.scan_done = True
            st.session_state.results = pd.DataFrame()

    # ========================================================
    # SCAN RESULTS
    # ========================================================

    if st.session_state.scan_done:

        market = st.session_state.market_data.copy()

        st.success(
            f"{len(st.session_state.candidates)} "
            "liquid USDT spot markets selected."
        )

        market_display = market.copy()

        market_display["lastPrice"] = (
            market_display["lastPrice"].map(
                lambda x: f"{x:.8g}"
            )
        )

        market_display["quoteVolume"] = (
            market_display["quoteVolume"].map(
                lambda x: f"{x:,.0f}"
            )
        )

        market_display["priceChangePercent"] = (
            market_display[
                "priceChangePercent"
            ].map(
                lambda x: f"{x:.2f}%"
            )
        )

        st.dataframe(
            market_display,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # BACKTEST
        # ====================================================

        st.header(
            "3. Decision-Support Analysis"
        )

        if st.button(
            "🚀 Run V6 Analysis",
            type="primary"
        ):

            all_results = []

            progress = st.progress(0)
            status = st.empty()

            candidates = (
                st.session_state.candidates
            )

            filters = (
                st.session_state.filters
            )

            targets = {
                "tight": tight_target,
                "moderate": moderate_target,
                "wide": wide_target,
            }

            for i, symbol in enumerate(
                candidates
            ):

                status.write(
                    f"Analyzing {symbol} "
                    f"({i + 1}/{len(candidates)})..."
                )

                df = fetch_klines(
                    symbol,
                    INTERVAL,
                    KLINE_LIMIT
                )

                if not df.empty:

                    liquidity_row = market[
                        market["symbol"] == symbol
                    ]

                    if (
                        not liquidity_row.empty
                    ):
                        liquidity_score = float(
                            liquidity_row[
                                "Liquidity Score"
                            ].iloc[0]
                        )
                    else:
                        liquidity_score = 2.5

                    results = evaluate_coin(
                        symbol=symbol,
                        df=df,
                        wallet=wallet,
                        fee_pct=fee_pct,
                        grid_type=grid_type,
                        atr_multiplier=atr_multiplier,
                        targets=targets,
                        exchange_filter=filters.get(
                            symbol,
                            {}
                        ),
                        train_ratio=train_ratio,
                        liquidity_score=liquidity_score,
                    )

                    all_results.extend(
                        results
                    )

                progress.progress(
                    (i + 1) /
                    len(candidates)
                )

            status.empty()

            if not all_results:
                st.warning(
                    "No valid results. Try more capital, "
                    "different target spacing, or more markets."
                )
                return

            result_df = pd.DataFrame(
                all_results
            )

            if only_ranging:
                result_df = result_df[
                    result_df["Regime"].isin(
                        [
                            "Weak Trend / Range",
                            "Transition",
                        ]
                    )
                ]

            if result_df.empty:
                st.warning(
                    "No strategies passed the market-regime filter."
                )
                return

            result_df = result_df[
                (result_df["Max DD %"] <= max_dd_filter) &
                (result_df["Skipped Order %"] <= max_skipped_filter) &
                (result_df["Completed Cycles"] >= min_cycles_filter)
            ]

            if result_df.empty:
                st.warning(
                    "No strategies passed the drawdown filter."
                )
                return

            result_df = result_df.sort_values(
                [
                    "Score",
                    "Total Return %",
                ],
                ascending=False
            ).reset_index(
                drop=True
            )

            st.session_state.results = result_df

        # ====================================================
        # RESULTS DISPLAY
        # ====================================================

        result_df = st.session_state.results

        if not result_df.empty:

            st.header(
                "4. Final Decision Ranking"
            )

            display_cols = [
                "Coin",
                "Strategy",
                "Decision",
                "Score",
                "Regime",
                "ADX",
                "RSI",
                "EMA Structure",
                "MACD State",
                "Current Price",
                "Lower",
                "Upper",
                "Price Position %",
                "Grid Levels",
                "USDT/Grid",
                "Grid Step %",
                "Net Grid Edge %",
                "Completed Cycles",
                "Profit Factor",
                "Skipped Order %",
                "Test Alpha %",
                "Path Spread %",
                "Total Return %",
                "Realized Profit",
                "Max DD %",
                "Range Escape Bars",
                "Skipped Orders",
            ]

            st.dataframe(
                display_results(
                    result_df[
                        display_cols
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # BEST CANDIDATE
            # =================================================

            best = result_df.iloc[0]

            st.header(
                "5. Best Candidate Explanation"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Decision",
                    best["Decision"]
                )

            with c2:
                st.metric(
                    "Suitability Score",
                    f'{best["Score"]:.1f}/100'
                )

            with c3:
                st.metric(
                    "Historical Return",
                    f'{best["Total Return %"]:.2f}%'
                )

            with c4:
                st.metric(
                    "Max Drawdown",
                    f'{best["Max DD %"]:.2f}%'
                )

            st.subheader(
                f'{best["Coin"]} — {best["Strategy"]}'
            )

            st.write(
                f"""
**Current price:** `{best["Current Price"]:.8g}`

**Proposed grid range:** `{best["Lower"]:.8g}`
to `{best["Upper"]:.8g}`

**Current price position:** `{best["Price Position %"]:.1f}%`
through the proposed range.

**Grid levels:** `{best["Grid Levels"]}`

**Approximate capital per grid:** `{best["USDT/Grid"]:.2f} USDT`

**Grid spacing:** `{best["Grid Step %"]:.3f}%`

**Estimated grid edge after assumed fees:**
`{best["Net Grid Edge %"]:.3f}%`

**Historical completed cycles:** `{best["Completed Cycles"]}`

**Historical realized profit:**
`{best["Realized Profit"]:.4f} USDT`

**Historical maximum drawdown:**
`{best["Max DD %"]:.2f}%`

**Out-of-sample test return:** `{best["Total Return %"]:.2f}%`

**Buy-and-hold test return:** `{best["Buy Hold Return %"]:.2f}%`

**Grid alpha vs buy-and-hold:** `{best["Test Alpha %"]:.2f}%`

**Worst-vs-best intrabar path spread:** `{best["Path Spread %"]:.2f}%`

**Profit factor:** `{best["Profit Factor"]:.2f}`

**Skipped-order rate:** `{best["Skipped Order %"]:.2f}%`
"""
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "ADX",
                    f'{best["ADX"]:.1f}'
                )

            with c2:
                st.metric(
                    "RSI",
                    f'{best["RSI"]:.1f}'
                )

            with c3:
                st.metric(
                    "EMA Structure",
                    best["EMA Structure"]
                )

            with c4:
                st.metric(
                    "MACD",
                    best["MACD State"]
                )

            st.subheader(
                "Why the score was assigned"
            )

            for item in best["_good"]:
                st.success(
                    "✅ " + item
                )

            for item in best["_bad"]:
                st.error(
                    "❌ " + item
                )

            for item in best["_warnings"]:
                st.warning(
                    "⚠️ " + item
                )

            # =================================================
            # GRID LEVELS
            # =================================================

            st.subheader(
                "6. Proposed Grid Levels"
            )

            grid = best["_grid"]

            grid_df = pd.DataFrame(
                {
                    "Level": range(
                        1,
                        len(grid) + 1
                    ),
                    "Price": grid,
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
            # INDICATOR DETAIL
            # =================================================

            st.subheader(
                "7. Indicator Details"
            )

            indicator_rows = [
                ["ATR", f'{best["ATR %"]:.2f}%'],
                ["ADX", f'{best["ADX"]:.2f}'],
                ["RSI", f'{best["RSI"]:.2f}'],
                ["EMA Structure", best["EMA Structure"]],
                ["MACD State", best["MACD State"]],
                ["MACD Histogram", f'{best["MACD Histogram"]:.8g}'],
                ["Bollinger Width", f'{best["BB Width %"]:.2f}%'],
                ["Volume Ratio", f'{best["Volume Ratio"]:.2f}x'],
                ["Market Regime", best["Regime"]],
            ]

            indicator_df = pd.DataFrame(
                indicator_rows,
                columns=[
                    "Indicator",
                    "Current Reading",
                ]
            )

            st.dataframe(
                indicator_df,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # CSV
            # =================================================

            csv_df = result_df.drop(
                columns=[
                    "_grid",
                    "_good",
                    "_bad",
                    "_warnings",
                ],
                errors="ignore"
            )

            st.download_button(
                "⬇️ Download Full Ranking CSV",
                data=csv_df.to_csv(
                    index=False
                ),
                file_name=(
                    "binance_grid_decision_v5.csv"
                ),
                mime="text/csv",
            )

            st.warning(
                "This is decision-support software, not a "
                "guarantee of future profit. A historical "
                "backtest can differ materially from live "
                "trading because of slippage, fees, order "
                "execution, liquidity, and unknown intrabar "
                "price paths."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
