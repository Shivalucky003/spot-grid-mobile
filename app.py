import streamlit as st
import pandas as pd
import requests
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Spot Grid Assistant", layout="wide")

BINANCE_EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
BINANCE_24HR_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

STABLECOIN_BLACKLIST = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", "USDT", "PAX", "USD1", "RLUSD", "PYUSD", "USDE", "USDS", "USDD", "GUSD", "LUSD", "FRAX", "USDJ", "USDB", "DEUSD", "SUSD", "EUSD", "CUSD", "EURS", "TRY", "BRL", "BIDR", "U"}

# --- BACKEND ENGINE ---

@st.cache_data(ttl=300)
def fetch_spot_universe():
    try:
        response = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = []
        for s in data['symbols']:
            base_asset = s.get('baseAsset', '').upper()
            quote_asset = s.get('quoteAsset', '').upper()
            is_stablecoin = base_asset in STABLECOIN_BLACKLIST or "USD" in base_asset or "EUR" in base_asset
            if s.get('status') == 'TRADING' and quote_asset == 'USDT' and s.get('isSpotTradingAllowed', True) and not is_stablecoin:
                symbols.append(s['symbol'])
        return symbols
    except Exception as e:
        return []

@st.cache_data(ttl=300)
def fetch_24h_data():
    try:
        response = requests.get(BINANCE_24HR_URL, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        numeric_cols = ['lastPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'priceChangePercent']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']]
    except Exception as e:
        return pd.DataFrame()

def fetch_klines(symbol, interval="1h", limit=336): # Increased to 14 days (336 hours)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if response.status_code == 200:
            columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NumTrades', 'TBB', 'TBQ', 'Ignore']
            df = pd.DataFrame(response.json(), columns=columns)
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col])
            return df
    except Exception:
        pass
    return pd.DataFrame()

# --- FRONTEND ---

def main():
    st.title("⚡ Spot Grid Assistant V2")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Data Freshness:** {current_time} UTC | Engine: Operational")
    
    st.markdown("### 1. Market Universe")
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe..."):
            universe = fetch_spot_universe()
            market_data = fetch_24h_data()
            if not market_data.empty and universe:
                valid_markets = market_data[market_data['symbol'].isin(universe)]
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(5) # Reduced to top 5 to keep mobile view clean
                st.success(f"Success: {len(valid_markets)} USDT pairs validated. Processing Top 5.")
                st.dataframe(top_candidates[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']], use_container_width=True)
                st.session_state['candidates'] = top_candidates['symbol'].tolist()
                
    if 'candidates' in st.session_state:
        st.markdown("### 2. Strategy Evaluator (Phases 3-9)")
        if st.button("Generate 3-Tier Strategies"):
            results = []
            progress_bar = st.progress(0)
            candidates = st.session_state['candidates']
            
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=336)
                if not df_klines.empty:
                    current_price = df_klines['Close'].iloc[-1]
                    
                    # Define the 3 time horizons for simulation
                    df_3d = df_klines.tail(72)  # Tight
                    df_7d = df_klines.tail(168) # Moderate
                    df_14d = df_klines.tail(336) # Wide
                    
                    strategies = [
                        {"Type": "Tight (3D)", "df": df_3d, "Target_Pct": 0.6},
                        {"Type": "Mod (7D)", "df": df_7d, "Target_Pct": 1.0},
                        {"Type": "Wide (14D)", "df": df_14d, "Target_Pct": 1.5}
                    ]
                    
                    for strat in strategies:
                        # Range Math
                        high_max = strat["df"]['High'].max()
                        low_min = strat["df"]['Low'].min()
                        upper_price = high_max * 1.01 # 1% safety buffer
                        lower_price = low_min * 0.99
                        
                        # Simulation: Calculate grids based on target profit step
                        range_width_pct = ((upper_price - lower_price) / lower_price) * 100
                        ideal_grids = int(range_width_pct / strat["Target_Pct"])
                        grid_count = max(5, min(ideal_grids, 100)) # Keep within Binance limits
                        
                        # Feasibility Math: Calculate required capital
                        req_capital = grid_count * 15.0 # $15 minimum per grid
                        
                        # Performance Math
                        grid_spacing = (upper_price - lower_price) / grid_count
                        strat["df"].loc[:, 'price_change'] = strat["df"]['Close'].diff().abs()
                        total_movement = strat["df"]['price_change'].sum()
                        estimated_crosses = int(total_movement / grid_spacing) if grid_spacing > 0 else 0
                        
                        # Append to results
                        results.append({"Coin": coin, "Strategy": strat["Type"], "Price": round(current_price, 4), "Lower": round(lower_price, 4), "Upper": round(upper_price, 4), "Grids": grid_count, "Req USDT": round(req_capital, 0), "Crosses": estimated_crosses})
                        
                progress_bar.progress((i + 1) / len(candidates))
                
            st.markdown("### 🏆 3-Tier Final Ranking")
            if results:
                final_df = pd.DataFrame(results)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No valid data could be calculated.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import requests
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Spot Grid Assistant", layout="wide")

BINANCE_EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
BINANCE_24HR_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

STABLECOIN_BLACKLIST = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", "USDT", "PAX", "USD1", "RLUSD", "PYUSD", "USDE", "USDS", "USDD", "GUSD", "LUSD", "FRAX", "USDJ", "USDB", "DEUSD", "SUSD", "EUSD", "CUSD", "EURS", "TRY", "BRL", "BIDR", "U"}

# --- BACKEND ENGINE ---

@st.cache_data(ttl=300)
def fetch_spot_universe():
    try:
        response = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = []
        for s in data['symbols']:
            base_asset = s.get('baseAsset', '').upper()
            quote_asset = s.get('quoteAsset', '').upper()
            is_stablecoin = base_asset in STABLECOIN_BLACKLIST or "USD" in base_asset or "EUR" in base_asset
            if s.get('status') == 'TRADING' and quote_asset == 'USDT' and s.get('isSpotTradingAllowed', True) and not is_stablecoin:
                symbols.append(s['symbol'])
        return symbols
    except Exception as e:
        return []

@st.cache_data(ttl=300)
def fetch_24h_data():
    try:
        response = requests.get(BINANCE_24HR_URL, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        numeric_cols = ['lastPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'priceChangePercent']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']]
    except Exception as e:
        return pd.DataFrame()

def fetch_klines(symbol, interval="1h", limit=336): # Increased to 14 days (336 hours)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if response.status_code == 200:
            columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NumTrades', 'TBB', 'TBQ', 'Ignore']
            df = pd.DataFrame(response.json(), columns=columns)
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col])
            return df
    except Exception:
        pass
    return pd.DataFrame()

# --- FRONTEND ---

def main():
    st.title("⚡ Spot Grid Assistant V2")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Data Freshness:** {current_time} UTC | Engine: Operational")
    
    st.markdown("### 1. Market Universe")
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe..."):
            universe = fetch_spot_universe()
            market_data = fetch_24h_data()
            if not market_data.empty and universe:
                valid_markets = market_data[market_data['symbol'].isin(universe)]
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(5) # Reduced to top 5 to keep mobile view clean
                st.success(f"Success: {len(valid_markets)} USDT pairs validated. Processing Top 5.")
                st.dataframe(top_candidates[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']], use_container_width=True)
                st.session_state['candidates'] = top_candidates['symbol'].tolist()
                
    if 'candidates' in st.session_state:
        st.markdown("### 2. Strategy Evaluator (Phases 3-9)")
        if st.button("Generate 3-Tier Strategies"):
            results = []
            progress_bar = st.progress(0)
            candidates = st.session_state['candidates']
            
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=336)
                if not df_klines.empty:
                    current_price = df_klines['Close'].iloc[-1]
                    
                    # Define the 3 time horizons for simulation
                    df_3d = df_klines.tail(72)  # Tight
                    df_7d = df_klines.tail(168) # Moderate
                    df_14d = df_klines.tail(336) # Wide
                    
                    strategies = [
                        {"Type": "Tight (3D)", "df": df_3d, "Target_Pct": 0.6},
                        {"Type": "Mod (7D)", "df": df_7d, "Target_Pct": 1.0},
                        {"Type": "Wide (14D)", "df": df_14d, "Target_Pct": 1.5}
                    ]
                    
                    for strat in strategies:
                        # Range Math
                        high_max = strat["df"]['High'].max()
                        low_min = strat["df"]['Low'].min()
                        upper_price = high_max * 1.01 # 1% safety buffer
                        lower_price = low_min * 0.99
                        
                        # Simulation: Calculate grids based on target profit step
                        range_width_pct = ((upper_price - lower_price) / lower_price) * 100
                        ideal_grids = int(range_width_pct / strat["Target_Pct"])
                        grid_count = max(5, min(ideal_grids, 100)) # Keep within Binance limits
                        
                        # Feasibility Math: Calculate required capital
                        req_capital = grid_count * 15.0 # $15 minimum per grid
                        
                        # Performance Math
                        grid_spacing = (upper_price - lower_price) / grid_count
                        strat["df"].loc[:, 'price_change'] = strat["df"]['Close'].diff().abs()
                        total_movement = strat["df"]['price_change'].sum()
                        estimated_crosses = int(total_movement / grid_spacing) if grid_spacing > 0 else 0
                        
                        # Append to results
                        results.append({"Coin": coin, "Strategy": strat["Type"], "Price": round(current_price, 4), "Lower": round(lower_price, 4), "Upper": round(upper_price, 4), "Grids": grid_count, "Req USDT": round(req_capital, 0), "Crosses": estimated_crosses})
                        
                progress_bar.progress((i + 1) / len(candidates))
                
            st.markdown("### 🏆 3-Tier Final Ranking")
            if results:
                final_df = pd.DataFrame(results)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No valid data could be calculated.")

if __name__ == "__main__":
    main()
    import streamlit as st
import pandas as pd
import requests
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Spot Grid Assistant", layout="wide")

BINANCE_EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
BINANCE_24HR_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

STABLECOIN_BLACKLIST = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", "USDT", "PAX", "USD1", "RLUSD", "PYUSD", "USDE", "USDS", "USDD", "GUSD", "LUSD", "FRAX", "USDJ", "USDB", "DEUSD", "SUSD", "EUSD", "CUSD", "EURS", "TRY", "BRL", "BIDR", "U"}

# --- BACKEND ENGINE ---

@st.cache_data(ttl=300)
def fetch_spot_universe():
    try:
        response = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = []
        for s in data['symbols']:
            base_asset = s.get('baseAsset', '').upper()
            quote_asset = s.get('quoteAsset', '').upper()
            is_stablecoin = base_asset in STABLECOIN_BLACKLIST or "USD" in base_asset or "EUR" in base_asset
            if s.get('status') == 'TRADING' and quote_asset == 'USDT' and s.get('isSpotTradingAllowed', True) and not is_stablecoin:
                symbols.append(s['symbol'])
        return symbols
    except Exception as e:
        return []

@st.cache_data(ttl=300)
def fetch_24h_data():
    try:
        response = requests.get(BINANCE_24HR_URL, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        numeric_cols = ['lastPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'priceChangePercent']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']]
    except Exception as e:
        return pd.DataFrame()

def fetch_klines(symbol, interval="1h", limit=336): # Increased to 14 days (336 hours)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if response.status_code == 200:
            columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NumTrades', 'TBB', 'TBQ', 'Ignore']
            df = pd.DataFrame(response.json(), columns=columns)
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col])
            return df
    except Exception:
        pass
    return pd.DataFrame()

# --- FRONTEND ---

def main():
    st.title("⚡ Spot Grid Assistant V2")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Data Freshness:** {current_time} UTC | Engine: Operational")
    
    st.markdown("### 1. Market Universe")
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe..."):
            universe = fetch_spot_universe()
            market_data = fetch_24h_data()
            if not market_data.empty and universe:
                valid_markets = market_data[market_data['symbol'].isin(universe)]
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(5) # Reduced to top 5 to keep mobile view clean
                st.success(f"Success: {len(valid_markets)} USDT pairs validated. Processing Top 5.")
                st.dataframe(top_candidates[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']], use_container_width=True)
                st.session_state['candidates'] = top_candidates['symbol'].tolist()
                
    if 'candidates' in st.session_state:
        st.markdown("### 2. Strategy Evaluator (Phases 3-9)")
        if st.button("Generate 3-Tier Strategies"):
            results = []
            progress_bar = st.progress(0)
            candidates = st.session_state['candidates']
            
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=336)
                if not df_klines.empty:
                    current_price = df_klines['Close'].iloc[-1]
                    
                    # Define the 3 time horizons for simulation
                    df_3d = df_klines.tail(72)  # Tight
                    df_7d = df_klines.tail(168) # Moderate
                    df_14d = df_klines.tail(336) # Wide
                    
                    strategies = [
                        {"Type": "Tight (3D)", "df": df_3d, "Target_Pct": 0.6},
                        {"Type": "Mod (7D)", "df": df_7d, "Target_Pct": 1.0},
                        {"Type": "Wide (14D)", "df": df_14d, "Target_Pct": 1.5}
                    ]
                    
                    for strat in strategies:
                        # Range Math
                        high_max = strat["df"]['High'].max()
                        low_min = strat["df"]['Low'].min()
                        upper_price = high_max * 1.01 # 1% safety buffer
                        lower_price = low_min * 0.99
                        
                        # Simulation: Calculate grids based on target profit step
                        range_width_pct = ((upper_price - lower_price) / lower_price) * 100
                        ideal_grids = int(range_width_pct / strat["Target_Pct"])
                        grid_count = max(5, min(ideal_grids, 100)) # Keep within Binance limits
                        
                        # Feasibility Math: Calculate required capital
                        req_capital = grid_count * 15.0 # $15 minimum per grid
                        
                        # Performance Math
                        grid_spacing = (upper_price - lower_price) / grid_count
                        strat["df"].loc[:, 'price_change'] = strat["df"]['Close'].diff().abs()
                        total_movement = strat["df"]['price_change'].sum()
                        estimated_crosses = int(total_movement / grid_spacing) if grid_spacing > 0 else 0
                        
                        # Append to results
                        results.append({"Coin": coin, "Strategy": strat["Type"], "Price": round(current_price, 4), "Lower": round(lower_price, 4), "Upper": round(upper_price, 4), "Grids": grid_count, "Req USDT": round(req_capital, 0), "Crosses": estimated_crosses})
                        
                progress_bar.progress((i + 1) / len(candidates))
                
            st.markdown("### 🏆 3-Tier Final Ranking")
            if results:
                final_df = pd.DataFrame(results)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No valid data could be calculated.")

if __name__ == "__main__":
    main()
    import streamlit as st
import pandas as pd
import requests
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Spot Grid Assistant", layout="wide")

BINANCE_EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
BINANCE_24HR_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

STABLECOIN_BLACKLIST = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", "USDT", "PAX", "USD1", "RLUSD", "PYUSD", "USDE", "USDS", "USDD", "GUSD", "LUSD", "FRAX", "USDJ", "USDB", "DEUSD", "SUSD", "EUSD", "CUSD", "EURS", "TRY", "BRL", "BIDR", "U"}

# --- BACKEND ENGINE ---

@st.cache_data(ttl=300)
def fetch_spot_universe():
    try:
        response = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = []
        for s in data['symbols']:
            base_asset = s.get('baseAsset', '').upper()
            quote_asset = s.get('quoteAsset', '').upper()
            is_stablecoin = base_asset in STABLECOIN_BLACKLIST or "USD" in base_asset or "EUR" in base_asset
            if s.get('status') == 'TRADING' and quote_asset == 'USDT' and s.get('isSpotTradingAllowed', True) and not is_stablecoin:
                symbols.append(s['symbol'])
        return symbols
    except Exception as e:
        return []

@st.cache_data(ttl=300)
def fetch_24h_data():
    try:
        response = requests.get(BINANCE_24HR_URL, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        numeric_cols = ['lastPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'priceChangePercent']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']]
    except Exception as e:
        return pd.DataFrame()

def fetch_klines(symbol, interval="1h", limit=336): # Increased to 14 days (336 hours)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if response.status_code == 200:
            columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NumTrades', 'TBB', 'TBQ', 'Ignore']
            df = pd.DataFrame(response.json(), columns=columns)
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col])
            return df
    except Exception:
        pass
    return pd.DataFrame()

# --- FRONTEND ---

def main():
    st.title("⚡ Spot Grid Assistant V2")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Data Freshness:** {current_time} UTC | Engine: Operational")
    
    st.markdown("### 1. Market Universe")
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe..."):
            universe = fetch_spot_universe()
            market_data = fetch_24h_data()
            if not market_data.empty and universe:
                valid_markets = market_data[market_data['symbol'].isin(universe)]
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(5) # Reduced to top 5 to keep mobile view clean
                st.success(f"Success: {len(valid_markets)} USDT pairs validated. Processing Top 5.")
                st.dataframe(top_candidates[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']], use_container_width=True)
                st.session_state['candidates'] = top_candidates['symbol'].tolist()
                
    if 'candidates' in st.session_state:
        st.markdown("### 2. Strategy Evaluator (Phases 3-9)")
        if st.button("Generate 3-Tier Strategies"):
            results = []
            progress_bar = st.progress(0)
            candidates = st.session_state['candidates']
            
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=336)
                if not df_klines.empty:
                    current_price = df_klines['Close'].iloc[-1]
                    
                    # Define the 3 time horizons for simulation
                    df_3d = df_klines.tail(72)  # Tight
                    df_7d = df_klines.tail(168) # Moderate
                    df_14d = df_klines.tail(336) # Wide
                    
                    strategies = [
                        {"Type": "Tight (3D)", "df": df_3d, "Target_Pct": 0.6},
                        {"Type": "Mod (7D)", "df": df_7d, "Target_Pct": 1.0},
                        {"Type": "Wide (14D)", "df": df_14d, "Target_Pct": 1.5}
                    ]
                    
                    for strat in strategies:
                        # Range Math
                        high_max = strat["df"]['High'].max()
                        low_min = strat["df"]['Low'].min()
                        upper_price = high_max * 1.01 # 1% safety buffer
                        lower_price = low_min * 0.99
                        
                        # Simulation: Calculate grids based on target profit step
                        range_width_pct = ((upper_price - lower_price) / lower_price) * 100
                        ideal_grids = int(range_width_pct / strat["Target_Pct"])
                        grid_count = max(5, min(ideal_grids, 100)) # Keep within Binance limits
                        
                        # Feasibility Math: Calculate required capital
                        req_capital = grid_count * 15.0 # $15 minimum per grid
                        
                        # Performance Math
                        grid_spacing = (upper_price - lower_price) / grid_count
                        strat["df"].loc[:, 'price_change'] = strat["df"]['Close'].diff().abs()
                        total_movement = strat["df"]['price_change'].sum()
                        estimated_crosses = int(total_movement / grid_spacing) if grid_spacing > 0 else 0
                        
                        # Append to results
                        results.append({"Coin": coin, "Strategy": strat["Type"], "Price": round(current_price, 4), "Lower": round(lower_price, 4), "Upper": round(upper_price, 4), "Grids": grid_count, "Req USDT": round(req_capital, 0), "Crosses": estimated_crosses})
                        
                progress_bar.progress((i + 1) / len(candidates))
                
            st.markdown("### 🏆 3-Tier Final Ranking")
            if results:
                final_df = pd.DataFrame(results)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No valid data could be calculated.")

if __name__ == "__main__":
    main()
    grid_count = max(5, min(max_feasible_grids, 50))
                    
                    if grid_count > 0:
                        grid_spacing = (upper_price - lower_price) / grid_count
                        profit_per_step_pct = (grid_spacing / lower_price) * 100 
                    else:
                        grid_spacing = 0.01
                        profit_per_step_pct = 0.0
                        
                    df_klines['price_change'] = df_klines['Close'].diff().abs()
                    total_movement = df_klines['price_change'].sum()
                    estimated_crosses = int(total_movement / grid_spacing) if grid_spacing > 0 else 0
                    
                    estimated_profit_usdt = estimated_crosses * (allocated_capital / max(1, grid_count)) * (profit_per_step_pct / 100)
                    est_roi_pct = (estimated_profit_usdt / wallet_balance) * 100
                    
                    results.append({"Coin": coin, "Price": round(current_price, 4), "Lower": round(lower_price, 4), "Upper": round(upper_price, 4), "Grids": grid_count, "Crosses": estimated_crosses, "Est ROI %": round(est_roi_pct, 2)})
                progress_bar.progress((i + 1) / len(candidates))
                
            st.markdown("### 🏆 Final Ranking")
            if results:
                final_df = pd.DataFrame(results).sort_values(by="Est ROI %", ascending=False).reset_index(drop=True)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No valid data could be calculated for the candidates.")

if __name__ == "__main__":
    main()
  
