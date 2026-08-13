import streamlit as st
import pandas as pd
import requests
import datetime

# --- CONFIGURATION (The "Control Panel") ---
st.set_page_config(page_title="Spot Grid Assistant", layout="wide")

# Using the data-vision endpoints to avoid US geo-blocking issues on cloud servers
BINANCE_EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
BINANCE_24HR_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

# Expanded list of stablecoins and fiat tokens to exclude
STABLECOIN_BLACKLIST = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", 
    "USDT", "PAX", "USD1", "RLUSD", "PYUSD", "USDE", "USDS", "USDD", 
    "GUSD", "LUSD", "FRAX", "USDJ", "USDB", "DEUSD", "SUSD", "EUSD", 
    "CUSD", "EURS", "TRY", "BRL", "BIDR", "U"
}

# --- BACKEND ENGINE (Data GuardDog & API Pipeline) ---

@st.cache_data(ttl=300) # Caches data for 5 minutes (The "Clock Buffer")
def fetch_spot_universe():
    """Fetches and filters Binance exchange info for USDT spot pairs, excluding stablecoins."""
    try:
        response = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        symbols = []
        for s in data['symbols']:
            base_asset = s.get('baseAsset', '').upper()
            quote_asset = s.get('quoteAsset', '').upper()
            
            # Smart stablecoin filter:
            # 1. Must not be in the blacklist
            # 2. Base asset must not contain 'USD' (catches new/unlisted stablecoins)
            is_stablecoin = (
                base_asset in STABLECOIN_BLACKLIST or 
                "USD" in base_asset or 
                "EUR" in base_asset
            )
            
            # Validation: Must be TRADING, USDT quote, spot allowed, AND not a stablecoin
            if (s.get('status') == 'TRADING' and 
                quote_asset == 'USDT' and 
                s.get('isSpotTradingAllowed', True) and 
                not is_stablecoin):
                
                symbols.append(s['symbol'])
        return symbols
    except Exception as e:
        st.error(f"Fallback Plan Triggered: Failed to fetch exchange info. Error: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_24h_data():
    """Fetches 24h ticker data and maps it to our universe."""
    try:
        response = requests.get(BINANCE_24HR_URL, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        
        # Convert numeric columns safely
        numeric_cols = ['lastPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'priceChangePercent']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return df[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']]
    except Exception as e:
        st.error(f"Guard Dog Alert: Market Data API failed. Error: {e}")
        return pd.DataFrame()

def fetch_klines(symbol, interval="1h", limit=500):
    """The Conveyor Belt: Fetches historical klines for a single coin at a time."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if response.status_code == 200:
            columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NumTrades', 'TBB', 'TBQ', 'Ignore']
            df = pd.DataFrame(response.json(), columns=columns)
            # Ensure price columns are numeric for calculation
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col])
            return df
    except Exception:
        pass
    return pd.DataFrame()

# --- FRONTEND (The Mobile Dashboard) ---

def main():
    st.title("⚡ Spot Grid Assistant V2")
    
    # Audit Stamp / Flight Log
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Audit Stamp:** Data retrieved at {current_time} UTC | Engine: Operational")

    st.markdown("### 1. Account Configuration")
    col1, col2 = st.columns(2)
    with col1:
        wallet_balance = st.number_input("Available USDT", min_value=0.0, value=1000.0)
    with col2:
        risk_pct = st.slider("Risk per Grid (%)", 1, 10, 5)

    st.markdown("### 2. Market Universe")
    
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe..."):
            universe = fetch_spot_universe()
            market_data = fetch_24h_data()
            
            if not market_data.empty and universe:
                # Vectorized Filtering (Replaces VBA Loops)
                valid_markets = market_data[market_data['symbol'].isin(universe)]
                
                # Simple filter: Top 10 by Quote Volume
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(10)
                
                st.success(f"Success: {len(valid_markets)} USDT pairs validated.")
                st.dataframe(top_candidates, use_container_width=True)
                
                # Save to session state for the next phase
                st.session_state['candidates'] = top_candidates['symbol'].tolist()

    if 'candidates' in st.session_state:
        st.markdown("### 3. Strategy Evaluator (Conveyor Belt)")
        if st.button("Run Backtest & Ranking (Phases 3-9)"):
            results = []
            progress_bar = st.progress(0)
            
            # Block-by-Coin Processing to save RAM
            candidates = st.session_state['candidates']
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=168) # 7 days of 1h candles
                
                if not df_klines.empty:
                    current_price = df_klines['Close'].iloc[-1]
                    
                    # --- PHASE 4 & 5: INDICATORS & RANGE GENERATION ---
                    # Calculate 7-day High/Low bounds with a 2% safety buffer
                    high_max = df_klines['High'].max()
                    low_min = df_klines['Low'].min()
                    upper_price = high_max * 1.02
                    lower_price = low_min * 0.98
                    
                    # --- PHASE 6: CAPITAL FEASIBILITY ---
                    # Calculate Grid Count based on wallet balance and risk percentage
                    # Assuming a $15 minimum trade size per grid step to satisfy Binance API rules
                    allocated_capital = wallet_balance * (risk_pct / 100)
                    max_feasible_grids = int(allocated_capital / 15)
                    
                    # Bound the grid count to a reasonable range (minimum 5, maximum 50)
                    grid_count = max(5, min(max_feasible_grids, 50))
                    
                    # Calculate grid spacing and expected profit per step
                    if grid_count > 0:
                        grid_spacing = (upper_price - lower_price) / grid_count
                        profit_per_step_pct = (grid_spacing / lower_price) * 100 
                    else:
                        grid_spacing = 0.01
                        profit_per_step_pct = 0.0
                    
                    # --- PHASE 7 & 8: VECTORIZED BACKTEST & SIMULATION ---
                    # Calculate total absolute price movement to estimate grid crosses
                    df_klines['price_change'] = df_klines['Close'].diff().abs()
                    total_movement = df_klines['price_change'].sum()
                    
                    # Estimate crosses: Total Movement / Grid Spacing
                    estimated_crosses = int(total_movement / grid_spacing) if grid_spacing > 0 else 0
                    
                    # Estimate 7-day ROI based on crosses and capital per grid
                    estimated_profit_usdt = estimated_crosses * (allocated_capital / max(1, grid_count)) * (profit_per_step_pct / 100)
                    est_roi_pct = (estimated_profit_usdt / wallet_balance) * 100
                    
                    # --- PHASE 9: FINAL RANKING OUTPUT ---
                    results.append({
                        "Coin": coin,
                        "Price": round(current_price, 4),
                        "Lower": round(lower_price, 4),
                        "Upper": round(upper_price, 4),
                        "Grids": grid_count,
                        "Crosses": estimated_crosses,
                        "Est ROI %": round(est_roi_pct, 2)
                    })
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(candidates))
            
            st.markdown("### 🏆 Final Ranking")
            
            # Sort the final results by Estimated ROI (Highest to Lowest)
            if results:
                final_df = pd.DataFrame(results)
                final_df = final_df.sort_values(by="Est ROI %", ascending=False).reset_index(drop=True)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No valid data could be calculated for the candidates.")

if __name__ == "__main__":
    main()
        return symbols
    except Exception as e:
        st.error(f"Fallback Plan Triggered: Failed to fetch exchange info. Error: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_24h_data():
    """Fetches 24h ticker data and maps it to our universe."""
    try:
        response = requests.get(BINANCE_24HR_URL, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        
        # Convert numeric columns safely
        numeric_cols = ['lastPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'priceChangePercent']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return df[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']]
    except Exception as e:
        st.error(f"Guard Dog Alert: Market Data API failed. Error: {e}")
        return pd.DataFrame()

def fetch_klines(symbol, interval="1h", limit=500):
    """The Conveyor Belt: Fetches historical klines for a single coin at a time."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if response.status_code == 200:
            columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NumTrades', 'TBB', 'TBQ', 'Ignore']
            df = pd.DataFrame(response.json(), columns=columns)
            # Ensure price columns are numeric for calculation
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col])
            return df
    except Exception:
        pass
    return pd.DataFrame()

# --- FRONTEND (The Mobile Dashboard) ---

def main():
    st.title("⚡ Spot Grid Assistant V2")
    
    # Audit Stamp / Flight Log
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Audit Stamp:** Data retrieved at {current_time} UTC | Engine: Operational")

    st.markdown("### 1. Account Configuration")
    col1, col2 = st.columns(2)
    with col1:
        wallet_balance = st.number_input("Available USDT", min_value=0.0, value=1000.0)
    with col2:
        risk_pct = st.slider("Risk per Grid (%)", 1, 10, 5)

    st.markdown("### 2. Market Universe")
    
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe..."):
            universe = fetch_spot_universe()
            market_data = fetch_24h_data()
            
            if not market_data.empty and universe:
                # Vectorized Filtering (Replaces VBA Loops)
                valid_markets = market_data[market_data['symbol'].isin(universe)]
                
                # Simple filter: Top 10 by Quote Volume
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(10)
                
                st.success(f"Success: {len(valid_markets)} USDT pairs validated.")
                st.dataframe(top_candidates, use_container_width=True)
                
                # Save to session state for the next phase
                st.session_state['candidates'] = top_candidates['symbol'].tolist()

    if 'candidates' in st.session_state:
        st.markdown("### 3. Strategy Evaluator (Conveyor Belt)")
        if st.button("Run Backtest & Ranking (Phases 3-9)"):
            results = []
            progress_bar = st.progress(0)
            
            # Block-by-Coin Processing to save RAM
            candidates = st.session_state['candidates']
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=168) # 7 days of 1h candles
                
                if not df_klines.empty:
                    current_price = df_klines['Close'].iloc[-1]
                    
                    # --- PHASE 4 & 5: INDICATORS & RANGE GENERATION ---
                    # Calculate 7-day High/Low bounds with a 2% safety buffer
                    high_max = df_klines['High'].max()
                    low_min = df_klines['Low'].min()
                    upper_price = high_max * 1.02
                    lower_price = low_min * 0.98
                    
                    # --- PHASE 6: CAPITAL FEASIBILITY ---
                    # Calculate Grid Count based on wallet balance and risk percentage
                    # Assuming a $15 minimum trade size per grid step to satisfy Binance API rules
                    allocated_capital = wallet_balance * (risk_pct / 100)
                    max_feasible_grids = int(allocated_capital / 15)
                    
                    # Bound the grid count to a reasonable range (minimum 5, maximum 50)
                    grid_count = max(5, min(max_feasible_grids, 50))
                    
                    # Calculate grid spacing and expected profit per step
                    if grid_count > 0:
                        grid_spacing = (upper_price - lower_price) / grid_count
                        profit_per_step_pct = (grid_spacing / lower_price) * 100 
                    else:
                        grid_spacing = 0.01
                        profit_per_step_pct = 0.0
                    
                    # --- PHASE 7 & 8: VECTORIZED BACKTEST & SIMULATION ---
                    # Calculate total absolute price movement to estimate grid crosses
                    df_klines['price_change'] = df_klines['Close'].diff().abs()
                    total_movement = df_klines['price_change'].sum()
                    
                    # Estimate crosses: Total Movement / Grid Spacing
                    estimated_crosses = int(total_movement / grid_spacing) if grid_spacing > 0 else 0
                    
                    # Estimate 7-day ROI based on crosses and capital per grid
                    estimated_profit_usdt = estimated_crosses * (allocated_capital / max(1, grid_count)) * (profit_per_step_pct / 100)
                    est_roi_pct = (estimated_profit_usdt / wallet_balance) * 100
                    
                    # --- PHASE 9: FINAL RANKING OUTPUT ---
                    results.append({
                        "Coin": coin,
                        "Price": round(current_price, 4),
                        "Lower": round(lower_price, 4),
                        "Upper": round(upper_price, 4),
                        "Grids": grid_count,
                        "Crosses": estimated_crosses,
                        "Est ROI %": round(est_roi_pct, 2)
                    })
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(candidates))
            
            st.markdown("### 🏆 Final Ranking")
            
            # Sort the final results by Estimated ROI (Highest to Lowest)
            if results:
                final_df = pd.DataFrame(results)
                final_df = final_df.sort_values(by="Est ROI %", ascending=False).reset_index(drop=True)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No valid data could be calculated for the candidates.")

if __name__ == "__main__":
    main()
            # Smart stablecoin filter:
            # 1. Must not be in the blacklist
            # 2. Base asset must not contain 'USD' (catches new/unlisted stablecoins)
            is_stablecoin = (
                base_asset in STABLECOIN_BLACKLIST or 
                "USD" in base_asset or 
                "EUR" in base_asset
            )
            
            # Validation: Must be TRADING, USDT quote, spot allowed, AND not a stablecoin
            if (s.get('status') == 'TRADING' and 
                quote_asset == 'USDT' and 
                s.get('isSpotTradingAllowed', True) and 
                not is_stablecoin):
                
                symbols.append(s['symbol'])
        return symbols
    except Exception as e:
        st.error(f"Fallback Plan Triggered: Failed to fetch exchange info. Error: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_24h_data():
    """Fetches 24h ticker data and maps it to our universe."""
    try:
        response = requests.get(BINANCE_24HR_URL, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        
        # Convert numeric columns safely
        numeric_cols = ['lastPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'priceChangePercent']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return df[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']]
    except Exception as e:
        st.error(f"Guard Dog Alert: Market Data API failed. Error: {e}")
        return pd.DataFrame()

def fetch_klines(symbol, interval="1h", limit=500):
    """The Conveyor Belt: Fetches historical klines for a single coin at a time."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if response.status_code == 200:
            columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NumTrades', 'TBB', 'TBQ', 'Ignore']
            df = pd.DataFrame(response.json(), columns=columns)
            df['Close'] = pd.to_numeric(df['Close'])
            return df
    except Exception:
        pass
    return pd.DataFrame()

# --- FRONTEND (The Mobile Dashboard) ---

def main():
    st.title("⚡ Spot Grid Assistant V2")
    
    # Audit Stamp / Flight Log
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Audit Stamp:** Data retrieved at {current_time} UTC | Engine: Operational")

    st.markdown("### 1. Account Configuration")
    col1, col2 = st.columns(2)
    with col1:
        wallet_balance = st.number_input("Available USDT", min_value=0.0, value=1000.0)
    with col2:
        risk_pct = st.slider("Risk per Grid (%)", 1, 10, 5)

    st.markdown("### 2. Market Universe")
    
    if st.button("Scan Binance (Phase 1 & 2)"):
        with st.spinner("Fetching USDT Market Universe..."):
            universe = fetch_spot_universe()
            market_data = fetch_24h_data()
            
            if not market_data.empty and universe:
                # Vectorized Filtering (Replaces VBA Loops)
                valid_markets = market_data[market_data['symbol'].isin(universe)]
                
                # Simple filter: Top 10 by Quote Volume
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(10)
                
                st.success(f"Success: {len(valid_markets)} USDT pairs validated.")
                st.dataframe(top_candidates, use_container_width=True)
                
                # Save to session state for the next phase
                st.session_state['candidates'] = top_candidates['symbol'].tolist()

    if 'candidates' in st.session_state:
        st.markdown("### 3. Strategy Evaluator (Conveyor Belt)")
        if st.button("Run Backtest & Ranking (Phases 3-9)"):
            results = []
            progress_bar = st.progress(0)
            
            # Block-by-Coin Processing to save RAM
            candidates = st.session_state['candidates']
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=168) # 7 days of 1h candles
                
                if not df_klines.empty:
                    current_price = df_klines['Close'].iloc[-1]
                    
                    results.append({
                        "Coin": coin,
                        "Current Price": current_price,
                        "Status": 1 # Binary status indicator
                    })
                
                progress_bar.progress((i + 1) / len(candidates))
            
            st.markdown("### 🏆 Final Ranking")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

if __name__ == "__main__":
    main()
