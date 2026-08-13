import streamlit as st
import pandas as pd
import requests
import datetime

# --- CONFIGURATION (The "Control Panel") ---
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

def fetch_klines(symbol, interval="1h", limit=336):
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
    st.info(f"**Data Freshness:** {current_time} UTC")
    
    # --- INTERACTIVE SETTINGS MENU ---
    st.markdown("### 1. Strategy Parameters")
    with st.expander("⚙️ Tap to Edit Trading Rules", expanded=False):
        fee_choice = st.selectbox("Trading Fee Tier (Round-Trip)", ["Standard (0.20%)", "BNB Discount (0.15%)", "Zero Fee (0.00%)"])
        
        # Determine fee float based on selection
        if "0.20" in fee_choice:
            round_trip_fee_pct = 0.20
        elif "0.15" in fee_choice:
            round_trip_fee_pct = 0.15
        else:
            round_trip_fee_pct = 0.00
            
        min_grid_inv = st.number_input("Minimum USDT per Grid Step", min_value=5.0, value=15.0, step=1.0)
        
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
        with st.spinner("Fetching USDT Market Universe..."):
            universe = fetch_spot_universe()
            market_data = fetch_24h_data()
            if not market_data.empty and universe:
                valid_markets = market_data[market_data['symbol'].isin(universe)]
                top_candidates = valid_markets.sort_values(by='quoteVolume', ascending=False).head(5)
                st.success(f"Success: {len(valid_markets)} USDT pairs validated.")
                st.dataframe(top_candidates[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']], use_container_width=True)
                st.session_state['candidates'] = top_candidates['symbol'].tolist()
                
    if 'candidates' in st.session_state:
        st.markdown("### 3. Strategy Evaluator")
        if st.button("Generate 3-Tier Strategies"):
            results = []
            progress_bar = st.progress(0)
            candidates = st.session_state['candidates']
            
            for i, coin in enumerate(candidates):
                df_klines = fetch_klines(coin, interval="1h", limit=336)
                if not df_klines.empty:
                    current_price = df_klines['Close'].iloc[-1]
                    
                    df_3d = df_klines.tail(72)
                    df_7d = df_klines.tail(168)
                    df_14d = df_klines.tail(336)
                    
                    # Using the dynamic targets from the UI
                    strategies = [
                        {"Type": "Tight (3D)", "df": df_3d, "Target_Pct": tight_pct},
                        {"Type": "Mod (7D)", "df": df_7d, "Target_Pct": mod_pct},
                        {"Type": "Wide (14D)", "df": df_14d, "Target_Pct": wide_pct}
                    ]
                    
                    for strat in strategies:
                        high_max = strat["df"]['High'].max()
                        low_min = strat["df"]['Low'].min()
                        upper_price = high_max * 1.01
                        lower_price = low_min * 0.99
                        
                        range_width_pct = ((upper_price - lower_price) / lower_price) * 100
                        ideal_grids = int(range_width_pct / strat["Target_Pct"])
                        grid_count = max(5, min(ideal_grids, 150))
                        
                        # Using the dynamic minimum investment from the UI
                        req_capital = grid_count * min_grid_inv
                        
                        grid_spacing = (upper_price - lower_price) / grid_count
                        strat["df"].loc[:, 'price_change'] = strat["df"]['Close'].diff().abs()
                        total_movement = strat["df"]['price_change'].sum()
                        estimated_crosses = int(total_movement / grid_spacing) if grid_spacing > 0 else 0
                        
                        # Apply dynamic fee selection
                        net_profit_per_cross_pct = strat["Target_Pct"] - round_trip_fee_pct
                        if net_profit_per_cross_pct < 0:
                            net_profit_per_cross_pct = 0
                            
                        net_profit_usdt = estimated_crosses * min_grid_inv * (net_profit_per_cross_pct / 100)
                        net_roi_pct = (net_profit_usdt / req_capital) * 100 if req_capital > 0 else 0
                        
                        results.append({"Coin": coin, "Strategy": strat["Type"], "Price": round(current_price, 4), "Lower": round(lower_price, 4), "Upper": round(upper_price, 4), "Grids": grid_count, "Req USDT": round(req_capital, 0), "Crosses": estimated_crosses, "Net ROI %": round(net_roi_pct, 2)})
                        
                progress_bar.progress((i + 1) / len(candidates))
                
            st.markdown("### 🏆 3-Tier Final Ranking (Fee-Adjusted)")
            if results:
                final_df = pd.DataFrame(results).sort_values(by="Net ROI %", ascending=False).reset_index(drop=True)
                st.dataframe(final_df, use_container_width=True)

if __name__ == "__main__":
    main()
    
