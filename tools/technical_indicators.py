# tools/technical_indicators.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# --- Helper Function to Standardize Price Data ---

def _convert_to_dataframe_and_standardize(
    price_data_list_of_dicts: List[Dict[str, Any]]
) -> Optional[pd.DataFrame]:
    """
    Converts a list of price data dictionaries into a Pandas DataFrame,
    standardizes column names to 'open', 'high', 'low', 'close', 'volume',
    and 'date', converts to numeric types, and sets 'date' as index.
    """
    if not price_data_list_of_dicts:
        return None

    df = pd.DataFrame(price_data_list_of_dicts)

    if df.empty:
        return None

    column_map = {
        'timestamp': 'date', 'Date': 'date', 'datetime': 'date',
        'open_price': 'open', 'Open': 'open',
        'high_price': 'high', 'High': 'high',
        'low_price': 'low', 'Low': 'low',
        'close_price': 'close', 'Close': 'close', 'adj close': 'close', 
        'adj_close': 'close', 'Adj Close': 'close',
        'Volume': 'volume', 'vol': 'volume'
    }
    
    df.rename(columns=lambda c: column_map.get(str(c).lower().replace(' ', '_'), str(c).lower().replace(' ', '_')), inplace=True)
    
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
        except Exception as e:
            print(f"Warning: Could not process 'date' column: {e}. Proceeding without date index.")
    else:
        print("Warning: 'date' column not found in price data. Calculations will use row order.")

    for col_name in ['open', 'high', 'low', 'close', 'volume']:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    if 'close' in df.columns:
        df.dropna(subset=['close'], inplace=True) 
    
    if df.empty:
        return None
        
    return df

# --- Indicator Calculation Functions (Manual Implementations) ---

def calculate_rsi(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates RSI (Relative Strength Index) manually."""
    period = 14
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "RSI calculation requires 'close' price data."}
    if len(df) < period + 1:
        return {"error": f"Insufficient data for RSI (need {period + 1} periods, got {len(df)})."}

    try:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs))
        
        latest_rsi = rsi_val.iloc[-1]
        if pd.isna(latest_rsi):
            if pd.isna(rs.iloc[-1]) and avg_loss.iloc[-1] == 0 and avg_gain.iloc[-1] > 0:
                latest_rsi = 100.0
            elif pd.isna(rs.iloc[-1]) and avg_loss.iloc[-1] == 0 and avg_gain.iloc[-1] == 0:
                latest_rsi = 50.0
            else:
                return {"error": "RSI calculation resulted in NaN."}
        
        signal = "neutral"
        if latest_rsi < 30: signal = "buy" # Oversold
        elif latest_rsi > 70: signal = "sell" # Overbought
        
        return {"rsi": round(latest_rsi, 2), "signal": signal}
    except Exception as e:
        return {"error": f"Error during manual RSI calculation: {str(e)}"}


def calculate_ema(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    period = 20
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "EMA calculation requires 'close' price data."}
    if len(df) < period:
        return {"error": f"Insufficient data for EMA (need {period} periods, got {len(df)})."}

    try:
        ema_series = df['close'].ewm(span=period, adjust=False, min_periods=period).mean()
        latest_ema = ema_series.iloc[-1]
        latest_close = df['close'].iloc[-1]

        if pd.isna(latest_ema):
            return {"error": "EMA calculation resulted in NaN."}

        signal = "neutral"
        # EMA crossover logic is more complex, requiring multiple EMAs or price vs EMA.
        # Simple signal: if price is above EMA, bullish tendency; below, bearish.
        if latest_close > latest_ema: signal = "buy_bias" # General bullish bias
        elif latest_close < latest_ema: signal = "sell_bias" # General bearish bias
        
        return {"ema": round(latest_ema, 2), "signal": signal, "note": "Signal is price vs EMA; for crossovers, use MACD or multiple EMAs."}
    except Exception as e:
        return {"error": f"Error during manual EMA calculation: {str(e)}"}


def calculate_bollinger_bands(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    period = 20
    std_dev_multiplier = 2.0
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "Bollinger Bands calculation requires 'close' price data."}
    if len(df) < period:
        return {"error": f"Insufficient data for Bollinger Bands (need {period} periods, got {len(df)})."}

    try:
        middle_band = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std_dev_multiplier)
        lower_band = middle_band - (std_dev * std_dev_multiplier)

        latest_close = df['close'].iloc[-1]
        lb = lower_band.iloc[-1]
        mb = middle_band.iloc[-1]
        ub = upper_band.iloc[-1]

        if pd.isna(lb) or pd.isna(mb) or pd.isna(ub):
            return {"error": "Bollinger Bands calculation resulted in NaN values."}

        signal = "neutral"
        if latest_close < lb: signal = "buy" # Price touched/crossed lower band
        elif latest_close > ub: signal = "sell" # Price touched/crossed upper band
        elif latest_close > mb and latest_close < ub : signal = "hold_bullish_range" # In upper half
        elif latest_close < mb and latest_close > lb : signal = "hold_bearish_range" # In lower half

        return {
            "lower_band": round(lb, 2),
            "middle_band": round(mb, 2),
            "upper_band": round(ub, 2),
            "signal": signal
        }
    except Exception as e:
        return {"error": f"Error during manual Bollinger Bands calculation: {str(e)}"}

def calculate_atr(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    period = 14
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "ATR calculation requires 'high', 'low', and 'close' price data."}
    if len(df) < period + 1 :
        return {"error": f"Insufficient data for ATR (need {period + 1} periods, got {len(df)})."}

    try:
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        atr_series = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        latest_atr = atr_series.iloc[-1]
        
        if pd.isna(latest_atr):
            return {"error": "ATR calculation failed or resulted in NaN."}
        # ATR itself doesn't give buy/sell signals but indicates volatility.
        return {"atr": round(latest_atr, 4), "signal": "volatility_measure", "note": "Higher ATR means higher volatility."}
    except Exception as e:
        return {"error": f"Error during manual ATR calculation: {str(e)}"}

def calculate_macd(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    fast_period = 12
    slow_period = 26
    signal_period = 9
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "MACD calculation requires 'close' price data."}
    if len(df) < slow_period + signal_period: # Heuristic
        return {"error": f"Insufficient data for MACD (need ~{slow_period + signal_period} periods, got {len(df)})."}

    try:
        ema_fast = df['close'].ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
        histogram = macd_line - signal_line

        ml = macd_line.iloc[-1]
        sl = signal_line.iloc[-1]
        hist = histogram.iloc[-1]
        prev_ml = macd_line.iloc[-2] if len(macd_line) > 1 else np.nan
        prev_sl = signal_line.iloc[-2] if len(signal_line) > 1 else np.nan


        if pd.isna(ml) or pd.isna(sl) or pd.isna(hist):
            return {"error": "MACD calculation resulted in NaN values."}

        signal = "neutral"
        # Bullish Crossover: MACD line crosses above Signal line
        if not pd.isna(prev_ml) and not pd.isna(prev_sl) and prev_ml <= prev_sl and ml > sl:
            signal = "buy"
        # Bearish Crossover: MACD line crosses below Signal line
        elif not pd.isna(prev_ml) and not pd.isna(prev_sl) and prev_ml >= prev_sl and ml < sl:
            signal = "sell"
        # General tendency based on histogram or MACD vs zero
        elif ml > 0 and sl > 0 and hist > 0: signal = "hold_bullish"
        elif ml < 0 and sl < 0 and hist < 0: signal = "hold_bearish"
        
        return {
            "macd_line": round(ml, 4),
            "histogram": round(hist, 4),
            "signal_line": round(sl, 4),
            "signal": signal
        }
    except Exception as e:
        return {"error": f"Error during manual MACD calculation: {str(e)}"}

def calculate_adx(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    di_len = 14
    adx_len = 14
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "ADX calculation requires 'high', 'low', and 'close' price data."}
    if len(df) < di_len + adx_len:
        return {"error": f"Insufficient data for ADX (need ~{di_len + adx_len} periods, got {len(df)})."}

    try:
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        atr = tr.ewm(com=di_len - 1, adjust=False, min_periods=di_len).mean()

        move_up = df['high'].diff()
        move_down = -df['low'].diff()

        plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=df.index)

        plus_di_smooth = plus_dm.ewm(com=di_len - 1, adjust=False, min_periods=di_len).mean()
        minus_di_smooth = minus_dm.ewm(com=di_len - 1, adjust=False, min_periods=di_len).mean()
        
        # Add small epsilon to atr to prevent division by zero if atr is exactly 0
        epsilon = 1e-9 
        plus_di = (plus_di_smooth / (atr + epsilon)) * 100
        minus_di = (minus_di_smooth / (atr + epsilon)) * 100
        
        # For DX, if (plus_di + minus_di) is zero, DX is conventionally 0 or 100 depending on definition.
        # Here, setting to 0 if sum is 0 to avoid NaN/Inf.
        dx_denominator = (plus_di + minus_di).replace(0, np.nan) # Avoid division by zero
        dx = (abs(plus_di - minus_di) / dx_denominator) * 100
        dx.fillna(0, inplace=True) # If plus_di + minus_di was 0, dx becomes 0.

        adx = dx.ewm(com=adx_len - 1, adjust=False, min_periods=adx_len).mean()
        
        latest_adx = adx.iloc[-1]
        latest_pdi = plus_di.iloc[-1]
        latest_mdi = minus_di.iloc[-1]

        if pd.isna(latest_adx) or pd.isna(latest_pdi) or pd.isna(latest_mdi):
             return {"error": "ADX calculation resulted in NaN values."}

        signal = "weak_trend_or_ranging"
        if latest_adx > 25:
            if latest_pdi > latest_mdi: signal = "strong_uptrend"
            else: signal = "strong_downtrend"
        elif latest_adx < 20:
            signal = "weak_trend_or_ranging"
        
        return {
            "adx": round(latest_adx, 2),
            "plus_di": round(latest_pdi, 2),
            "minus_di": round(latest_mdi, 2),
            "signal": signal
        }
    except Exception as e:
        return {"error": f"Error during manual ADX calculation: {str(e)}"}

def calculate_supertrend(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    atr_period = 10
    multiplier = 3.0
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Supertrend requires 'high', 'low', and 'close' price data."}
    if len(df) < atr_period + 1:
        return {"error": f"Insufficient data for Supertrend (need {atr_period + 1} periods, got {len(df)})."}

    try:
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        atr = tr.ewm(com=atr_period - 1, adjust=False, min_periods=atr_period).mean()

        basic_upper_band = (df['high'] + df['low']) / 2 + multiplier * atr
        basic_lower_band = (df['high'] + df['low']) / 2 - multiplier * atr

        supertrend = pd.Series(np.nan, index=df.index)
        final_upper_band = basic_upper_band.copy()
        final_lower_band = basic_lower_band.copy()
        
        # Initialize first supertrend value to avoid NaN issues in loop
        if len(df) > 0:
             supertrend.iloc[0] = final_upper_band.iloc[0] # Arbitrary start, will correct

        for i in range(1, len(df)):
            if basic_upper_band.iloc[i] < final_upper_band.iloc[i-1] or df['close'].iloc[i-1] > final_upper_band.iloc[i-1]:
                final_upper_band.iloc[i] = basic_upper_band.iloc[i]
            else:
                final_upper_band.iloc[i] = final_upper_band.iloc[i-1]

            if basic_lower_band.iloc[i] > final_lower_band.iloc[i-1] or df['close'].iloc[i-1] < final_lower_band.iloc[i-1]:
                final_lower_band.iloc[i] = basic_lower_band.iloc[i]
            else:
                final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
            
            # Supertrend Logic
            prev_supertrend = supertrend.iloc[i-1]
            if pd.isna(prev_supertrend) and i > 0: # If previous is NaN, use earlier valid one or re-initialize
                 # Fallback: if prior ST is NaN, re-evaluate based on current close vs bands
                if df['close'].iloc[i] > final_upper_band.iloc[i]: # Trend might be up
                    prev_supertrend = final_lower_band.iloc[i-1] # Assume prev was lower band
                else: # Trend might be down
                    prev_supertrend = final_upper_band.iloc[i-1] # Assume prev was upper band


            if prev_supertrend == final_upper_band.iloc[i-1]: # Previous trend was down
                if df['close'].iloc[i] <= final_upper_band.iloc[i]:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
                else: # Close crossed above upper band
                    supertrend.iloc[i] = final_lower_band.iloc[i]
            elif prev_supertrend == final_lower_band.iloc[i-1]: # Previous trend was up
                if df['close'].iloc[i] >= final_lower_band.iloc[i]:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                else: # Close crossed below lower band
                    supertrend.iloc[i] = final_upper_band.iloc[i]
            else: # Initial or ambiguous state, default based on close vs bands
                 if df['close'].iloc[i] > final_upper_band.iloc[i]:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                 elif df['close'].iloc[i] < final_lower_band.iloc[i]:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
                 else: # Inside bands, maintain previous trend direction if possible
                    supertrend.iloc[i] = prev_supertrend # maintain

        latest_st = supertrend.iloc[-1]
        if pd.isna(latest_st):
             # Try to resolve NaN from last ST value
            if df['close'].iloc[-1] > final_upper_band.iloc[-1]: latest_st = final_lower_band.iloc[-1]
            elif df['close'].iloc[-1] < final_lower_band.iloc[-1]: latest_st = final_upper_band.iloc[-1]
            else: return {"error": "Supertrend calculation resulted in unresolvable NaN."}
        
        trend_direction = "buy" if df['close'].iloc[-1] > latest_st else "sell"

        return {
            "supertrend_value": round(latest_st, 2),
            "signal": trend_direction
        }
    except Exception as e:
        return {"error": f"Error during manual Supertrend calculation: {str(e)}"}

def calculate_ichimoku(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Ichimoku Cloud requires 'high', 'low', and 'close' price data."}
    if len(df) < senkou_b_period:
        return {"error": f"Insufficient data for Ichimoku (need {senkou_b_period} periods, got {len(df)})."}

    try:
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        chikou_span_value = df['close'] # Current close value, plotted 26 periods back

        senkou_span_a = (tenkan_sen + kijun_sen) / 2 # Plotted 26 periods ahead
        
        senkou_b_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = df['low'].rolling(window=senkou_b_period).min()
        senkou_span_b = (senkou_b_high + senkou_b_low) / 2 # Plotted 26 periods ahead
        
        ts = tenkan_sen.iloc[-1]
        ks = kijun_sen.iloc[-1]
        cs_val = chikou_span_value.iloc[-1]
        sa = senkou_span_a.iloc[-1] # This is value for current date (to be plotted ahead)
        sb = senkou_span_b.iloc[-1] # This is value for current date (to be plotted ahead)
        price = df['close'].iloc[-1]

        # Chikou span comparison: current close vs close 26 periods ago
        chikou_compared_to_past_price = df['close'].shift(kijun_period).iloc[-1] if len(df) > kijun_period else np.nan


        if pd.isna(ts) or pd.isna(ks) or pd.isna(sa) or pd.isna(sb):
             return {"error": "Ichimoku calculation resulted in NaN values for key components."}

        signal = "neutral"
        # Strong Buy: Price > Cloud (SA & SB), SA > SB (Cloud bullish), TS > KS (bullish crossover), CS > Past Price
        if price > max(sa, sb) and sa > sb and ts > ks and (pd.isna(chikou_compared_to_past_price) or cs_val > chikou_compared_to_past_price):
            signal = "strong_buy"
        # Strong Sell: Price < Cloud (SA & SB), SA < SB (Cloud bearish), TS < KS (bearish crossover), CS < Past Price
        elif price < min(sa, sb) and sa < sb and ts < ks and (pd.isna(chikou_compared_to_past_price) or cs_val < chikou_compared_to_past_price):
            signal = "strong_sell"
        # Weaker signals
        elif ts > ks and price > max(sa,sb): signal = "buy_bias" # Tenkan/Kijun bullish cross, price above cloud
        elif ts < ks and price < min(sa,sb): signal = "sell_bias" # Tenkan/Kijun bearish cross, price below cloud

        return {
            "tenkan_sen": round(ts, 2),
            "kijun_sen": round(ks, 2),
            "senkou_span_a_current_value": round(sa, 2), # Value for current period, plotted ahead
            "senkou_span_b_current_value": round(sb, 2), # Value for current period, plotted ahead
            "chikou_span_current_value": round(cs_val, 2), # Value for current period, plotted behind
            "signal": signal,
            "note": "SA, SB, CS values are for current candle; their plot positions are shifted."
        }
    except Exception as e:
        return {"error": f"Error during manual Ichimoku Cloud calculation: {str(e)}"}

def calculate_parabolic_sar(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    initial_af = 0.02
    increment_af = 0.02
    max_af = 0.20
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Parabolic SAR requires 'high', 'low', and 'close' price data."}
    if len(df) < 5:
        return {"error": f"Insufficient data for Parabolic SAR (need at least 5 periods, got {len(df)})."}

    try:
        high = df['high']
        low = df['low']
        close = df['close'] 

        sar = pd.Series(np.nan, index=df.index)
        ep = pd.Series(np.nan, index=df.index) 
        af = pd.Series(np.nan, index=df.index) 
        trend = pd.Series(0, index=df.index)

        idx0 = df.index[0]
        
        if len(df) > 1:
            idx1 = df.index[1]
            if close.loc[idx1] > close.loc[idx0]:
                trend.loc[idx1] = 1 
                sar.loc[idx1] = low.loc[idx0] 
                ep.loc[idx1] = high.loc[idx1]
            else:
                trend.loc[idx1] = -1 
                sar.loc[idx1] = high.loc[idx0]
                ep.loc[idx1] = low.loc[idx1]
            af.loc[idx1] = initial_af
        else:
            return {"error": "Parabolic SAR needs at least 2 data points."}


        for i in range(2, len(df)):
            idx_curr = df.index[i]
            idx_prev = df.index[i-1]
            idx_prev_prev = df.index[i-2]

            prev_sar = sar.loc[idx_prev]
            prev_ep = ep.loc[idx_prev]
            prev_af = af.loc[idx_prev]
            prev_trend = trend.loc[idx_prev]

            current_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            if prev_trend == 1:
                current_sar = min(current_sar, low.loc[idx_prev], low.loc[idx_prev_prev])
                if current_sar > low.loc[idx_curr]: 
                    trend.loc[idx_curr] = -1
                    sar.loc[idx_curr] = max(prev_ep, high.loc[idx_curr]) # SAR is highest high of previous trend on reversal
                    ep.loc[idx_curr] = low.loc[idx_curr]
                    af.loc[idx_curr] = initial_af
                else: 
                    trend.loc[idx_curr] = 1
                    sar.loc[idx_curr] = current_sar
                    if high.loc[idx_curr] > prev_ep:
                        ep.loc[idx_curr] = high.loc[idx_curr]
                        af.loc[idx_curr] = min(max_af, prev_af + increment_af)
                    else:
                        ep.loc[idx_curr] = prev_ep
                        af.loc[idx_curr] = prev_af
            else: 
                current_sar = max(current_sar, high.loc[idx_prev], high.loc[idx_prev_prev])
                if current_sar < high.loc[idx_curr]: 
                    trend.loc[idx_curr] = 1
                    sar.loc[idx_curr] = min(prev_ep, low.loc[idx_curr]) # SAR is lowest low of previous trend on reversal
                    ep.loc[idx_curr] = high.loc[idx_curr]
                    af.loc[idx_curr] = initial_af
                else: 
                    trend.loc[idx_curr] = -1
                    sar.loc[idx_curr] = current_sar
                    if low.loc[idx_curr] < prev_ep:
                        ep.loc[idx_curr] = low.loc[idx_curr]
                        af.loc[idx_curr] = min(max_af, prev_af + increment_af)
                    else:
                        ep.loc[idx_curr] = prev_ep
                        af.loc[idx_curr] = prev_af
        
        latest_sar = sar.iloc[-1]
        latest_trend_val = trend.iloc[-1]
        
        if pd.isna(latest_sar):
            return {"error": "Parabolic SAR calculation resulted in NaN."}
        
        signal = "buy" if latest_trend_val == 1 and close.iloc[-1] > latest_sar else \
                 "sell" if latest_trend_val == -1 and close.iloc[-1] < latest_sar else "hold"

        return {
            "sar_value": round(latest_sar, 2),
            "signal": signal 
        }
    except Exception as e:
        return {"error": f"Error during manual Parabolic SAR calculation: {str(e)}"}

def calculate_williams_r(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    period = 14
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Williams %R requires 'high', 'low', and 'close' price data."}
    if len(df) < period:
        return {"error": f"Insufficient data for Williams %R (need {period} periods, got {len(df)})."}

    try:
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        close = df['close']

        wr_val = ((highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)) * -100
        wr_val.fillna(-50, inplace=True) 
        latest_wr = wr_val.iloc[-1]

        if pd.isna(latest_wr):
             return {"error": "Williams %R calculation resulted in NaN."}

        signal = "neutral"
        if latest_wr < -80: signal = "buy" # Oversold
        elif latest_wr > -20: signal = "sell" # Overbought

        return {"williams_r": round(latest_wr, 2), "signal": signal}
    except Exception as e:
        return {"error": f"Error during manual Williams %R calculation: {str(e)}"}

def calculate_vwap(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    period = 20
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close', 'volume']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "VWAP calculation requires 'high', 'low', 'close', and 'volume' data."}
    if any(df[col].isnull().all() for col in required_cols):
        return {"error": f"One or more required columns for VWAP ({required_cols}) contains all NaN values."}
    if len(df) < period:
        return {"error": f"Insufficient data for rolling VWAP (need {period} periods, got {len(df)})."}

    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_volume = typical_price * df['volume']
        
        sum_tp_volume = tp_volume.rolling(window=period).sum()
        sum_volume = df['volume'].rolling(window=period).sum()

        vwap_series = sum_tp_volume / sum_volume.replace(0, np.nan)
        vwap_series.fillna(method='ffill', inplace=True) 
        latest_vwap = vwap_series.iloc[-1]
        latest_close = df['close'].iloc[-1]

        if pd.isna(latest_vwap):
            return {"error": "VWAP calculation resulted in NaN."}

        signal = "neutral"
        if latest_close > latest_vwap: signal = "buy_bias" # Price above VWAP = bullish bias
        elif latest_close < latest_vwap: signal = "sell_bias" # Price below VWAP = bearish bias
        
        return {"vwap": round(latest_vwap, 2), "signal": signal, "note": "Signal is price vs VWAP for general bias."}
    except Exception as e:
        return {"error": f"Error during manual VWAP calculation: {str(e)}"}

# --- New Indicators ---
def calculate_stochastic_oscillator(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Stochastic Oscillator (%K and %D)."""
    k_period = 14  # Standard period for %K
    d_period = 3   # Standard period for %D (smoothing of %K)
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Stochastic Oscillator requires 'high', 'low', and 'close' price data."}
    if len(df) < k_period:
        return {"error": f"Insufficient data for Stochastic %K (need {k_period} periods, got {len(df)})."}

    try:
        lowest_low_k = df['low'].rolling(window=k_period).min()
        highest_high_k = df['high'].rolling(window=k_period).max()
        
        percent_k = ((df['close'] - lowest_low_k) / (highest_high_k - lowest_low_k).replace(0, np.nan)) * 100
        percent_k.fillna(50, inplace=True) # Neutral if range is zero
        
        percent_d = percent_k.rolling(window=d_period).mean()

        latest_k = percent_k.iloc[-1]
        latest_d = percent_d.iloc[-1]

        if pd.isna(latest_k) or pd.isna(latest_d):
            return {"error": "Stochastic Oscillator calculation resulted in NaN."}

        signal = "neutral"
        # Oversold/Overbought
        if latest_k < 20 and latest_d < 20: signal = "buy" # Both in oversold
        elif latest_k > 80 and latest_d > 80: signal = "sell" # Both in overbought
        # Crossovers (simplified)
        elif len(percent_k) > 1 and len(percent_d) > 1:
            prev_k, prev_d = percent_k.iloc[-2], percent_d.iloc[-2]
            if not pd.isna(prev_k) and not pd.isna(prev_d):
                if prev_k <= prev_d and latest_k > latest_d and latest_d < 30: signal = "buy_crossover" # Bullish crossover in oversold
                elif prev_k >= prev_d and latest_k < latest_d and latest_d > 70: signal = "sell_crossover" # Bearish crossover in overbought
        
        return {
            "percent_k": round(latest_k, 2),
            "percent_d": round(latest_d, 2),
            "signal": signal
        }
    except Exception as e:
        return {"error": f"Error during manual Stochastic Oscillator calculation: {str(e)}"}

def calculate_standard_deviation(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates rolling standard deviation of closing prices."""
    period = 20 # Standard period, often matches Bollinger Bands SMA
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "Standard Deviation calculation requires 'close' price data."}
    if len(df) < period:
        return {"error": f"Insufficient data for Standard Deviation (need {period} periods, got {len(df)})."}

    try:
        std_dev_series = df['close'].rolling(window=period).std()
        latest_std_dev = std_dev_series.iloc[-1]

        if pd.isna(latest_std_dev):
            return {"error": "Standard Deviation calculation resulted in NaN."}
        
        # Standard deviation itself is a measure of volatility, not a direct buy/sell signal.
        return {"std_dev": round(latest_std_dev, 2), "signal": "volatility_measure", "note": "Higher value means higher volatility."}
    except Exception as e:
        return {"error": f"Error during manual Standard Deviation calculation: {str(e)}"}

def calculate_obv(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates On-Balance Volume (OBV)."""
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['close', 'volume']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "OBV calculation requires 'close' and 'volume' data."}
    if len(df) < 2: # Need at least two periods for a change
        return {"error": f"Insufficient data for OBV (need at least 2 periods, got {len(df)})."}
    
    try:
        obv = pd.Series(0.0, index=df.index)
        obv.iloc[0] = df['volume'].iloc[0] # Start with first day's volume or 0

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        latest_obv = obv.iloc[-1]
        
        # OBV trend is more important than absolute value. Compare recent trend.
        signal = "neutral"
        if len(obv) > 5: # Look at short trend of OBV (e.g., 5 periods)
            obv_trend = obv.iloc[-1] - obv.iloc[-6]
            price_trend = df['close'].iloc[-1] - df['close'].iloc[-6]
            if obv_trend > 0 and price_trend > 0: signal = "bullish_confirmation" # OBV confirms price uptrend
            elif obv_trend < 0 and price_trend < 0: signal = "bearish_confirmation" # OBV confirms price downtrend
            elif obv_trend > 0 and price_trend < 0: signal = "bullish_divergence_potential_buy"
            elif obv_trend < 0 and price_trend > 0: signal = "bearish_divergence_potential_sell"

        return {"obv": latest_obv, "signal": signal, "note": "OBV value is cumulative; trend and divergence are key."}
    except Exception as e:
        return {"error": f"Error during manual OBV calculation: {str(e)}"}

# --- Mapping indicator names to functions ---
INDICATOR_TOOLS_MAP = {
    "rsi": calculate_rsi,
    "ema": calculate_ema,
    "bollingerbands": calculate_bollinger_bands,
    "atr": calculate_atr,
    "macd": calculate_macd,
    "adx": calculate_adx,
    "supertrend": calculate_supertrend,
    "ichimoku": calculate_ichimoku,
    "parabolicsar": calculate_parabolic_sar,
    "williamsr": calculate_williams_r,
    "vwap": calculate_vwap,
    "stochastic": calculate_stochastic_oscillator, # New
    "stddev": calculate_standard_deviation,      # New
    "obv": calculate_obv                         # New
}

# Default indicators per category (for vague queries in technical_agent.py)
# Categories: Momentum, Trend, Volatility, Volume, Other/Chart Patterns
DEFAULT_INDICATORS_BY_CATEGORY = {
    "momentum": "rsi",        # For questions like "is it overbought/oversold?"
    "trend": "supertrend",    # For questions like "is it trending up/down?" or "what's the trend?"
    "volatility": "atr",      # For questions like "is it very volatile?"
    "volume": "obv",          # For questions about volume pressure.
    "strength": "adx",        # For "how strong is the trend?"
    "general_outlook": "macd" # A good all-rounder if category is unclear but specific calculation needed
}


if __name__ == '__main__':
    sample_data = []
    start_close = 100
    for i in range(1, 71): # Generate 70 data points
        change = np.random.uniform(-2, 2.1)
        prev_close = sample_data[-1]['close'] if sample_data else start_close
        close = round(max(1, prev_close + change),2) # Ensure price > 0
        high = round(close + np.random.uniform(0.1,3),2)
        low = round(close - np.random.uniform(0.1,min(2.9, close-0.1)),2) # Ensure low < close
        open_val = round( (sample_data[-1]['close'] + close)/2 + np.random.uniform(-1,1) ,2)
        open_val = round(max(1, open_val), 2)

        if low >= close : low = round(close - np.random.uniform(0.1, 0.5),2)
        if high <= close : high = round(close + np.random.uniform(0.1, 0.5),2)
        if open_val > high : open_val = high
        if open_val < low : open_val = low


        month = (i-1)//30 + 1
        day = (i-1)%30 + 1
        date_str = f'2023-{month:02d}-{day:02d}'
        if month > 12: date_str = f'2024-{(month-12):02d}-{day:02d}'


        sample_data.append({
            'date': date_str,
            'open': open_val,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(800, 1500)
        })


    print("--- Testing Manual Indicator Calculations ---")
    for name, func in INDICATOR_TOOLS_MAP.items():
        print(f"\nCalculating {name.upper()}:")
        try:
            result = func(sample_data)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error testing {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Default Indicators by Category ---")
    print(DEFAULT_INDICATORS_BY_CATEGORY)