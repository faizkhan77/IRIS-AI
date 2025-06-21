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

    # Drop rows where essential data (e.g., close) might be NaN after conversion
    # Indicators will check for their specific required columns.
    if 'close' in df.columns: # A common essential column
        df.dropna(subset=['close'], inplace=True) 
    
    if df.empty:
        return None
        
    return df

# --- Indicator Calculation Functions (Manual Implementations) ---

def calculate_rsi(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates RSI (Relative Strength Index) manually."""
    period = 14  # Standard default parameter
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "RSI calculation requires 'close' price data."}
    if len(df) < period + 1: # Need period + 1 for first change
        return {"error": f"Insufficient data for RSI (need {period + 1} periods, got {len(df)})."}

    try:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Wilder's smoothing: com (center of mass) = period - 1
        avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        
        # Handle case where avg_loss is zero to prevent division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan) # replace 0 with NaN to propagate NaN if loss is consistently 0
        rsi = 100 - (100 / (1 + rs))
        
        rsi_val = rsi.iloc[-1]
        if pd.isna(rsi_val):
             # Check if avg_loss was zero leading to NaN rs
            if pd.isna(rs.iloc[-1]) and avg_loss.iloc[-1] == 0 and avg_gain.iloc[-1] > 0:
                rsi_val = 100.0 # If all losses are zero (strong uptrend), RSI is 100
            elif pd.isna(rs.iloc[-1]) and avg_loss.iloc[-1] == 0 and avg_gain.iloc[-1] == 0:
                rsi_val = 50.0 # Or some other neutral value if no change
            else:
                return {"error": "RSI calculation resulted in NaN (possibly insufficient varying data)."}
        
        return {"rsi": round(rsi_val, 2)}
    except Exception as e:
        return {"error": f"Error during manual RSI calculation: {str(e)}"}


def calculate_ema(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates EMA (Exponential Moving Average) manually using pandas ewm."""
    period = 20  # Standard default parameter
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "EMA calculation requires 'close' price data."}
    if len(df) < period:
        return {"error": f"Insufficient data for EMA (need {period} periods, got {len(df)})."}

    try:
        # Standard EMA: span = period
        ema_series = df['close'].ewm(span=period, adjust=False, min_periods=period).mean()
        if ema_series.empty or pd.isna(ema_series.iloc[-1]):
            return {"error": "EMA calculation resulted in NaN (possibly insufficient data)."}
        return {"ema": round(ema_series.iloc[-1], 2)}
    except Exception as e:
        return {"error": f"Error during manual EMA calculation: {str(e)}"}


def calculate_bollinger_bands(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Bollinger Bands manually."""
    period = 20         # Standard default parameter
    std_dev_multiplier = 2.0  # Standard default parameter
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

        if pd.isna(middle_band.iloc[-1]) or pd.isna(upper_band.iloc[-1]) or pd.isna(lower_band.iloc[-1]):
            return {"error": "Bollinger Bands calculation resulted in NaN values."}

        return {
            "lower_band": round(lower_band.iloc[-1], 2),
            "middle_band": round(middle_band.iloc[-1], 2),
            "upper_band": round(upper_band.iloc[-1], 2)
        }
    except Exception as e:
        return {"error": f"Error during manual Bollinger Bands calculation: {str(e)}"}


def calculate_atr(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates ATR (Average True Range) manually."""
    period = 14  # Standard default parameter
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "ATR calculation requires 'high', 'low', and 'close' price data."}
    if len(df) < period + 1 : # Need period for smoothing + 1 for prev_close
        return {"error": f"Insufficient data for ATR (need {period + 1} periods, got {len(df)})."}

    try:
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        
        # Wilder's smoothing for ATR: com = period - 1
        atr_series = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            return {"error": "ATR calculation failed or resulted in NaN."}
        return {"atr": round(atr_series.iloc[-1], 4)}
    except Exception as e:
        return {"error": f"Error during manual ATR calculation: {str(e)}"}


def calculate_macd(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates MACD (Moving Average Convergence Divergence) manually."""
    fast_period = 12    # Standard default
    slow_period = 26    # Standard default
    signal_period = 9   # Standard default
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "MACD calculation requires 'close' price data."}
    if len(df) < slow_period + signal_period:
        return {"error": f"Insufficient data for MACD (need ~{slow_period + signal_period} periods, got {len(df)})."}

    try:
        ema_fast = df['close'].ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
        histogram = macd_line - signal_line

        if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]) or pd.isna(histogram.iloc[-1]):
            return {"error": "MACD calculation resulted in NaN values."}

        return {
            "macd_line": round(macd_line.iloc[-1], 4),
            "histogram": round(histogram.iloc[-1], 4),
            "signal_line": round(signal_line.iloc[-1], 4)
        }
    except Exception as e:
        return {"error": f"Error during manual MACD calculation: {str(e)}"}


def calculate_adx(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates ADX (Average Directional Index) manually."""
    di_len = 14     # Standard default for DI calculation
    adx_len = 14    # Standard default for ADX smoothing of DX
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "ADX calculation requires 'high', 'low', and 'close' price data."}
    # ADX smoothing needs di_len for TR/DM, then adx_len for DX smoothing.
    # Total effective periods needed is roughly di_len + adx_len.
    if len(df) < di_len + adx_len:
        return {"error": f"Insufficient data for ADX (need ~{di_len + adx_len} periods, got {len(df)})."}

    try:
        # Calculate True Range (TR)
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        atr = tr.ewm(com=di_len - 1, adjust=False, min_periods=di_len).mean() # Smoothed TR

        # Calculate Directional Movement (+DM, -DM)
        move_up = df['high'].diff()
        move_down = -df['low'].diff() # low.diff() is current_low - prev_low. We need prev_low - current_low

        plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=df.index)

        # Smooth +DM and -DM (Wilder's)
        plus_di_smooth = plus_dm.ewm(com=di_len - 1, adjust=False, min_periods=di_len).mean()
        minus_di_smooth = minus_dm.ewm(com=di_len - 1, adjust=False, min_periods=di_len).mean()

        # Calculate Directional Indicators (+DI, -DI)
        plus_di = (plus_di_smooth / atr) * 100
        minus_di = (minus_di_smooth / atr) * 100

        # Calculate Directional Movement Index (DX)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        dx.fillna(0, inplace=True) # If plus_di + minus_di is 0, DX is 0

        # Calculate ADX (Smoothed DX - Wilder's)
        adx = dx.ewm(com=adx_len - 1, adjust=False, min_periods=adx_len).mean()

        if pd.isna(adx.iloc[-1]) or pd.isna(plus_di.iloc[-1]) or pd.isna(minus_di.iloc[-1]):
             return {"error": "ADX calculation resulted in NaN values."}

        return {
            "adx": round(adx.iloc[-1], 2),
            "plus_di": round(plus_di.iloc[-1], 2),
            "minus_di": round(minus_di.iloc[-1], 2)
        }
    except Exception as e:
        return {"error": f"Error during manual ADX calculation: {str(e)}"}


def calculate_supertrend(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Supertrend manually."""
    atr_period = 10     # Standard default
    multiplier = 3.0    # Standard default
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Supertrend requires 'high', 'low', and 'close' price data."}
    if len(df) < atr_period + 1: # For ATR and one prev close
        return {"error": f"Insufficient data for Supertrend (need {atr_period + 1} periods, got {len(df)})."}

    try:
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        atr = tr.ewm(com=atr_period - 1, adjust=False, min_periods=atr_period).mean()

        # Basic Upper & Lower Bands
        basic_upper_band = (df['high'] + df['low']) / 2 + multiplier * atr
        basic_lower_band = (df['high'] + df['low']) / 2 - multiplier * atr

        # Initialize Supertrend series
        supertrend = pd.Series(np.nan, index=df.index)
        final_upper_band = basic_upper_band.copy()
        final_lower_band = basic_lower_band.copy()

        # Iterative calculation for Supertrend
        for i in range(1, len(df)):
            # Final Upper Band
            if basic_upper_band.iloc[i] < final_upper_band.iloc[i-1] or df['close'].iloc[i-1] > final_upper_band.iloc[i-1]:
                final_upper_band.iloc[i] = basic_upper_band.iloc[i]
            else:
                final_upper_band.iloc[i] = final_upper_band.iloc[i-1]

            # Final Lower Band
            if basic_lower_band.iloc[i] > final_lower_band.iloc[i-1] or df['close'].iloc[i-1] < final_lower_band.iloc[i-1]:
                final_lower_band.iloc[i] = basic_lower_band.iloc[i]
            else:
                final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
            
            # Supertrend
            if pd.isna(supertrend.iloc[i-1]): # Initial case
                 if df['close'].iloc[i] > final_upper_band.iloc[i]:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                 else:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
            elif supertrend.iloc[i-1] == final_upper_band.iloc[i-1]:
                if df['close'].iloc[i] <= final_upper_band.iloc[i]:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
                else:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
            elif supertrend.iloc[i-1] == final_lower_band.iloc[i-1]:
                if df['close'].iloc[i] >= final_lower_band.iloc[i]:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
        
        if pd.isna(supertrend.iloc[-1]):
             return {"error": "Supertrend calculation resulted in NaN."}
        
        # Determine trend direction
        trend_direction = "up" if df['close'].iloc[-1] > supertrend.iloc[-1] else \
                          "down" if df['close'].iloc[-1] < supertrend.iloc[-1] else "neutral"

        return {
            "supertrend_value": round(supertrend.iloc[-1], 2),
            "trend": trend_direction
        }
    except Exception as e:
        return {"error": f"Error during manual Supertrend calculation: {str(e)}"}


def calculate_ichimoku(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Ichimoku Cloud components manually."""
    tenkan_period = 9    # Standard default
    kijun_period = 26    # Standard default
    senkou_b_period = 52 # Standard default for Senkou Span B calculation window
    # chikou_shift = -kijun_period (26) (how it's plotted, not its value)
    # senkou_shift = kijun_period (26) (how spans A/B are plotted ahead)

    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Ichimoku Cloud requires 'high', 'low', and 'close' price data."}
    if len(df) < senkou_b_period: # Longest lookback period
        return {"error": f"Insufficient data for Ichimoku (need {senkou_b_period} periods, got {len(df)})."}

    try:
        # Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Chikou Span (Lagging Span) - The value is current close.
        # Its plotted position is 26 periods in the past.
        chikou_span_value = df['close'] 

        # Senkou Span A (Leading Span A)
        # Value is (current Tenkan + current Kijun) / 2. Plotted 26 periods ahead.
        senkou_span_a = (tenkan_sen + kijun_sen) / 2

        # Senkou Span B (Leading Span B)
        # Value is (52-period high + 52-period low) / 2. Plotted 26 periods ahead.
        senkou_b_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = df['low'].rolling(window=senkou_b_period).min()
        senkou_span_b = (senkou_b_high + senkou_b_low) / 2
        
        last_tenkan = tenkan_sen.iloc[-1]
        last_kijun = kijun_sen.iloc[-1]
        last_chikou_val = chikou_span_value.iloc[-1]
        last_senkou_a = senkou_span_a.iloc[-1] # This is value for current date, plotted ahead
        last_senkou_b = senkou_span_b.iloc[-1] # This is value for current date, plotted ahead

        if pd.isna(last_tenkan) or pd.isna(last_kijun) or pd.isna(last_senkou_a) or pd.isna(last_senkou_b):
             return {"error": "Ichimoku calculation resulted in NaN values for key components."}

        return {
            "tenkan_sen": round(last_tenkan, 2),
            "kijun_sen": round(last_kijun, 2),
            "senkou_span_a": round(last_senkou_a, 2),
            "senkou_span_b": round(last_senkou_b, 2),
            "chikou_span_value": round(last_chikou_val, 2)
        }
    except Exception as e:
        return {"error": f"Error during manual Ichimoku Cloud calculation: {str(e)}"}


def calculate_parabolic_sar(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Parabolic SAR manually."""
    initial_af = 0.02
    increment_af = 0.02
    max_af = 0.20
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close'] # Close is used in trend determination
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Parabolic SAR requires 'high', 'low', and 'close' price data."}
    if len(df) < 5: # Needs some runway, at least a few periods
        return {"error": f"Insufficient data for Parabolic SAR (need at least 5 periods, got {len(df)})."}

    try:
        high = df['high']
        low = df['low']
        close = df['close'] # Not directly in formula but for trend/reversal checks implicitly

        sar = pd.Series(np.nan, index=df.index)
        ep = pd.Series(np.nan, index=df.index) # Extreme Point
        af = pd.Series(np.nan, index=df.index) # Acceleration Factor
        trend = pd.Series(0, index=df.index) # 1 for uptrend, -1 for downtrend

        # Initial values (for the first valid point, typically index 1 after diffs)
        # Start with a downtrend assumption (or uptrend, matters less for long series)
        # Let's start with first available full bar.
        idx0 = df.index[0] # first actual index
        
        # Start trend based on second day's movement relative to first
        if len(df) > 1:
            idx1 = df.index[1]
            if close.loc[idx1] > close.loc[idx0]:
                trend.loc[idx1] = 1 # Uptrend
                sar.loc[idx1] = low.loc[idx0] # Previous low
                ep.loc[idx1] = high.loc[idx1]
            else:
                trend.loc[idx1] = -1 # Downtrend
                sar.loc[idx1] = high.loc[idx0] # Previous high
                ep.loc[idx1] = low.loc[idx1]
            af.loc[idx1] = initial_af
        else: # Single data point, cannot calculate SAR
            return {"error": "Parabolic SAR needs at least 2 data points."}


        for i in range(2, len(df)):
            idx_curr = df.index[i]
            idx_prev = df.index[i-1]

            prev_sar = sar.loc[idx_prev]
            prev_ep = ep.loc[idx_prev]
            prev_af = af.loc[idx_prev]
            prev_trend = trend.loc[idx_prev]

            current_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            # Uptrend rules
            if prev_trend == 1:
                # Potential SAR cap (must be below previous two lows)
                current_sar = min(current_sar, low.loc[idx_prev], low.loc[df.index[i-2]] if i > 1 else low.loc[idx_prev])
                if current_sar > low.loc[idx_curr]: # Trend reversal
                    trend.loc[idx_curr] = -1
                    sar.loc[idx_curr] = prev_ep # New SAR is previous EP
                    ep.loc[idx_curr] = low.loc[idx_curr]
                    af.loc[idx_curr] = initial_af
                else: # Continue uptrend
                    trend.loc[idx_curr] = 1
                    sar.loc[idx_curr] = current_sar
                    if high.loc[idx_curr] > prev_ep:
                        ep.loc[idx_curr] = high.loc[idx_curr]
                        af.loc[idx_curr] = min(max_af, prev_af + increment_af)
                    else:
                        ep.loc[idx_curr] = prev_ep
                        af.loc[idx_curr] = prev_af
            # Downtrend rules
            else: # prev_trend == -1
                # Potential SAR floor (must be above previous two highs)
                current_sar = max(current_sar, high.loc[idx_prev], high.loc[df.index[i-2]] if i > 1 else high.loc[idx_prev])
                if current_sar < high.loc[idx_curr]: # Trend reversal
                    trend.loc[idx_curr] = 1
                    sar.loc[idx_curr] = prev_ep # New SAR is previous EP
                    ep.loc[idx_curr] = high.loc[idx_curr]
                    af.loc[idx_curr] = initial_af
                else: # Continue downtrend
                    trend.loc[idx_curr] = -1
                    sar.loc[idx_curr] = current_sar
                    if low.loc[idx_curr] < prev_ep:
                        ep.loc[idx_curr] = low.loc[idx_curr]
                        af.loc[idx_curr] = min(max_af, prev_af + increment_af)
                    else:
                        ep.loc[idx_curr] = prev_ep
                        af.loc[idx_curr] = prev_af
        
        last_sar = sar.iloc[-1]
        last_trend_val = trend.iloc[-1]
        
        if pd.isna(last_sar):
            return {"error": "Parabolic SAR calculation resulted in NaN."}
        
        trend_direction = "up" if last_trend_val == 1 else "down"

        return {
            "sar_value": round(last_sar, 2), # Typically 2 decimal places for price based
            "trend": trend_direction 
        }
    except Exception as e:
        return {"error": f"Error during manual Parabolic SAR calculation: {str(e)}"}


def calculate_williams_r(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Williams %R manually."""
    period = 14  # Standard default parameter
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

        wr = ((highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)) * -100
        wr.fillna(-50, inplace=True) # Or some other neutral value if highest_high == lowest_low

        if pd.isna(wr.iloc[-1]):
             return {"error": "Williams %R calculation resulted in NaN."}

        return {"williams_r": round(wr.iloc[-1], 2)}
    except Exception as e:
        return {"error": f"Error during manual Williams %R calculation: {str(e)}"}


def calculate_vwap(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates a rolling EOD VWAP (Volume Weighted Average Price) manually."""
    period = 20  # Standard default parameter for rolling EOD VWAP
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close', 'volume']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "VWAP calculation requires 'high', 'low', 'close', and 'volume' data."}
    if any(df[col].isnull().all() for col in required_cols): # Check if any required column is all NaN
        return {"error": f"One or more required columns for VWAP ({required_cols}) contains all NaN values."}
    if len(df) < period:
        return {"error": f"Insufficient data for rolling VWAP (need {period} periods, got {len(df)})."}

    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_volume = typical_price * df['volume']
        
        sum_tp_volume = tp_volume.rolling(window=period).sum()
        sum_volume = df['volume'].rolling(window=period).sum()

        vwap = sum_tp_volume / sum_volume.replace(0, np.nan) # Avoid division by zero if sum_volume is 0
        vwap.fillna(method='ffill', inplace=True) # Fill initial NaNs or if sum_volume was 0 for a period

        if pd.isna(vwap.iloc[-1]):
            # This might happen if all volumes in the window are zero or very early data points
            return {"error": "VWAP calculation resulted in NaN (possibly zero volume in window or insufficient data)."}

        return {"vwap": round(vwap.iloc[-1], 2)}
    except Exception as e:
        return {"error": f"Error during manual VWAP calculation: {str(e)}"}


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
    "parabolicsar": calculate_parabolic_sar, # Note: "parabolic_sar" in agent_technicals.py prompt example
    "williamsr": calculate_williams_r,     # Note: "williams_r" in agent_technicals.py prompt example
    "vwap": calculate_vwap,
}

# Ensure keys match those in technicals_agent.py (lowercase, no spaces/underscores)
# The agent normalizes to "rsi", "macd", "bollingerbands", etc.
# "parabolic_sar" becomes "parabolicsar"
# "williams_r" becomes "williamsr"

if __name__ == '__main__':
    # Example Test Data (replace with more comprehensive data for thorough testing)
    sample_data = [
        {'date': '2023-01-01', 'open': 100, 'high': 105, 'low': 98, 'close': 102, 'volume': 1000},
        {'date': '2023-01-02', 'open': 102, 'high': 108, 'low': 101, 'close': 107, 'volume': 1200},
        {'date': '2023-01-03', 'open': 107, 'high': 110, 'low': 105, 'close': 106, 'volume': 900},
        {'date': '2023-01-04', 'open': 106, 'high': 109, 'low': 103, 'close': 104, 'volume': 1100},
        {'date': '2023-01-05', 'open': 104, 'high': 106, 'low': 100, 'close': 101, 'volume': 1300},
    ]
    # Extend sample_data to at least ~60-70 points for meaningful calculation of all indicators
    start_close = 100
    for i in range(6, 70):
        change = np.random.uniform(-2, 2.1)
        prev_close = sample_data[-1]['close'] if sample_data else start_close
        close = round(prev_close + change,2)
        high = round(close + np.random.uniform(0,3),2)
        low = round(close - np.random.uniform(0,3),2)
        open_val = round( (sample_data[-1]['close'] + close)/2 + np.random.uniform(-1,1) ,2)

        sample_data.append({
            'date': f'2023-01-{i:02d}' if i < 32 else (f'2023-02-{(i-31):02d}' if i < 60 else f'2023-03-{(i-59):02d}'),
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

    # Test with insufficient data for one indicator
    print("\n--- Testing with insufficient data (RSI) ---")
    short_sample_data = sample_data[:5]
    result_short_rsi = calculate_rsi(short_sample_data)
    print(f"  RSI (short data): {result_short_rsi}")

    print("\n--- Testing with missing columns (VWAP) ---")
    data_no_volume = [{'date': d['date'], 'high':d['high'], 'low':d['low'], 'close':d['close']} for d in sample_data[:30]]
    result_no_vol_vwap = calculate_vwap(data_no_volume)
    print(f"  VWAP (no volume col): {result_no_vol_vwap}")