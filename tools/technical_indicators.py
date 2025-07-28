# tools/technical_indicators.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

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
    """Calculates RSI and provides a standardized buy/sell/neutral signal."""
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
        if latest_rsi < 30: signal = "buy"  # Oversold
        elif latest_rsi > 70: signal = "sell" # Overbought
        
        return {"rsi": round(latest_rsi, 2), "signal": signal}
    except Exception as e:
        return {"error": f"Error during manual RSI calculation: {str(e)}"}


def calculate_ema(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates EMA and provides a standardized signal based on price vs EMA."""
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
        if latest_close > latest_ema: signal = "buy"
        elif latest_close < latest_ema: signal = "sell"
        
        return {"ema": round(latest_ema, 2), "signal": signal, "note": "Signal is price vs EMA; for crossovers, use MACD or multiple EMAs."}
    except Exception as e:
        return {"error": f"Error during manual EMA calculation: {str(e)}"}

def calculate_bollinger_bands(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Bollinger Bands and provides a standardized signal."""
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
        ub = upper_band.iloc[-1]

        if pd.isna(lb) or pd.isna(ub):
            return {"error": "Bollinger Bands calculation resulted in NaN values."}

        signal = "neutral"
        if latest_close < lb: signal = "buy" # Price touched/crossed lower band
        elif latest_close > ub: signal = "sell" # Price touched/crossed upper band
        
        return {
            "lower_band": round(lb, 2),
            "middle_band": round(middle_band.iloc[-1], 2),
            "upper_band": round(ub, 2),
            "signal": signal
        }
    except Exception as e:
        return {"error": f"Error during manual Bollinger Bands calculation: {str(e)}"}

def calculate_atr(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates ATR. As a volatility measure, its signal is 'neutral' for aggregation."""
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
        
        return {"atr": round(latest_atr, 4), "signal": "neutral", "note": "This is a volatility measure, not a directional signal. Higher ATR means higher volatility."}
    except Exception as e:
        return {"error": f"Error during manual ATR calculation: {str(e)}"}

def calculate_macd(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates MACD and provides a standardized signal based on crossovers and histogram."""
    fast_period, slow_period, signal_period = 12, 26, 9
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    if df is None or 'close' not in df.columns:
        return {"error": "MACD calculation requires 'close' price data."}
    if len(df) < slow_period + signal_period:
        return {"error": f"Insufficient data for MACD (need ~{slow_period + signal_period} periods, got {len(df)})."}

    try:
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        ml, sl, hist = macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
        prev_ml = macd_line.iloc[-2] if len(macd_line) > 1 else np.nan
        prev_sl = signal_line.iloc[-2] if len(signal_line) > 1 else np.nan

        if pd.isna(ml) or pd.isna(sl) or pd.isna(hist):
            return {"error": "MACD calculation resulted in NaN values."}

        signal = "neutral"
        # Bullish Crossover: MACD line crosses above Signal line
        if not pd.isna(prev_ml) and not pd.isna(prev_sl) and prev_ml <= prev_sl and ml > sl:
            signal = "strong_buy"
        # Bearish Crossover: MACD line crosses below Signal line
        elif not pd.isna(prev_ml) and not pd.isna(prev_sl) and prev_ml >= prev_sl and ml < sl:
            signal = "strong_sell"
        # General tendency based on position vs zero line
        elif ml > 0 and sl > 0: signal = "buy"
        elif ml < 0 and sl < 0: signal = "sell"
        
        return {"macd_line": round(ml, 4), "signal_line": round(sl, 4), "histogram": round(hist, 4), "signal": signal}
    except Exception as e:
        return {"error": f"Error during manual MACD calculation: {str(e)}"}

def calculate_adx(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates ADX and provides a standardized signal for trend strength and direction."""
    di_len, adx_len = 14, 14
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "ADX calculation requires 'high', 'low', and 'close' price data."}
    if len(df) < di_len + adx_len:
        return {"error": f"Insufficient data for ADX (need ~{di_len + adx_len} periods, got {len(df)})."}

    try:
        tr = (df['high'] - df['low']).combine_first(abs(df['high'] - df['close'].shift(1))).combine_first(abs(df['low'] - df['close'].shift(1)))
        atr = tr.ewm(com=di_len - 1, adjust=False).mean()
        
        move_up, move_down = df['high'].diff(), -df['low'].diff()
        plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=df.index)
        
        plus_di = 100 * (plus_dm.ewm(com=di_len - 1, adjust=False).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(com=di_len - 1, adjust=False).mean() / atr.replace(0, np.nan))
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(com=adx_len - 1, adjust=False).mean()

        latest_adx, latest_pdi, latest_mdi = adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
        
        if pd.isna(latest_adx) or pd.isna(latest_pdi) or pd.isna(latest_mdi):
             return {"error": "ADX calculation resulted in NaN values."}

        signal = "neutral"
        if latest_adx > 25:
            if latest_pdi > latest_mdi: signal = "strong_buy"
            else: signal = "strong_sell"
        elif latest_adx < 20:
            signal = "neutral" # Weak trend
        
        return {"adx": round(latest_adx, 2), "plus_di": round(latest_pdi, 2), "minus_di": round(latest_mdi, 2), "signal": signal}
    except Exception as e:
        return {"error": f"Error during manual ADX calculation: {str(e)}"}


def calculate_supertrend(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Supertrend and provides a standardized signal based on trend direction."""
    atr_period, multiplier = 10, 3.0
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Supertrend requires 'high', 'low', and 'close' price data."}
    if len(df) < atr_period + 1:
        return {"error": f"Insufficient data for Supertrend (need {atr_period + 1} periods, got {len(df)})."}

    try:
        # ATR Calculation
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
        
        # Supertrend Calculation
        hl2 = (df['high'] + df['low']) / 2
        final_upper_band = hl2 + (multiplier * atr)
        final_lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(np.nan, index=df.index)
        trend = pd.Series(1, index=df.index) # Default to uptrend

        for i in range(1, len(df)):
            if df['close'].iloc[i] > final_upper_band.iloc[i-1]:
                trend.iloc[i] = 1
            elif df['close'].iloc[i] < final_lower_band.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
                if trend.iloc[i] == 1 and final_lower_band.iloc[i] < final_lower_band.iloc[i-1]:
                    final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
                if trend.iloc[i] == -1 and final_upper_band.iloc[i] > final_upper_band.iloc[i-1]:
                    final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
        
        supertrend = np.where(trend == 1, final_lower_band, final_upper_band)
        
        latest_st = supertrend[-1]
        signal = "buy" if df['close'].iloc[-1] > latest_st else "sell"

        return {"supertrend_value": round(latest_st, 2), "signal": signal}
    except Exception as e:
        return {"error": f"Error during manual Supertrend calculation: {str(e)}"}

def calculate_ichimoku(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Ichimoku Cloud and provides a comprehensive, standardized signal."""
    tenkan_period, kijun_period, senkou_b_period = 9, 26, 52
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)

    required_cols = ['high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
        return {"error": "Ichimoku Cloud requires 'high', 'low', and 'close' price data."}
    if len(df) < senkou_b_period:
        return {"error": f"Insufficient data for Ichimoku (need {senkou_b_period} periods, got {len(df)})."}

    try:
        tenkan_sen = (df['high'].rolling(window=tenkan_period).max() + df['low'].rolling(window=tenkan_period).min()) / 2
        kijun_sen = (df['high'].rolling(window=kijun_period).max() + df['low'].rolling(window=kijun_period).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        senkou_span_b = ((df['high'].rolling(window=senkou_b_period).max() + df['low'].rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)
        chikou_span = df['close'].shift(-kijun_period)
        
        price = df['close'].iloc[-1]
        ts, ks = tenkan_sen.iloc[-1], kijun_sen.iloc[-1]
        sa, sb = senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]
        
        if any(pd.isna(v) for v in [price, ts, ks, sa, sb]):
            return {"error": "Ichimoku calculation resulted in NaN values."}

        signal = "neutral"
        if price > max(sa, sb) and ts > ks and sa > sb: signal = "strong_buy"
        elif price < min(sa, sb) and ts < ks and sa < sb: signal = "strong_sell"
        elif price > max(sa, sb): signal = "buy"
        elif price < min(sa, sb): signal = "sell"

        return {"tenkan_sen": round(ts, 2), "kijun_sen": round(ks, 2), "signal": signal}
    except Exception as e:
        return {"error": f"Error during manual Ichimoku Cloud calculation: {str(e)}"}


# other indicator functions (parabolic_sar, williams_r, etc.) with standardized signals 


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


# --- NEW: Aggregation Function ---
def aggregate_signals(
    indicator_results: List[Tuple[str, Dict[str, Any]]],
    time_period: str = '1d'
) -> Dict[str, Any]:
    """
    Aggregates signals from multiple technical indicators into a single verdict.

    Args:
        indicator_results: A list of tuples, where each tuple contains the
                           indicator name and its result dictionary.
                           e.g., [('rsi', {'rsi': 35, 'signal': 'buy'}), 
                                  ('macd', {'signal': 'sell', ...})]
        time_period: The time frame of the analysis ('15m', '1h', '1d', '1w', '1m').
                     This is used for weighting the indicators.

    Returns:
        A dictionary containing the overall verdict, a composite score,
        and a breakdown of individual indicator signals and their contribution.
    """
    score_map = {
        "strong_buy": 2,
        "buy": 1,
        "neutral": 0,
        "sell": -1,
        "strong_sell": -2
    }

    # Define which indicators fall into which category for weighting
    momentum_indicators = {'rsi', 'stochastic', 'williamsr', 'macd'}
    trend_indicators = {'ema', 'supertrend', 'ichimoku', 'parabolicsar', 'adx'}
    
    total_score = 0.0
    total_weight = 0.0
    breakdown = []

    for name, result in indicator_results:
        if "error" in result:
            continue # Skip indicators that failed

        signal = result.get("signal", "neutral")
        score = score_map.get(signal, 0)
        
        # Determine weight based on time period
        weight = 1.0
        if time_period in ['15m', '1h'] and name in momentum_indicators:
            weight = 1.5
        elif time_period in ['1w', '1m'] and name in trend_indicators:
            weight = 1.5
            
        total_score += score * weight
        total_weight += weight
        
        breakdown.append({
            "indicator": name,
            "signal": signal,
            "score": score,
            "weight": weight
        })

    if total_weight == 0:
        return {
            "overall_verdict": "Neutral",
            "composite_score": 0,
            "summary": "No valid indicator signals to aggregate.",
            "breakdown": breakdown
        }

    # Normalize the score by total weight to keep it in the -2 to 2 range
    final_score = total_score / total_weight

    # Determine overall verdict based on the final score
    verdict = "Neutral"
    if final_score >= 1.5: verdict = "Strong Sell" # Corrected: Positive score is Buy
    elif final_score >= 0.5: verdict = "Buy"
    elif final_score <= -1.5: verdict = "Strong Sell"
    elif final_score <= -0.5: verdict = "Sell"
    
    # Corrected logic for verdict determination
    if final_score >= 1.5:
        verdict = "Strong Buy"
    elif final_score >= 0.5:
        verdict = "Buy"
    elif final_score > -0.5:
        verdict = "Neutral"
    elif final_score > -1.5:
        verdict = "Sell"
    else: # score <= -1.5
        verdict = "Strong Sell"


    summary = (f"Aggregated signal from {len(breakdown)} indicators "
               f"is '{verdict}' with a composite score of {final_score:.2f}.")

    return {
        "overall_verdict": verdict,
        "composite_score": round(final_score, 2),
        "summary": summary,
        "breakdown": breakdown
    }




def calculate_historical_performance_score(price_data_list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates a comprehensive technical performance score based on historical data.
    This is NOT a point-in-time signal, but an evaluation over the entire period.
    """
    df = _convert_to_dataframe_and_standardize(price_data_list_of_dicts)
    if df is None or len(df) < 200: # Require at least ~10 months of data for meaningful stats
        return {"error": "Insufficient data for historical performance analysis (requires at least 200 trading days)."}

    try:
        # Metric 1: Price Appreciation (CAGR)
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        num_years = len(df) / 252.0  # Approx. trading days in a year
        price_cagr = ((end_price / start_price) ** (1 / num_years) - 1) * 100 if num_years > 0 and start_price > 0 else 0

        # Metric 2: Volatility (Annualized Standard Deviation of daily returns)
        daily_returns = df['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100

        # Metric 3: Sharpe Ratio (simple version, assuming 5% risk-free rate)
        risk_free_rate = 0.05
        avg_daily_return = daily_returns.mean()
        std_dev_daily_return = daily_returns.std()
        sharpe_ratio = ((avg_daily_return * 252) - risk_free_rate) / (std_dev_daily_return * np.sqrt(252)) if std_dev_daily_return > 0 else 0

        # --- Scoring Logic (0-10 scale) ---
        # CAGR Score: Scaled up to 50% CAGR for a max score of 10
        cagr_score = min(10, max(0, price_cagr / 5.0))
        # Volatility Score: Lower is better. Score 10 for <15% volatility, 0 for >55%
        volatility_score = min(10, max(0, 10 - ((volatility - 15) / 4.0)))
        # Sharpe Ratio Score: Higher is better. Score 10 for Sharpe >= 1.5
        sharpe_score = min(10, max(0, (sharpe_ratio + 0.5) * 5))

        # --- Final Weighted Score ---
        # For "consistent performance", we value risk-adjusted returns (Sharpe) and growth (CAGR) highly.
        final_score = (0.4 * cagr_score) + (0.4 * sharpe_score) + (0.2 * volatility_score)

        return {
            "signal": "historical_performance",
            "technical_performance_score": round(final_score, 2),
            "price_cagr_perc": round(price_cagr, 2),
            "annualized_volatility_perc": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "note": f"Performance analysis over {num_years:.1f} years."
        }
    except Exception as e:
        return {"error": f"Error during historical performance calculation: {str(e)}"}


# Map of indicator names to their calculation functions
INDICATOR_TOOLS_MAP = {
    "rsi": calculate_rsi,
    "ema": calculate_ema,
    "bollinger bands": calculate_bollinger_bands,
    "atr": calculate_atr,
    "macd": calculate_macd,
    "adx": calculate_adx,
    "supertrend": calculate_supertrend,
    "ichimoku": calculate_ichimoku,
    "parabolic sar": calculate_parabolic_sar,
    "williams r": calculate_williams_r,
    "vwap": calculate_vwap,
    "stochastic": calculate_stochastic_oscillator,
    "stddev": calculate_standard_deviation,
    "obv": calculate_obv,
     "historical_performance_score": calculate_historical_performance_score
}

# Default set of indicators for a general "should I buy/sell" query
AGGREGATION_DEFAULT_INDICATORS = [
    "rsi",
    "macd",
    "supertrend",
    "adx"
]

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