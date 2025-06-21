# app/indicators.py
import math
from typing import List, Dict, Any, Optional, Tuple, Union

# --- Standard Default Parameters (mirrored for clarity, true source is agent) ---
# These are the values that will be hardcoded into the functions below.
# RSI_PERIOD = 14
# SMA_PERIOD = 20
# EMA_PERIOD = 20
# MACD_FAST_LENGTH = 12
# MACD_SLOW_LENGTH = 26
# MACD_SIGNAL_LENGTH = 9
# BOLLINGER_BANDS_PERIOD = 20
# BOLLINGER_BANDS_STD_DEV = 2.0
# ADX_DI_LEN = 14
# ADX_ADX_LEN = 14
# ATR_PERIOD = 14
# SUPERTREND_ATR_PERIOD = 10
# SUPERTREND_FACTOR = 3.0
# ICHIMOKU_CONVERSION_PERIODS = 9
# ICHIMOKU_BASE_PERIODS = 26
# ICHIMOKU_LAGGING_SPAN_2_PERIODS = 52
# ICHIMOKU_DISPLACEMENT = 26
# PSAR_START_AF = 0.02
# PSAR_INCREMENT_AF = 0.02
# PSAR_MAX_AF = 0.2
# WILLIAMS_R_LENGTH = 14


# Helper to get the last valid item from a list or None
def _get_latest(series: List[Any]) -> Optional[Any]:
    if not series:
        return None
    for val in reversed(series):
        if val is not None and (not isinstance(val, float) or not math.isnan(val)):
            return val
    return None

def _get_nth_latest(series: List[Any], n: int) -> Optional[Any]:
    """Gets the nth latest value (0 for latest, 1 for second latest, etc.) if valid."""
    if not series:
        return None
    
    valid_items_indices = [i for i, val in enumerate(series) if val is not None and (not isinstance(val, float) or not math.isnan(val))]
    if not valid_items_indices or n >= len(valid_items_indices):
        return None
    
    target_original_index = valid_items_indices[-(n + 1)]
    return series[target_original_index]

# --- Helper Calculation Functions (these retain period parameters as they are general purpose) ---

def _calculate_sma(data: List[Optional[float]], period: int) -> List[Optional[float]]:
    if not data or period <= 0:
        return [None] * len(data)
    
    sma_values_strict: List[Optional[float]] = [None] * (period -1)
    if len(data) < period:
        return [None] * len(data)

    initial_segment = [d for d in data[0:period] if d is not None]
    if len(initial_segment) < period: 
        for i in range(period -1, len(data)):
            window = [d for d in data[i-period+1 : i+1] if d is not None]
            if len(window) == period:
                sma_values_strict.append(sum(window) / period)
                start_idx_for_cont = i + 1
                for j in range(start_idx_for_cont, len(data)):
                    prev_val = data[j-period]
                    curr_val = data[j]
                    last_sma_val = sma_values_strict[-1]
                    if last_sma_val is not None and prev_val is not None and curr_val is not None:
                        sma_values_strict.append(last_sma_val + (curr_val - prev_val) / period)
                    else: 
                        window_cont = [d_cont for d_cont in data[j-period+1 : j+1] if d_cont is not None]
                        if len(window_cont) == period:
                            sma_values_strict.append(sum(window_cont) / period)
                        else:
                            sma_values_strict.append(None)
                # Pad remaining if loop broke early
                remaining_padding = len(data) - len(sma_values_strict)
                if remaining_padding > 0:
                    sma_values_strict.extend([None] * remaining_padding)
                break 
            else: 
                sma_values_strict.append(None)
        return sma_values_strict
    
    current_sum_val = sum(initial_segment) 
    sma_values_strict.append(current_sum_val / period)
    
    for i in range(period, len(data)):
        if data[i] is not None and data[i-period] is not None and sma_values_strict[-1] is not None:
            current_sum_val += data[i] - data[i-period]
            sma_values_strict.append(current_sum_val / period)
        else: 
            window = [d for d in data[i-period+1 : i+1] if d is not None]
            if len(window) == period:
                sma_values_strict.append(sum(window)/period)
            else:
                sma_values_strict.append(None)
    return sma_values_strict


def _calculate_ema(data: List[Optional[float]], period: int) -> List[Optional[float]]:
    if not data or period <= 0:
        return [None] * len(data)
    
    ema_values: List[Optional[float]] = [None] * len(data)
    multiplier = 2 / (period + 1)
    
    first_sma_val = None
    first_sma_idx = -1

    for i in range(period - 1, len(data)):
        window = [d for d in data[i - period + 1 : i + 1] if d is not None]
        if len(window) == period:
            first_sma_val = sum(window) / period
            first_sma_idx = i
            ema_values[i] = first_sma_val
            break
            
    if first_sma_idx == -1: 
        return ema_values

    for i in range(first_sma_idx + 1, len(data)):
        if data[i] is not None:
            if ema_values[i-1] is not None:
                ema_val = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
                ema_values[i] = ema_val
            else:
                # Attempt to re-initialize EMA
                window = [d for d in data[i - period + 1 : i + 1] if d is not None]
                if len(window) == period:
                     ema_values[i] = sum(window) / period 
                # else: ema_values[i] = None # Implicitly None already
        # else: ema_values[i] = None # Implicitly None already
            
    return ema_values

def _calculate_rma(data: List[Optional[float]], period: int) -> List[Optional[float]]:
    """Wilder's Smoothing Average (Running Moving Average)"""
    if not data or period <= 0 :
        return [None] * len(data)

    rma_values: List[Optional[float]] = [None] * len(data)
    alpha = 1 / period

    first_rma_val = None
    first_rma_idx = -1

    for i in range(period - 1, len(data)):
        window = [d for d in data[i - period + 1 : i + 1] if d is not None]
        if len(window) == period:
            first_rma_val = sum(window) / period
            first_rma_idx = i
            rma_values[i] = first_rma_val
            break
            
    if first_rma_idx == -1: 
        return rma_values

    for i in range(first_rma_idx + 1, len(data)):
        if data[i] is not None:
            if rma_values[i-1] is not None:
                rma_val = alpha * data[i] + (1 - alpha) * rma_values[i-1]
                rma_values[i] = rma_val
            else: 
                window = [d for d in data[i - period + 1 : i + 1] if d is not None]
                if len(window) == period:
                     rma_values[i] = sum(window) / period
                # else: rma_values[i] = None
        # else: rma_values[i] = None
            
    return rma_values


def _calculate_tr(highs: List[Optional[float]], lows: List[Optional[float]], closes: List[Optional[float]]) -> List[Optional[float]]:
    n = len(closes)
    if n == 0: return []
    
    tr_values: List[Optional[float]] = [None] * n
    if n > 0 and highs[0] is not None and lows[0] is not None:
         tr_values[0] = highs[0] - lows[0]

    for i in range(1, n):
        if highs[i] is None or lows[i] is None or closes[i-1] is None:
            tr_values[i] = None
            continue
        h_minus_l = highs[i] - lows[i]
        h_minus_pc = abs(highs[i] - closes[i-1])
        l_minus_pc = abs(lows[i] - closes[i-1])
        tr_values[i] = max(h_minus_l, h_minus_pc, l_minus_pc)
    return tr_values

def _calculate_directional_movement(highs: List[Optional[float]], lows: List[Optional[float]]) -> Dict[str, List[Optional[float]]]:
    n = len(highs)
    if n < 2:
        return {"plus_dm": [None]*n, "minus_dm": [None]*n}

    plus_dm_values: List[Optional[float]] = [None] * n
    minus_dm_values: List[Optional[float]] = [None] * n 

    for i in range(1, n):
        if highs[i] is None or lows[i] is None or highs[i-1] is None or lows[i-1] is None:
            # plus_dm_values[i] = None # Implicitly None
            # minus_dm_values[i] = None # Implicitly None
            continue

        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]

        current_plus_dm = 0.0
        current_minus_dm = 0.0

        if up_move > down_move and up_move > 0:
            current_plus_dm = up_move
        
        if down_move > up_move and down_move > 0:
            current_minus_dm = down_move
        
        plus_dm_values[i] = current_plus_dm
        minus_dm_values[i] = current_minus_dm
        
    return {"plus_dm": plus_dm_values, "minus_dm": minus_dm_values}

def _calculate_donchian_channel_ichimoku_helper(highs: List[Optional[float]], lows: List[Optional[float]], period: int) -> List[Optional[float]]:
    n = len(highs)
    if n < period or period <=0:
        return [None] * n
    
    results: List[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        high_slice_raw = highs[i - period + 1 : i + 1]
        low_slice_raw = lows[i - period + 1 : i + 1]

        high_slice = [x for x in high_slice_raw if x is not None]
        low_slice = [x for x in low_slice_raw if x is not None]

        if not high_slice or not low_slice : # Check if lists are empty after filtering
            results[i] = None
            continue
        
        # Ensure enough valid data points for the period
        # This check was slightly different from original, ensuring >0 length after filtering, not == period
        # Reverting to stricter "== period" if that's intended for Donchian.
        # However, for robustness if there are a few Nones, max/min of available valid points is often used.
        # Let's assume max/min of available if any valid point exists in window.
        # If strict 'period' count is needed:
        # if len(high_slice) < period or len(low_slice) < period:
        #     results[i] = None
        #     continue
            
        highest = max(high_slice)
        lowest = min(low_slice)
        results[i] = (highest + lowest) / 2
    return results

# --- Main Indicator Functions (using fixed default periods) ---

def calculate_rsi(closes: List[Optional[float]]) -> List[Optional[float]]:
    period = 14 # Standard RSI period
    n = len(closes)
    if n <= period : 
        return [None] * n

    rsi_values: List[Optional[float]] = [None] * n 
    deltas: List[Optional[float]] = [None] * n
    for i in range(1,n):
        if closes[i] is not None and closes[i-1] is not None:
            deltas[i] = closes[i] - closes[i-1]
        # else: deltas[i] = None # Implicitly None

    first_valid_avg_idx = -1
    avg_gain = 0.0 # Must be float
    avg_loss = 0.0 # Must be float

    for i in range(period, n): 
        if i == period: 
            current_gains_sum = 0.0
            current_losses_sum = 0.0
            valid_deltas_count = 0
            for k in range(1, period + 1): 
                if deltas[k] is not None:
                    valid_deltas_count +=1
                    if deltas[k] > 0: current_gains_sum += deltas[k]
                    else: current_losses_sum -= deltas[k] 
            
            if valid_deltas_count == period: # Strict check for full initial window
                avg_gain = current_gains_sum / period
                avg_loss = current_losses_sum / period
                first_valid_avg_idx = period 
                
                if avg_loss == 0: current_rsi_val = 100.0 if avg_gain > 0 else 50.0 # Avoid division by zero
                else: rs_val = avg_gain / avg_loss; current_rsi_val = 100.0 - (100.0 / (1.0 + rs_val))
                rsi_values[period] = current_rsi_val
            else: 
                continue # Cannot start, try next index if logic allows (it doesn't here for fixed start)
        
        elif first_valid_avg_idx != -1: 
            if deltas[i] is None: 
                # avg_gain = None # This would break future calculations, better to skip or carry forward
                # avg_loss = None
                rsi_values[i] = None # RSI for this point is None
                first_valid_avg_idx = -1 # Invalidate, attempt re-initialization if a future point allows
                continue

            change = deltas[i] # This is already checked for None above
            gain = change if change > 0 else 0.0
            loss = -change if change < 0 else 0.0

            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            if avg_loss == 0: current_rsi_val = 100.0 if avg_gain > 0 else 50.0
            else: rs_val = avg_gain / avg_loss; current_rsi_val = 100.0 - (100.0 / (1.0 + rs_val))
            rsi_values[i] = current_rsi_val
        # else: we are still looking for the first valid starting point
            
    return rsi_values


def calculate_macd(closes: List[Optional[float]]) -> Dict[str, List[Optional[float]]]:
    fast_length = 12
    slow_length = 26
    signal_length = 9
    sma_source = "EMA" 
    sma_signal = "EMA"
    
    n = len(closes)
    # Ensure enough data for the longest MA + signal line
    if n < slow_length + signal_length -1 : # Max length needed for EMA start + signal line start
        return {"macd": [None]*n, "signal": [None]*n, "hist": [None]*n}

    ma_function = _calculate_sma if sma_source == "SMA" else _calculate_ema
    signal_ma_function = _calculate_sma if sma_signal == "SMA" else _calculate_ema

    fast_ma = ma_function(closes, fast_length)
    slow_ma = ma_function(closes, slow_length)

    macd_line: List[Optional[float]] = [None] * n
    for i in range(n):
        if fast_ma[i] is not None and slow_ma[i] is not None:
            macd_line[i] = fast_ma[i] - slow_ma[i]
    
    signal_line = signal_ma_function(macd_line, signal_length)

    hist_line: List[Optional[float]] = [None] * n
    for i in range(n):
        if macd_line[i] is not None and signal_line[i] is not None:
            hist_line[i] = macd_line[i] - signal_line[i]
            
    return {"macd": macd_line, "signal": signal_line, "hist": hist_line}


def calculate_adx(
    highs: List[Optional[float]], 
    lows: List[Optional[float]], 
    closes: List[Optional[float]]
) -> Dict[str, List[Optional[float]]]:
    di_len = 14
    adx_len = 14
    
    n = len(closes)
    # Min length for ADX: di_len for smoothed DMs/TR, then adx_len for smoothing DX
    min_len_needed = di_len + adx_len -1 
    if n < min_len_needed:
        return {"plus_di": [None]*n, "minus_di": [None]*n, "adx": [None]*n}

    tr_series = _calculate_tr(highs, lows, closes)
    dm_data = _calculate_directional_movement(highs, lows)
    plus_dm_series = dm_data["plus_dm"]
    minus_dm_series = dm_data["minus_dm"]

    smoothed_tr = _calculate_rma(tr_series, di_len)
    smoothed_plus_dm = _calculate_rma(plus_dm_series, di_len)
    smoothed_minus_dm = _calculate_rma(minus_dm_series, di_len)

    plus_di: List[Optional[float]] = [None] * n
    minus_di: List[Optional[float]] = [None] * n
    
    for i in range(n):
        if smoothed_tr[i] is not None and smoothed_tr[i] != 0 and \
           smoothed_plus_dm[i] is not None and smoothed_minus_dm[i] is not None:
            plus_di[i] = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100
            minus_di[i] = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100

    dx_series: List[Optional[float]] = [None] * n
    for i in range(n):
        if plus_di[i] is not None and minus_di[i] is not None:
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx_series[i] = (abs(plus_di[i] - minus_di[i]) / di_sum) * 100
            else:
                dx_series[i] = 0.0 # DX is 0 if sum of DIs is 0
            
    adx_line = _calculate_rma(dx_series, adx_len)
    
    return {"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line}

def calculate_atr(highs: List[Optional[float]], lows: List[Optional[float]], closes: List[Optional[float]]) -> List[Optional[float]]:
    period = 14 # Standard ATR period
    if period <= 0:
        raise ValueError("ATR period must be positive.") # Should not happen with hardcoded positive value
    tr_series = _calculate_tr(highs, lows, closes)
    return _calculate_rma(tr_series, period)

def calculate_supertrend(
    highs: List[Optional[float]], 
    lows: List[Optional[float]], 
    closes: List[Optional[float]]
) -> Dict[str, List[Any]]: 
    atr_period = 10
    factor = 3.0

    n = len(closes)
    min_len = atr_period + 1 # ATR needs 'atr_period' data, Supertrend needs one ATR value to start
    if n < min_len :
        return {"supertrend": [None]*n, "direction": [None]*n}

    atr_values = calculate_atr(highs, lows, closes) # ATR uses its own fixed period (14) by default if not passed

    supertrend_line: List[Optional[float]] = [None] * n
    direction_line: List[Optional[int]] = [None] * n # 1 for uptrend, -1 for downtrend

    first_valid_atr_idx = -1
    for i, atr_val in enumerate(atr_values):
        if atr_val is not None:
            first_valid_atr_idx = i
            break
    
    if first_valid_atr_idx == -1 or first_valid_atr_idx >= n:
        return {"supertrend": [None]*n, "direction": [None]*n}
    
    # Start calculation from the first valid ATR (which depends on ATR's own period)
    # Supertrend itself uses `atr_period` (10 for this func) for its logic related to ATR array indexing if it were dynamic.
    # Since ATR is pre-calculated, we just need a valid ATR value.
    
    # Initial Supertrend:
    # Need highs, lows, closes, and atr_values at first_valid_atr_idx
    if highs[first_valid_atr_idx] is None or \
       lows[first_valid_atr_idx] is None or \
       closes[first_valid_atr_idx] is None or \
       atr_values[first_valid_atr_idx] is None:
        # Not enough data at the first ATR point, propagate None up to this point
        return {"supertrend": [None]*n, "direction": [None]*n} # Or fill up to first_valid_atr_idx

    # Initialize first Supertrend value and direction
    # Basic Upper Band = ((High + Low) / 2) + Multiplier * ATR
    # Basic Lower Band = ((High + Low) / 2) - Multiplier * ATR
    # Final Upper Band = if BasicUpperBand < PrevFinalUpperBand or PrevClose > PrevFinalUpperBand then BasicUpperBand else PrevFinalUpperBand
    # Final Lower Band = if BasicLowerBand > PrevFinalLowerBand or PrevClose < PrevFinalLowerBand then BasicLowerBand else PrevFinalLowerBand
    # SuperTrend = if PrevSuperTrend == PrevFinalUpperBand and Close < FinalUpperBand then FinalUpperBand
    #             else if PrevSuperTrend == PrevFinalUpperBand and Close > FinalUpperBand then FinalLowerBand
    #             else if PrevSuperTrend == PrevFinalLowerBand and Close > FinalLowerBand then FinalLowerBand
    #             else if PrevSuperTrend == PrevFinalLowerBand and Close < FinalLowerBand then FinalUpperBand
    #             else FinalLowerBand (initial state or if conditions fail)
    
    # Simpler common implementation:
    # Up = (high + low) / 2 - factor * atr
    # Dn = (high + low) / 2 + factor * atr
    # Trend up: if close > previous Up, then current Up = max(Up, previous Up)
    # Trend down: if close < previous Dn, then current Dn = min(Dn, previous Dn)
    # If close crosses previous Dn -> trend becomes up. If close crosses previous Up -> trend becomes down.

    # Using the logic from the user's original Python code which seemed to follow a common variant:
    supertrend_line[first_valid_atr_idx] = (highs[first_valid_atr_idx] + lows[first_valid_atr_idx]) / 2 - \
                                           factor * atr_values[first_valid_atr_idx]
    direction_line[first_valid_atr_idx] = 1 # Initial assumption: uptrend

    for i in range(first_valid_atr_idx + 1, n):
        if highs[i] is None or lows[i] is None or closes[i] is None or \
           closes[i-1] is None or atr_values[i] is None or \
           supertrend_line[i-1] is None or direction_line[i-1] is None:
            # Propagate None if data is missing for calculation
            # supertrend_line[i] = None # Implicitly
            # direction_line[i] = None # Implicitly
            continue

        hl2 = (highs[i] + lows[i]) / 2
        basic_upper_band = hl2 + factor * atr_values[i]
        basic_lower_band = hl2 - factor * atr_values[i]
        
        prev_st = supertrend_line[i-1]
        prev_dir = direction_line[i-1]

        current_st = prev_st
        current_dir = prev_dir

        if prev_dir == 1: # Previous trend was up
            if closes[i] < prev_st: # Price crossed below previous Supertrend (which was a lower band)
                current_st = basic_upper_band # Switch to upper band
                current_dir = -1
            else: # Price stayed above
                current_st = max(basic_lower_band, prev_st) # Trail the lower band up
                current_dir = 1 
        elif prev_dir == -1: # Previous trend was down
            if closes[i] > prev_st: # Price crossed above previous Supertrend (which was an upper band)
                current_st = basic_lower_band # Switch to lower band
                current_dir = 1
            else: # Price stayed below
                current_st = min(basic_upper_band, prev_st) # Trail the upper band down
                current_dir = -1
        
        supertrend_line[i] = current_st
        direction_line[i] = current_dir
            
    return {"supertrend": supertrend_line, "direction": direction_line}


def calculate_ichimoku_cloud(
    highs: List[Optional[float]], 
    lows: List[Optional[float]], 
    closes: List[Optional[float]]
) -> Dict[str, List[Optional[float]]]:
    conversion_periods = 9
    base_periods = 26
    lagging_span_2_periods = 52 # Senkou Span B period
    displacement = 26 # For Senkou Spans and Chikou Span plotting reference

    n = len(closes)
    # Minimum length needed is max of all lookback periods for calculation, 
    # plus displacement for plotting the cloud forward.
    min_len_for_calc = max(conversion_periods, base_periods, lagging_span_2_periods)
    if n < min_len_for_calc : # Not enough data even for initial calculations
        return {
            "conversion_line": [None]*n, "base_line": [None]*n, "lead_line1": [None]*n, 
            "lead_line2": [None]*n, "lagging_span": [None]*n, "displacement_val": displacement
        }

    conversion_line = _calculate_donchian_channel_ichimoku_helper(highs, lows, conversion_periods)
    base_line = _calculate_donchian_channel_ichimoku_helper(highs, lows, base_periods)
    
    # Senkou Span A (Lead Line 1) Calculation part
    lead_line1_calc: List[Optional[float]] = [None] * n
    for i in range(n):
        if conversion_line[i] is not None and base_line[i] is not None:
            lead_line1_calc[i] = (conversion_line[i] + base_line[i]) / 2
    
    # Senkou Span B (Lead Line 2) Calculation part
    lead_line2_calc = _calculate_donchian_channel_ichimoku_helper(highs, lows, lagging_span_2_periods)

    # Displace Senkou Spans forward
    lead_line1_displaced: List[Optional[float]] = [None] * n
    lead_line2_displaced: List[Optional[float]] = [None] * n

    for i in range(n):
        if i + displacement < n:
            if lead_line1_calc[i] is not None:
                lead_line1_displaced[i + displacement] = lead_line1_calc[i]
            if lead_line2_calc[i] is not None:
                lead_line2_displaced[i + displacement] = lead_line2_calc[i]

    # Lagging Span (Chikou Span): Current close plotted `displacement` periods in the PAST.
    lagging_span: List[Optional[float]] = [None] * n
    for i in range(n):
        if i - displacement >= 0:
            lagging_span[i - displacement] = closes[i] # Plot closes[i] at index i-displacement
        
    return {
        "conversion_line": conversion_line, 
        "base_line": base_line,             
        "lead_line1": lead_line1_displaced, # Senkou Span A (displaced)
        "lead_line2": lead_line2_displaced, # Senkou Span B (displaced)
        "lagging_span": lagging_span,       # Chikou Span (displaced)
        "displacement_val": displacement    # For reference
    }


def calculate_parabolic_sar(
    highs: List[Optional[float]], 
    lows: List[Optional[float]]
) -> List[Optional[float]]:
    start_af = 0.02 
    increment_af = 0.02 
    max_af = 0.2
    
    n = len(highs)
    if n < 2: return [None]*n # Need at least 2 points for first calculation step
    
    sar_values: List[Optional[float]] = [None] * n
    # Initial values needed for the first element if data exists
    if highs[0] is None or lows[0] is None: return sar_values # Cannot start

    # Initialization (common practice: SAR for day 1 is previous day's Low if uptrend, High if downtrend.
    # If day 1 is the very first data point, some assume initial trend or use first price.)
    # User's code: sar_values[0] = lows[0] - implies assuming an initial uptrend was just prior or starting.
    # This is a common simplification for the very first point.
    
    # We need to determine initial trend for day 1 to set sar_values[1]
    # For day 1 (index 0), SAR is not typically calculated. Calculation starts from day 2 (index 1).
    # Let's assume sar_values[0] is None, and calculation starts for sar_values[1].
    # For sar_values[1]:
    #   If C[0] > C[-1] (hypothetical previous day), uptrend. SAR[1] = Low[0]. EP = High[0].
    #   If C[0] < C[-1], downtrend. SAR[1] = High[0]. EP = Low[0].
    # Since we don't have C[-1], a common approach is to look at C[1] vs C[0].
    # If C[1] > C[0], assume uptrend starting, SAR[1] = Low[0], EP = High[0].
    # Else, assume downtrend, SAR[1] = High[0], EP = Low[0].

    # User's code starts SAR at index 0 and EP based on index 0 data. This means SAR[0] is more of a placeholder.
    # Let's follow the user's provided logic structure closely.
    
    # sar_values[0] = lows[0] # User's original init
    # ep = highs[0] # User's original init (if assuming uptrend from this point)
    
    # More robust initialization for loop starting at i=1:
    # Need sar_values[i-1], which means sar_values[0] must be set.
    # If we determine trend based on first two points (highs[1]/lows[1] vs highs[0]/lows[0]):
    if highs[1] is None or lows[1] is None: return sar_values # Need second point

    is_uptrend = True # Initial assumption
    if (highs[1] + lows[1]) / 2 < (highs[0] + lows[0]) / 2 : # If second day's mid is lower
        is_uptrend = False
    
    if is_uptrend:
        sar_values[0] = lows[0] # SAR for the period ending at highs[0]/lows[0]
        ep = highs[0]       # Extreme price during this first period
    else:
        sar_values[0] = highs[0]
        ep = lows[0]
    
    af = start_af

    for i in range(1, n):
        if highs[i] is None or lows[i] is None or sar_values[i-1] is None:
            # Propagate None, reset state for potential re-initialization
            is_uptrend = True # Default assumption for next valid point
            ep = None 
            af = start_af
            sar_values[i] = None
            continue
        
        # If previous SAR was None, we need to re-initialize based on current point `i`
        # This part is tricky. For simplicity, if prev_sar is None, current SAR is also None.
        # A full re-init would require looking at highs[i] vs highs[i-1] again.

        prev_sar = sar_values[i-1]
        
        if is_uptrend:
            sar_candidate = prev_sar + af * (ep - prev_sar)
            # PSAR cannot be higher than current or previous low in an uptrend
            sar_values[i] = min(sar_candidate, lows[i], lows[i-1] if lows[i-1] is not None else lows[i])


            if highs[i] > ep: # New extreme high
                ep = highs[i]
                af = min(af + increment_af, max_af)
            
            if sar_values[i] > lows[i]: # Trend reversal
                is_uptrend = False
                sar_values[i] = ep # SAR is the prior EP
                ep = lows[i]       # New EP is current low
                af = start_af
        else: # Downtrend
            sar_candidate = prev_sar + af * (ep - prev_sar)
            # PSAR cannot be lower than current or previous high in a downtrend
            sar_values[i] = max(sar_candidate, highs[i], highs[i-1] if highs[i-1] is not None else highs[i])

            if lows[i] < ep: # New extreme low
                ep = lows[i]
                af = min(af + increment_af, max_af)

            if sar_values[i] < highs[i]: # Trend reversal
                is_uptrend = True
                sar_values[i] = ep  # SAR is the prior EP
                ep = highs[i]       # New EP is current high
                af = start_af
                
    return sar_values


def calculate_williams_r(highs: List[Optional[float]], lows: List[Optional[float]], closes: List[Optional[float]]) -> List[Optional[float]]:
    length = 14
    n = len(closes)
    if n < length or length <=0: # length check implicitly handles n=0
        return [None]*n
        
    percent_r_values: List[Optional[float]] = [None] * n
    for i in range(length - 1, n):
        if closes[i] is None:
            # percent_r_values[i] = None # Implicitly
            continue
        
        h_slice_raw = highs[i - length + 1 : i + 1]
        l_slice_raw = lows[i - length + 1 : i + 1]

        h_slice = [x for x in h_slice_raw if x is not None]
        l_slice = [x for x in l_slice_raw if x is not None]

        if not h_slice or not l_slice: # Not enough valid data in window
            # percent_r_values[i] =None # Implicitly
            continue

        highest_high = max(h_slice)
        lowest_low = min(l_slice)
        
        if highest_high == lowest_low: # Avoid division by zero
            # Behavior can vary: 0, -50, or -100 depending on interpretation.
            # If HH=LL and Close=HH, then num is 0, so %R is 0.
            # If HH=LL and Close=LL, then num is 0, so %R is 0.
            # If close is also HH/LL, then result is 0 * -100 = 0.
            # Let's use 0 as per user's previous code.
            percent_r_values[i] = 0.0 
        else:
            pr = ((highest_high - closes[i]) / (highest_high - lowest_low)) * -100
            percent_r_values[i] = pr
            
    return percent_r_values


def calculate_vwap(
    highs: List[Optional[float]], 
    lows: List[Optional[float]], 
    closes: List[Optional[float]], 
    volumes: List[Optional[float]]
) -> Dict[str, List[Optional[float]]]: 
    # VWAP is typically session-based (resets daily). This is a continuous cumulative VWAP.
    # The agent might provide daily data, so it effectively becomes daily if data is single day.
    # If data spans multiple days, this will be cumulative over all days.
    
    n = len(closes)
    if n < 1:
        return {"vwap": []}

    vwap_values: List[Optional[float]] = [None] * n
    cumulative_price_volume = 0.0
    cumulative_volume = 0.0

    for i in range(n):
        current_high = highs[i]
        current_low = lows[i]
        current_close = closes[i]
        current_volume_val = volumes[i]

        if current_high is None or current_low is None or current_close is None or \
           current_volume_val is None or current_volume_val == 0:
            # If any essential component is missing or volume is zero, VWAP for this point is None.
            # Cumulative sums do not update with this point's data.
            # The VWAP will carry over the previous valid value if sums are > 0.
            if cumulative_volume > 0 : # If there was a previous valid VWAP
                 vwap_values[i] = cumulative_price_volume / cumulative_volume
            # else: vwap_values[i] is None (implicitly)
            continue

        typical_price = (current_high + current_low + current_close) / 3
        
        cumulative_price_volume += typical_price * current_volume_val
        cumulative_volume += current_volume_val
        
        if cumulative_volume == 0: # Should not happen if current_volume_val > 0 check passed
            vwap_values[i] = None
        else:
            vwap_values[i] = cumulative_price_volume / cumulative_volume
            
    return {"vwap": vwap_values} 


def calculate_bollinger_bands(closes: List[Optional[float]]) -> Dict[str, List[Optional[float]]]:
    length = 20
    ma_type = "SMA" # Standard BB uses SMA
    mult = 2.0

    n = len(closes)
    if n < length or length <= 0:
        return {"basis": [None]*n, "upper": [None]*n, "lower": [None]*n}

    if ma_type == "SMA": ma_function = _calculate_sma
    elif ma_type == "EMA": ma_function = _calculate_ema
    # elif ma_type == "RMA" or ma_type == "SMMA": ma_function = _calculate_rma # Not typically used for BB basis
    else: ma_function = _calculate_sma 

    basis_line = ma_function(closes, length)
    
    upper_band: List[Optional[float]] = [None] * n
    lower_band: List[Optional[float]] = [None] * n

    for i in range(length - 1, n): # Start from where basis_line can be calculated
        if basis_line[i] is None: 
            continue # If basis MA is None, bands cannot be calculated
        
        # Calculate standard deviation for the window ending at `i`
        price_slice_raw = closes[i - length + 1 : i + 1]
        price_slice = [p for p in price_slice_raw if p is not None]

        if len(price_slice) < length: # Not enough valid data points in window for stdev
            # This check ensures we use 'length' data points for stdev, matching typical BB.
            # If allowing fewer points, stdev calculation might change (e.g. sample stdev).
            continue

        mean_of_slice = sum(price_slice) / len(price_slice) # Mean of the actual valid data points in the slice
                                                           # Note: basis_line[i] is MA of full window if all valid.
                                                           # For stdev, use actual mean of the slice used for stdev.
                                                           # However, standard BB typically uses the MA (basis_line[i]) as the mean for variance calc.

        # Standard BB uses the MA (basis_line[i]) as the mean for variance calculation.
        variance = sum([(p - basis_line[i])**2 for p in price_slice]) / len(price_slice) # Using len(price_slice) not 'length' if Nones exist
                                                                                          # Or, if strict, ensure len(price_slice) == length
        stdev = math.sqrt(variance)
        
        upper_band[i] = basis_line[i] + mult * stdev
        lower_band[i] = basis_line[i] - mult * stdev
        
    return {"basis": basis_line, "upper": upper_band, "lower": lower_band}


# --- Master Decision Logic (get_technical_analysis_summary) ---
# This function calls the above main indicator functions.
# Since those functions now use hardcoded periods, no period arguments are passed here.
def get_technical_analysis_summary(
    opens: List[Optional[float]],
    highs: List[Optional[float]],
    lows: List[Optional[float]],
    closes: List[Optional[float]],
    volumes: List[Optional[float]],
    selected_indicators_list: Optional[List[str]] = None
) -> Dict[str, Any]:

    num_data_points = len(closes)
    all_possible_indicators = [
        "RSI", "EMA", "SMA", "MACD", "ADX", "Supertrend", 
        "BollingerBands", "VWAP", "WilliamsR", "PSAR", "Ichimoku", "ATR"
    ]

    if num_data_points < 60: # Heuristic minimum for multiple indicators
        return {
            "signals": [{"name": indicator, "value": None, "decision": "Neutral"} 
                        for indicator in all_possible_indicators],
            "overallSignal": "Neutral",
            "raw_score_debug": 0.0
        }
    
    indicators_to_process = selected_indicators_list if selected_indicators_list else all_possible_indicators

    # Calculate all indicators - no period arguments passed
    rsi14_series = calculate_rsi(closes) # Uses fixed 14 period
    ema9_series = _calculate_ema(closes, 9) # EMA 9 for decision logic
    ema20_series = _calculate_ema(closes, 20) # EMA 20 if needed (or a calculate_ema_standard could exist)
    sma20_series = _calculate_sma(closes, 20) # SMA 20 for decision logic

    macd_data = calculate_macd(closes) # Uses fixed 12,26,9
    adx_data = calculate_adx(highs, lows, closes) # Uses fixed 14,14
    supertrend_data = calculate_supertrend(highs, lows, closes) # Uses fixed 10,3 with ATR 14
    bollinger_data = calculate_bollinger_bands(closes) # Uses fixed 20,2
    vwap_data = calculate_vwap(highs, lows, closes, volumes) # No period
    williams_r_series = calculate_williams_r(highs, lows, closes) # Uses fixed 14
    psar_series = calculate_parabolic_sar(highs, lows) # Uses fixed 0.02,0.02,0.2
    ichimoku_data = calculate_ichimoku_cloud(highs, lows, closes) # Uses fixed 9,26,52,26
    atr_series = calculate_atr(highs, lows, closes) # Uses fixed 14

    latest_price = _get_latest(closes)
    prev_close = _get_nth_latest(closes, 1) 
    
    displacement_val = ichimoku_data.get("displacement_val", 26) 

    latest_rsi = _get_latest(rsi14_series)
    latest_ema9 = _get_latest(ema9_series) # EMA 9 is used in decision logic
    latest_sma20 = _get_latest(sma20_series) # SMA 20 is used
    
    latest_macd_line = _get_latest(macd_data.get("macd", []))
    latest_macd_signal = _get_latest(macd_data.get("signal", []))
    latest_adx_line = _get_latest(adx_data.get("adx", []))
    latest_plus_di = _get_latest(adx_data.get("plus_di", []))
    latest_minus_di = _get_latest(adx_data.get("minus_di", []))
    latest_supertrend_val = _get_latest(supertrend_data.get("supertrend", []))
    latest_supertrend_dir = _get_latest(supertrend_data.get("direction", []))
    latest_bb_upper = _get_latest(bollinger_data.get("upper", []))
    latest_bb_lower = _get_latest(bollinger_data.get("lower", []))
    latest_bb_middle = _get_latest(bollinger_data.get("basis", []))
    latest_vwap = _get_latest(vwap_data.get("vwap", []))
    latest_williams_r = _get_latest(williams_r_series)
    latest_psar = _get_latest(psar_series)
    latest_tenkan_sen = _get_latest(ichimoku_data.get("conversion_line", []))
    latest_kijun_sen = _get_latest(ichimoku_data.get("base_line", []))
    latest_senkou_a = _get_latest(ichimoku_data.get("lead_line1",[])) 
    latest_senkou_b = _get_latest(ichimoku_data.get("lead_line2",[])) 
    
    chikou_latest_relevant_close = None
    if displacement_val > 0: # This refers to Ichimoku's displacement
         chikou_latest_relevant_close = _get_nth_latest(closes, displacement_val - 1) 
    price_to_compare_chikou_against = _get_nth_latest(closes, displacement_val)
    
    latest_atr = _get_latest(atr_series)

    indicator_decisions_map = {key: "Neutral" for key in all_possible_indicators}
    indicator_values_map = {key: None for key in all_possible_indicators}

    if latest_price is None: # Cannot make decisions without current price
        # Return neutral for all indicators if latest price is missing
        signals_list = [{"name": i_name, "value": None, "decision": "Neutral"} for i_name in all_possible_indicators]
        return {"signals": signals_list, "overallSignal": "Neutral", "raw_score_debug": 0.0}

    # --- Decision Logic (remains largely the same, uses the calculated latest values) ---

    # RSI (14)
    indicator_values_map["RSI"] = latest_rsi
    if latest_rsi is not None:
        if latest_rsi < 20: indicator_decisions_map["RSI"] = "Strong Buy"
        elif latest_rsi < 30: indicator_decisions_map["RSI"] = "Buy"
        elif latest_rsi > 80: indicator_decisions_map["RSI"] = "Strong Sell"
        elif latest_rsi > 70: indicator_decisions_map["RSI"] = "Sell"

    # EMA (9 for this decision logic)
    indicator_values_map["EMA"] = latest_ema9 
    if latest_ema9 is not None:
        if latest_price > latest_ema9 * 1.05: indicator_decisions_map["EMA"] = "Strong Buy"
        elif latest_price > latest_ema9: indicator_decisions_map["EMA"] = "Buy"
        elif latest_price < latest_ema9 * 0.95: indicator_decisions_map["EMA"] = "Strong Sell"
        elif latest_price < latest_ema9: indicator_decisions_map["EMA"] = "Sell"

    # SMA (20)
    indicator_values_map["SMA"] = latest_sma20
    if latest_sma20 is not None:
        if latest_price > latest_sma20 * 1.05: indicator_decisions_map["SMA"] = "Strong Buy"
        elif latest_price > latest_sma20: indicator_decisions_map["SMA"] = "Buy"
        elif latest_price < latest_sma20 * 0.95: indicator_decisions_map["SMA"] = "Strong Sell"
        elif latest_price < latest_sma20: indicator_decisions_map["SMA"] = "Sell"
    
    # MACD (12,26,9)
    indicator_values_map["MACD"] = {"macd": latest_macd_line, "signal": latest_macd_signal}
    if latest_macd_line is not None and latest_macd_signal is not None:
        macd_diff = latest_macd_line - latest_macd_signal
        if latest_macd_line > latest_macd_signal and latest_macd_line > 0 and macd_diff > abs(latest_macd_line) * 0.1:
            indicator_decisions_map["MACD"] = "Strong Buy"
        elif latest_macd_line > latest_macd_signal:
            indicator_decisions_map["MACD"] = "Buy"
        elif latest_macd_line < latest_macd_signal and latest_macd_line < 0 and abs(macd_diff) > abs(latest_macd_line) * 0.1:
            indicator_decisions_map["MACD"] = "Strong Sell"
        elif latest_macd_line < latest_macd_signal:
            indicator_decisions_map["MACD"] = "Sell"

    # ADX (14,14)
    indicator_values_map["ADX"] = {"adx": latest_adx_line, "plusDI": latest_plus_di, "minusDI": latest_minus_di}
    if latest_adx_line is not None and latest_plus_di is not None and latest_minus_di is not None:
        if latest_adx_line > 40 and latest_plus_di > latest_minus_di: indicator_decisions_map["ADX"] = "Strong Buy"
        elif latest_adx_line > 25 and latest_plus_di > latest_minus_di: indicator_decisions_map["ADX"] = "Buy"
        elif latest_adx_line > 40 and latest_minus_di > latest_plus_di: indicator_decisions_map["ADX"] = "Strong Sell"
        elif latest_adx_line > 25 and latest_minus_di > latest_plus_di: indicator_decisions_map["ADX"] = "Sell"

    # Supertrend (10,3 with ATR 14)
    indicator_values_map["Supertrend"] = {"value": latest_supertrend_val, "direction": latest_supertrend_dir}
    if latest_supertrend_dir is not None and latest_supertrend_val is not None:
        if latest_supertrend_dir == 1 and latest_price > latest_supertrend_val:
            indicator_decisions_map["Supertrend"] = "Buy"
        elif latest_supertrend_dir == -1 and latest_price < latest_supertrend_val:
            indicator_decisions_map["Supertrend"] = "Sell"
    
    # Bollinger Bands (20,2)
    indicator_values_map["BollingerBands"] = {"upper": latest_bb_upper, "middle": latest_bb_middle, "lower": latest_bb_lower}
    if latest_bb_upper is not None and latest_bb_lower is not None and latest_bb_middle is not None:
        if latest_price < latest_bb_lower: indicator_decisions_map["BollingerBands"] = "Strong Buy"
        elif latest_price > latest_bb_upper: indicator_decisions_map["BollingerBands"] = "Strong Sell"
        elif latest_price < latest_bb_middle: indicator_decisions_map["BollingerBands"] = "Sell" # Between lower and middle
        elif latest_price > latest_bb_middle: indicator_decisions_map["BollingerBands"] = "Buy"  # Between middle and upper

    # VWAP
    indicator_values_map["VWAP"] = latest_vwap
    if latest_vwap is not None:
        if latest_price > latest_vwap * 1.03: indicator_decisions_map["VWAP"] = "Strong Buy"
        elif latest_price > latest_vwap: indicator_decisions_map["VWAP"] = "Buy"
        elif latest_price < latest_vwap * 0.97: indicator_decisions_map["VWAP"] = "Strong Sell"
        elif latest_price < latest_vwap: indicator_decisions_map["VWAP"] = "Sell"
    
    # Williams %R (14)
    indicator_values_map["WilliamsR"] = latest_williams_r
    if latest_williams_r is not None:
        if latest_williams_r < -80: indicator_decisions_map["WilliamsR"] = "Strong Buy" 
        elif latest_williams_r < -70: indicator_decisions_map["WilliamsR"] = "Buy"
        elif latest_williams_r > -20: indicator_decisions_map["WilliamsR"] = "Strong Sell"
        elif latest_williams_r > -30: indicator_decisions_map["WilliamsR"] = "Sell"

    # Parabolic SAR (0.02,0.02,0.2)
    indicator_values_map["PSAR"] = latest_psar
    if latest_psar is not None and prev_close is not None: 
        # Buy if price crosses above SAR (SAR was resistance, now support)
        # Sell if price crosses below SAR (SAR was support, now resistance)
        # Simpler: current price vs current SAR, and also consider if SAR just flipped
        if latest_price > latest_psar and prev_close < _get_nth_latest(psar_series, 1) if _get_nth_latest(psar_series, 1) is not None else latest_psar : # SAR just flipped below price
             indicator_decisions_map["PSAR"] = "Buy"
        elif latest_price < latest_psar and prev_close > _get_nth_latest(psar_series, 1) if _get_nth_latest(psar_series, 1) is not None else latest_psar: # SAR just flipped above price
             indicator_decisions_map["PSAR"] = "Sell"
        elif latest_price > latest_psar: # General bullish if above SAR
             indicator_decisions_map["PSAR"] = "Buy" # Could be weaker buy
        elif latest_price < latest_psar: # General bearish if below SAR
             indicator_decisions_map["PSAR"] = "Sell" # Could be weaker sell
    
    # Ichimoku Cloud (9,26,52,26)
    indicator_values_map["Ichimoku"] = {
        "tenkan": latest_tenkan_sen, "kijun": latest_kijun_sen,
        "senkouA": latest_senkou_a, "senkouB": latest_senkou_b,
        "chikou_val_for_cond": chikou_latest_relevant_close, 
        "price_for_chikou_cond": price_to_compare_chikou_against 
    }
    if latest_senkou_a is not None and latest_senkou_b is not None and \
       latest_tenkan_sen is not None and latest_kijun_sen is not None and \
       chikou_latest_relevant_close is not None and price_to_compare_chikou_against is not None:
        
        cloud_top = max(latest_senkou_a, latest_senkou_b)
        cloud_bottom = min(latest_senkou_a, latest_senkou_b)
        
        # Tenkan/Kijun cross
        tk_cross_bullish = latest_tenkan_sen > latest_kijun_sen
        tk_cross_bearish = latest_tenkan_sen < latest_kijun_sen

        # Price vs Cloud
        price_above_cloud = latest_price > cloud_top
        price_below_cloud = latest_price < cloud_bottom
        # price_in_cloud = latest_price <= cloud_top and latest_price >= cloud_bottom (Neutral)

        # Chikou Span vs Price (displaced)
        chikou_bullish_confirmation = chikou_latest_relevant_close > price_to_compare_chikou_against
        chikou_bearish_confirmation = chikou_latest_relevant_close < price_to_compare_chikou_against
            
        # Strong signals require alignment of multiple conditions
        if price_above_cloud and tk_cross_bullish and chikou_bullish_confirmation:
            indicator_decisions_map["Ichimoku"] = "Strong Buy"
        elif price_above_cloud and tk_cross_bullish: # Buy without full Chikou confirmation yet
            indicator_decisions_map["Ichimoku"] = "Buy"
        elif price_below_cloud and tk_cross_bearish and chikou_bearish_confirmation:
            indicator_decisions_map["Ichimoku"] = "Strong Sell"
        elif price_below_cloud and tk_cross_bearish: # Sell without full Chikou confirmation yet
            indicator_decisions_map["Ichimoku"] = "Sell"
    
    # ATR (14) - used for volatility context, not direct buy/sell usually
    indicator_values_map["ATR"] = latest_atr # Store the value
    if latest_atr is not None and latest_ema9 is not None: # Decision logic from user's original
        if latest_atr < latest_price * 0.01: 
            indicator_decisions_map["ATR"] = "Neutral" # Low volatility implies Neutral for this logic
        elif latest_price > latest_ema9 and latest_atr > latest_price * 0.02: 
            indicator_decisions_map["ATR"] = "Buy" # High volatility breakout with price above EMA
        elif latest_price < latest_ema9 and latest_atr > latest_price * 0.02: 
            indicator_decisions_map["ATR"] = "Sell" # High volatility breakdown with price below EMA

    # --- Scoring ---
    decision_scores = {"Strong Buy": 2, "Buy": 1, "Neutral": 0, "Sell": -1, "Strong Sell": -2}
    weights = { # Weights can be adjusted based on perceived reliability
        "RSI": 1.0, "EMA": 1.2, "SMA": 1.2, "MACD": 1.5, "ADX": 1.0, "Supertrend": 1.5,
        "BollingerBands": 1.0, "VWAP": 0.8, "WilliamsR": 1.0, "PSAR": 1.2, "Ichimoku": 1.8, "ATR": 0.5
    } # Note: EMA and SMA here refer to the generic MA signals, not specific period MAs if used differently.
      # The decision logic for EMA/SMA is for latest_price vs latest_ema9/latest_sma20.
    
    total_score = 0.0
    active_weight_sum = 0.0 # To normalize score if not all indicators give a signal

    for indicator_name in indicators_to_process: 
        decision = indicator_decisions_map.get(indicator_name, "Neutral")
        # Only consider indicators that gave a non-None value and thus a potential decision
        if indicator_values_map.get(indicator_name) is not None:
            score = decision_scores.get(decision, 0)
            weight = weights.get(indicator_name, 0)
            if score != 0 : # Only include in score if not Neutral for this indicator
                total_score += score * weight
                active_weight_sum += weight

    overall_signal = "Neutral"
    if active_weight_sum > 0: # Avoid division by zero if all are neutral or None
        normalized_score = total_score / active_weight_sum
        if normalized_score >= 1.5 : overall_signal = "Strong Buy"
        elif normalized_score >= 0.5 : overall_signal = "Buy"
        elif normalized_score <= -1.5 : overall_signal = "Strong Sell"
        elif normalized_score <= -0.5 : overall_signal = "Sell"
    
    # Ensure all indicators are in the final list, even if not processed or value is None
    signals_list = []
    for name in all_possible_indicators: 
        signals_list.append({
            "name": name, 
            "value": indicator_values_map.get(name), 
            "decision": indicator_decisions_map.get(name, "Neutral")
        })
        
    return {"signals": signals_list, "overallSignal": overall_signal, "raw_score_debug": total_score, "normalized_score_debug": normalized_score if active_weight_sum > 0 else 0}