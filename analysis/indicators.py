# app/indicators.py
import math
from typing import List, Dict, Any, Optional, Tuple, Union

# Helper to get the last valid item from a list or None
def _get_latest(series: List[Any]) -> Optional[Any]:
    if not series:
        return None
    # Iterate backwards to find the last non-None, non-NaN value
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
    
    # Get the index from the original series corresponding to the nth latest valid item
    target_original_index = valid_items_indices[-(n + 1)]
    return series[target_original_index]

# --- Helper Calculation Functions ---

def _calculate_sma(data: List[Optional[float]], period: int) -> List[Optional[float]]:
    if not data or period <= 0:
        return [None] * len(data)
    
    sma_values: List[Optional[float]] = [None] * len(data)
    current_sum = 0.0
    count = 0
    
    for i in range(len(data)):
        if data[i] is not None:
            current_sum += data[i]
            count += 1
        
        if i >= period and data[i-period] is not None: # Subtract element that is sliding out
            current_sum -= data[i-period]
            count -=1
        
        if i >= period -1 and count == period: # Check count == period to ensure full window of valid numbers
             sma_values[i] = current_sum / period
        elif i >= period -1 and count > 0 and i < period + (period -1): # handle initial part if some Nones exist but still enough points
            # This part can be tricky if Nones are in the initial window.
            # For simplicity, only calculate if a full valid window is available.
            # Let's refine: strict SMA requires 'period' valid numbers.
            # If data can have Nones, a more robust rolling window is needed.
            # The JS map approach (value - slowMa[index] || 0) implies MA funcs return full length arrays.
            pass # Keep as None if not full valid window

    # More standard SMA that pads initial values
    sma_values_strict: List[Optional[float]] = [None] * (period -1)
    if len(data) < period:
        return [None] * len(data)

    # Check if initial segment has enough non-None values
    initial_segment = [d for d in data[0:period] if d is not None]
    if len(initial_segment) < period: # Not enough data for first SMA
        # Try to find the first possible SMA
        for i in range(period -1, len(data)):
            window = [d for d in data[i-period+1 : i+1] if d is not None]
            if len(window) == period:
                sma_values_strict.append(sum(window) / period)
                # Now continue from here
                start_idx_for_cont = i + 1
                for j in range(start_idx_for_cont, len(data)):
                    prev_val = data[j-period]
                    curr_val = data[j]
                    last_sma = sma_values_strict[-1]
                    if last_sma is not None and prev_val is not None and curr_val is not None:
                        sma_values_strict.append(last_sma + (curr_val - prev_val) / period)
                    else: # If gap, try re-calculating SMA for window
                        window_cont = [d for d in data[j-period+1 : j+1] if d is not None]
                        if len(window_cont) == period:
                            sma_values_strict.append(sum(window_cont) / period)
                        else:
                            sma_values_strict.append(None)
                break # Found first SMA, loop for continuation done
            else: # Not enough for this window
                sma_values_strict.append(None)
        return sma_values_strict if len(sma_values_strict) == len(data) else [None]* (len(data) - len(sma_values_strict)) + sma_values_strict

    current_sum_val = sum(initial_segment) # sum of first 'period' valid numbers
    sma_values_strict.append(current_sum_val / period)
    
    for i in range(period, len(data)):
        if data[i] is not None and data[i-period] is not None:
            current_sum_val += data[i] - data[i-period]
            sma_values_strict.append(current_sum_val / period)
        else: # If there's a None in the window, SMA becomes None or needs recalculation
              # For simplicity matching JS `|| 0` behavior for subtraction, better to ensure MA funcs handle internal Nones
              # or input data is pre-cleaned. The current JS MACD implies MA funcs return full length arrays.
            # Recalculate for window if None encountered
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
    
    # Find first valid segment for initial SMA
    first_sma_val = None
    first_sma_idx = -1

    for i in range(period - 1, len(data)):
        window = [d for d in data[i - period + 1 : i + 1] if d is not None]
        if len(window) == period:
            first_sma_val = sum(window) / period
            first_sma_idx = i
            ema_values[i] = first_sma_val
            break
            
    if first_sma_idx == -1: # No valid segment found
        return ema_values

    for i in range(first_sma_idx + 1, len(data)):
        if data[i] is not None:
            if ema_values[i-1] is not None:
                ema_val = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
                ema_values[i] = ema_val
            else:
                # Attempt to re-initialize EMA if previous was None but current data is valid
                # This implies restarting SMA for 'period' lookback if possible
                window = [d for d in data[i - period + 1 : i + 1] if d is not None]
                if len(window) == period:
                     ema_values[i] = sum(window) / period # Restart with SMA
                else:
                     ema_values[i] = None # Cannot restart
        else:
            ema_values[i] = None # Propagate None if current data is None
            
    return ema_values

def _calculate_rma(data: List[Optional[float]], period: int) -> List[Optional[float]]:
    """Wilder's Smoothing Average (Running Moving Average)"""
    if not data or period <= 0 :
        return [None] * len(data)

    rma_values: List[Optional[float]] = [None] * len(data)
    alpha = 1 / period

    # Find first valid segment for initial SMA (which is the first RMA)
    first_rma_val = None
    first_rma_idx = -1

    for i in range(period - 1, len(data)):
        window = [d for d in data[i - period + 1 : i + 1] if d is not None]
        if len(window) == period:
            first_rma_val = sum(window) / period
            first_rma_idx = i
            rma_values[i] = first_rma_val
            break
            
    if first_rma_idx == -1: # No valid segment found
        return rma_values

    for i in range(first_rma_idx + 1, len(data)):
        if data[i] is not None:
            if rma_values[i-1] is not None:
                rma_val = alpha * data[i] + (1 - alpha) * rma_values[i-1]
                rma_values[i] = rma_val
            else: # Attempt to re-initialize if previous was None
                window = [d for d in data[i - period + 1 : i + 1] if d is not None]
                if len(window) == period:
                     rma_values[i] = sum(window) / period # Restart with SMA
                else:
                     rma_values[i] = None
        else:
            rma_values[i] = None
            
    return rma_values


def _calculate_tr(highs: List[Optional[float]], lows: List[Optional[float]], closes: List[Optional[float]]) -> List[Optional[float]]:
    n = len(closes)
    if n == 0: return []
    
    tr_values: List[Optional[float]] = [None] * n
    if n > 0 and highs[0] is not None and lows[0] is not None:
         tr_values[0] = highs[0] - lows[0] # First TR is H-L

    for i in range(1, n):
        if highs[i] is None or lows[i] is None or closes[i-1] is None:
            tr_values[i] = None
            continue
        h_minus_l = highs[i] - lows[i]
        h_minus_pc = abs(highs[i] - closes[i-1])
        l_minus_pc = abs(lows[i] - closes[i-1])
        tr_values[i] = max(h_minus_l, h_minus_pc, l_minus_pc)
    return tr_values

def calculate_atr(highs: List[Optional[float]], lows: List[Optional[float]], closes: List[Optional[float]], period: int = 14) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("ATR period must be positive.")
    tr_series = _calculate_tr(highs, lows, closes)
    return _calculate_rma(tr_series, period)


def _calculate_directional_movement(highs: List[Optional[float]], lows: List[Optional[float]]) -> Dict[str, List[Optional[float]]]:
    n = len(highs)
    if n < 2: # Needs at least two data points for one DM value
        return {"plus_dm": [None]*n, "minus_dm": [None]*n}

    plus_dm_values: List[Optional[float]] = [None] * n
    minus_dm_values: List[Optional[float]] = [None] * n # First value is undefined (None)

    for i in range(1, n):
        if highs[i] is None or lows[i] is None or highs[i-1] is None or lows[i-1] is None:
            plus_dm_values[i] = None
            minus_dm_values[i] = None
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

# --- Main Indicator Functions --- (largely from user's Python, assuming faithful porting of core logic)

def calculate_rsi(closes: List[Optional[float]], period: int = 14) -> List[Optional[float]]:
    n = len(closes)
    if n <= period : 
        return [None] * n

    rsi_values: List[Optional[float]] = [None] * n 
    
    # Calculate deltas ensuring no None propagation issues for delta itself
    deltas: List[Optional[float]] = [None] * n
    for i in range(1,n):
        if closes[i] is not None and closes[i-1] is not None:
            deltas[i] = closes[i] - closes[i-1]
        else:
            deltas[i] = None # Delta is None if components are None

    # Find first period valid deltas for initial avg_gain/loss
    first_valid_avg_idx = -1
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(period, n): # Start loop from where we could have 'period' deltas
        # Window of deltas is deltas[i-period+1 : i+1]
        # However, RSI's first value corresponds to closes[period], using deltas up to delta[period] (which is C[period]-C[period-1])
        # The JS version sums `period` changes, so `deltas[1]` to `deltas[period]`.
        # This means the first RSI can be calculated at index `period` of `closes`.
        
        # Initial gains/losses for first RSI (at index `period` for closes)
        # uses deltas from index 1 to `period`.
        if i == period: # Calculate initial averages
            current_gains_sum = 0.0
            current_losses_sum = 0.0
            valid_deltas_count = 0
            for k in range(1, period + 1): # Deltas from index 1 to period
                if deltas[k] is not None:
                    valid_deltas_count +=1
                    if deltas[k] > 0: current_gains_sum += deltas[k]
                    else: current_losses_sum -= deltas[k] # losses are positive
            
            if valid_deltas_count == period:
                avg_gain = current_gains_sum / period
                avg_loss = current_losses_sum / period
                first_valid_avg_idx = period # RSI computed at this index of `closes`
                
                if avg_loss == 0: current_rsi = 100.0 if avg_gain > 0 else 50.0
                else: rs_val = avg_gain / avg_loss; current_rsi = 100.0 - (100.0 / (1.0 + rs_val))
                rsi_values[period] = current_rsi
            else: # Cannot calculate initial RSI, try next period
                continue 
        
        elif first_valid_avg_idx != -1: # Subsequent RSIs if initial was successful
            if deltas[i] is None: # Current delta is None, cannot update avg_gain/loss smoothly
                avg_gain = None # Invalidate averages
                avg_loss = None
                rsi_values[i] = None
                first_valid_avg_idx = -1 # Force re-initialization attempt on next non-None delta
                continue

            change = deltas[i]
            gain = change if change > 0 else 0.0
            loss = -change if change < 0 else 0.0

            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            if avg_loss == 0: current_rsi = 100.0 if avg_gain > 0 else 50.0
            else: rs_val = avg_gain / avg_loss; current_rsi = 100.0 - (100.0 / (1.0 + rs_val))
            rsi_values[i] = current_rsi
        # If first_valid_avg_idx is still -1, we are searching for a new starting point (after a gap)
        # This re-initialization logic for gaps might be complex and differ from JS if it doesn't handle it.
        # The JS version seems to run continuously. If data is missing, results will be NaN.
    return rsi_values


def calculate_macd(
    closes: List[Optional[float]],
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
    sma_source: str = "EMA", # "EMA" or "SMA"
    sma_signal: str = "EMA"  # "EMA" or "SMA"
) -> Dict[str, List[Optional[float]]]:
    
    n = len(closes)
    if n < slow_length + signal_length: 
        return {"macd": [None]*n, "signal": [None]*n, "hist": [None]*n}

    ma_function = _calculate_sma if sma_source == "SMA" else _calculate_ema
    signal_ma_function = _calculate_sma if sma_signal == "SMA" else _calculate_ema

    fast_ma = ma_function(closes, fast_length)
    slow_ma = ma_function(closes, slow_length)

    macd_line: List[Optional[float]] = [None] * n
    for i in range(n):
        if fast_ma[i] is not None and slow_ma[i] is not None:
            macd_line[i] = fast_ma[i] - slow_ma[i]
    
    signal_line = signal_ma_function(macd_line, signal_length) # macd_line can have Nones, EMA/SMA helpers should handle

    hist_line: List[Optional[float]] = [None] * n
    for i in range(n):
        if macd_line[i] is not None and signal_line[i] is not None:
            hist_line[i] = macd_line[i] - signal_line[i]
            
    return {"macd": macd_line, "signal": signal_line, "hist": hist_line}


def calculate_adx(
    highs: List[Optional[float]], 
    lows: List[Optional[float]], 
    closes: List[Optional[float]], 
    di_len: int = 14, 
    adx_len: int = 14
) -> Dict[str, List[Optional[float]]]:
    
    n = len(closes)
    min_len = di_len + adx_len 
    if n < min_len:
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
           smoothed_plus_dm[i] is not None and smoothed_minus_dm[i] is not None: # Check all components exist
            plus_di[i] = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100
            minus_di[i] = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100

    dx_series: List[Optional[float]] = [None] * n
    for i in range(n):
        if plus_di[i] is not None and minus_di[i] is not None:
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx_series[i] = (abs(plus_di[i] - minus_di[i]) / di_sum) * 100
            else:
                dx_series[i] = 0.0 
            
    adx_line = _calculate_rma(dx_series, adx_len)
    
    return {"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line}


def calculate_supertrend(
    highs: List[Optional[float]], 
    lows: List[Optional[float]], 
    closes: List[Optional[float]], 
    atr_period: int = 10, 
    factor: float = 3.0
) -> Dict[str, List[Any]]: 

    n = len(closes)
    min_len = atr_period + 1 
    if n < min_len :
        return {"supertrend": [None]*n, "direction": [None]*n}

    atr_values = calculate_atr(highs, lows, closes, atr_period)

    supertrend_line: List[Optional[float]] = [None] * n
    direction_line: List[Optional[int]] = [None] * n

    # Find first valid ATR to start Supertrend calculation
    first_valid_atr_idx = -1
    for i, atr_val in enumerate(atr_values):
        if atr_val is not None:
            first_valid_atr_idx = i
            break
    
    if first_valid_atr_idx == -1 or first_valid_atr_idx >= n:
        return {"supertrend": [None]*n, "direction": [None]*n}
    
    start_index = first_valid_atr_idx 

    # Initial Supertrend value
    # Ensure all components are valid at start_index
    if highs[start_index] is None or lows[start_index] is None or atr_values[start_index] is None or closes[start_index] is None:
         return {"supertrend": [None]*n, "direction": [None]*n}

    # JS: supertrend[i] = lowerBand; direction[i] = 1; (for i === atrPeriod)
    # Assuming ATR values are valid from atr_period index onwards in JS
    # Python first_valid_atr_idx could be that start.
    initial_mid_price = (highs[start_index] + lows[start_index]) / 2
    supertrend_line[start_index] = initial_mid_price - factor * atr_values[start_index] # Initial value based on lower band
    direction_line[start_index] = 1 # Initial assumption: uptrend (JS also does this)

    for i in range(start_index + 1, n):
        # Check all required data points for calculation
        if highs[i] is None or lows[i] is None or closes[i] is None or closes[i-1] is None or \
           atr_values[i] is None or supertrend_line[i-1] is None:
            # If data is missing, cannot continue trend, might need to reset or carry forward ST if appropriate.
            # For now, set to None and let next valid point re-evaluate.
            supertrend_line[i] = None 
            direction_line[i] = None
            continue

        basic_upper_band = ((highs[i] + lows[i]) / 2) + (factor * atr_values[i])
        basic_lower_band = ((highs[i] + lows[i]) / 2) - (factor * atr_values[i])
        
        prev_st = supertrend_line[i-1]
        
        # Following the exact JS logic provided in the prompt for supertrend value update:
        if closes[i-1] > prev_st: # If previous close was above previous supertrend value
            supertrend_line[i] = min(basic_lower_band, prev_st) # JS: Math.min(lowerBand, supertrend[i-1])
        else: # Previous close was below or equal to previous supertrend value
            supertrend_line[i] = max(basic_upper_band, prev_st) # JS: Math.max(upperBand, supertrend[i-1])
        
        # Determine current direction based on current close and current ST
        if closes[i] > supertrend_line[i]:
            direction_line[i] = 1
        elif closes[i] < supertrend_line[i]:
            direction_line[i] = -1
        else: # close == supertrend_line[i]
            direction_line[i] = direction_line[i-1] if i > 0 and direction_line[i-1] is not None else 1 

    return {"supertrend": supertrend_line, "direction": direction_line}

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

        if len(high_slice) < period or len(low_slice) < period : # Ensure full valid window
            results[i] = None
            continue
            
        highest = max(high_slice)
        lowest = min(low_slice)
        results[i] = (highest + lowest) / 2
    return results


def calculate_ichimoku_cloud(
    highs: List[Optional[float]], 
    lows: List[Optional[float]], 
    closes: List[Optional[float]],
    conversion_periods: int = 9,
    base_periods: int = 26,
    lagging_span_2_periods: int = 52, # Senkou Span B period
    displacement: int = 26 # For Senkou Spans and Chikou
) -> Dict[str, List[Optional[float]]]:

    n = len(closes)
    min_len = max(conversion_periods, base_periods, lagging_span_2_periods) + displacement
    if n < min_len:
        return {
            "conversion_line": [None]*n, "base_line": [None]*n, "lead_line1": [None]*n, 
            "lead_line2": [None]*n, "lagging_span": [None]*n, "displacement_val": displacement
        }

    conversion_line = _calculate_donchian_channel_ichimoku_helper(highs, lows, conversion_periods)
    base_line = _calculate_donchian_channel_ichimoku_helper(highs, lows, base_periods)
    
    lead_line1_calc: List[Optional[float]] = [None] * n
    for i in range(n):
        if conversion_line[i] is not None and base_line[i] is not None:
            lead_line1_calc[i] = (conversion_line[i] + base_line[i]) / 2
    
    lead_line2_calc = _calculate_donchian_channel_ichimoku_helper(highs, lows, lagging_span_2_periods)

    lead_line1_displaced: List[Optional[float]] = [None] * n
    lead_line2_displaced: List[Optional[float]] = [None] * n

    for i in range(n - displacement): # Plot calculated value `displacement` periods ahead
        lead_line1_displaced[i + displacement] = lead_line1_calc[i]
        lead_line2_displaced[i + displacement] = lead_line2_calc[i]

    # Lagging Span (Chikou): current close plotted `displacement` periods in the PAST.
    # This is the standard interpretation.
    lagging_span: List[Optional[float]] = [None] * n
    for i in range(displacement, n):
        lagging_span[i - displacement] = closes[i]
        
    return {
        "conversion_line": conversion_line, 
        "base_line": base_line,             
        "lead_line1": lead_line1_displaced, 
        "lead_line2": lead_line2_displaced, 
        "lagging_span": lagging_span, # This is standard Chikou
        "displacement_val": displacement
    }


def calculate_parabolic_sar(
    highs: List[Optional[float]], 
    lows: List[Optional[float]], 
    start_af: float = 0.02, 
    increment_af: float = 0.02, 
    max_af: float = 0.2
) -> List[Optional[float]]:
    
    n = len(highs)
    if n < 2: return [None]*n
    
    sar_values: List[Optional[float]] = [None] * n
    if highs[0] is None or lows[0] is None: return sar_values

    sar_values[0] = lows[0] 
    ep = lows[0] 
    af = start_af
    is_uptrend = True 

    for i in range(1, n):
        if highs[i] is None or lows[i] is None or sar_values[i-1] is None or \
           highs[i-1] is None or lows[i-1] is None : 
            sar_values[i] = None
            # Invalidate EP, AF, trend if SAR is None to force re-evaluation or use last valid?
            # JS seems to continue if sar[i-1] is valid. If SAR becomes NaN, it might stop.
            # Python will propagate None. For a robust PSAR, if sar_values[i-1] is None,
            # it might need to re-initialize (e.g. set sar_values[i] to lows[i] or highs[i] and reset ep, af)
            # For now, strict propagation if prev_sar is None.
            if sar_values[i-1] is None: # If previous SAR is None, cannot calculate current
                is_uptrend = True # Reset to initial assumption for a potential restart
                ep = lows[i] if lows[i] is not None else (highs[i] if highs[i] is not None else None)
                af = start_af
                sar_values[i] = None # Still None as it depends on prev_sar for calculation
                continue

        prev_sar = sar_values[i-1]
        
        if is_uptrend:
            sar_values[i] = prev_sar + af * (ep - prev_sar)
            if sar_values[i] > lows[i]: 
                is_uptrend = False
                sar_values[i] = ep 
                ep = lows[i]   
                af = start_af
            else: 
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + increment_af, max_af)
                sar_values[i] = min(sar_values[i], lows[i-1], lows[i]) 
        else: 
            sar_values[i] = prev_sar + af * (ep - prev_sar)
            if sar_values[i] < highs[i]: 
                is_uptrend = True
                sar_values[i] = ep 
                ep = highs[i]  
                af = start_af
            else: 
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + increment_af, max_af)
                sar_values[i] = max(sar_values[i], highs[i-1], highs[i])
                
    return sar_values


def calculate_williams_r(highs: List[Optional[float]], lows: List[Optional[float]], closes: List[Optional[float]], length: int = 14) -> List[Optional[float]]:
    n = len(closes)
    if n < length or length <=0:
        return [None]*n
        
    percent_r_values: List[Optional[float]] = [None] * n
    for i in range(length - 1, n):
        if closes[i] is None:
            percent_r_values[i] = None
            continue
        
        h_slice_raw = highs[i - length + 1 : i + 1]
        l_slice_raw = lows[i - length + 1 : i + 1]

        h_slice = [x for x in h_slice_raw if x is not None]
        l_slice = [x for x in l_slice_raw if x is not None]

        if len(h_slice) < length or len(l_slice) < length: # Need full valid window
            percent_r_values[i] =None
            continue

        highest_high = max(h_slice)
        lowest_low = min(l_slice)
        
        if highest_high == lowest_low:
            percent_r_values[i] = 0.0 # JS uses 0
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
    
    n = len(closes)
    if n < 1:
        return {"vwap": []}

    vwap_values: List[Optional[float]] = [None] * n
    cumulative_price_volume = 0.0
    cumulative_volume = 0.0

    for i in range(n):
        # JS Version implies session VWAP (resets daily). This is continuous.
        # For daily reset, `cumulative_price_volume` and `cumulative_volume` would reset if new day.
        # Assuming continuous VWAP as per JS `calculateVWAP` not showing reset logic.
        if highs[i] is None or lows[i] is None or closes[i] is None or \
           volumes[i] is None or volumes[i] == 0:
            vwap_values[i] = None 
            # If a value is None, should cumulative sums be affected or held?
            # JS version: it would likely make that iteration's price or volume 0 or NaN, affecting sums.
            # This Python: if any component is None, VWAP for the day is None. Sums don't update.
            # If VWAP is expected to be robust to some missing H/L/C but not volume, logic changes.
            # Given JS `prices[i].volume || 1`, it defaults volume to 1 if missing/zero.
            # Let's replicate that:
            current_volume = volumes[i] if volumes[i] is not None and volumes[i] > 0 else 1.0

            if highs[i] is None or lows[i] is None or closes[i] is None:
                vwap_values[i] = None
                continue # Skip updating cumulative sums if price components are missing

            typical_price = (highs[i] + lows[i] + closes[i]) / 3
            
            cumulative_price_volume += typical_price * current_volume
            cumulative_volume += current_volume
            
            if cumulative_volume == 0: 
                vwap_values[i] = None
            else:
                vwap_values[i] = cumulative_price_volume / cumulative_volume
        else: # All values are present and volume > 0
            typical_price = (highs[i] + lows[i] + closes[i]) / 3
            volume = volumes[i]
            
            cumulative_price_volume += typical_price * volume
            cumulative_volume += volume
            vwap_values[i] = cumulative_price_volume / cumulative_volume
            
    return {"vwap": vwap_values} 


def calculate_bollinger_bands(
    closes: List[Optional[float]], 
    length: int = 20, 
    ma_type: str = "SMA", 
    mult: float = 2.0
) -> Dict[str, List[Optional[float]]]:

    n = len(closes)
    if n < length or length <= 0:
        return {"basis": [None]*n, "upper": [None]*n, "lower": [None]*n}

    if ma_type == "SMA": ma_function = _calculate_sma
    elif ma_type == "EMA": ma_function = _calculate_ema
    elif ma_type == "RMA" or ma_type == "SMMA": ma_function = _calculate_rma
    else: ma_function = _calculate_sma 

    basis_line = ma_function(closes, length)
    
    upper_band: List[Optional[float]] = [None] * n
    lower_band: List[Optional[float]] = [None] * n

    for i in range(length - 1, n):
        if basis_line[i] is None: 
            continue
        
        price_slice_raw = closes[i - length + 1 : i + 1]
        price_slice = [p for p in price_slice_raw if p is not None]

        if len(price_slice) < length: # Not enough valid data points in window for stdev
            continue

        mean = basis_line[i] 
        variance = sum([(p - mean)**2 for p in price_slice]) / length # Use actual number of items in price_slice? Standard BB uses 'length'.
        stdev = math.sqrt(variance)
        
        upper_band[i] = mean + mult * stdev
        lower_band[i] = mean - mult * stdev
        
    return {"basis": basis_line, "upper": upper_band, "lower": lower_band}


# --- Master Decision Logic ---
def get_technical_analysis_summary(
    opens: List[Optional[float]],
    highs: List[Optional[float]],
    lows: List[Optional[float]],
    closes: List[Optional[float]],
    volumes: List[Optional[float]],
    selected_indicators_list: Optional[List[str]] = None # Added parameter
) -> Dict[str, Any]:

    num_data_points = len(closes)
    # Default list of all indicators we can calculate
    all_possible_indicators = [
        "RSI", "EMA", "SMA", "MACD", "ADX", "Supertrend", 
        "BollingerBands", "VWAP", "WilliamsR", "PSAR", "Ichimoku", "ATR"
    ]

    if num_data_points < 60: # Minimum data points for meaningful analysis
        return {
            "signals": [{"name": indicator, "value": None, "decision": "Neutral"} 
                        for indicator in all_possible_indicators],
            "overallSignal": "Neutral",
            "raw_score_debug": 0.0
        }
    
    # Determine which indicators to process
    indicators_to_process = selected_indicators_list
    if not indicators_to_process: # If None or empty, process all
        indicators_to_process = all_possible_indicators

    # Calculate all indicators
    rsi14_series = calculate_rsi(closes, 14)
    ema9_series = _calculate_ema(closes, 9)
    sma20_series = _calculate_sma(closes, 20)
    macd_data = calculate_macd(closes, 12, 26, 9, "EMA", "EMA")
    adx_data = calculate_adx(highs, lows, closes, 14, 14)
    supertrend_data = calculate_supertrend(highs, lows, closes, 10, 3.0)
    bollinger_data = calculate_bollinger_bands(closes, 20, "SMA", 2.0)
    vwap_data = calculate_vwap(highs, lows, closes, volumes)
    williams_r_series = calculate_williams_r(highs, lows, closes, 14)
    psar_series = calculate_parabolic_sar(highs, lows) 
    ichimoku_data = calculate_ichimoku_cloud(highs, lows, closes) 
    atr_series = calculate_atr(highs, lows, closes, 14)

    latest_price = _get_latest(closes)
    prev_close = _get_nth_latest(closes, 1) 
    
    displacement_val = ichimoku_data.get("displacement_val", 26) # Default to 26 if not found

    # Get latest values for decision making
    latest_rsi = _get_latest(rsi14_series)
    latest_ema9 = _get_latest(ema9_series)
    latest_sma20 = _get_latest(sma20_series)
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
    latest_senkou_a = _get_latest(ichimoku_data.get("lead_line1",[])) # Already displaced
    latest_senkou_b = _get_latest(ichimoku_data.get("lead_line2",[])) # Already displaced
    
    # Corrected Ichimoku Chikou Span related values for decision logic to match JS intent:
    # JS `latestChikouSpan` (which is `closes[L-1-(displacement-1)]`)
    # compared against `closes[L-1-displacement]`
    chikou_latest_relevant_close = None
    if displacement_val > 0:
         chikou_latest_relevant_close = _get_nth_latest(closes, displacement_val - 1)

    price_to_compare_chikou_against = _get_nth_latest(closes, displacement_val)
    
    latest_atr = _get_latest(atr_series)

    indicator_decisions_map = {key: "Neutral" for key in all_possible_indicators}
    indicator_values_map = {key: None for key in all_possible_indicators}

    if latest_price is None:
        signals_list = [{"name": i_name, "value": indicator_values_map.get(i_name), "decision": "Neutral"} for i_name in all_possible_indicators]
        return {"signals": signals_list, "overallSignal": "Neutral", "raw_score_debug": 0.0}

    # RSI
    indicator_values_map["RSI"] = latest_rsi
    if latest_rsi is not None:
        if latest_rsi < 20: indicator_decisions_map["RSI"] = "Strong Buy"
        elif latest_rsi < 30: indicator_decisions_map["RSI"] = "Buy"
        elif latest_rsi > 80: indicator_decisions_map["RSI"] = "Strong Sell"
        elif latest_rsi > 70: indicator_decisions_map["RSI"] = "Sell"

    # EMA
    indicator_values_map["EMA"] = latest_ema9
    if latest_ema9 is not None:
        if latest_price > latest_ema9 * 1.05: indicator_decisions_map["EMA"] = "Strong Buy"
        elif latest_price > latest_ema9: indicator_decisions_map["EMA"] = "Buy"
        elif latest_price < latest_ema9 * 0.95: indicator_decisions_map["EMA"] = "Strong Sell"
        elif latest_price < latest_ema9: indicator_decisions_map["EMA"] = "Sell"

    # SMA
    indicator_values_map["SMA"] = latest_sma20
    if latest_sma20 is not None:
        if latest_price > latest_sma20 * 1.05: indicator_decisions_map["SMA"] = "Strong Buy"
        elif latest_price > latest_sma20: indicator_decisions_map["SMA"] = "Buy"
        elif latest_price < latest_sma20 * 0.95: indicator_decisions_map["SMA"] = "Strong Sell"
        elif latest_price < latest_sma20: indicator_decisions_map["SMA"] = "Sell"
    
    # MACD
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

    # ADX
    indicator_values_map["ADX"] = {"adx": latest_adx_line, "plusDI": latest_plus_di, "minusDI": latest_minus_di}
    if latest_adx_line is not None and latest_plus_di is not None and latest_minus_di is not None:
        if latest_adx_line > 40 and latest_plus_di > latest_minus_di: indicator_decisions_map["ADX"] = "Strong Buy"
        elif latest_adx_line > 25 and latest_plus_di > latest_minus_di: indicator_decisions_map["ADX"] = "Buy"
        elif latest_adx_line > 40 and latest_minus_di > latest_plus_di: indicator_decisions_map["ADX"] = "Strong Sell"
        elif latest_adx_line > 25 and latest_minus_di > latest_plus_di: indicator_decisions_map["ADX"] = "Sell"

    # Supertrend
    indicator_values_map["Supertrend"] = {"value": latest_supertrend_val, "direction": latest_supertrend_dir}
    if latest_supertrend_dir is not None and latest_supertrend_val is not None:
        if latest_supertrend_dir == 1 and latest_price > latest_supertrend_val:
            indicator_decisions_map["Supertrend"] = "Buy"
        elif latest_supertrend_dir == -1 and latest_price < latest_supertrend_val:
            indicator_decisions_map["Supertrend"] = "Sell"
    
    # Bollinger Bands
    indicator_values_map["BollingerBands"] = {"upper": latest_bb_upper, "middle": latest_bb_middle, "lower": latest_bb_lower}
    if latest_bb_upper is not None and latest_bb_lower is not None and latest_bb_middle is not None:
        if latest_price < latest_bb_lower: indicator_decisions_map["BollingerBands"] = "Strong Buy"
        elif latest_price > latest_bb_upper: indicator_decisions_map["BollingerBands"] = "Strong Sell"
        elif latest_price < latest_bb_middle: indicator_decisions_map["BollingerBands"] = "Sell"
        elif latest_price > latest_bb_middle: indicator_decisions_map["BollingerBands"] = "Buy"

    # VWAP
    indicator_values_map["VWAP"] = latest_vwap
    if latest_vwap is not None:
        if latest_price > latest_vwap * 1.03: indicator_decisions_map["VWAP"] = "Strong Buy"
        elif latest_price > latest_vwap: indicator_decisions_map["VWAP"] = "Buy"
        elif latest_price < latest_vwap * 0.97: indicator_decisions_map["VWAP"] = "Strong Sell"
        elif latest_price < latest_vwap: indicator_decisions_map["VWAP"] = "Sell"
    
    # Williams %R
    indicator_values_map["WilliamsR"] = latest_williams_r
    if latest_williams_r is not None:
        if latest_williams_r > -20: indicator_decisions_map["WilliamsR"] = "Strong Sell" 
        elif latest_williams_r > -30: indicator_decisions_map["WilliamsR"] = "Sell"
        elif latest_williams_r < -80: indicator_decisions_map["WilliamsR"] = "Strong Buy"
        elif latest_williams_r < -70: indicator_decisions_map["WilliamsR"] = "Buy"

    # Parabolic SAR
    indicator_values_map["PSAR"] = latest_psar
    if latest_psar is not None and prev_close is not None: 
        if latest_price > latest_psar and latest_psar < prev_close : 
             indicator_decisions_map["PSAR"] = "Buy"
        elif latest_price < latest_psar and latest_psar > prev_close: 
             indicator_decisions_map["PSAR"] = "Sell"
    
    # Ichimoku Cloud
    indicator_values_map["Ichimoku"] = {
        "tenkan": latest_tenkan_sen, "kijun": latest_kijun_sen,
        "senkouA": latest_senkou_a, "senkouB": latest_senkou_b,
        "chikou_val_for_cond": chikou_latest_relevant_close, # For debug/display
        "price_for_chikou_cond": price_to_compare_chikou_against # For debug/display
    }
    if latest_senkou_a is not None and latest_senkou_b is not None and \
       latest_tenkan_sen is not None and latest_kijun_sen is not None and \
       chikou_latest_relevant_close is not None and price_to_compare_chikou_against is not None:
        
        cloud_top = max(latest_senkou_a, latest_senkou_b)
        cloud_bottom = min(latest_senkou_a, latest_senkou_b)
        
        chikou_bullish_confirmation = chikou_latest_relevant_close > price_to_compare_chikou_against
        chikou_bearish_confirmation = chikou_latest_relevant_close < price_to_compare_chikou_against
            
        if latest_price > cloud_top and latest_tenkan_sen > latest_kijun_sen and chikou_bullish_confirmation:
            indicator_decisions_map["Ichimoku"] = "Strong Buy"
        elif latest_price > cloud_top and latest_tenkan_sen > latest_kijun_sen: # No chikou confirmation for "Buy"
            indicator_decisions_map["Ichimoku"] = "Buy"
        elif latest_price < cloud_bottom and latest_tenkan_sen < latest_kijun_sen and chikou_bearish_confirmation:
            indicator_decisions_map["Ichimoku"] = "Strong Sell"
        elif latest_price < cloud_bottom and latest_tenkan_sen < latest_kijun_sen: # No chikou confirmation for "Sell"
            indicator_decisions_map["Ichimoku"] = "Sell"
    
    # ATR
    indicator_values_map["ATR"] = latest_atr
    if latest_atr is not None and latest_ema9 is not None: # Depends on EMA9 for context
        if latest_atr < latest_price * 0.01: 
            indicator_decisions_map["ATR"] = "Neutral" # Low volatility
        # JS logic: if price > ema9 AND atr > price*0.02 -> Buy
        # JS logic: if price < ema9 AND atr > price*0.02 -> Sell
        elif latest_price > latest_ema9 and latest_atr > latest_price * 0.02: 
            indicator_decisions_map["ATR"] = "Buy" 
        elif latest_price < latest_ema9 and latest_atr > latest_price * 0.02: 
            indicator_decisions_map["ATR"] = "Sell"

    decision_scores = {"Strong Buy": 2, "Buy": 1, "Neutral": 0, "Sell": -1, "Strong Sell": -2}
    weights = {
        "RSI": 1.0, "EMA": 1.5, "SMA": 1.5, "MACD": 1.0, "ADX": 1.0, "Supertrend": 1.5,
        "BollingerBands": 1.0, "VWAP": 1.0, "WilliamsR": 1.0, "PSAR": 1.0, "Ichimoku": 1.5, "ATR": 1.0
    }
    
    total_score = 0.0
    for indicator_name in indicators_to_process: # Use the dynamically determined list
        decision = indicator_decisions_map.get(indicator_name, "Neutral")
        score = decision_scores.get(decision, 0)
        weight = weights.get(indicator_name, 0)
        total_score += score * weight

    overall_signal = "Neutral"
    if not indicators_to_process: 
        overall_signal = "Neutral"
    elif total_score >= 1.5 : overall_signal = "Strong Buy"
    elif total_score >= 0.5 : overall_signal = "Buy"
    elif total_score <= -1.5 : overall_signal = "Strong Sell"
    elif total_score <= -0.5 : overall_signal = "Sell"

    signals_list = []
    for name in all_possible_indicators: # Return all, but decisions based on selected
        signals_list.append({
            "name": name, 
            "value": indicator_values_map.get(name), 
            "decision": indicator_decisions_map.get(name, "Neutral")
        })
        
    return {"signals": signals_list, "overallSignal": overall_signal, "raw_score_debug": total_score}   