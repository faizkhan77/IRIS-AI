# app/calculations.py
from typing import List, Dict, Any, Union, Optional
from sqlmodel import Session, select
# Import your SQLModel classes if type hinting needs them
# Import your SQLModel classes from db_models.py
from analysis.models.fundamentals.masters import (
    CompanyMaster, CompanyAddress, CompanyListings,
    HouseMaster, IndustryMaster, StockExchangeMaster, CompanyProfile
)
from analysis.models.fundamentals.equity import CompanyEquity, CompanyEquityCons
from analysis.models.fundamentals.monthlyshareprice import MonthlyPriceBSE, MonthlyPriceNSE
from analysis.models.fundamentals.registrars import RegistrarMaster, CompanyRegistrar, CompanyDirector
from analysis.models.fundamentals.results import FinancialResult, FinancialResultCons
from analysis.models.fundamentals.shareholding import ShareholdingCategoryMaster, CompanyShareholding, ShpSummary
from analysis.models.fundamentals.financials.balance_sheet import CompanyBalanceSheet, CompanyBalanceSheetCons
from analysis.models.fundamentals.financials.cash_flow import CompanyCashflow, CompanyCashflowCons
from analysis.models.fundamentals.financials.finance_ratio import CompanyFinanceRatio, CompanyFinanceRatioCons
from analysis.models.fundamentals.financials.profit_loss import CompanyProfitLoss, CompanyProfitLossCons
from analysis.models.stockprice.base import (
    IndicesMaster, CompanyIndexPart, BSEPrice, BSEIntradayIndex,
    BSEIndicesEOD, BSEAdjustedPrice
)

from .database_queries import format_date_end_int # Import helper
from datetime import datetime



def get_safe_attr_for_number(obj: Optional[Any], attr_name: str, default: Any = None) -> Any:
    """Helper to get attribute, defaulting to None for missing numbers, suitable for charts."""
    if obj is None:
        return default
    value = getattr(obj, attr_name, default)
    return default if value is None else value

def format_yyyymm_to_mmm_yyyy(date_int_yyyymm: Optional[int]) -> str:
    if date_int_yyyymm is None:
        return "N/A"
    s = str(date_int_yyyymm)
    if len(s) == 6: # YYYYMM
        try:
            dt_obj = datetime.strptime(s + "01", "%Y%m%d") 
            return dt_obj.strftime("%b %Y") 
        except ValueError:
            return s 
    elif len(s) == 8: # YYYYMMDD
        try:
            dt_obj = datetime.strptime(s, "%Y%m%d")
            return dt_obj.strftime("%b %Y")
        except ValueError:
            return s
    return s 



# Helper (can be kept or modified)
def get_safe_attr(obj: Optional[Any], attr_name: str, default: Any = "N/A") -> Any:
    # Changed default to "N/A" for most cases, can be overridden if 0.0 is better
    if obj is None:
        return default
    value = getattr(obj, attr_name, default)
    return default if value is None else value

def format_lakhs_to_crores_display(value_in_lakhs: Optional[Union[int, float]]) -> str:
    """
    Converts a value from Lakhs to Crores and formats it for display
    similar to Screener.in (usually no decimals for Cr values in P&L table).
    Returns "N/A" if input is None or not a number.
    """
    if not isinstance(value_in_lakhs, (int, float)):
        return "N/A"
    
    if value_in_lakhs == 0:
        return "0" # Screener often shows "0" instead of "0.00" for these

    value_in_crores = value_in_lakhs / 100.0
    
    # Screener style: usually no decimals for Cr. amounts in P&L table, uses comma for thousands separator
    # For very large numbers, f-string with comma formatting is fine.
    # Example: 7045.92 Crores should be displayed as "7,046" (rounded)
    # Example: 85877.2 Lakhs -> 858.772 Crores -> "859" Cr
    # Example: 30939.4 Lakhs -> 309.394 Crores -> "309" Cr
    
    # Let's round to nearest integer for Cr display in this table context
    formatted_value = f"{value_in_crores:,.0f}" 
    return formatted_value

# --- Metric Calculation Functions ---
# These now mostly format or do light calculations based on fetched DB objects.


def calculate_moving_average(data: List[float], window: int) -> List[Optional[float]]:
    if not data or window <= 0: # Correct
        return [None] * len(data) 
    
    # if len(data) < window: # Not enough data to calculate full MA for the *entire series*
    #     return [None] * len(data) # This was a bit too restrictive. MA can start later.
        
    averages: List[Optional[float]] = [None] * (window - 1) # Pad initial points where MA cannot be calculated
    
    # Check if enough data for *at least one* MA value
    if len(data) < window:
        return [None] * len(data) # Not enough to calculate even one point

    current_sum = sum(data[0:window])
    averages.append(current_sum / window)
    
    for i in range(window, len(data)):
        current_sum += data[i] - data[i-window]
        averages.append(current_sum / window)
    return averages

def prepare_price_chart_data_with_dma(
    price_history_db: List[BSEAdjustedPrice],
    dma_periods: List[int] = [50, 200] 
) -> List[Dict[str, Any]]:
    if not price_history_db:
        return []

    # Extract closing prices for DMA calculation, ensuring they are valid numbers
    # This list is what calculate_moving_average will receive
    closing_prices_for_dma_calc = [p.close for p in price_history_db if p.close is not None and isinstance(p.close, (float, int))]
    
    dma_series_calculated = {}
    for period in dma_periods:
        # calculate_moving_average returns a list of the same length as its input (closing_prices_for_dma_calc)
        # with leading Nones where MA isn't available.
        dma_series_calculated[f"dma{period}"] = calculate_moving_average(closing_prices_for_dma_calc, period)

    chart_data = []
    
    # We need to map the calculated DMA values (based on filtered closing_prices_for_dma_calc)
    # back to the original price_history_db items.
    
    # dma_idx will track the current position in the dma_series_calculated arrays
    dma_idx = 0 
    for item_db in price_history_db:
        data_point = {
            "date": item_db.date.strftime('%Y-%m-%d') if item_db.date else None,
            "price": item_db.close, # This can be None if original data had it
            "volume": item_db.volume, # Can be None
        }

        # If the current item_db.close contributed to closing_prices_for_dma_calc,
        # then we can potentially get a DMA value for it.
        if item_db.close is not None and isinstance(item_db.close, (float, int)):
            for period in dma_periods:
                dma_key = f"dma{period}"
                calculated_dmas_for_period = dma_series_calculated[dma_key]
                if dma_idx < len(calculated_dmas_for_period):
                    data_point[dma_key] = calculated_dmas_for_period[dma_idx]
                else:
                    data_point[dma_key] = None # Should not happen if lengths are managed
            dma_idx += 1 # Increment only if this price point was used for DMA calculation
        else:
            # If item_db.close was None or not a number, it wasn't in closing_prices_for_dma_calc
            # So, it doesn't have a directly corresponding DMA value from that calculation.
            # DMAs for such points will be None.
            for period in dma_periods:
                data_point[f"dma{period}"] = None
            
        chart_data.append(data_point)
        
    return chart_data

def calculate_market_cap_formatted(current_price: Union[float, str], outstanding_shares: Optional[float]) -> str:
    if isinstance(current_price, (int, float)) and isinstance(outstanding_shares, (int, float)) and current_price > 0 and outstanding_shares > 0:
        market_cap_value = current_price * outstanding_shares
        return f"{(market_cap_value / 1_00_00_000):,.0f} Cr" 
    return "N/A"

def get_current_price_val(latest_eod_price: Optional[BSEAdjustedPrice]) -> Union[float, str]:
    # Ensure "N/A" is returned if price is 0 or None to avoid calculation issues
    price = get_safe_attr(latest_eod_price, 'close', "N/A")
    return price if price != 0.0 else "N/A"


def get_price_change_details(latest_eod: Optional[BSEAdjustedPrice], prev_eod: Optional[BSEAdjustedPrice]) -> Dict[str, Any]:
    current_price = get_safe_attr(latest_eod, 'close') # Default 0.0 from get_safe_attr
    previous_close = get_safe_attr(prev_eod, 'close') # Default 0.0 from get_safe_attr

    # Ensure prices are valid numbers and previous_close is not zero for division
    if isinstance(current_price, (int, float)) and current_price > 0 and \
       isinstance(previous_close, (int, float)) and previous_close > 0:
        change = current_price - previous_close
        percent_change = (change / previous_close) * 100
        return { "absolute": f"{change:.2f}", "percent": f"{percent_change:.2f}", "isPositive": change >= 0 }
    return {"absolute": "N/A", "percent": "N/A", "isPositive": True}

def get_year_high_low_val(high_52w: Optional[float], low_52w: Optional[float]) -> str:
    h = f"{high_52w:.2f}" if isinstance(high_52w, float) and high_52w > 0 else "N/A"
    l = f"{low_52w:.2f}" if isinstance(low_52w, float) and low_52w > 0 else "N/A"
    if h == "N/A" and l == "N/A":
        return "N/A"
    return f"{h} / {l}"

def get_day_high_low_val(latest_eod_price: Optional[BSEAdjustedPrice]) -> str:
    h_val = get_safe_attr(latest_eod_price, 'high', "N/A")
    l_val = get_safe_attr(latest_eod_price, 'low', "N/A")
    h_str = f"{h_val:.2f}" if isinstance(h_val, float) and h_val > 0 else "N/A"
    l_str = f"{l_val:.2f}" if isinstance(l_val, float) and l_val > 0 else "N/A"
    if h_str == "N/A" and l_str == "N/A":
        return "N/A"
    return f"{h_str} / {l_str}"

def calculate_stock_pe_val(current_price: Union[float, str], ttm_eps: Optional[float]) -> Union[float, str]:
    if isinstance(current_price, (int, float)) and current_price > 0 and \
       isinstance(ttm_eps, (int, float)) and ttm_eps != 0: # Allow non-positive EPS for P/E
        return round(current_price / ttm_eps, 2)
    return "N/A"

def get_book_value_formatted_val(latest_ratios: Optional[CompanyFinanceRatioCons]) -> str:
    bvps = get_safe_attr(latest_ratios, 'book_nav_share')
    # Consider if bvps can be 0 or negative. If so, "N/A" for 0 might be too restrictive.
    return f"{bvps:.2f}" if isinstance(bvps, (int, float)) and bvps != 0 else "N/A"


def calculate_dividend_yield_formatted_val(current_price: Union[float, str], annual_dps: Optional[float]) -> str:
    if isinstance(current_price, (int, float)) and current_price > 0 and \
       isinstance(annual_dps, (int, float)) and annual_dps > 0 : # DPS must be positive
        return f"{((annual_dps / current_price) * 100):.2f}" 
    return "0.00" # Standard to show 0.00 if no dividend or N/A price

def get_roce_formatted_val(latest_ratios: Optional[CompanyFinanceRatioCons]) -> str:
    if latest_ratios is None: return "N/A"
    roce_val = getattr(latest_ratios, 'roce', None)
    if roce_val is None: return "N/A"
    return f"{roce_val:.2f}" if isinstance(roce_val, (int, float)) else "N/A"


def get_roe_formatted_val(latest_ratios: Optional[CompanyFinanceRatioCons]) -> str:
    if latest_ratios is None: return "N/A"
    roe_val = getattr(latest_ratios, 'roe', None)
    if roe_val is None: return "N/A"
    return f"{roe_val:.2f}" if isinstance(roe_val, (int, float)) else "N/A"

def get_face_value_val(company_master: Optional[CompanyMaster]) -> Union[float, str]:
    if company_master is None: return "N/A"
    fv = getattr(company_master, 'fv', None)
    if fv is None: return "N/A"
    return fv if isinstance(fv, (int, float)) else "N/A"

def calculate_rsi_value(price_history_db: List[BSEAdjustedPrice], period: int = 14) -> Union[float, str]:
    if not price_history_db or len(price_history_db) < period +1: return "N/A" # Need period+1 prices for period changes
    
    # Ensure prices are valid and there are enough of them
    prices = [p.close for p in price_history_db if p.close is not None and isinstance(p.close, (int, float))]
    if len(prices) < period + 1: return "N/A"

    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    if len(deltas) < period: return "N/A"

    gains_sum = 0.0
    losses_sum = 0.0

    for i in range(period): # Calculate initial average gain/loss over the first 'period' deltas
        change = deltas[i]
        if change > 0:
            gains_sum += change
        else:
            losses_sum -= change # Losses are positive values

    avg_gain = gains_sum / period
    avg_loss = losses_sum / period
    
    # Smooth subsequent gains/losses
    for i in range(period, len(deltas)):
        change = deltas[i]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # Or N/A if both are 0
        
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi, 2)


# --- Data Formatting Functions for Charts (now accept DB model lists) ---
def format_price_volume_data_for_chart(price_history_db: List[BSEAdjustedPrice]) -> List[Dict[str, Any]]:
    return [
        {"date": item.date.strftime('%Y-%m-%d'), "price": item.close, "volume": item.volume}
        for item in price_history_db if item.date and item.close is not None # Ensure date and close are not None
    ]

# Pros, Cons, About, KeyPoints are fetched directly or still manual
# So functions like get_pros_list might just return a fixed list or data passed from a specific field

def format_peer_comparison_data_for_api(
    db: Session, # Pass DB session to fetch peer details
    main_fincode: int,
    peer_master_list: List[CompanyMaster] # List of CompanyMaster for peers
    ) -> List[Dict[str, Any]]:
    from . import database_queries as db_queries # Local import to avoid circular dependency
    
    peers_data_for_api = []
    # Include main company in peer comparison table
    all_fincodes_for_peer_table = [main_fincode] + [p.fincode for p in peer_master_list if p.fincode]


    for peer_fincode_val in all_fincodes_for_peer_table:
        peer_master = db_queries.get_company_master_info(db, peer_fincode_val)
        if not peer_master: continue

        latest_eod = db_queries.get_latest_eod_price_data(db, peer_fincode_val)
        current_price = get_current_price_val(latest_eod)
        
        latest_ratios = db_queries.get_latest_annual_ratios(db, peer_fincode_val)
        # Try to get TTM EPS from ratios, fallback to quarterly calculation
        ttm_eps_from_ratios = get_safe_attr(latest_ratios, 'reported_eps', None) # default to None
        ttm_eps = ttm_eps_from_ratios if ttm_eps_from_ratios is not None else db_queries.get_ttm_eps_from_quarterly(db, peer_fincode_val)

        stock_pe = calculate_stock_pe_val(current_price, ttm_eps)
        
        outstanding_shares = db_queries.get_latest_outstanding_shares(db, peer_fincode_val)
        market_cap_str = calculate_market_cap_formatted(current_price, outstanding_shares)
        
        try:
            market_cap_numeric = float(market_cap_str.split(' ')[0].replace(',', '')) if market_cap_str != "N/A" and " Cr" in market_cap_str else None
        except ValueError:
            market_cap_numeric = None


        annual_dps = get_safe_attr(latest_ratios, 'dps', None) # default to None
        div_yield_str = calculate_dividend_yield_formatted_val(current_price, annual_dps)
        div_yield_float = float(div_yield_str) if div_yield_str != "0.00" and div_yield_str != "N/A" else 0.0 # Keep as 0.0 if "0.00"

        roce_str = get_roce_formatted_val(latest_ratios)
        roce_float = float(roce_str) if roce_str != "N/A" else None


        # Latest quarterly P&L for NP Qtr, Sales Qtr
        latest_q_results_list = db_queries.get_quarterly_financial_results(db, peer_fincode_val, num_quarters=2) 
        latest_q = latest_q_results_list[-1] if latest_q_results_list else None
        prev_q = latest_q_results_list[-2] if len(latest_q_results_list) > 1 else None

        net_profit_qtr_val = get_safe_attr(latest_q, 'net_profit', "N/A")
        sales_qtr_val = get_safe_attr(latest_q, 'net_sales', "N/A")
        
        qtr_profit_var_str = "N/A"
        latest_q_np = get_safe_attr(latest_q, 'net_profit', None)
        prev_q_np = get_safe_attr(prev_q, 'net_profit', None)

        if isinstance(latest_q_np, (int, float)) and isinstance(prev_q_np, (int, float)) and prev_q_np != 0:
            qtr_profit_var_str = f"{((latest_q_np - prev_q_np) / abs(prev_q_np) * 100):.2f}"
        
        qtr_profit_var_float = float(qtr_profit_var_str) if qtr_profit_var_str != "N/A" else None


        qtr_sales_var_str = "N/A"
        latest_q_sales = get_safe_attr(latest_q, 'net_sales', None)
        prev_q_sales = get_safe_attr(prev_q, 'net_sales', None)
        if isinstance(latest_q_sales, (int,float)) and isinstance(prev_q_sales, (int,float)) and prev_q_sales !=0:
            qtr_sales_var_str = f"{((latest_q_sales - prev_q_sales) / abs(prev_q_sales) * 100):.2f}"
        
        qtr_sales_var_float = float(qtr_sales_var_str) if qtr_sales_var_str != "N/A" else None


        peers_data_for_api.append({
            "id": str(peer_fincode_val), 
            "name": get_safe_attr(peer_master, 'compname', "N/A"),
            "cmp": current_price, # can be float or "N/A"
            "pe": stock_pe,       # can be float or "N/A"
            "marketCap": market_cap_numeric, 
            "marketCapFormatted": market_cap_str,
            "dividendYield": div_yield_float if div_yield_float is not None else "N/A", # Send float or "N/A"
            "netProfitQtr": net_profit_qtr_val, # can be float or "N/A"
            "qtrProfitVar": qtr_profit_var_float if qtr_profit_var_float is not None else "N/A",
            "salesQtr": sales_qtr_val, # can be float or "N/A"
            "qtrSalesVar": qtr_sales_var_float if qtr_sales_var_float is not None else "N/A",
            "roce": roce_float if roce_float is not None else "N/A",
        })
    return peers_data_for_api

# MODIFIED FUNCTION
def format_quarterly_financials_for_chart(
    quarterly_results_db: List[FinancialResultCons], 
    data_keys_map: Dict[str, str] # TargetKey (for chart): SourceKey (from DB model)
) -> List[Dict[str, Any]]:
    formatted = []
    for q_res in quarterly_results_db:
        period_label = format_yyyymm_to_mmm_yyyy(q_res.date_end)
        entry = {"name": period_label} 
        for target_key, source_attr_name in data_keys_map.items():
            # For charts, None is better than "N/A" string if connectNulls is used
            entry[target_key] = get_safe_attr(q_res, source_attr_name, None) 
        formatted.append(entry)
    return formatted # Data is already ascending from db_queries

def format_annual_financials_for_chart(
    annual_results_db: List[Union[CompanyProfitLossCons, CompanyCashflowCons, CompanyBalanceSheetCons]], # Added CompanyBalanceSheetCons
    data_keys_map: Dict[str,str]
) -> List[Dict[str, Any]]:
    formatted = []
    for res in annual_results_db: # Should be oldest first for charts
        period_label = format_yyyymm_to_mmm_yyyy(res.year_end) 
        
        entry = {"name": period_label} 
        for target_key, source_attr_name in data_keys_map.items():
            raw_value = get_safe_attr_for_number(res, source_attr_name, None)
            if isinstance(raw_value, (int, float)):
                # Convert to Crores for chart values if source is in Lakhs
                entry[target_key] = round(raw_value / 100.0, 2) 
            else:
                entry[target_key] = None
        formatted.append(entry)
    return formatted


def get_annual_ratios_table_data(annual_ratios_db: List[CompanyFinanceRatioCons]) -> List[Dict[str, Any]]:
    data = []
    # annual_ratios_db is sorted oldest year first
    for ratio_rec in reversed(annual_ratios_db): # Process latest year first for table display
        year_label = format_yyyymm_to_mmm_yyyy(ratio_rec.year_end)

        debtor_days = get_safe_attr_for_number(ratio_rec, 'receivable_days', None)
        inventory_days = get_safe_attr_for_number(ratio_rec, 'inventory_days', None)
        payable_days = get_safe_attr_for_number(ratio_rec, 'payable_days', None)
        
        cash_conversion_cycle = "N/A"
        if isinstance(debtor_days, (int, float)) and \
           isinstance(inventory_days, (int, float)) and \
           isinstance(payable_days, (int, float)):
            cash_conversion_cycle = round(debtor_days + inventory_days - payable_days)

        # Assuming 'working_capital_days' field exists. If not, this will be "N/A"
        # or you use 'working_capital_sales' and label it appropriately.
        # For this example, let's assume a hypothetical 'working_capital_days' field
        # matching the Screener.in label. If it's not in your model, this will need adjustment.
        working_capital_days = get_safe_attr_for_number(ratio_rec, 'working_capital_days', None) # HYPOTHETICAL FIELD
        # If you only have 'working_capital_sales' (which is a ratio, not days):
        # working_capital_sales_ratio = get_safe_attr_for_number(ratio_rec, 'working_capital_sales', None)


        roce_percent = get_safe_attr_for_number(ratio_rec, 'roce', None)

        data.append({
            "year": year_label,
            "debtorDays": f"{debtor_days:.0f}" if isinstance(debtor_days, (float, int)) else "N/A",
            "inventoryDays": f"{inventory_days:.0f}" if isinstance(inventory_days, (float, int)) else "N/A",
            "daysPayable": f"{payable_days:.0f}" if isinstance(payable_days, (float, int)) else "N/A",
            "cashConversionCycle": f"{cash_conversion_cycle}" if cash_conversion_cycle != "N/A" else "N/A",
            "workingCapitalDays": f"{working_capital_days:.0f}" if isinstance(working_capital_days, (float, int)) else "N/A",
            "roce": f"{roce_percent:.0f}%" if isinstance(roce_percent, (float, int)) else "N/A", # Screener shows ROCE as int % here
        })
    return data


def format_annual_ratios_for_chart(
    annual_ratios_db: List[CompanyFinanceRatioCons],
    ratio_keys_map: Dict[str, str] # e.g., {"Debtor Days": "receivable_days", "ROCE %": "roce"}
) -> List[Dict[str, Any]]:
    formatted_data = []
    # annual_ratios_db is sorted oldest year first
    for ratio_rec in annual_ratios_db:
        year_label = format_yyyymm_to_mmm_yyyy(ratio_rec.year_end)
        entry = {"name": year_label} # For X-axis

        for chart_key, db_field_name in ratio_keys_map.items():
            value = get_safe_attr_for_number(ratio_rec, db_field_name, None)
            if isinstance(value, (int, float)):
                # For percentages like ROCE, keep as is (e.g., 15.2 for 15.2%)
                # For days, they are already numbers.
                entry[chart_key] = round(value, 2) if chart_key.endswith('%') else round(value) # Simple rounding
            else:
                entry[chart_key] = None
        
        if len(entry) > 1: # Add if there's more than just the name (year)
            formatted_data.append(entry)
    return formatted_data


def format_shareholding_for_pie_chart(latest_shp: Optional[ShpSummary]) -> List[Dict[str, Any]]:
    if not latest_shp:
        print("[CALC_SHP_PIE] latest_shp is None, returning empty list.")
        return []

    # Use the _for_chart helper to ensure None for missing values that should be numbers
    promoters_val = get_safe_attr_for_number(latest_shp, 'tp_f_total_promoter', 0.0)
    fii_val = (get_safe_attr_for_number(latest_shp, 'tp_in_fii', 0.0) or 0.0) + \
              (get_safe_attr_for_number(latest_shp, 'tp_in_foreign_port_inv', 0.0) or 0.0)
    dii_val = (get_safe_attr_for_number(latest_shp, 'tp_in_mf_uti', 0.0) or 0.0) + \
              (get_safe_attr_for_number(latest_shp, 'tp_in_fi_banks', 0.0) or 0.0) + \
              (get_safe_attr_for_number(latest_shp, 'tp_in_insurance', 0.0) or 0.0)
    public_val = get_safe_attr_for_number(latest_shp, 'tp_nin_subtotal', 0.0)

    data = []
    if promoters_val > 0: data.append({"name": "Promoters", "value": promoters_val})
    if fii_val > 0: data.append({"name": "FII", "value": fii_val})
    if dii_val > 0: data.append({"name": "DII", "value": dii_val})
    if public_val > 0: data.append({"name": "Public", "value": public_val})
    
    current_total = sum(item['value'] for item in data)
    others_value = 0.0

    # Try to get a grand total from the DB if available for more accurate "Others"
    # Assuming 'tp_grand_total' might be the sum of all percentages from the source
    # or 'tp_total_prom_public' plus any custodian holdings.
    # For simplicity, if a field like 'tp_total_prom_public' (usually 100 or close) exists:
    grand_total_field = get_safe_attr_for_number(latest_shp, 'tp_total_prom_public', 100.0) # Default to 100 if not found
    if grand_total_field is not None and grand_total_field > current_total:
        others_value = grand_total_field - current_total
    elif 0 < current_total < 100 : # Fallback if no better total field
        others_value = 100.0 - current_total
        
    if others_value > 0.01: # Only add "Others" if it's a significant slice
        data.append({"name": "Others", "value": round(others_value, 2)})
    
    print(f"[CALC_SHP_PIE] Pie chart data: {data}")
    return data

def format_shareholding_trend_for_chart(shp_history_db: List[ShpSummary]) -> List[Dict[str, Any]]:
    formatted_data = []
    # shp_history_db is sorted ascending by date (oldest first)
    for shp in shp_history_db:
        date_label = format_yyyymm_to_mmm_yyyy(shp.date_end) 
        
        # Use the _for_chart helper to ensure None for missing values
        promoters = get_safe_attr_for_number(shp, 'tp_f_total_promoter', None)
        fii = (get_safe_attr_for_number(shp, 'tp_in_fii', 0.0) or 0.0) + \
              (get_safe_attr_for_number(shp, 'tp_in_foreign_port_inv', 0.0) or 0.0)
        fii = fii if fii > 0.005 else None # So small values don't clutter chart if not meaningful
        
        dii = (get_safe_attr_for_number(shp, 'tp_in_mf_uti', 0.0) or 0.0) + \
              (get_safe_attr_for_number(shp, 'tp_in_fi_banks', 0.0) or 0.0) + \
              (get_safe_attr_for_number(shp, 'tp_in_insurance', 0.0) or 0.0)
        dii = dii if dii > 0.005 else None
        
        public = get_safe_attr_for_number(shp, 'tp_nin_subtotal', None)
        
        # Only add data point if there's something to plot
        if any(val is not None for val in [promoters, fii, dii, public]):
            formatted_data.append({
                "date": date_label, # This is the X-axis "name"
                "Promoters": promoters,
                "FII": fii,
                "DII": dii,
                "Public": public,
            })
    # print(f"[CALC_SHP_TREND] Trend data: {formatted_data}") # Can be very verbose
    return formatted_data



def get_quarterly_results_table_data(quarterly_results_db: List[FinancialResultCons]) -> List[Dict[str, Any]]:
    data = []
    # quarterly_results_db is sorted ascending by date (oldest first) from db_queries
    # For table display, we usually want latest first, so iterate in reverse or reverse the list at the end.
    for q_res in reversed(quarterly_results_db): # Iterate reversed for latest first
        quarter_label = format_yyyymm_to_mmm_yyyy(q_res.date_end)

        data.append({
            "quarter": quarter_label,
            "revenue": get_safe_attr(q_res, 'net_sales', "N/A"), # Target: Revenue
            "interest": get_safe_attr(q_res, 'interest', "N/A"),
            "netProfit": get_safe_attr(q_res, 'net_profit', "N/A"),
            "eps": get_safe_attr(q_res, 'eps_basic', "N/A"),
        })
    return data # Now returns latest quarter first

def get_annual_profit_loss_table_data(
    annual_pl_db: List[CompanyProfitLossCons], 
    annual_ratios_db: Optional[List[CompanyFinanceRatioCons]] = None
) -> List[Dict[str, Any]]:
    data = []
    ratios_lookup = {}
    if annual_ratios_db:
        for ratio_entry in annual_ratios_db:
            if ratio_entry.year_end:
                ratios_lookup[ratio_entry.year_end] = ratio_entry
                
    for pl_statement in reversed(annual_pl_db):
        year_label = format_yyyymm_to_mmm_yyyy(pl_statement.year_end)
        
        net_sales_lakhs = get_safe_attr_for_number(pl_statement, 'net_sales', None)
        expenses_lakhs = get_safe_attr_for_number(pl_statement, 'total_expendiure', None)
        op_profit_lakhs = get_safe_attr_for_number(pl_statement, 'operating_profit', None)
        other_income_lakhs = get_safe_attr_for_number(pl_statement, 'other_income', None)
        interest_lakhs = get_safe_attr_for_number(pl_statement, 'interest', None)
        depreciation_lakhs = get_safe_attr_for_number(pl_statement, 'depreciation', None)
        pbt_lakhs = get_safe_attr_for_number(pl_statement, 'profit_before_tax', None)
        tax_amount_lakhs = get_safe_attr_for_number(pl_statement, 'taxation', None)
        net_profit_lakhs = get_safe_attr_for_number(pl_statement, 'profit_after_tax', None)
        eps_raw = get_safe_attr_for_number(pl_statement, 'reported_eps', None)

        opm_percentage_str = "N/A"
        if isinstance(net_sales_lakhs, (float, int)) and net_sales_lakhs != 0 and \
           isinstance(op_profit_lakhs, (float, int)):
            opm_val = (op_profit_lakhs / net_sales_lakhs) * 100
            opm_percentage_str = f"{opm_val:.0f}%" # Screener shows OPM as integer %

        tax_percentage_str = "N/A"
        if isinstance(tax_amount_lakhs, (float, int)) and \
           isinstance(pbt_lakhs, (float, int)) and pbt_lakhs != 0:
            tax_val = (tax_amount_lakhs / abs(pbt_lakhs)) * 100
            tax_percentage_str = f"{tax_val:.0f}%" # Screener shows Tax % as integer

        dividend_payout_str = "N/A"
        if pl_statement.year_end and pl_statement.year_end in ratios_lookup:
            year_ratios = ratios_lookup[pl_statement.year_end]
            div_payout_num = get_safe_attr_for_number(year_ratios, 'dividend_payout_per', None)
            if isinstance(div_payout_num, (float, int)):
                dividend_payout_str = f"{div_payout_num:.0f}%"

        data.append({
            "year": year_label,
            "sales": format_lakhs_to_crores_display(net_sales_lakhs),
            "expenses": format_lakhs_to_crores_display(expenses_lakhs),
            "operatingProfit": format_lakhs_to_crores_display(op_profit_lakhs),
            "opm": opm_percentage_str,
            "otherIncome": format_lakhs_to_crores_display(other_income_lakhs),
            "interest": format_lakhs_to_crores_display(interest_lakhs),
            "depreciation": format_lakhs_to_crores_display(depreciation_lakhs),
            "profitBeforeTax": format_lakhs_to_crores_display(pbt_lakhs),
            "tax": tax_percentage_str,
            "netProfit": format_lakhs_to_crores_display(net_profit_lakhs),
            "eps": f"{eps_raw:.2f}" if isinstance(eps_raw, (float, int)) else "N/A",
            "dividendPayout": dividend_payout_str
        })
    return data

# app/calculations.py
# ... (ensure format_yyyymm_to_mmm_yyyy, get_safe_attr_for_number, format_lakhs_to_crores_display are defined) ...

def get_annual_balance_sheet_table_data(annual_bs_db: List[CompanyBalanceSheetCons]) -> List[Dict[str, Any]]:
    data = []
    for bs_statement in reversed(annual_bs_db): # Latest year first
        year_label = format_yyyymm_to_mmm_yyyy(bs_statement.year_end)

        # LIABILITIES SIDE
        equity_capital_lakhs = get_safe_attr_for_number(bs_statement, 'share_capital', None)
        reserves_lakhs = get_safe_attr_for_number(bs_statement, 'reserve', None) # Total Reserves
        borrowings_lakhs = get_safe_attr_for_number(bs_statement, 'total_debt_inclst_nf', None) # Total Debt
        
        total_liabilities_lakhs = get_safe_attr_for_number(bs_statement, 'total_sounces_funds_nf', None) # Corrected typo: sounces -> sources
        shareholders_funds_lakhs = get_safe_attr_for_number(bs_statement, 'shareholders_funds', None)

        other_liabilities_lakhs = None
        if isinstance(total_liabilities_lakhs, (float, int)) and \
           isinstance(shareholders_funds_lakhs, (float, int)) and \
           isinstance(borrowings_lakhs, (float, int)):
            other_liabilities_lakhs = total_liabilities_lakhs - shareholders_funds_lakhs - borrowings_lakhs
        elif isinstance(total_liabilities_lakhs, (float, int)) and \
             isinstance(shareholders_funds_lakhs, (float, int)) and \
             borrowings_lakhs is None: # If borrowings is None but others are present
             other_liabilities_lakhs = total_liabilities_lakhs - shareholders_funds_lakhs
        else: # Fallback if calculation not possible, try direct field if one exists for "Other Liabilities sum"
              # From your doc, field 15 'Other_Liabilities' is "Other Liabilities & Provisions"
              # And field 39 'Current_Liabilities_Nf' is "Other Current Liabilities"
              # And field 34 'Oth_Lt_Liab' is "Other Long Term Liabilities"
              # A common definition would be Oth_Lt_Liab + Current_Liabilities_Nf (excluding ST Debt, Trade Payables if shown separately)
              # For now, using field 15 which seems more aggregated for this purpose.
              other_liabilities_lakhs = get_safe_attr_for_number(bs_statement, 'other_liabilities', None)


        # ASSETS SIDE
        fixed_assets_lakhs = get_safe_attr_for_number(bs_statement, 'net_block', None)
        cwip_lakhs = get_safe_attr_for_number(bs_statement, 'cwip_nf', None)
        
        investments_non_current_lakhs = get_safe_attr_for_number(bs_statement, 'investments_nf', 0.0) or 0.0
        investments_current_lakhs = get_safe_attr_for_number(bs_statement, 'current_investments', 0.0) or 0.0
        total_investments_lakhs = investments_non_current_lakhs + investments_current_lakhs
        
        total_assets_lakhs = get_safe_attr_for_number(bs_statement, 'total_applications_nf', None)
        other_assets_lakhs = None
        if isinstance(total_assets_lakhs, (float, int)):
            fa_l = fixed_assets_lakhs if isinstance(fixed_assets_lakhs, (float, int)) else 0.0
            cw_l = cwip_lakhs if isinstance(cwip_lakhs, (float, int)) else 0.0
            inv_l = total_investments_lakhs # Already a number
            other_assets_lakhs = total_assets_lakhs - fa_l - cw_l - inv_l
        else: # Fallback if total_assets_lakhs is None, sum key components
            other_assets_lakhs = (get_safe_attr_for_number(bs_statement, 'other_noc_assets', 0.0) or 0.0) + \
                                 (get_safe_attr_for_number(bs_statement, 'other_current_nf', 0.0) or 0.0)


        data.append({
            "year": year_label,
            "equityCapital": format_lakhs_to_crores_display(equity_capital_lakhs),
            "reserves": format_lakhs_to_crores_display(reserves_lakhs),
            "borrowings": format_lakhs_to_crores_display(borrowings_lakhs),
            "otherLiabilities": format_lakhs_to_crores_display(other_liabilities_lakhs),
            "totalLiabilities": format_lakhs_to_crores_display(total_liabilities_lakhs),
            "fixedAssets": format_lakhs_to_crores_display(fixed_assets_lakhs),
            "cwip": format_lakhs_to_crores_display(cwip_lakhs),
            "investments": format_lakhs_to_crores_display(total_investments_lakhs if total_investments_lakhs > 0 else None),
            "otherAssets": format_lakhs_to_crores_display(other_assets_lakhs),
            "totalAssets": format_lakhs_to_crores_display(total_assets_lakhs),
        })
    return data


def format_annual_balance_sheet_for_chart(
    annual_bs_db: List[CompanyBalanceSheetCons]
    # data_keys_map: Dict[str, str] # e.g. {"equity": "share_capital", "reserves": "reserve"}
) -> Dict[str, List[Dict[str, Any]]]: # Return a dict with data for liabilities and assets charts
    """
    Formats annual consolidated balance sheet data for stacked bar charts.
    Assumes input figures are in Lakhs and converts them to Crores for chart display.
    """
    liabilities_chart_data = []
    assets_chart_data = []
    
    # annual_bs_db is sorted oldest year first from db_queries.get_annual_balance_sheet
    for bs in annual_bs_db:
        year_label = format_yyyymm_to_mmm_yyyy(bs.year_end)

        # Convert to Crores for chart values, keeping 2 decimal places for chart precision
        def to_cr(val_lakhs: Optional[Union[int, float]]) -> Optional[float]:
            if isinstance(val_lakhs, (int, float)):
                return round(val_lakhs / 100.0, 2)
            return None

        # LIABILITIES SIDE for Chart
        equity_capital_cr_chart = to_cr(get_safe_attr_for_number(bs, 'share_capital', None))
        reserves_cr_chart = to_cr(get_safe_attr_for_number(bs, 'reserve', None))
        borrowings_cr_chart = to_cr(get_safe_attr_for_number(bs, 'total_debt_inclst_nf', None)) # Using total debt
        
        # Calculate Other Liabilities for chart (same logic as table for consistency)
        total_liabilities_lakhs_chart = get_safe_attr_for_number(bs, 'total_sounces_funds_nf', None)
        shareholders_funds_lakhs_chart = get_safe_attr_for_number(bs, 'shareholders_funds', None)
        other_liabilities_lakhs_chart = None
        if isinstance(total_liabilities_lakhs_chart, (float, int)) and \
           isinstance(shareholders_funds_lakhs_chart, (float, int)) and \
           isinstance(borrowings_cr_chart, (float, int)): # Use the crore value's original type for check
             borrowings_lakhs_for_calc = get_safe_attr_for_number(bs, 'total_debt_inclst_nf', None)
             if isinstance(borrowings_lakhs_for_calc, (float, int)):
                other_liabilities_lakhs_chart = total_liabilities_lakhs_chart - shareholders_funds_lakhs_chart - borrowings_lakhs_for_calc
        elif isinstance(total_liabilities_lakhs_chart, (float, int)) and \
             isinstance(shareholders_funds_lakhs_chart, (float, int)) and \
             borrowings_cr_chart is None:
             other_liabilities_lakhs_chart = total_liabilities_lakhs_chart - shareholders_funds_lakhs_chart
        else:
             other_liabilities_lakhs_chart = get_safe_attr_for_number(bs, 'other_liabilities', None)
        other_liabilities_cr_chart = to_cr(other_liabilities_lakhs_chart)


        liabilities_entry = {"name": year_label} # 'name' for X-axis
        if equity_capital_cr_chart is not None: liabilities_entry["Equity Capital"] = equity_capital_cr_chart
        if reserves_cr_chart is not None: liabilities_entry["Reserves"] = reserves_cr_chart
        if borrowings_cr_chart is not None: liabilities_entry["Borrowings"] = borrowings_cr_chart
        if other_liabilities_cr_chart is not None: liabilities_entry["Other Liabilities"] = other_liabilities_cr_chart
        
        if len(liabilities_entry) > 1: # Only add if there's data beyond just the name
            liabilities_chart_data.append(liabilities_entry)

        # ASSETS SIDE for Chart
        fixed_assets_cr_chart = to_cr(get_safe_attr_for_number(bs, 'net_block', None))
        cwip_cr_chart = to_cr(get_safe_attr_for_number(bs, 'cwip_nf', None))
        
        inv_non_current_lakhs = get_safe_attr_for_number(bs, 'investments_nf', 0.0) or 0.0
        inv_current_lakhs = get_safe_attr_for_number(bs, 'current_investments', 0.0) or 0.0
        total_inv_lakhs_chart = inv_non_current_lakhs + inv_current_lakhs
        investments_cr_chart = to_cr(total_inv_lakhs_chart if total_inv_lakhs_chart > 0 else None)

        # Calculate Other Assets for chart (same logic as table)
        total_assets_lakhs_chart = get_safe_attr_for_number(bs, 'total_applications_nf', None)
        other_assets_lakhs_for_chart_calc = None
        if isinstance(total_assets_lakhs_chart, (float, int)):
            fa_l_chart = get_safe_attr_for_number(bs, 'net_block', 0.0) or 0.0
            cw_l_chart = get_safe_attr_for_number(bs, 'cwip_nf', 0.0) or 0.0
            other_assets_lakhs_for_chart_calc = total_assets_lakhs_chart - fa_l_chart - cw_l_chart - total_inv_lakhs_chart
        else:
            other_assets_lakhs_for_chart_calc = (get_safe_attr_for_number(bs, 'other_noc_assets', 0.0) or 0.0) + \
                                                (get_safe_attr_for_number(bs, 'other_current_nf', 0.0) or 0.0)
        other_assets_cr_chart = to_cr(other_assets_lakhs_for_chart_calc)
        
        assets_entry = {"name": year_label}
        if fixed_assets_cr_chart is not None: assets_entry["Fixed Assets"] = fixed_assets_cr_chart
        if cwip_cr_chart is not None: assets_entry["CWIP"] = cwip_cr_chart
        if investments_cr_chart is not None: assets_entry["Investments"] = investments_cr_chart
        if other_assets_cr_chart is not None: assets_entry["Other Assets"] = other_assets_cr_chart
        
        if len(assets_entry) > 1:
            assets_chart_data.append(assets_entry)
            
    return {
        "liabilitiesChart": liabilities_chart_data,
        "assetsChart": assets_chart_data
    }

def get_annual_cash_flow_table_data(annual_cf_db: List[CompanyCashflowCons]) -> List[Dict[str, Any]]:
    """
    Formats annual consolidated cash flow data for table display.
    Assumes input figures from DB are in Lakhs and will be displayed in Crores.
    """
    data = []
    # annual_cf_db is sorted oldest year first from db_queries.get_annual_cash_flow
    # Reverse for table to show latest year first.
    for cf_statement in reversed(annual_cf_db):
        year_label = format_yyyymm_to_mmm_yyyy(cf_statement.year_end)

        data.append({
            "year": year_label,
            "cashFromOperating": format_lakhs_to_crores_display(get_safe_attr_for_number(cf_statement, 'cash_from_operation', None)),
            "cashFromInvesting": format_lakhs_to_crores_display(get_safe_attr_for_number(cf_statement, 'cash_from_investment', None)),
            "cashFromFinancing": format_lakhs_to_crores_display(get_safe_attr_for_number(cf_statement, 'cash_from_financing', None)),
            "netCashFlow": format_lakhs_to_crores_display(get_safe_attr_for_number(cf_statement, 'net_cash_inflow_outflow', None)),
        })
    return data

def get_shareholding_history_table_data(shp_history_db: List[ShpSummary]) -> List[Dict[str,Any]]:
    data = []
    # shp_history_db is sorted ascending (oldest first). For table, show latest first.
    for shp in reversed(shp_history_db):
        date_label = format_yyyymm_to_mmm_yyyy(shp.date_end)
        
        promoters_val = get_safe_attr(shp, 'tp_f_total_promoter', "N/A") # Use "N/A" for table
        fii_val_num = (get_safe_attr(shp, 'tp_in_fii', 0.0) or 0.0) + \
                      (get_safe_attr(shp, 'tp_in_foreign_port_inv', 0.0) or 0.0)
        dii_val_num = (get_safe_attr(shp, 'tp_in_mf_uti', 0.0) or 0.0) + \
                      (get_safe_attr(shp, 'tp_in_fi_banks', 0.0) or 0.0) + \
                      (get_safe_attr(shp, 'tp_in_insurance', 0.0) or 0.0)
        public_val = get_safe_attr(shp, 'tp_nin_subtotal', "N/A")
        
        current_total_percent = 0
        if isinstance(promoters_val, (int, float)): current_total_percent += promoters_val
        current_total_percent += fii_val_num
        current_total_percent += dii_val_num
        if isinstance(public_val, (int, float)): current_total_percent += public_val

        others_val_str = "0.00" # Default to "0.00"
        if 0 < current_total_percent < 100.1 : # Allow for small rounding diffs
            others_val_num = 100.0 - current_total_percent
            others_val_str = f"{max(0, others_val_num):.2f}" # Ensure not negative, format
        elif current_total_percent == 0 and promoters_val == "N/A": # If all main categories are N/A
             others_val_str = "N/A"


        data.append({
            "date": date_label,
            "promoters": f"{promoters_val:.2f}" if isinstance(promoters_val, (int,float)) else promoters_val,
            "fii": f"{fii_val_num:.2f}",
            "dii": f"{dii_val_num:.2f}",
            "public": f"{public_val:.2f}" if isinstance(public_val, (int,float)) else public_val,
            "others": others_val_str
        })
    return data