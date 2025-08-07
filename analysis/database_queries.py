# app/database_queries.py
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from sqlmodel import Session, select
from sqlalchemy import desc, func, text # Import text for raw SQL if needed

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

# app/database_queries.py
def get_daily_price_history_for_range(
    db: Session, fincode: int, time_range: str
) -> List[BSEAdjustedPrice]: 
    now = datetime.now().date()
    start_date_dt: Optional[datetime.date] = None
    # ... time_range logic determines start_date_dt ...
    # THIS start_date_dt IS FOR THE VISIBLE CHART RANGE
    
    # To calculate DMAs for the beginning of this range, we need earlier data.
    # Let's calculate an earlier_start_date for fetching data for DMA calculation.
    # Max DMA period we are considering is 200 days.
    # Approx 200 trading days = 200 * (7/5) calendar days = 280 calendar days.
    # Add some buffer, say 300 days.
    
    earliest_data_needed_for_dma_offset = timedelta(days=300) # For 200 DMA buffer

    if start_date_dt: # If not "max" range
        fetch_from_date = start_date_dt - earliest_data_needed_for_dma_offset
    else: # For "max" range, no specific start_date_dt, so fetch_from_date is also None
        fetch_from_date = None

    statement = select(BSEAdjustedPrice).where(BSEAdjustedPrice.fincode == fincode)
    if fetch_from_date: # If we have a date to fetch from (i.e., not "max" and start_date_dt was valid)
        statement = statement.where(BSEAdjustedPrice.date >= fetch_from_date)
    
    statement = statement.order_by(BSEAdjustedPrice.date)
    all_data_for_dma_and_chart = db.exec(statement).all()

    # Now, filter this `all_data_for_dma_and_chart` to return only the data for the selected visible `time_range`
    # The `prepare_price_chart_data_with_dma` will use this full dataset to calculate DMAs,
    # but the API will return chart points starting from `start_date_dt`.
    # Actually, it's better if `prepare_price_chart_data_with_dma` receives ALL data needed
    # and the slicing for final output happens AFTER DMA calculation, or the function is smart.

    # The `prepare_price_chart_data_with_dma` function is designed to take the full history
    # (including lookback period for DMAs) and will produce DMA values aligned with that history.
    # The frontend will then display the relevant part. So, returning `all_data_for_dma_and_chart` is correct here.
    
    return all_data_for_dma_and_chart
# --- Helper to format date_end (YYYYMMDD int) to YYYY-MM-DD string ---
def format_date_end_int(date_int: Optional[int]) -> Optional[str]:
    if date_int is None:
        return None
    s = str(date_int)
    if len(s) == 8: # YYYYMMDD
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    elif len(s) == 6: # YYYYMM (less common for date_end)
        return f"{s[:4]}-{s[4:6]}-01" # Default to 1st of month
    return str(date_int) # Fallback

def get_company_master_info(db: Session, fincode: int) -> Optional[CompanyMaster]:
    statement = select(CompanyMaster).where(CompanyMaster.fincode == fincode)
    return db.exec(statement).first()

def get_industry_info(db: Session, ind_code: Optional[int]) -> Optional[IndustryMaster]:
    if ind_code is None:
        return None
    statement = select(IndustryMaster).where(IndustryMaster.ind_code == ind_code)
    return db.exec(statement).first()

def get_company_profile_text(db: Session, fincode: int) -> Optional[str]:
    statement = select(CompanyProfile.details).where(CompanyProfile.fincode == fincode)
    profile = db.exec(statement).first()
    return profile if profile else None

# app/database_queries.py
# ...
def get_all_annual_ratios_consolidated(db: Session, fincode: int) -> List[CompanyFinanceRatioCons]:
    print(f"[DB_QUERY] Getting ALL annual consolidated ratios for fincode: {fincode}")
    statement = select(CompanyFinanceRatioCons)\
                .where(CompanyFinanceRatioCons.fincode == fincode)\
                .where(CompanyFinanceRatioCons.type == 'C')\
                .order_by(CompanyFinanceRatioCons.year_end) 
    results = db.exec(statement).all()
    print(f"[DB_QUERY] Found {len(results)} total annual consolidated ratio records for fincode {fincode}.")
    return results

# --- Price Data ---
# IMPORTANT: Prioritize DailyPriceNSE/BSE if it exists.
def get_latest_eod_price_data(db: Session, fincode: int) -> Optional[BSEAdjustedPrice]: # Change to DailyPriceBSE if needed
    # This function assumes you have a DailyPriceNSE (or BSE) table
    statement = select(BSEAdjustedPrice).where(BSEAdjustedPrice.fincode == fincode).order_by(desc(BSEAdjustedPrice.date)).limit(1)
    return db.exec(statement).first()

def get_previous_eod_price_data(db: Session, fincode: int) -> Optional[BSEAdjustedPrice]:
    statement = select(BSEAdjustedPrice).where(BSEAdjustedPrice.fincode == fincode).order_by(desc(BSEAdjustedPrice.date)).offset(1).limit(1)
    return db.exec(statement).first()

def get_52_week_high_low(db: Session, fincode: int) -> Tuple[Optional[float], Optional[float]]:
    one_year_ago = datetime.now() - timedelta(days=365)
    statement = select(func.max(BSEAdjustedPrice.high), func.min(BSEAdjustedPrice.low))\
                .where(BSEAdjustedPrice.fincode == fincode)\
                .where(BSEAdjustedPrice.date >= one_year_ago)
    result = db.exec(statement).first()
    return (result[0], result[1]) if result else (None, None)

def get_price_history_for_chart(db: Session, fincode: int, days_limit: int = 365*2) -> List[BSEAdjustedPrice]:
    # Fetches last N days of price history
    start_date = datetime.now() - timedelta(days=days_limit)
    statement = select(BSEAdjustedPrice)\
                .where(BSEAdjustedPrice.fincode == fincode)\
                .where(BSEAdjustedPrice.date >= start_date)\
                .order_by(BSEAdjustedPrice.date)
    return db.exec(statement).all()

# Fallback if only Monthly Price data is available (less accurate for current metrics)
def get_latest_monthly_price_data(db: Session, fincode: int) -> Optional[MonthlyPriceBSE]:
    statement = select(MonthlyPriceBSE).where(MonthlyPriceBSE.fincode == fincode)\
                .order_by(desc(MonthlyPriceBSE.year), desc(MonthlyPriceBSE.month)).limit(1)
    return db.exec(statement).first()


def get_annual_ratios_history_consolidated(db: Session, fincode: int, num_years: int = 10) -> List[CompanyFinanceRatioCons]:
    """
    Fetches the 'num_years' latest annual CONSOLIDATED financial ratios.
    Returns sorted oldest year first.
    """
    print(f"[DB_QUERY] Getting annual CONSOLIDATED RATIOS history for fincode: {fincode}, num_years: {num_years}")
    statement = select(CompanyFinanceRatioCons)\
                .where(CompanyFinanceRatioCons.fincode == fincode)\
                .where(CompanyFinanceRatioCons.type == 'C')\
                .order_by(desc(CompanyFinanceRatioCons.year_end))\
                .limit(num_years)
    results = db.exec(statement).all()
    if not results:
        print(f"[DB_QUERY] NO annual consolidated RATIO records found for fincode {fincode}")
    return results[::-1] # Oldest of the N years first

# --- Financials ---
# Strategy:
# 1. Quarterly: Use FinancialResultCons (assuming it has consolidated quarterly with 'Q' type)
# 2. Annual P&L: Use CompanyProfitLoss (type='C')
# 3. Annual Balance Sheet: Use CompanyBalanceSheet (type='C')
# 4. Annual Cash Flow: Use CompanyCashflow (type='C')
# 5. Annual Ratios: Use CompanyFinanceRatio (type='C')
def get_quarterly_financial_results(db: Session, fincode: int, num_quarters: int = 8) -> List[FinancialResultCons]: # Defaulted num_quarters to 8 for clarity
    """
    Fetches the latest 'num_quarters' distinct quarterly financial results for a given fincode.
    It prefers 'QR' (revised) records over 'Q' (original) records if both exist for the same date_end.
    The results are returned sorted by date_end in ascending order (oldest of the set first),
    which is suitable for time-series charts.
    """
    print(f"[DB_QUERY] Getting quarterly results for fincode: {fincode}, target num_quarters: {num_quarters}")

    # Step 1: Fetch a slightly larger set of Q and QR records, ordered to help pick the preferred one.
    # Fetch more than num_quarters initially because Q/QR pairs count as two but represent one period.
    # Example: if num_quarters is 8, fetch maybe 16 or 2*num_quarters to be safe.
    initial_fetch_limit = num_quarters * 2 
    
    statement_all_q_qr = (
        select(FinancialResultCons)
        .where(FinancialResultCons.fincode == fincode)
        .where(FinancialResultCons.result_type.in_(['Q', 'QR']))
        .order_by(desc(FinancialResultCons.date_end), desc(FinancialResultCons.result_type)) # Date newest, then 'QR' before 'Q'
        .limit(initial_fetch_limit) # Fetch more to allow for Python-side de-duplication
    )
    
    all_potentially_relevant_results = db.exec(statement_all_q_qr).all()
    
    if not all_potentially_relevant_results:
        print(f"[DB_QUERY] No Q or QR results found for fincode: {fincode}")
        return []

    # Step 2: Filter in Python to get distinct date_end periods, respecting the 'QR' preference.
    distinct_period_results_latest_first = []
    seen_date_ends = set()

    for result_item in all_potentially_relevant_results:
        if result_item.date_end not in seen_date_ends:
            # This is the first time we're seeing this date_end.
            # Due to the SQL sorting (date_end DESC, result_type DESC),
            # if a 'QR' exists for this date_end, it will be encountered before a 'Q'.
            distinct_period_results_latest_first.append(result_item)
            seen_date_ends.add(result_item.date_end)
            
            # Stop if we have collected enough distinct quarters
            if len(distinct_period_results_latest_first) >= num_quarters:
                break
    
    # The list is currently latest period first. Reverse it to be oldest first for chart consumption.
    results_oldest_first = distinct_period_results_latest_first[::-1]
    
    print(f"[DB_QUERY] Filtered to {len(results_oldest_first)} distinct quarterly results (oldest first):")
    # for r_debug in results_oldest_first: # Optional: keep for detailed debugging if needed
    #     print(f"  Date: {r_debug.date_end}, Type: {r_debug.result_type}, Sales: {r_debug.net_sales}")
        
    return results_oldest_first

def get_annual_profit_loss(db: Session, fincode: int, num_years: int = 10) -> List[CompanyProfitLossCons]: # MODIFIED: Return type and default num_years
    """
    Fetches the latest 'num_years' annual CONSOLIDATED profit & loss statements.
    """
    print(f"[DB_QUERY] Getting annual CONSOLIDATED P&L for fincode: {fincode}, num_years: {num_years}")
    statement = select(CompanyProfitLossCons)\
                .where(CompanyProfitLossCons.fincode == fincode)\
                .where(CompanyProfitLossCons.type == 'C')\
                .order_by(desc(CompanyProfitLossCons.year_end))\
                .limit(num_years)
    results = db.exec(statement).all()
    
    if results:
        print(f"[DB_QUERY] Found {len(results)} annual consolidated P&L records for fincode {fincode}.")
    else:
        print(f"[DB_QUERY] NO annual consolidated P&L records found for fincode {fincode} with type 'C'.")
        
    return results[::-1] # Return in ascending year_end order (oldest of the set first) for charts

def get_latest_annual_profit_loss(db: Session, fincode: int) -> Optional[CompanyProfitLoss]:
    statement = select(CompanyProfitLoss)\
                .where(CompanyProfitLoss.fincode == fincode)\
                .where(CompanyProfitLoss.type == 'C')\
                .order_by(desc(CompanyProfitLoss.year_end))\
                .limit(1)
    return db.exec(statement).first()


def get_annual_balance_sheet(db: Session, fincode: int, num_years: int = 10) -> List[CompanyBalanceSheetCons]: # Use Cons model
    print(f"[DB_QUERY] Getting annual CONSOLIDATED Balance Sheet for fincode: {fincode}, num_years: {num_years}")
    statement = select(CompanyBalanceSheetCons)\
                .where(CompanyBalanceSheetCons.fincode == fincode)\
                .where(CompanyBalanceSheetCons.type == 'C')\
                .order_by(desc(CompanyBalanceSheetCons.year_end))\
                .limit(num_years)
    results = db.exec(statement).all()
    if not results:
        print(f"[DB_QUERY] NO annual consolidated BS records found for fincode {fincode}")
    return results[::-1] # Return oldest of the N years first
    
def get_latest_annual_balance_sheet(db: Session, fincode: int) -> Optional[CompanyBalanceSheet]:
    statement = select(CompanyBalanceSheet)\
                .where(CompanyBalanceSheet.fincode == fincode)\
                .where(CompanyBalanceSheet.type == 'C')\
                .order_by(desc(CompanyBalanceSheet.year_end))\
                .limit(1)
    return db.exec(statement).first()


def get_annual_cash_flow(db: Session, fincode: int, num_years: int = 10) -> List[CompanyCashflowCons]: # Use Cons model
    print(f"[DB_QUERY] Getting annual CONSOLIDATED Cash Flow for fincode: {fincode}, num_years: {num_years}")
    statement = select(CompanyCashflowCons)\
                .where(CompanyCashflowCons.fincode == fincode)\
                .where(CompanyCashflowCons.type == 'C')\
                .order_by(desc(CompanyCashflowCons.year_end))\
                .limit(num_years)
    results = db.exec(statement).all()
    if not results:
        print(f"[DB_QUERY] NO annual consolidated CF records found for fincode {fincode}")
    return results[::-1] # Return oldest of the N years first

def get_latest_annual_ratios(db: Session, fincode: int) -> Optional[CompanyFinanceRatioCons]:
    """
    Fetches the latest CONSOLIDATED annual ratios for a company.
    """
    print(f"[DB_QUERY] Attempting to get latest annual consolidated ratios for fincode: {fincode} from company_finance_ratio_cons") # Added table name to print
    statement = select(CompanyFinanceRatioCons)\
                .where(CompanyFinanceRatioCons.fincode == fincode)\
                .where(CompanyFinanceRatioCons.type == 'C')\
                .order_by(desc(CompanyFinanceRatioCons.year_end))\
                .limit(1)
    result = db.exec(statement).first()
    if result:
        print(f"[DB_QUERY] Found CONSOLIDATED ratios for fincode {fincode}, year_end: {result.year_end}, ROCE: {result.roce}, DPS: {result.dps}, EPS: {result.reported_eps}")
    else:
        print(f"[DB_QUERY] NO CONSOLIDATED annual ratios found for fincode {fincode} with type 'C' from company_finance_ratio_cons.")
    return result

# --- Shareholding ---
def get_shareholding_pattern_history(db: Session, fincode: int, num_periods: int = 5) -> List[ShpSummary]:
    # ShpSummary seems to be the aggregated table for patterns
    statement = select(ShpSummary)\
                .where(ShpSummary.fincode == fincode)\
                .order_by(desc(ShpSummary.date_end))\
                .limit(num_periods)
    results = db.exec(statement).all()
    return results[::-1] # Return in ascending date order

def get_latest_shareholding_pattern(db: Session, fincode: int) -> Optional[ShpSummary]:
    statement = select(ShpSummary)\
                .where(ShpSummary.fincode == fincode)\
                .order_by(desc(ShpSummary.date_end))\
                .limit(1)
    return db.exec(statement).first()

# --- For TTM EPS calculation (if not directly available in CompanyFinanceRatio) ---
def get_ttm_eps_from_quarterly(db: Session, fincode: int) -> Optional[float]:
    # This assumes FinancialResultCons has quarterly data and 'eps_basic' or 'epsabs'
    # and we need consolidated quarterly EPS
    statement = select(FinancialResultCons.eps_basic) \
        .where(FinancialResultCons.fincode == fincode) \
        .where(FinancialResultCons.result_type.in_(['Q', 'QR'])) \
        .order_by(desc(FinancialResultCons.date_end)) \
        .limit(4)
    quarterly_eps_records = db.exec(statement).all()
    if len(quarterly_eps_records) == 4:
        # Ensure all are numbers before summing
        valid_eps = [eps for eps in quarterly_eps_records if isinstance(eps, (int, float))]
        if len(valid_eps) == 4:
            return sum(valid_eps)
    return None

# --- For Outstanding Shares ---
def get_latest_outstanding_shares(db: Session, fincode: int) -> Optional[float]:
    # Try FinancialResultCons first for latest (quarterly or annual)
    statement_cons = select(FinancialResultCons.ns_grand_total) \
        .where(FinancialResultCons.fincode == fincode) \
        .order_by(desc(FinancialResultCons.date_end)) \
        .limit(1)
    shares_cons = db.exec(statement_cons).first()
    if shares_cons and isinstance(shares_cons, (int, float)):
        return shares_cons

    # Fallback to FinancialResult (standalone, might be more frequent)
    statement_sa = select(FinancialResult.ns_grand_total) \
        .where(FinancialResult.fincode == fincode) \
        .order_by(desc(FinancialResult.date_end)) \
        .limit(1)
    shares_sa = db.exec(statement_sa).first()
    if shares_sa and isinstance(shares_sa, (int, float)):
        return shares_sa
    return None

# --- For Peer Comparison (Simplified: just get company names from same industry) ---
def get_peer_companies_basic(db: Session, fincode: int, limit: int = 5) -> List[CompanyMaster]:
    main_company = get_company_master_info(db, fincode)
    if not main_company or not main_company.ind_code:
        return []
    
    statement = select(CompanyMaster)\
                .where(CompanyMaster.ind_code == main_company.ind_code)\
                .where(CompanyMaster.fincode != fincode).limit(limit) 
    return db.exec(statement).all()