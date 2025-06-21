from models.fundamentals.masters import (
    CompanyAddress,
    CompanyMaster,
    HouseMaster,
    IndustryMaster,
    StockExchangeMaster,
    CompanyListings,
    CompanyProfile,
)
from models.fundamentals.equity import CompanyEquity, CompanyEquityCons
from models.fundamentals.registrars import (
    CompanyDirector,
    CompanyRegistrar,
    RegistrarMaster,
)
from models.fundamentals.results import FinancialResult, FinancialResultCons
from models.fundamentals.monthlyshareprice import MonthlyPriceBSE, MonthlyPriceNSE
from models.fundamentals.shareholding import (
    CompanyShareholding,
    ShareholdingCategoryMaster,
    ShpSummary,
)
from models.fundamentals.financials.balance_sheet import (
    CompanyBalanceSheet,
    CompanyBalanceSheetCons,
)
from models.fundamentals.financials.cash_flow import (
    CompanyCashflow,
    CompanyCashflowCons,
)
from models.fundamentals.financials.finance_ratio import (
    CompanyFinanceRatio,
    CompanyFinanceRatioCons,
)
from models.fundamentals.financials.profit_loss import (
    CompanyProfitLoss,
    CompanyProfitLossCons,
)
from models.stockprice.base import (
    CompanyIndexPart,
    IndicesMaster,
    BSEAdjustedPrice,
    BSEIndicesEOD,
)


fundamental = "fundamental"
stockprice = "stockprice"

FILEMAPPING = [
    {
        "table_name":"company_master",
        "type": fundamental,
        "model_class": CompanyMaster,
        "filename": "Company_master.json",
        "unique_keys": ["fincode"],
    },
    {
        "table_name":"industry_master",
        "type": fundamental,
        "model_class": IndustryMaster,
        "filename": "Industrymaster_Ex1.json",
        "unique_keys": ["ind_code"],
    },
    {
        "table_name":"company_address",
        "type": fundamental,
        "model_class": CompanyAddress,
        "filename": "Companyaddress.json",
        "unique_keys": ["fincode"],
    },
    {
        "table_name":"house_master",
        "type": fundamental,
        "model_class": HouseMaster,
        "filename": "Housemaster.json",
        "unique_keys": ["house_code"],
    },
    {
        "table_name":"stock_exchange_master",
        "type": fundamental,
        "model_class": StockExchangeMaster,
        "filename": "Stockexchangemaster.json",
        "unique_keys": ["stk_id"],
    },
    {
        "table_name":"company_listings",
        "type": fundamental,
        "model_class": CompanyListings,
        "filename": "Complistings.json",
        "unique_keys": ["fincode","stk_id"],
    },
    {
        "table_name":"shareholding_category_master",
        "type": fundamental,
        "model_class": ShareholdingCategoryMaster,
        "filename": "Shp_catmaster_2.json",
        "unique_keys": ["shp_catid"],
    },
    {
        "table_name":"company_shareholders_details",
        "type": fundamental,
        "model_class": CompanyShareholding,
        "filename": "Shp_details.json",
        "unique_keys": ["fincode", "date_end", "srno"],
    },
    {
        "table_name":"company_registrar_master",
        "type": fundamental,
        "model_class": RegistrarMaster,
        "filename": "Registrarmaster.json",
        "unique_keys": ["registrar_no"],
    },
    {
        "table_name":"company_registrar_data",
        "type": fundamental,
        "model_class": CompanyRegistrar,
        "filename": "Registrardata.json",
        "unique_keys": ["fincode","registrar_no"],
    },
    {
        "table_name":"company_board_director",
        "type": fundamental,
        "model_class": CompanyDirector,
        "filename": "Board.json",
        "unique_keys": ["fincode", "serialno", "dirtype_id", "yrc"],
    },
    {
        "table_name":"monthly_price_bse",
        "type": fundamental,
        "model_class": MonthlyPriceBSE,
        "filename": "Monthlyprice.json",
        "unique_keys": ["fincode","year","month"],
    },
    {
        "table_name":"monthly_price_nse",
        "type": fundamental,
        "model_class": MonthlyPriceNSE,
        "filename": "Nse_Monthprice.json",
        "unique_keys": ["fincode","year","month"],
    },
    {
        "table_name":"company_equity_cons",
        "type": fundamental,
        "model_class": CompanyEquityCons,
        "filename": "company_equity_cons.json",
        "unique_keys": ["fincode"],
    },
    {
        "table_name":"company_equity",
        "type": fundamental,
        "model_class": CompanyEquity,
        "filename": "company_equity.json",
        "unique_keys": ["fincode"],
    },
    {
        "table_name":"company_finance_balancesheet",
        "type": fundamental,
        "model_class": CompanyBalanceSheet,
        "filename": "Finance_bs.json",
        "unique_keys": ["fincode","year_end","type"],
    },
    {
        "table_name":"company_finance_balancesheet_cons",
        "type": fundamental,
        "model_class": CompanyBalanceSheetCons,
        "filename": "Finance_cons_bs.json",
        "unique_keys": ["fincode","year_end","type"],
    },
    {
        "table_name":"company_finance_cashflow",
        "type": fundamental,
        "model_class": CompanyCashflow,
        "filename": "Finance_cf.json",
        "unique_keys": ["fincode","year_end","type"],
    },
    {
        "table_name":"company_finance_cashflow_cons",
        "type": fundamental,
        "model_class": CompanyCashflowCons,
        "filename": "Finance_cons_cf.json",
        "unique_keys": ["fincode","year_end","type"],
    },
    {
        "table_name":"company_finance_profitloss",
        "type": fundamental,
        "model_class": CompanyProfitLoss,
        "filename": "Finance_pl.json",
        "unique_keys": ["fincode","year_end","type"],
    },
    {
        "table_name":"company_finance_profitloss_cons",
        "type": fundamental,
        "model_class": CompanyProfitLossCons,
        "filename": "Finance_cons_pl.json",
        "unique_keys": ["fincode","year_end","type"],
    },
    {
        "table_name":"company_finance_ratio",
        "type": fundamental,
        "model_class": CompanyFinanceRatio,
        "filename": "Finance_fr.json",
        "unique_keys": ["fincode","year_end","type"],
    },
    {
        "table_name":"company_finance_ratio_cons",
        "type": fundamental,
        "model_class": CompanyFinanceRatioCons,
        "filename": "Finance_cons_fr.json",
        "unique_keys": ["fincode","year_end","type"],
    },
    {
        "table_name":"company_results",
        "type": fundamental,
        "model_class": FinancialResult,
        "filename": "Resultsf_IND_Ex1.json",
        "unique_keys": ["fincode","date_end","result_type"],
    },
    {
        "table_name":"company_results_cons",
        "type": fundamental,
        "model_class": FinancialResultCons,
        "filename": "Resultsf_IND_Cons_Ex1.json",
        "unique_keys": ["fincode","date_end","result_type"],
    },
    {
        "table_name":"indices_master",
        "type": stockprice,
        "model_class": IndicesMaster,
        "filename": "Indicesmaster.json",
        "unique_keys": ["index_code"],
    },
    {
        "table_name":"company_index_part",
        "type": stockprice,
        "model_class": CompanyIndexPart,
        "filename": "Comp_Indexpart.json",
        "unique_keys": ["fincode","index_code"],
    },
    {
        "table_name":"bse_indices_price_eod",
        "type": stockprice,
        "model_class": BSEIndicesEOD,
        "filename": "Indices_hst(BSE).json",
        "unique_keys": ["scripcode","date"],
    },
    {
        "table_name":"company_shareholding_pattern",
        "type": fundamental,
        "model_class": ShpSummary,
        "filename": "Shpsummary.json",
        "unique_keys": ["fincode","date_end"],
    },
    {
        "table_name":"bse_abjusted_price_eod",
        "type": stockprice,
        "model_class": BSEAdjustedPrice,
        "filename": "Bseadjprice.json",
        "unique_keys": ["fincode","date"],
    },
    {
        "table_name":"company_profile",
        "type": fundamental,
        "model_class": CompanyProfile,
        "filename": "CompanyBriefProfile.json",
        "unique_keys": ["fincode"],
    },
]
