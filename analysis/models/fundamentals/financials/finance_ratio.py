from typing import Optional
from sqlmodel import SQLModel, Field


class CompanyFinanceRatio(SQLModel, table=True):
    __tablename__ = "company_finance_ratio"

    fincode: int = Field(primary_key=True, description="Accord Company Code")  # 1
    year_end: int = Field(primary_key=True, description="Year End")  # 2
    type: str = Field(primary_key=True, max_length=1, description="Standalone or Consolidated")  # 3
    reported_eps: Optional[float] = Field(default=None, description="Reported EPS")  # 4
    adjusted_eps: Optional[float] = Field(default=None, description="Adjusted EPS")  # 5
    ceps: Optional[float] = Field(default=None, description="Cash Earnings per share")  # 6
    dps: Optional[float] = Field(default=None, description="Dividend Per share")  # 7
    book_nav_share: Optional[float] = Field(default=None, description="Book NAV Share")  # 8
    tax_rate: Optional[float] = Field(default=None, description="Tax Rate")  # 9
    core_ebitda_margin: Optional[float] = Field(default=None, description="Core EBITDA Margin")  # 10
    ebit_margin: Optional[float] = Field(default=None, description="EBIT Margin")  # 11
    pre_tax_margin: Optional[float] = Field(default=None, description="Pre Tax Margin")  # 12
    pat_margin: Optional[float] = Field(default=None, description="PAT Margin")  # 13
    cash_profit_margin: Optional[float] = Field(default=None, description="Cash Profit Margin")  # 14
    roa: Optional[float] = Field(default=None, description="Return On Assets")  # 15
    roe: Optional[float] = Field(default=None, description="Return On Equity")  # 16
    roce: Optional[float] = Field(default=None, description="Return On Capital Employed")  # 17
    asset_turnover: Optional[float] = Field(default=None, description="Asset Turnover")  # 18
    sales_fixed_asset: Optional[float] = Field(default=None, description="Sales Fixed Asset")  # 19
    working_capital_sales: Optional[float] = Field(default=None, description="Working Capital Sales")  # 20
    fixed_capital_sales: Optional[float] = Field(default=None, description="Fixed Capital Sales")  # 21
    receivable_days: Optional[float] = Field(default=None, description="Receivable days")  # 22
    inventory_days: Optional[float] = Field(default=None, description="Inventory Days")  # 23
    payable_days: Optional[float] = Field(default=None, description="Payable days")  # 24
    per: Optional[float] = Field(default=None, description="Price earning ratio")  # 25
    pce: Optional[float] = Field(default=None, description="Price to cash earnings")  # 26
    price_book: Optional[float] = Field(default=None, description="Price Book")  # 27
    yield_: Optional[float] = Field(default=None, description="Dividend Yield")  # 28
    ev_net_sales: Optional[float] = Field(default=None, description="EV to Net Sales")  # 29
    ev_core_ebitda: Optional[float] = Field(default=None, description="EV to Core EBITDA")  # 30
    ev_ebit: Optional[float] = Field(default=None, description="EV to EBIT")  # 31
    ev_ce: Optional[float] = Field(default=None, description="EV to Capital Employed")  # 32
    mcap_sales: Optional[float] = Field(default=None, description="MCap to Sales")  # 33
    net_sales_growth: Optional[float] = Field(default=None, description="Net Sales Growth")  # 34
    pbidt_excl_oi_growth: Optional[float] = Field(default=None, description="PBIDT to Excl OI Growth")  # 35
    core_ebitda_growth: Optional[float] = Field(default=None, description="Core EBITDA Growth")  # 36
    ebit_growth: Optional[float] = Field(default=None, description="EBIT Growth")  # 37
    pat_growth: Optional[float] = Field(default=None, description="PAT Growth")  # 38
    adj_pat_growth: Optional[float] = Field(default=None, description="Adj PAT Growth")  # 39
    adj_eps_growth: Optional[float] = Field(default=None, description="Adj EPS Growth")  # 40
    reported_eps_growth: Optional[float] = Field(default=None, description="Reported EPS Growth")  # 41
    total_debt_equity: Optional[float] = Field(default=None, description="Total Debt Equity")  # 42
    current_ratio: Optional[float] = Field(default=None, description="Current Ratio")  # 43
    quick_ratio: Optional[float] = Field(default=None, description="Quick Ratio")  # 44
    interest_cover: Optional[float] = Field(default=None, description="Interest Cover ratio")  # 45
    total_debt_mcap: Optional[float] = Field(default=None, description="Total Debt to Mcap")  # 46
    yield_adv: Optional[float] = Field(default=None, description="Yield Adv")  # 47
    yield_inv: Optional[float] = Field(default=None, description="Yield Inv")  # 48
    cost_liab: Optional[float] = Field(default=None, description="Cost Liab")  # 49
    nim: Optional[float] = Field(default=None, description="Net Interest margin")  # 50
    int_spread: Optional[float] = Field(default=None, description="Interest Spread ratio")  # 51
    cost_incratio: Optional[float] = Field(default=None, description="Cost Income ratio")  # 52
    core_cost_incratio: Optional[float] = Field(default=None, description="Core Cost Income ratio")  # 53
    op_cost_asset: Optional[float] = Field(default=None, description="Operating Cost to Asset")  # 54
    adj_per: Optional[float] = Field(default=None, description="Adj PER")  # 55
    tier1_ratio: Optional[float] = Field(default=None, description="Tier1Ratio")  # 56
    tier2_ratio: Optional[float] = Field(default=None, description="Tier2Ratio")  # 57
    car: Optional[float] = Field(default=None, description="CAR")  # 58
    core_op_income_growth: Optional[float] = Field(default=None, description="Operating income growth")  # 59
    eps_growth: Optional[float] = Field(default=None, description="EPS Growth")  # 60
    bvps_growth: Optional[float] = Field(default=None, description="BVPS Growth")  # 61
    adv_growth: Optional[float] = Field(default=None, description="Advances Growth")  # 62
    loan_deposits: Optional[float] = Field(default=None, description="Loan to Deposits")  # 63
    cash_deposits: Optional[float] = Field(default=None, description="Cash to Deposits")  # 64
    investment_deposits: Optional[float] = Field(default=None, description="Investment to Deposits")  # 65
    inc_loan_deposits: Optional[float] = Field(default=None, description="IncLoan to Deposits")  # 66
    gross_npa: Optional[float] = Field(default=None, description="Gross NPA %")  # 67
    net_npa: Optional[float] = Field(default=None, description="Net NPA %")  # 68
    ownersfund_total_source: Optional[float] = Field(default=None, description="Ownersfund total Source")  # 69
    fixed_assets_ta: Optional[float] = Field(default=None, description="Gross sales to Net block")  # 70
    inventory_tr: Optional[float] = Field(default=None, description="Inventory Turnover")  # 124
    dividend_pr_np: Optional[float] = Field(default=None, description="Dividend PR NP")  # 72
    dividend_pr_cp: Optional[float] = Field(default=None, description="Dividend PR CP")  # 73
    earning_retention_ratio: Optional[float] = Field(default=None, description="Earning Retention Ratio")  # 74
    cash_earnings_retention: Optional[float] = Field(default=None, description="Cash Earnings Retention")  # 75
    price_bv: Optional[float] = Field(default=None, description="Price BV")  # 76
    return_sales: Optional[float] = Field(default=None, description="Return Sales")  # 77
    debt_ta: Optional[float] = Field(default=None, description="Debt TA")  # 78

    ev: Optional[float] = Field(default=None, description="Enterprise value (Units will be as per Annual Financial data of specific company)")  # 79
    price_sales_ratio: Optional[float] = Field(default=None, description="PriceSalesRatio")  # 80
    ev_ebita: Optional[float] = Field(default=None, description="EV EBITA")  # 81
    cf_per_share: Optional[float] = Field(default=None, description="CF PerShare")  # 82
    pcf_ratio: Optional[float] = Field(default=None, description="PCF RATIO")  # 83
    fcf_share: Optional[float] = Field(default=None, description="FCF Share")  # 84
    pfcf_ratio: Optional[float] = Field(default=None, description="PFCF Ratio")  # 85
    fcf_yield: Optional[float] = Field(default=None, description="FCF Yield")  # 86
    scf_ratio: Optional[float] = Field(default=None, description="SCF Ratio")  # 87
    gpm: Optional[float] = Field(default=None, description="GPM")  # 88
    sales_to_total_assets: Optional[float] = Field(default=None, description="Sales To Total Assets")  # 89
    sales_to_current_assets: Optional[float] = Field(default=None, description="Sales To Current Assets")  # 90
    total_car_base_i: Optional[float] = Field(default=None, description="Total CAR baseI")  # 91
    ti_car_base_i: Optional[float] = Field(default=None, description="TI CAR baseI")  # 92
    tii_car_base_i: Optional[float] = Field(default=None, description="TII CAR baseI")  # 93
    total_car: Optional[float] = Field(default=None, description="Total CAR")  # 94
    ti_car: Optional[float] = Field(default=None, description="TI CAR")  # 95
    tii_car: Optional[float] = Field(default=None, description="TII CAR")  # 96
    npa_gross: Optional[float] = Field(default=None, description="Gross NPA Amount")  # 97
    npa_net: Optional[float] = Field(default=None, description="Net NPA Amount")  # 98
    npa_advances: Optional[float] = Field(default=None, description="Net NPA %")  # 99
    net_profit_to_pbt: Optional[float] = Field(default=None, description="NetProfit To PBT")  # 100
    pbt_to_pbit: Optional[float] = Field(default=None, description="PBT To PBIT")  # 101
    pbit_to_sales: Optional[float] = Field(default=None, description="PBIT To Sales")  # 102
    net_profit_to_sales: Optional[float] = Field(default=None, description="NetProfit To Sales")  # 103
    net_sales_to_total_assets: Optional[float] = Field(default=None, description="NetSales To TotalAssets")  # 104
    return_on_assets: Optional[float] = Field(default=None, description="Return On Assets. Ignore this field and refer ROA field.")  # 105
    assets_to_equity: Optional[float] = Field(default=None, description="Assets To Equity")  # 106
    return_on_equity: Optional[float] = Field(default=None, description="Return On Equity")  # 107
    interest_to_debt: Optional[float] = Field(default=None, description="Interest To Debt")  # 108
    debt_to_assets: Optional[float] = Field(default=None, description="Debt To Assets")  # 109
    interest_to_assets: Optional[float] = Field(default=None, description="Interest To Assets")  # 110
    roe_after_interest: Optional[float] = Field(default=None, description="ROE After Interest")  # 111
    dividend_payout_per: Optional[float] = Field(default=None, description="Dividend Payout Per")  # 112
    other_income_to_net_worth: Optional[float] = Field(default=None, description="OtherIncome To NetWorth")  # 113
    roa_after_interest: Optional[float] = Field(default=None, description="ROA After Interest")  # 114
    roe_before_other_inc: Optional[float] = Field(default=None, description="ROE Before OtherInc")  # 115
    roe_after_other_inc: Optional[float] = Field(default=None, description="ROE After OtherInc")  # 116
    roe_after_tax_rate: Optional[float] = Field(default=None, description="ROE After Tax Rate")  # 117
    credit_deposit: Optional[float] = Field(default=None, description="Credit to Deposit")  # 118
    interest_expended_interest_earned: Optional[float] = Field(default=None, description="InterestExpended to InterestEarned")  # 119
    interest_income_total_funds: Optional[float] = Field(default=None, description="InterestIncome to TotalFunds")  # 120

    interest_expended_total_funds: Optional[float] = Field(default=None, description="InterestExpended to TotalFunds")  # 121
    net_interest_income_total_funds: Optional[float] = Field(default=None, description="NetInterestIncome to TotalFunds")  # 122
    casa: Optional[float] = Field(default=None, description="CASA")  # 123
    inventory_turnover: Optional[float] = Field(default=None, description="Inventory Turnover")  # 124
    debtors_turnover: Optional[float] = Field(default=None, description="Debtors Turnover")  # 125
    adj_pe: Optional[float] = Field(default=None, description="Adj PE")  # 126
    adjusted_bv: Optional[float] = Field(default=None, description="Adjusted bv")  # 127
    adj_dps: Optional[float] = Field(default=None, description="Adj DPS")  # 128
    networth_growth: Optional[float] = Field(default=None, description="Networth Growth")  # 129
    ce_growth: Optional[float] = Field(default=None, description="CE Growth")  # 130
    gb_growth: Optional[float] = Field(default=None, description="GB Growth")  # 131
    pbdt_growth: Optional[float] = Field(default=None, description="PBDT Growth")  # 132
    pbit_growth: Optional[float] = Field(default=None, description="PBIT Growth")  # 133
    pbt_growth: Optional[float] = Field(default=None, description="PBT Growth")  # 134
    cp_growth: Optional[float] = Field(default=None, description="CP Growth")  # 135
    exports_growth: Optional[float] = Field(default=None, description="Exports Growth")  # 136
    imports_growth: Optional[float] = Field(default=None, description="Imports Growth")  # 137
    mcap_growth: Optional[float] = Field(default=None, description="MCap Growth")  # 138
    ebidtam: Optional[float] = Field(default=None, description="EBIDTAM")  # 139
    pbdtm: Optional[float] = Field(default=None, description="PBDTM")  # 140
    lt_debt_equity_ratio: Optional[float] = Field(default=None, description="LT Debt Equity Ratio")  # 141
    roic: Optional[float] = Field(default=None, description="ROIC")  # 142
    total_car_b: Optional[float] = Field(default=None, description="Total CAR (Basel III)")  # 143
    ti_car_b: Optional[float] = Field(default=None, description="Tier - 1 (Basel III)")  # 144
    tii_car_b: Optional[float] = Field(default=None, description="Tier - 2 (Basel III)")  # 145
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 146
    


class CompanyFinanceRatioCons(SQLModel, table=True):
    __tablename__ = "company_finance_ratio_cons"

    fincode: int = Field(primary_key=True, description="Accord Company Code")  # 1
    year_end: int = Field(primary_key=True, description="Year End")  # 2
    type: str = Field(primary_key=True, max_length=1, description="Standalone or Consolidated")  # 3
    reported_eps: Optional[float] = Field(default=None, description="Reported EPS")  # 4
    adjusted_eps: Optional[float] = Field(default=None, description="Adjusted EPS")  # 5
    ceps: Optional[float] = Field(default=None, description="Cash Earnings per share")  # 6
    dps: Optional[float] = Field(default=None, description="Dividend Per share")  # 7
    book_nav_share: Optional[float] = Field(default=None, description="Book NAV Share")  # 8
    tax_rate: Optional[float] = Field(default=None, description="Tax Rate")  # 9
    core_ebitda_margin: Optional[float] = Field(default=None, description="Core EBITDA Margin")  # 10
    ebit_margin: Optional[float] = Field(default=None, description="EBIT Margin")  # 11
    pre_tax_margin: Optional[float] = Field(default=None, description="Pre Tax Margin")  # 12
    pat_margin: Optional[float] = Field(default=None, description="PAT Margin")  # 13
    cash_profit_margin: Optional[float] = Field(default=None, description="Cash Profit Margin")  # 14
    roa: Optional[float] = Field(default=None, description="Return On Assets")  # 15
    roe: Optional[float] = Field(default=None, description="Return On Equity")  # 16
    roce: Optional[float] = Field(default=None, description="Return On Capital Employed")  # 17
    asset_turnover: Optional[float] = Field(default=None, description="Asset Turnover")  # 18
    sales_fixed_asset: Optional[float] = Field(default=None, description="Sales Fixed Asset")  # 19
    working_capital_sales: Optional[float] = Field(default=None, description="Working Capital Sales")  # 20
    fixed_capital_sales: Optional[float] = Field(default=None, description="Fixed Capital Sales")  # 21
    receivable_days: Optional[float] = Field(default=None, description="Receivable days")  # 22
    inventory_days: Optional[float] = Field(default=None, description="Inventory Days")  # 23
    payable_days: Optional[float] = Field(default=None, description="Payable days")  # 24
    per: Optional[float] = Field(default=None, description="Price earning ratio")  # 25
    pce: Optional[float] = Field(default=None, description="Price to cash earnings")  # 26
    price_book: Optional[float] = Field(default=None, description="Price Book")  # 27
    yield_: Optional[float] = Field(default=None, description="Dividend Yield")  # 28
    ev_net_sales: Optional[float] = Field(default=None, description="EV to Net Sales")  # 29
    ev_core_ebitda: Optional[float] = Field(default=None, description="EV to Core EBITDA")  # 30
    ev_ebit: Optional[float] = Field(default=None, description="EV to EBIT")  # 31
    ev_ce: Optional[float] = Field(default=None, description="EV to Capital Employed")  # 32
    mcap_sales: Optional[float] = Field(default=None, description="MCap to Sales")  # 33
    net_sales_growth: Optional[float] = Field(default=None, description="Net Sales Growth")  # 34
    pbidt_excl_oi_growth: Optional[float] = Field(default=None, description="PBIDT to Excl OI Growth")  # 35
    core_ebitda_growth: Optional[float] = Field(default=None, description="Core EBITDA Growth")  # 36
    ebit_growth: Optional[float] = Field(default=None, description="EBIT Growth")  # 37
    pat_growth: Optional[float] = Field(default=None, description="PAT Growth")  # 38
    adj_pat_growth: Optional[float] = Field(default=None, description="Adj PAT Growth")  # 39
    adj_eps_growth: Optional[float] = Field(default=None, description="Adj EPS Growth")  # 40
    reported_eps_growth: Optional[float] = Field(default=None, description="Reported EPS Growth")  # 41
    total_debt_equity: Optional[float] = Field(default=None, description="Total Debt Equity")  # 42
    current_ratio: Optional[float] = Field(default=None, description="Current Ratio")  # 43
    quick_ratio: Optional[float] = Field(default=None, description="Quick Ratio")  # 44
    interest_cover: Optional[float] = Field(default=None, description="Interest Cover ratio")  # 45
    total_debt_mcap: Optional[float] = Field(default=None, description="Total Debt to Mcap")  # 46
    yield_adv: Optional[float] = Field(default=None, description="Yield Adv")  # 47
    yield_inv: Optional[float] = Field(default=None, description="Yield Inv")  # 48
    cost_liab: Optional[float] = Field(default=None, description="Cost Liab")  # 49
    nim: Optional[float] = Field(default=None, description="Net Interest margin")  # 50
    int_spread: Optional[float] = Field(default=None, description="Interest Spread ratio")  # 51
    cost_incratio: Optional[float] = Field(default=None, description="Cost Income ratio")  # 52
    core_cost_incratio: Optional[float] = Field(default=None, description="Core Cost Income ratio")  # 53
    op_cost_asset: Optional[float] = Field(default=None, description="Operating Cost to Asset")  # 54
    adj_per: Optional[float] = Field(default=None, description="Adj PER")  # 55
    tier1_ratio: Optional[float] = Field(default=None, description="Tier1Ratio")  # 56
    tier2_ratio: Optional[float] = Field(default=None, description="Tier2Ratio")  # 57
    car: Optional[float] = Field(default=None, description="CAR")  # 58
    core_op_income_growth: Optional[float] = Field(default=None, description="Operating income growth")  # 59
    eps_growth: Optional[float] = Field(default=None, description="EPS Growth")  # 60
    bvps_growth: Optional[float] = Field(default=None, description="BVPS Growth")  # 61
    adv_growth: Optional[float] = Field(default=None, description="Advances Growth")  # 62
    loan_deposits: Optional[float] = Field(default=None, description="Loan to Deposits")  # 63
    cash_deposits: Optional[float] = Field(default=None, description="Cash to Deposits")  # 64
    investment_deposits: Optional[float] = Field(default=None, description="Investment to Deposits")  # 65
    inc_loan_deposits: Optional[float] = Field(default=None, description="IncLoan to Deposits")  # 66
    gross_npa: Optional[float] = Field(default=None, description="Gross NPA %")  # 67
    net_npa: Optional[float] = Field(default=None, description="Net NPA %")  # 68
    ownersfund_total_source: Optional[float] = Field(default=None, description="Ownersfund total Source")  # 69
    fixed_assets_ta: Optional[float] = Field(default=None, description="Gross sales to Net block")  # 70
    inventory_tr: Optional[float] = Field(default=None, description="Inventory Turnover")  # 124
    dividend_pr_np: Optional[float] = Field(default=None, description="Dividend PR NP")  # 72
    dividend_pr_cp: Optional[float] = Field(default=None, description="Dividend PR CP")  # 73
    earning_retention_ratio: Optional[float] = Field(default=None, description="Earning Retention Ratio")  # 74
    cash_earnings_retention: Optional[float] = Field(default=None, description="Cash Earnings Retention")  # 75
    price_bv: Optional[float] = Field(default=None, description="Price BV")  # 76
    return_sales: Optional[float] = Field(default=None, description="Return Sales")  # 77
    debt_ta: Optional[float] = Field(default=None, description="Debt TA")  # 78

    ev: Optional[float] = Field(default=None, description="Enterprise value (Units will be as per Annual Financial data of specific company)")  # 79
    price_sales_ratio: Optional[float] = Field(default=None, description="PriceSalesRatio")  # 80
    ev_ebita: Optional[float] = Field(default=None, description="EV EBITA")  # 81
    cf_per_share: Optional[float] = Field(default=None, description="CF PerShare")  # 82
    pcf_ratio: Optional[float] = Field(default=None, description="PCF RATIO")  # 83
    fcf_share: Optional[float] = Field(default=None, description="FCF Share")  # 84
    pfcf_ratio: Optional[float] = Field(default=None, description="PFCF Ratio")  # 85
    fcf_yield: Optional[float] = Field(default=None, description="FCF Yield")  # 86
    scf_ratio: Optional[float] = Field(default=None, description="SCF Ratio")  # 87
    gpm: Optional[float] = Field(default=None, description="GPM")  # 88
    sales_to_total_assets: Optional[float] = Field(default=None, description="Sales To Total Assets")  # 89
    sales_to_current_assets: Optional[float] = Field(default=None, description="Sales To Current Assets")  # 90
    total_car_base_i: Optional[float] = Field(default=None, description="Total CAR baseI")  # 91
    ti_car_base_i: Optional[float] = Field(default=None, description="TI CAR baseI")  # 92
    tii_car_base_i: Optional[float] = Field(default=None, description="TII CAR baseI")  # 93
    total_car: Optional[float] = Field(default=None, description="Total CAR")  # 94
    ti_car: Optional[float] = Field(default=None, description="TI CAR")  # 95
    tii_car: Optional[float] = Field(default=None, description="TII CAR")  # 96
    npa_gross: Optional[float] = Field(default=None, description="Gross NPA Amount")  # 97
    npa_net: Optional[float] = Field(default=None, description="Net NPA Amount")  # 98
    npa_advances: Optional[float] = Field(default=None, description="Net NPA %")  # 99
    net_profit_to_pbt: Optional[float] = Field(default=None, description="NetProfit To PBT")  # 100
    pbt_to_pbit: Optional[float] = Field(default=None, description="PBT To PBIT")  # 101
    pbit_to_sales: Optional[float] = Field(default=None, description="PBIT To Sales")  # 102
    net_profit_to_sales: Optional[float] = Field(default=None, description="NetProfit To Sales")  # 103
    net_sales_to_total_assets: Optional[float] = Field(default=None, description="NetSales To TotalAssets")  # 104
    return_on_assets: Optional[float] = Field(default=None, description="Return On Assets. Ignore this field and refer ROA field.")  # 105
    assets_to_equity: Optional[float] = Field(default=None, description="Assets To Equity")  # 106
    return_on_equity: Optional[float] = Field(default=None, description="Return On Equity")  # 107
    interest_to_debt: Optional[float] = Field(default=None, description="Interest To Debt")  # 108
    debt_to_assets: Optional[float] = Field(default=None, description="Debt To Assets")  # 109
    interest_to_assets: Optional[float] = Field(default=None, description="Interest To Assets")  # 110
    roe_after_interest: Optional[float] = Field(default=None, description="ROE After Interest")  # 111
    dividend_payout_per: Optional[float] = Field(default=None, description="Dividend Payout Per")  # 112
    other_income_to_net_worth: Optional[float] = Field(default=None, description="OtherIncome To NetWorth")  # 113
    roa_after_interest: Optional[float] = Field(default=None, description="ROA After Interest")  # 114
    roe_before_other_inc: Optional[float] = Field(default=None, description="ROE Before OtherInc")  # 115
    roe_after_other_inc: Optional[float] = Field(default=None, description="ROE After OtherInc")  # 116
    roe_after_tax_rate: Optional[float] = Field(default=None, description="ROE After Tax Rate")  # 117
    credit_deposit: Optional[float] = Field(default=None, description="Credit to Deposit")  # 118
    interest_expended_interest_earned: Optional[float] = Field(default=None, description="InterestExpended to InterestEarned")  # 119
    interest_income_total_funds: Optional[float] = Field(default=None, description="InterestIncome to TotalFunds")  # 120

    interest_expended_total_funds: Optional[float] = Field(default=None, description="InterestExpended to TotalFunds")  # 121
    net_interest_income_total_funds: Optional[float] = Field(default=None, description="NetInterestIncome to TotalFunds")  # 122
    casa: Optional[float] = Field(default=None, description="CASA")  # 123
    inventory_turnover: Optional[float] = Field(default=None, description="Inventory Turnover")  # 124
    debtors_turnover: Optional[float] = Field(default=None, description="Debtors Turnover")  # 125
    adj_pe: Optional[float] = Field(default=None, description="Adj PE")  # 126
    adjusted_bv: Optional[float] = Field(default=None, description="Adjusted bv")  # 127
    adj_dps: Optional[float] = Field(default=None, description="Adj DPS")  # 128
    networth_growth: Optional[float] = Field(default=None, description="Networth Growth")  # 129
    ce_growth: Optional[float] = Field(default=None, description="CE Growth")  # 130
    gb_growth: Optional[float] = Field(default=None, description="GB Growth")  # 131
    pbdt_growth: Optional[float] = Field(default=None, description="PBDT Growth")  # 132
    pbit_growth: Optional[float] = Field(default=None, description="PBIT Growth")  # 133
    pbt_growth: Optional[float] = Field(default=None, description="PBT Growth")  # 134
    cp_growth: Optional[float] = Field(default=None, description="CP Growth")  # 135
    exports_growth: Optional[float] = Field(default=None, description="Exports Growth")  # 136
    imports_growth: Optional[float] = Field(default=None, description="Imports Growth")  # 137
    mcap_growth: Optional[float] = Field(default=None, description="MCap Growth")  # 138
    ebidtam: Optional[float] = Field(default=None, description="EBIDTAM")  # 139
    pbdtm: Optional[float] = Field(default=None, description="PBDTM")  # 140
    lt_debt_equity_ratio: Optional[float] = Field(default=None, description="LT Debt Equity Ratio")  # 141
    roic: Optional[float] = Field(default=None, description="ROIC")  # 142
    total_car_b: Optional[float] = Field(default=None, description="Total CAR (Basel III)")  # 143
    ti_car_b: Optional[float] = Field(default=None, description="Tier - 1 (Basel III)")  # 144
    tii_car_b: Optional[float] = Field(default=None, description="Tier - 2 (Basel III)")  # 145
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 146