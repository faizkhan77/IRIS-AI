from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, Text
from sqlalchemy.dialects.mysql import MEDIUMTEXT


class FinancialResult(SQLModel, table=True):
    __tablename__ = "company_results"

    fincode: int = Field(primary_key=True, description="Company Code")  # 1
    result_type: str = Field(primary_key=True, max_length=2, description="Result Type. Q-Quarterly, QR-Quarterly revised, H-Halfyearly, HR-Halfyearly revised, A-Annual, AR-Annual revised.")  # 2
    no_of_months: Optional[int] = Field(default=None, description="No of Months")  # 3
    date_end: int = Field(primary_key=True, description="Date End")  # 4

    net_sales: Optional[float] = Field(default=None, description="Total Revenue from Operations")  # 5
    gross_sale: Optional[float] = Field(default=None, description="Interest Income")  # 6
    excise_duty: Optional[float] = Field(default=None, description="GST")  # 7
    other_net_sales: Optional[float] = Field(default=None, description="Income from Operations_net")  # 8
    job_works: Optional[float] = Field(default=None, description="Other Operating Income")  # 9
    other_income: Optional[float] = Field(default=None, description="Other Income")  # 10
    dividend_income: Optional[float] = Field(default=None, description="Dividend Income")  # 11
    export_incentives: Optional[float] = Field(default=None, description="Export Incentives")  # 12
    foreign_exchange_gain: Optional[float] = Field(default=None, description="Foreign Exchange Gain")  # 13
    profit_investment: Optional[float] = Field(default=None, description="Profit on sale of Investments")  # 14
    other_interest_income: Optional[float] = Field(default=None, description="Interest Income")  # 15
    rental_income: Optional[float] = Field(default=None, description="Lease / Rental Income")  # 16
    sale_shares_units: Optional[float] = Field(default=None, description="Profit on sale of Share")  # 17
    prov_written_back: Optional[float] = Field(default=None, description="Provision Written Back")  # 18
    sale_assets: Optional[float] = Field(default=None, description="Profit on sale of Asset")  # 19
    other_other_income: Optional[float] = Field(default=None, description="Other Income")  # 20
    total_income: Optional[float] = Field(default=None, description="Total Income")  # 21
    expenditure: Optional[float] = Field(default=None, description="Total Expenditure")  # 22
    inc_dec_inventory: Optional[float] = Field(default=None, description="(Increase) / Decrease In Stocks")  # 23
    excise: Optional[float] = Field(default=None, description="Excise Duty")  # 24
    raw_material_cost: Optional[float] = Field(default=None, description="Raw Material Cost")  # 25
    purchase_fin_good: Optional[float] = Field(default=None, description="Purchase of Finished Goods")  # 26
    mfg_exps: Optional[float] = Field(default=None, description="Manufacturing Expenses")  # 27
    electricity_power_fuel: Optional[float] = Field(default=None, description="Electricity , Power & Fuel Cost")  # 28
    exployee_cost: Optional[float] = Field(default=None, description="Employees Cost")  # 29
    interest: Optional[float] = Field(default=None, description="Interest")  # 30
    depreciation: Optional[float] = Field(default=None, description="Depreciation")  # 31
    gen_admin_expenses: Optional[float] = Field(default=None, description="General Administration Expenses")  # 32
    selling_dist_expenses: Optional[float] = Field(default=None, description="Selling & Distribution Expenses")  # 33
    misc_expenses: Optional[float] = Field(default=None, description="Misc Expenses")  # 34
    loss_foreign_exchange: Optional[float] = Field(default=None, description="Loss_Foreign_exchange")  # 35
    loss_foreign_exchange_loan: Optional[float] = Field(default=None, description="Loss_Foreign_exchange_loan")  # 36
    expence_capitalised: Optional[float] = Field(default=None, description="Expenses Capitalised")  # 37
    pb_exitem_satax: Optional[float] = Field(default=None, description="Profit before Exceptional Items, Share of Associates and tax")  # 38
    exception_items: Optional[float] = Field(default=None, description="Exceptional Items")  # 39
    exc_foreign_exch_gain: Optional[float] = Field(default=None, description="Foreign Exchange Gain/Loss")  # 40
    exc_other: Optional[float] = Field(default=None, description="Other Exceptional items")  # 41
    pb_satax: Optional[float] = Field(default=None, description="Profit before Share of Associates and tax")  # 42
    shares_associate: Optional[float] = Field(default=None, description="Share of (loss)/profit in Associates and Joint Ventures")  # 43
    pbt: Optional[float] = Field(default=None, description="Profit before tax")  # 44
    tax: Optional[float] = Field(default=None, description="Tax")  # 45
    curr_tax: Optional[float] = Field(default=None, description="Current Tax")  # 46
    def_tax: Optional[float] = Field(default=None, description="Deferred Tax")  # 47
    fringe_benefits: Optional[float] = Field(default=None, description="Fringe Benefit Tax")  # 48
    prior_period_tax: Optional[float] = Field(default=None, description="Prior Period / Year Tax")  # 49
    mat_credit: Optional[float] = Field(default=None, description="MAT Credit")  # 50
    other_tax: Optional[float] = Field(default=None, description="Other Tax")  # 51
    pat: Optional[float] = Field(default=None, description="Profit after Tax")  # 52
    pbt_disc_operations: Optional[float] = Field(default=None, description="Profit before tax from discontinued operations")  # 53
    taxexp_discoperations: Optional[float] = Field(default=None, description="Tax expense of discontinued operations")  # 54
    discontinued_op: Optional[float] = Field(default=None, description="Profit for the period from discontinued operations")  # 55
    other_related_items: Optional[float] = Field(default=None, description="Other related Items")  # 56
    net_profit: Optional[float] = Field(default=None, description="Profit for the period")  # 57
    other_compincome: Optional[float] = Field(default=None, description="Other Comprehensive Incomes (Net of tax)")  # 58
    items_notclassif_pl: Optional[float] = Field(default=None, description="Items that will not be reclassified to profit or loss")  # 59
    it_items_notclassif_pl: Optional[float] = Field(default=None, description="Income tax relating to items that will not be reclassified to profit or loss")  # 60
    items_classif_pl: Optional[float] = Field(default=None, description="Items that will be reclassified to profit or loss")  # 61
    it_items_classif_pl: Optional[float] = Field(default=None, description="Income tax relating to Items that will be reclassified to profit or loss")  # 62
    minint_compincome: Optional[float] = Field(default=None, description="Minority Interest Comprehensive Income")  # 63
    othcinc_other: Optional[float] = Field(default=None, description="Other")  # 64
    total_compincome: Optional[float] = Field(default=None, description="Total Comprehensive Income")  # 65
    netprofit_ownofparent: Optional[float] = Field(default=None, description="Net Profit - Owners of the Parent")  # 66
    netprofit_noncontint: Optional[float] = Field(default=None, description="Net Profit - Non-controlling interests")  # 67
    other_compinc_ownofparent: Optional[float] = Field(default=None, description="Other Comprehensive Income - Owners of the Parent")  # 68
    other_compinc_noncontint: Optional[float] = Field(default=None, description="Other Comprehensive Income - Non-controlling interests")  # 69
    totalcompincome_ownofparent: Optional[float] = Field(default=None, description="Total Comprehensive Income - Owners of the Parent")  # 70
    minority_interest: Optional[float] = Field(default=None, description="Non-controlling interests")  # 71
    equity_cap: Optional[float] = Field(default=None, description="Equity Capital")  # 72
    fv: Optional[float] = Field(default=None, description="Face Value (In Rs)")  # 73
    reserves: Optional[float] = Field(default=None, description="Reserves")  # 74
    epsabs: Optional[float] = Field(default=None, description="Calculated EPS (Unit.Curr.)")  # 75
    epsann: Optional[float] = Field(default=None, description="Calculated EPS Annualised (Unit.Curr.)")  # 76
    adj_eps_abs: Optional[float] = Field(default=None, description="Adj Calculated EPS (Unit.Curr.)")  # 77
    adj_eps_ann: Optional[float] = Field(default=None, description="Adj Calculated EPS Annualised (Unit.Curr.)")  # 78
    eps_basic: Optional[float] = Field(default=None, description="Basic EPS")  # 79
    eps_diluted_extraord: Optional[float] = Field(default=None, description="Diluted EPS")  # 80
    pbidtmexoi: Optional[float] = Field(default=None, description="PBIDTM% (Excl OI)")  # 81
    pbidtm: Optional[float] = Field(default=None, description="PBIDTM%")  # 82
    pbdtm: Optional[float] = Field(default=None, description="PBDTM%")  # 83
    pbtm: Optional[float] = Field(default=None, description="PBTM%")  # 84
    patm: Optional[float] = Field(default=None, description="PATM%")  # 85
    rev_div_income: Optional[float] = Field(default=None, description="Dividend Income")  # 86
    rev_fee_comm: Optional[float] = Field(default=None, description="Fees and Commission Income")  # 87
    rev_gain_fairvalue: Optional[float] = Field(default=None, description="Gain on Fair value changes")  # 88
    rev_gain_finassets: Optional[float] = Field(default=None, description="Gain on de-recognised of Financial Instruments")  # 89
    inc_inv: Optional[float] = Field(default=None, description="Profit on sale of Investments")  # 90
    int_bal: Optional[float] = Field(default=None, description="Interest on Balances With RBI Other Inter Bank Funds")  # 91
    int_adv: Optional[float] = Field(default=None, description="Interest / Discount on Advances / Bills")  # 92
    int_others: Optional[float] = Field(default=None, description="Others")  # 93
    operating_ex: Optional[float] = Field(default=None, description="Operating Expenses")  # 94
    prov_emp: Optional[float] = Field(default=None, description="Payment To Provisions For Employees")  # 95
    other_exp: Optional[float] = Field(default=None, description="Other Operating Expenses")  # 96
    operating_profit: Optional[float] = Field(default=None, description="Operating Profit before Prov.& Cont.")  # 97
    pac: Optional[float] = Field(default=None, description="Provisions and Contingencies")  # 98
    ei: Optional[float] = Field(default=None, description="Extraordinary Items")  # 99
    ppi: Optional[float] = Field(default=None, description="Prior Period Items")  # 100
    govt_percent_of_shares: Optional[float] = Field(default=None, description="% of Shares held by Govt")  # 101
    cap_ratio_percent: Optional[float] = Field(default=None, description="Capital Adequacy Ratio Basel II")  # 102
    cap_ratio_percent3: Optional[float] = Field(default=None, description="Capital Adequacy Ratio Basel III")  # 103
    tier1basel3: Optional[float] = Field(default=None, description="Tier I Basel III")  # 104
    tier2basel3: Optional[float] = Field(default=None, description="Tier 2 Basel III")  # 105
    gross_net_npa: Optional[float] = Field(default=None, description="Gross / Net NPA")  # 106
    npa_gross: Optional[float] = Field(default=None, description="Amount of Gross NPA")  # 107
    npa_net: Optional[float] = Field(default=None, description="Amount of Net NPA")  # 108
    perc_gross_net_npa: Optional[float] = Field(default=None, description="Percentage of Gross/Net NPA")  # 109
    npa_net_perc: Optional[float] = Field(default=None, description="% of Net NPAs")  # 110
    npa_gross_perc: Optional[float] = Field(default=None, description="% of Gross NPAs")  # 111
    roa: Optional[float] = Field(default=None, description="Return on Assets")  # 112
    prom_no_of_shares: Optional[float] = Field(default=None, description="Number of Public Share Holding")  # 113
    prom_percent_of_shares: Optional[float] = Field(default=None, description="% of Public Share Holding")  # 114JJ
    eps_basic_extraord: Optional[float] = Field(default=None, description="Basic EPS before Extraordinary Items")  # 115
    eps_diluted: Optional[float] = Field(default=None, description="Diluted EPS before Extraordinary Items")  # 116
    promoter_nos: Optional[float] = Field(default=None, description="Promoters No of Shares")  # 117
    encumbered_nos: Optional[float] = Field(default=None, description="Encumbered No of Shares")  # 118
    percentage_pledgedpromoter: Optional[float] = Field(default=None, description="Encumbered % of Promoter Holdings")  # 119
    percentage_pledgedcapital: Optional[float] = Field(default=None, description="Encumbered % of Share Capital")  # 120
    nonpledgedencum: Optional[float] = Field(default=None, description="Non Encumbered")  # 121
    nonpledged_nos: Optional[float] = Field(default=None, description="Non Encumbered No of Shares")  # 122
    percentage_nonpledgedpromoter: Optional[float] = Field(default=None, description="Non Encumbered % of Promoter Holdings")  # 123
    percentage_nonpledgedcapital: Optional[float] = Field(default=None, description="Non Encumbered % of Share Capital")  # 124
    casa: Optional[float] = Field(default=None, description="CASA%")  # 125
    casa_amount: Optional[float] = Field(default=None, description="CASA Amount")  # 126
    nim: Optional[float] = Field(default=None, description="NIM %")  # 127
    gross_profit: Optional[float] = Field(default=None, description="Gross Profit")  # 128
    pref_cap: Optional[float] = Field(default=None, description="Preference Capital")  # 129
    misc_expd_woff: Optional[float] = Field(default=None, description="Misc. Expenses Written off")  # 130
    consolidated_net_profit: Optional[float] = Field(default=None, description="Consolidated Net Profit")  # 131
    notes: Optional[str] = Field(sa_column=Column(MEDIUMTEXT))  # Changed to TEXT  # 132
    return_on_capital_employed: Optional[float] = Field(default=None, description="Return on Capital Employed")  # 133 - Ignored
    debt_equity_ratio: Optional[float] = Field(default=None, description="Debt/Equity Ratio")  # 134
    interest_coverage_ratio: Optional[float] = Field(default=None, description="Interest Coverage Ratio")  # 135
    inventory_turnover_ratio: Optional[float] = Field(default=None, description="Inventory Turnover Ratio")  # 136
    debtor_turnover_ratio: Optional[float] = Field(default=None, description="Debtor Turnover Ratio")  # 137
    dividend_per_share: Optional[float] = Field(default=None, description="Dividend per share")  # 138
    dividend_payout_ratio: Optional[float] = Field(default=None, description="Dividend payout ratio")  # 139
    other_adjustments: Optional[float] = Field(default=None, description="Other Adjustments")  # 140
    pbidtxoi: Optional[float] = Field(default=None, description="PBIDT (Excl OI)")  # 141
    operating_profit_margin: Optional[float] = Field(default=None, description="Operating Profit Margin")  # 142 - Ignored
    net_profit_margin: Optional[float] = Field(default=None, description="Net Profit Margin")  # 143 - Ignored
    cash_bank: Optional[float] = Field(default=None, description="Cash and Cash Equivalents")  # 144
    debtors: Optional[float] = Field(default=None, description="Debtors")  # 145
    inventory: Optional[float] = Field(default=None, description="Inventory")  # 146
    loans_adv: Optional[float] = Field(default=None, description="Loans and Advances")  # 147
    ns_grand_total: Optional[float] = Field(default=None, description="Number of shares outstanding")  # 148
    ns_totalpublic: Optional[float] = Field(default=None, description="nsTotalpublic")  # 149
    tp_totalpublic: Optional[float] = Field(default=None, description="tpTotalpublic")  # 150
    provisions_coverage: Optional[float] = Field(default=None, description="Provisions Coverage %")  # 151
    no_of_atms: Optional[float] = Field(default=None, description="No of ATMs")  # 152
    no_of_branches: Optional[float] = Field(default=None, description="No of Branches")  # 153
    tier_i_basel_ii: Optional[float] = Field(default=None, description="Tier I Basel II")  # 154
    tier_2_basel_ii: Optional[float] = Field(default=None, description="Tier 2 Basel II")  # 155
    no_of_employes: Optional[float] = Field(default=None, description="No of Employes")  # 156
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 157




class FinancialResultCons(SQLModel, table=True):
    __tablename__ = "company_results_cons"

    fincode: int = Field(primary_key=True, description="Company Code")  # 1
    result_type: str = Field(primary_key=True, max_length=2, description="Result Type. Q-Quarterly, QR-Quarterly revised, H-Halfyearly, HR-Halfyearly revised, A-Annual, AR-Annual revised.")  # 2
    no_of_months: Optional[int] = Field(default=None, description="No of Months")  # 3
    date_end: int = Field(primary_key=True, description="Date End")  # 4

    net_sales: Optional[float] = Field(default=None, description="Total Revenue from Operations")  # 5
    gross_sale: Optional[float] = Field(default=None, description="Interest Income")  # 6
    excise_duty: Optional[float] = Field(default=None, description="GST")  # 7
    other_net_sales: Optional[float] = Field(default=None, description="Income from Operations_net")  # 8
    job_works: Optional[float] = Field(default=None, description="Other Operating Income")  # 9
    other_income: Optional[float] = Field(default=None, description="Other Income")  # 10
    dividend_income: Optional[float] = Field(default=None, description="Dividend Income")  # 11
    export_incentives: Optional[float] = Field(default=None, description="Export Incentives")  # 12
    foreign_exchange_gain: Optional[float] = Field(default=None, description="Foreign Exchange Gain")  # 13
    profit_investment: Optional[float] = Field(default=None, description="Profit on sale of Investments")  # 14
    other_interest_income: Optional[float] = Field(default=None, description="Interest Income")  # 15
    rental_income: Optional[float] = Field(default=None, description="Lease / Rental Income")  # 16
    sale_shares_units: Optional[float] = Field(default=None, description="Profit on sale of Share")  # 17
    prov_written_back: Optional[float] = Field(default=None, description="Provision Written Back")  # 18
    sale_assets: Optional[float] = Field(default=None, description="Profit on sale of Asset")  # 19
    other_other_income: Optional[float] = Field(default=None, description="Other Income")  # 20
    total_income: Optional[float] = Field(default=None, description="Total Income")  # 21
    expenditure: Optional[float] = Field(default=None, description="Total Expenditure")  # 22
    inc_dec_inventory: Optional[float] = Field(default=None, description="(Increase) / Decrease In Stocks")  # 23
    excise: Optional[float] = Field(default=None, description="Excise Duty")  # 24
    raw_material_cost: Optional[float] = Field(default=None, description="Raw Material Cost")  # 25
    purchase_fin_good: Optional[float] = Field(default=None, description="Purchase of Finished Goods")  # 26
    mfg_exps: Optional[float] = Field(default=None, description="Manufacturing Expenses")  # 27
    electricity_power_fuel: Optional[float] = Field(default=None, description="Electricity , Power & Fuel Cost")  # 28
    exployee_cost: Optional[float] = Field(default=None, description="Employees Cost")  # 29
    interest: Optional[float] = Field(default=None, description="Interest")  # 30
    depreciation: Optional[float] = Field(default=None, description="Depreciation")  # 31
    gen_admin_expenses: Optional[float] = Field(default=None, description="General Administration Expenses")  # 32
    selling_dist_expenses: Optional[float] = Field(default=None, description="Selling & Distribution Expenses")  # 33
    misc_expenses: Optional[float] = Field(default=None, description="Misc Expenses")  # 34
    loss_foreign_exchange: Optional[float] = Field(default=None, description="Loss_Foreign_exchange")  # 35
    loss_foreign_exchange_loan: Optional[float] = Field(default=None, description="Loss_Foreign_exchange_loan")  # 36
    expence_capitalised: Optional[float] = Field(default=None, description="Expenses Capitalised")  # 37
    pb_exitem_satax: Optional[float] = Field(default=None, description="Profit before Exceptional Items, Share of Associates and tax")  # 38
    exception_items: Optional[float] = Field(default=None, description="Exceptional Items")  # 39
    exc_foreign_exch_gain: Optional[float] = Field(default=None, description="Foreign Exchange Gain/Loss")  # 40
    exc_other: Optional[float] = Field(default=None, description="Other Exceptional items")  # 41
    pb_satax: Optional[float] = Field(default=None, description="Profit before Share of Associates and tax")  # 42
    shares_associate: Optional[float] = Field(default=None, description="Share of (loss)/profit in Associates and Joint Ventures")  # 43
    pbt: Optional[float] = Field(default=None, description="Profit before tax")  # 44
    tax: Optional[float] = Field(default=None, description="Tax")  # 45
    curr_tax: Optional[float] = Field(default=None, description="Current Tax")  # 46
    def_tax: Optional[float] = Field(default=None, description="Deferred Tax")  # 47
    fringe_benefits: Optional[float] = Field(default=None, description="Fringe Benefit Tax")  # 48
    prior_period_tax: Optional[float] = Field(default=None, description="Prior Period / Year Tax")  # 49
    mat_credit: Optional[float] = Field(default=None, description="MAT Credit")  # 50
    other_tax: Optional[float] = Field(default=None, description="Other Tax")  # 51
    pat: Optional[float] = Field(default=None, description="Profit after Tax")  # 52
    pbt_disc_operations: Optional[float] = Field(default=None, description="Profit before tax from discontinued operations")  # 53
    taxexp_discoperations: Optional[float] = Field(default=None, description="Tax expense of discontinued operations")  # 54
    discontinued_op: Optional[float] = Field(default=None, description="Profit for the period from discontinued operations")  # 55
    other_related_items: Optional[float] = Field(default=None, description="Other related Items")  # 56
    net_profit: Optional[float] = Field(default=None, description="Profit for the period")  # 57
    other_compincome: Optional[float] = Field(default=None, description="Other Comprehensive Incomes (Net of tax)")  # 58
    items_notclassif_pl: Optional[float] = Field(default=None, description="Items that will not be reclassified to profit or loss")  # 59
    it_items_notclassif_pl: Optional[float] = Field(default=None, description="Income tax relating to items that will not be reclassified to profit or loss")  # 60
    items_classif_pl: Optional[float] = Field(default=None, description="Items that will be reclassified to profit or loss")  # 61
    it_items_classif_pl: Optional[float] = Field(default=None, description="Income tax relating to Items that will be reclassified to profit or loss")  # 62
    minint_compincome: Optional[float] = Field(default=None, description="Minority Interest Comprehensive Income")  # 63
    othcinc_other: Optional[float] = Field(default=None, description="Other")  # 64
    total_compincome: Optional[float] = Field(default=None, description="Total Comprehensive Income")  # 65
    netprofit_ownofparent: Optional[float] = Field(default=None, description="Net Profit - Owners of the Parent")  # 66
    netprofit_noncontint: Optional[float] = Field(default=None, description="Net Profit - Non-controlling interests")  # 67
    other_compinc_ownofparent: Optional[float] = Field(default=None, description="Other Comprehensive Income - Owners of the Parent")  # 68
    other_compinc_noncontint: Optional[float] = Field(default=None, description="Other Comprehensive Income - Non-controlling interests")  # 69
    totalcompincome_ownofparent: Optional[float] = Field(default=None, description="Total Comprehensive Income - Owners of the Parent")  # 70
    minority_interest: Optional[float] = Field(default=None, description="Non-controlling interests")  # 71
    equity_cap: Optional[float] = Field(default=None, description="Equity Capital")  # 72
    fv: Optional[float] = Field(default=None, description="Face Value (In Rs)")  # 73
    reserves: Optional[float] = Field(default=None, description="Reserves")  # 74
    epsabs: Optional[float] = Field(default=None, description="Calculated EPS (Unit.Curr.)")  # 75
    epsann: Optional[float] = Field(default=None, description="Calculated EPS Annualised (Unit.Curr.)")  # 76
    adj_eps_abs: Optional[float] = Field(default=None, description="Adj Calculated EPS (Unit.Curr.)")  # 77
    adj_eps_ann: Optional[float] = Field(default=None, description="Adj Calculated EPS Annualised (Unit.Curr.)")  # 78
    eps_basic: Optional[float] = Field(default=None, description="Basic EPS")  # 79
    eps_diluted_extraord: Optional[float] = Field(default=None, description="Diluted EPS")  # 80
    pbidtmexoi: Optional[float] = Field(default=None, description="PBIDTM% (Excl OI)")  # 81
    pbidtm: Optional[float] = Field(default=None, description="PBIDTM%")  # 82
    pbdtm: Optional[float] = Field(default=None, description="PBDTM%")  # 83
    pbtm: Optional[float] = Field(default=None, description="PBTM%")  # 84
    patm: Optional[float] = Field(default=None, description="PATM%")  # 85
    rev_div_income: Optional[float] = Field(default=None, description="Dividend Income")  # 86
    rev_fee_comm: Optional[float] = Field(default=None, description="Fees and Commission Income")  # 87
    rev_gain_fairvalue: Optional[float] = Field(default=None, description="Gain on Fair value changes")  # 88
    rev_gain_finassets: Optional[float] = Field(default=None, description="Gain on de-recognised of Financial Instruments")  # 89
    inc_inv: Optional[float] = Field(default=None, description="Profit on sale of Investments")  # 90
    int_bal: Optional[float] = Field(default=None, description="Interest on Balances With RBI Other Inter Bank Funds")  # 91
    int_adv: Optional[float] = Field(default=None, description="Interest / Discount on Advances / Bills")  # 92
    int_others: Optional[float] = Field(default=None, description="Others")  # 93
    operating_ex: Optional[float] = Field(default=None, description="Operating Expenses")  # 94
    prov_emp: Optional[float] = Field(default=None, description="Payment To Provisions For Employees")  # 95
    other_exp: Optional[float] = Field(default=None, description="Other Operating Expenses")  # 96
    operating_profit: Optional[float] = Field(default=None, description="Operating Profit before Prov.& Cont.")  # 97
    pac: Optional[float] = Field(default=None, description="Provisions and Contingencies")  # 98
    ei: Optional[float] = Field(default=None, description="Extraordinary Items")  # 99
    ppi: Optional[float] = Field(default=None, description="Prior Period Items")  # 100
    govt_percent_of_shares: Optional[float] = Field(default=None, description="% of Shares held by Govt")  # 101
    cap_ratio_percent: Optional[float] = Field(default=None, description="Capital Adequacy Ratio Basel II")  # 102
    cap_ratio_percent3: Optional[float] = Field(default=None, description="Capital Adequacy Ratio Basel III")  # 103
    tier1basel3: Optional[float] = Field(default=None, description="Tier I Basel III")  # 104
    tier2basel3: Optional[float] = Field(default=None, description="Tier 2 Basel III")  # 105
    gross_net_npa: Optional[float] = Field(default=None, description="Gross / Net NPA")  # 106
    npa_gross: Optional[float] = Field(default=None, description="Amount of Gross NPA")  # 107
    npa_net: Optional[float] = Field(default=None, description="Amount of Net NPA")  # 108
    perc_gross_net_npa: Optional[float] = Field(default=None, description="Percentage of Gross/Net NPA")  # 109
    npa_net_perc: Optional[float] = Field(default=None, description="% of Net NPAs")  # 110
    npa_gross_perc: Optional[float] = Field(default=None, description="% of Gross NPAs")  # 111
    roa: Optional[float] = Field(default=None, description="Return on Assets")  # 112
    prom_no_of_shares: Optional[float] = Field(default=None, description="Number of Public Share Holding")  # 113
    prom_percent_of_shares: Optional[float] = Field(default=None, description="% of Public Share Holding")  # 114
    eps_basic_extraord: Optional[float] = Field(default=None, description="Basic EPS before Extraordinary Items")  # 115
    eps_diluted: Optional[float] = Field(default=None, description="Diluted EPS before Extraordinary Items")  # 116
    promoter_nos: Optional[float] = Field(default=None, description="Promoters No of Shares")  # 117
    encumbered_nos: Optional[float] = Field(default=None, description="Encumbered No of Shares")  # 118
    percentage_pledgedpromoter: Optional[float] = Field(default=None, description="Encumbered % of Promoter Holdings")  # 119
    percentage_pledgedcapital: Optional[float] = Field(default=None, description="Encumbered % of Share Capital")  # 120
    nonpledgedencum: Optional[float] = Field(default=None, description="Non Encumbered")  # 121
    nonpledged_nos: Optional[float] = Field(default=None, description="Non Encumbered No of Shares")  # 122
    percentage_nonpledgedpromoter: Optional[float] = Field(default=None, description="Non Encumbered % of Promoter Holdings")  # 123
    percentage_nonpledgedcapital: Optional[float] = Field(default=None, description="Non Encumbered % of Share Capital")  # 124
    casa: Optional[float] = Field(default=None, description="CASA%")  # 125
    casa_amount: Optional[float] = Field(default=None, description="CASA Amount")  # 126
    nim: Optional[float] = Field(default=None, description="NIM %")  # 127
    gross_profit: Optional[float] = Field(default=None, description="Gross Profit")  # 128
    pref_cap: Optional[float] = Field(default=None, description="Preference Capital")  # 129
    misc_expd_woff: Optional[float] = Field(default=None, description="Misc. Expenses Written off")  # 130
    consolidated_net_profit: Optional[float] = Field(default=None, description="Consolidated Net Profit")  # 131
    notes: Optional[str] = Field(sa_column=Column(MEDIUMTEXT))  # Changed to TEXT # 132
    return_on_capital_employed: Optional[float] = Field(default=None, description="Return on Capital Employed")  # 133 - Ignored
    debt_equity_ratio: Optional[float] = Field(default=None, description="Debt/Equity Ratio")  # 134
    interest_coverage_ratio: Optional[float] = Field(default=None, description="Interest Coverage Ratio")  # 135
    inventory_turnover_ratio: Optional[float] = Field(default=None, description="Inventory Turnover Ratio")  # 136
    debtor_turnover_ratio: Optional[float] = Field(default=None, description="Debtor Turnover Ratio")  # 137
    dividend_per_share: Optional[float] = Field(default=None, description="Dividend per share")  # 138
    dividend_payout_ratio: Optional[float] = Field(default=None, description="Dividend payout ratio")  # 139
    other_adjustments: Optional[float] = Field(default=None, description="Other Adjustments")  # 140
    pbidtxoi: Optional[float] = Field(default=None, description="PBIDT (Excl OI)")  # 141
    operating_profit_margin: Optional[float] = Field(default=None, description="Operating Profit Margin")  # 142 - Ignored
    net_profit_margin: Optional[float] = Field(default=None, description="Net Profit Margin")  # 143 - Ignored
    cash_bank: Optional[float] = Field(default=None, description="Cash and Cash Equivalents")  # 144
    debtors: Optional[float] = Field(default=None, description="Debtors")  # 145
    inventory: Optional[float] = Field(default=None, description="Inventory")  # 146
    loans_adv: Optional[float] = Field(default=None, description="Loans and Advances")  # 147
    ns_grand_total: Optional[float] = Field(default=None, description="Number of shares outstanding")  # 148
    ns_totalpublic: Optional[float] = Field(default=None, description="nsTotalpublic")  # 149
    tp_totalpublic: Optional[float] = Field(default=None, description="tpTotalpublic")  # 150
    provisions_coverage: Optional[float] = Field(default=None, description="Provisions Coverage %")  # 151
    no_of_atms: Optional[float] = Field(default=None, description="No of ATMs")  # 152
    no_of_branches: Optional[float] = Field(default=None, description="No of Branches")  # 153
    tier_i_basel_ii: Optional[float] = Field(default=None, description="Tier I Basel II")  # 154
    tier_2_basel_ii: Optional[float] = Field(default=None, description="Tier 2 Basel II")  # 155
    no_of_employes: Optional[float] = Field(default=None, description="No of Employes")  # 156
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 157

