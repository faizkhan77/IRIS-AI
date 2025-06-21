from typing import Optional
from sqlmodel import SQLModel, Field

class CompanyBalanceSheet(SQLModel, table=True):
    __tablename__ = "company_finance_balancesheet"

    fincode: int = Field(primary_key=True, description="Accord Company Code")  # 1
    year_end: int = Field(primary_key=True, description="Year End")  # 2
    type: str = Field(primary_key=True, max_length=1, description="Balance Sheet Type: S=Standalone, C=Consolidated")  # 3

    no_months: Optional[int] = Field(default=None, description="Number of months the balance sheet covers")  # 4
    unit: Optional[int] = Field(default=None, description="Unit of figures")  # 5
    share_capital: Optional[float] = Field(default=None, description="Share Capital")  # 6
    shares_warrants: Optional[float] = Field(default=None, description="Share Warrants")  # 7
    shares_application_money: Optional[float] = Field(default=None, description="Share Application Money")  # 8
    esop_outstanding: Optional[float] = Field(default=None, description="ESOP Outstanding")  # 9
    share_suspense: Optional[float] = Field(default=None, description="Share Capital Suspense Account")  # 10
    reserve: Optional[float] = Field(default=None, description="Total Reserves")  # 11
    shareholders_funds: Optional[float] = Field(default=None, description="Shareholder's Funds")  # 12
    deposits: Optional[float] = Field(default=None, description="Deposits")  # 13
    borrowings: Optional[float] = Field(default=None, description="Borrowings")  # 14
    other_liabilities: Optional[float] = Field(default=None, description="Other Liabilities & Provisions")  # 15
    total_sources_funds: Optional[float] = Field(default=None, description="Total Liabilities")  # 16
    cash_balances_rbi: Optional[float] = Field(default=None, description="Cash and balances with RBI")  # 17
    bank_call_money: Optional[float] = Field(default=None, description="Balances with banks and money at call")  # 18
    investments: Optional[float] = Field(default=None, description="Investments")  # 19
    advances: Optional[float] = Field(default=None, description="Advances")  # 20
    gross_block: Optional[float] = Field(default=None, description="Gross Block")  # 21
    acc_depreciation: Optional[float] = Field(default=None, description="Accumulated Depreciation")  # 22
    impairment_assets: Optional[float] = Field(default=None, description="Impairment of Assets")  # 23
    lease_adj: Optional[float] = Field(default=None, description="Lease Adjustment A/c")  # 24
    cwip: Optional[float] = Field(default=None, description="Capital Work in Progress")  # 25
    other_assets: Optional[float] = Field(default=None, description="Other Assets")  # 26
    total_applications: Optional[float] = Field(default=None, description="Total Assets")  # 27
    contingent_liabilities: Optional[float] = Field(default=None, description="Contingent Liabilities")  # 28
    bills_for_collection: Optional[float] = Field(default=None, description="Bills for Collection")  # 29
    ltrm_borrowing: Optional[float] = Field(default=None, description="Long-Term Borrowings")  # 30
    secured_loans: Optional[float] = Field(default=None, description="Secured Loans")  # 31
    unsecured_loans_nf: Optional[float] = Field(default=None, description="Unsecured Loans")  # 32
    net_def_tax_nf: Optional[float] = Field(default=None, description="Deferred Tax Assets and Liabilities")  # 33
    oth_lt_liab: Optional[float] = Field(default=None, description="Other Long Term Liabilities")  # 34
    lt_trade_payables: Optional[float] = Field(default=None, description="Long Term Trade Payables")  # 35
    lt_provisions: Optional[float] = Field(default=None, description="Long Term Provisions")  # 36
    total_nonliab: Optional[float] = Field(default=None, description="Total Non-Current Liabilities")  # 37
    trade_payables: Optional[float] = Field(default=None, description="Trade Payables")  # 38
    current_liabilities_nf: Optional[float] = Field(default=None, description="Other Current Liabilities")  # 39
    st_borrowings: Optional[float] = Field(default=None, description="Short Term Borrowings")  # 40
    provisions: Optional[float] = Field(default=None, description="Short Term Provisions")  # 41
    current_laib_prov_nf: Optional[float] = Field(default=None, description="Total Current Liabilities")  # 42
    total_sounces_funds_nf: Optional[float] = Field(default=None, description="Total Liabilities")  # 43
    loan_non_cuurent_assets: Optional[float] = Field(default=None, description="Loan Non-Current Assets")  # 44
    net_block: Optional[float] = Field(default=None, description="Net Block / Fixed Asset")  # 45
    cwip_nf: Optional[float] = Field(default=None, description="CWIP Non-Financial")  # 46
    intang_assetunderdev: Optional[float] = Field(default=None, description="Intangible assets under development")  # 47
    pre_operatieve_exps: Optional[float] = Field(default=None, description="Pre-operative Expenses pending")  # 48
    assetsintransit: Optional[float] = Field(default=None, description="Assets in Transit")  # 49
    investments_nf: Optional[float] = Field(default=None, description="Non-Current Investments")  # 50
    lt_loanadv: Optional[float] = Field(default=None, description="Long Term Loans & Advances")  # 51
    other_noc_assets: Optional[float] = Field(default=None, description="Other Non-Current Assets")  # 52
    total_nonassets: Optional[float] = Field(default=None, description="Total Non-Current Assets")  # 53
    current_investments: Optional[float] = Field(default=None, description="Current Investments")  # 54
    inventory: Optional[float] = Field(default=None, description="Inventories")  # 55
    debtors: Optional[float] = Field(default=None, description="Sundry Debtors")  # 56
    cash_bank: Optional[float] = Field(default=None, description="Cash and Bank")  # 57
    other_current_nf: Optional[float] = Field(default=None, description="Other Current Assets")  # 58
    loans_adv: Optional[float] = Field(default=None, description="Short Term Loans and Advances")  # 59
    currant_assets_nf: Optional[float] = Field(default=None, description="Total Current Assets")  # 60
    net_current_assets_nf: Optional[float] = Field(default=None, description="Net Current Assets incl. Investments")  # 61
    other_crnt_assets: Optional[float] = Field(default=None, description="Current Assets excl. Investments")  # 62
    misc_exps_not_wo: Optional[float] = Field(default=None, description="Miscellaneous Expenses not written off")  # 63
    total_applications_nf: Optional[float] = Field(default=None, description="Total Assets (New)")  # 64
    total_debt_inclst_nf: Optional[float] = Field(default=None, description="Total Debt incl. ST")  # 65
    book_nav_share: Optional[float] = Field(default=None, description="Book Value per Share")  # 66
    adj_bv: Optional[float] = Field(default=None, description="Adjusted Book Value")  # 67
    minority_interest: Optional[float] = Field(default=None, description="Minority Interest")  # 68
    equity_authorised: Optional[float] = Field(default=None, description="Equity - Authorised")  # 69
    equity_issued: Optional[float] = Field(default=None, description="Equity - Issued")  # 70
    equity_paidup: Optional[float] = Field(default=None, description="Equity - Paid Up")  # 71
    equity_forfeited: Optional[float] = Field(default=None, description="Equity - Forfeited")  # 72
    adj_equity: Optional[float] = Field(default=None, description="Adjusted Equity")  # 73
    preference_capital: Optional[float] = Field(default=None, description="Preference Capital")  # 74
    face_value: Optional[float] = Field(default=None, description="Face Value")  # 75
    share_premium: Optional[float] = Field(default=None, description="Securities Premium")  # 76
    capital_reserve: Optional[float] = Field(default=None, description="Capital Reserves")  # 77
    profit_loss: Optional[float] = Field(default=None, description="Profit & Loss Account Balances")  # 78
    general_reserve: Optional[float] = Field(default=None, description="General Reserves")  # 79
    reserve_excl_revaluation: Optional[float] = Field(default=None, description="Reserve excluding Revaluation")  # 80
    revaluation_reserve: Optional[float] = Field(default=None, description="Revaluation Reserve")  # 81
    demand_deposits: Optional[float] = Field(default=None, description="Demand Deposits")  # 82
    saving_deposits: Optional[float] = Field(default=None, description="Savings Deposits")  # 83
    term_fixed_deposits: Optional[float] = Field(default=None, description="Term/Fixed Deposits")  # 84
    current_deposits: Optional[float] = Field(default=None, description="Current Deposits")  # 85
    recurring_deposits: Optional[float] = Field(default=None, description="Recurring Deposits")  # 86
    borrowings_rbi: Optional[float] = Field(default=None, description="Borrowings from RBI")  # 87
    borrowings_banks: Optional[float] = Field(default=None, description="Borrowings from Banks")  # 88
    borrowings_goi: Optional[float] = Field(default=None, description="Borrowings from GOI")  # 89
    borrowings_fi: Optional[float] = Field(default=None, description="Borrowings from Financial Institutions")  # 90
    borrowings_bonds: Optional[float] = Field(default=None, description="Borrowings in Bonds/Debentures")  # 91
    borrowings_other: Optional[float] = Field(default=None, description="Borrowings - Other")  # 92
    borrowings_outof_india: Optional[float] = Field(default=None, description="Borrowings from Outside India")  # 93
    bills_payable: Optional[float] = Field(default=None, description="Bills Payable")  # 94
    inter_oofice_adjustments_liabilities: Optional[float] = Field(default=None, description="Inter-office Adjustments (Liabilities)")  # 95
    interest_accrued: Optional[float] = Field(default=None, description="Interest Accrued")  # 96
    proposed_dividend: Optional[float] = Field(default=None, description="Proposed Dividend")  # 97
    corporate_dividend_tax: Optional[float] = Field(default=None, description="Corporate Dividend Tax Payable")  # 98
    cash_rbi: Optional[float] = Field(default=None, description="Cash with RBI")  # 99
    cash_in_hand: Optional[float] = Field(default=None, description="Cash in Hand & Others")  # 100
    investments_india: Optional[float] = Field(default=None, description="Investments in India")  # 101
    goi_securities: Optional[float] = Field(default=None, description="GOI / State Government Securities")  # 102
    equity_shares_corporate: Optional[float] = Field(default=None, description="Equity Shares - Corporate")  # 103
    debentures_bonds: Optional[float] = Field(default=None, description="Debentures & Bonds")  # 104
    subsidiaries_joint_ventures: Optional[float] = Field(default=None, description="Subsidiaries / Joint Ventures / Associates")  # 105
    mutual_funds_insurance_units: Optional[float] = Field(default=None, description="Units - MF / Insurance / CP / PTC")  # 106
    investments_outside_india: Optional[float] = Field(default=None, description="Investments Outside India")  # 107
    goi_securities_outside_india: Optional[float] = Field(default=None, description="GOI Securities Outside India")  # 108
    subsidiaries_joint_ventures_outside_india: Optional[float] = Field(default=None, description="Foreign Subsidiaries / Joint Ventures")  # 109
    other_investments_outside_india: Optional[float] = Field(default=None, description="Other Foreign Investments")  # 110
    dimunition_investments: Optional[float] = Field(default=None, description="Provision for Diminution in Investment Value")  # 111
    bills_purchased_discounted: Optional[float] = Field(default=None, description="Bills Purchased & Discounted")  # 112
    cash_credit_overdraft: Optional[float] = Field(default=None, description="Cash Credit, Overdraft, Loans Repayable")  # 113
    term_loans: Optional[float] = Field(default=None, description="Term Loans")  # 114
    finance_lease_hire_purchase: Optional[float] = Field(default=None, description="Finance Lease and Hire Purchase Receivables")  # 115
    buildings: Optional[float] = Field(default=None, description="Premises / Buildings")  # 116
    assets_despose: Optional[float] = Field(default=None, description="Assets Given on Lease")  # 117
    other_fix_assets: Optional[float] = Field(default=None, description="Other Fixed Assets")  # 118
    inter_office_adjustments_assets: Optional[float] = Field(default=None, description="Inter-office Adjustment - Assets")  # 119
    interest_accrued_assets: Optional[float] = Field(default=None, description="Interest Accrued - Assets")  # 120
    advance_tax_tds: Optional[float] = Field(default=None, description="Advance Tax / TDS")  # 121
    stationery_stamps: Optional[float] = Field(default=None, description="Stationery & Stamps")  # 122
    non_banking_assets: Optional[float] = Field(default=None, description="Non-Banking Assets Acquired")  # 123
    deferred_tax_assets: Optional[float] = Field(default=None, description="Deferred Tax Asset")  # 124
    misc_expenditures_not_writtoff: Optional[float] = Field(default=None, description="Miscellaneous Expenditures Not Written Off")  # 125
    claims_not_acknowledge_debt: Optional[float] = Field(default=None, description="Claims Not Acknowledged as Debts")  # 126
    outstanding_forward_exchange_contract: Optional[float] = Field(default=None, description="Outstanding Forward Exchange Contracts")  # 127
    guarantees_constituents_india: Optional[float] = Field(default=None, description="Guarantees Given in India")  # 128
    guarantees_constituents_outside_india: Optional[float] = Field(default=None, description="Guarantees Given Outside India")  # 129
    acceptances_endorsements_obligations: Optional[float] = Field(default=None, description="Acceptances, Endorsements, Obligations")  # 130
    non_convertible_debenture: Optional[float] = Field(default=None, description="Non-Convertible Debentures")  # 131
    conv_debenture_bnds: Optional[float] = Field(default=None, description="Convertible Debentures & Bonds")  # 132
    packing_credit: Optional[float] = Field(default=None, description="Packing Credit - Bank")  # 133
    intercorpsecdeposits: Optional[float] = Field(default=None, description="Inter-Corporate and Security Deposits")  # 134
    term_loan_bank: Optional[float] = Field(default=None, description="Term Loans from Banks")  # 135
    term_loan_inst: Optional[float] = Field(default=None, description="Term Loans from Institutions")  # 136
    fixed_deposits: Optional[float] = Field(default=None, description="Fixed Deposits - Public")  # 137
    loans_susidiaries: Optional[float] = Field(default=None, description="Loans and Advances from Subsidiaries")  # 138
    inter_corp_deposits: Optional[float] = Field(default=None, description="Inter Corporate Deposits (Unsecured)")  # 139
    foreign_curr_convertible_notes: Optional[float] = Field(default=None, description="Foreign Currency Convertible Notes")  # 140
    foreign_curr_long_term_loans: Optional[float] = Field(default=None, description="Long Term Foreign Currency Loans")  # 141
    loans_bank: Optional[float] = Field(default=None, description="Loans from Banks")  # 142
    loans_govt: Optional[float] = Field(default=None, description="Loans from Government")  # 143
    loans_others: Optional[float] = Field(default=None, description="Loans from Others")  # 144
    def_tax_assets: Optional[float] = Field(default=None, description="Deferred Tax Assets")  # 145
    def_tax_liability: Optional[float] = Field(default=None, description="Deferred Tax Liability")  # 146
    sundry_crs: Optional[float] = Field(default=None, description="Sundry Creditors")  # 147
    acceptances: Optional[float] = Field(default=None, description="Acceptances")  # 148
    trade_paysubs: Optional[float] = Field(default=None, description="Due to Subsidiaries - Trade Payables")  # 149
    bank_od: Optional[float] = Field(default=None, description="Bank Overdraft / Short-term Credit")  # 150
    adv_customers: Optional[float] = Field(default=None, description="Advances received from customers")  # 151
    interest_not_due: Optional[float] = Field(default=None, description="Interest Accrued but Not Due")  # 152
    share_application: Optional[float] = Field(default=None, description="Share Application Money")  # 153
    cl_curmat_deb: Optional[float] = Field(default=None, description="Current Maturity of Debentures & Bonds")  # 154
    cl_curmat_others: Optional[float] = Field(default=None, description="Current Maturity - Others")  # 155
    loans: Optional[float] = Field(default=None, description="Loans")  # 156
    stbs_sec_loandemand: Optional[float] = Field(default=None, description="Secured ST Loans Repayable on Demand")  # 157
    stbs_wcap: Optional[float] = Field(default=None, description="Working Capital Loans - Secured")  # 158
    stbu_buycredit: Optional[float] = Field(default=None, description="Buyers Credits - Unsecured")  # 159
    stbu_commborr: Optional[float] = Field(default=None, description="Commercial Borrowings - Unsecured")  # 160
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 161
    proposed_divided: Optional[float] = Field(default=None, description="Proposed Equity Dividend")  # 162
    prov_corp_tax: Optional[float] = Field(default=None, description="Provision for Corporate Dividend Tax")  # 163
    prov_tax: Optional[float] = Field(default=None, description="Provision for Tax")  # 164
    prov_retirements: Optional[float] = Field(default=None, description="Provision for Post Retirement Benefits")  # 165
    prefence_dividend: Optional[float] = Field(default=None, description="Preference Dividend")  # 166
    long_term_investments_nf: Optional[float] = Field(default=None, description="Long Term Investments")  # 167
    long_term_quoted: Optional[float] = Field(default=None, description="Long Term Quoted Investments")  # 168
    long_term_unquoted: Optional[float] = Field(default=None, description="Long Term Unquoted Investments")  # 169
    current_quoted: Optional[float] = Field(default=None, description="Current Quoted Investments")  # 170
    raw_material: Optional[float] = Field(default=None, description="Raw Materials")  # 171
    wip: Optional[float] = Field(default=None, description="Work in Progress")  # 172
    fin_goods: Optional[float] = Field(default=None, description="Finished Goods")  # 173
    packing_materials_inventory: Optional[float] = Field(default=None, description="Packing Materials Inventory")  # 174
    store_spares: Optional[float] = Field(default=None, description="Stores and Spares")  # 175
    drs_six_months: Optional[float] = Field(default=None, description="Debtors > 6 months")  # 176
    drs_others: Optional[float] = Field(default=None, description="Other Debtors")  # 177
    cash_hand: Optional[float] = Field(default=None, description="Cash in Hand")  # 178
    bank_balance: Optional[float] = Field(default=None, description="Bank Balances")  # 179
    interest_investments: Optional[float] = Field(default=None, description="Interest on Investments")  # 180
    interest_debentures: Optional[float] = Field(default=None, description="Interest on Debentures")  # 181
    deposits_govt_other: Optional[float] = Field(default=None, description="Deposits with Government / Others")  # 182
    interest_accrued_due: Optional[float] = Field(default=None, description="Interest Accrued and Due on Loans")  # 183
    prepaid_exps: Optional[float] = Field(default=None, description="Prepaid Expenses")  # 184
    adv_cash_kind: Optional[float] = Field(default=None, description="Advances Recoverable in Cash or Kind")  # 185
    adv_tax: Optional[float] = Field(default=None, description="Advance Income Tax and TDS")  # 186
    due_director: Optional[float] = Field(default=None, description="Amounts Due from Directors")  # 187
    due_subsidiaries: Optional[float] = Field(default=None, description="Amounts Due from Subsidiaries")  # 188
    inter_corporate_deposits: Optional[float] = Field(default=None, description="Inter Corporate Deposits")  # 189
    non_current_assets : Optional[float] = Field(default=None, description="Inter Corporate Deposits")  # 190
    corporate_deposits: Optional[float] = Field(default=None, description="Corporate Deposits")  # 191




class CompanyBalanceSheetCons(SQLModel, table=True):
    __tablename__ = "company_finance_balancesheet_cons"

    fincode: int = Field(primary_key=True, description="Accord Company Code")  # 1
    year_end: int = Field(primary_key=True, description="Year End")  # 2
    type: str = Field(primary_key=True, max_length=1, description="Balance Sheet Type: S=Standalone, C=Consolidated")  # 3

    no_months: Optional[int] = Field(default=None, description="Number of months the balance sheet covers")  # 4
    unit: Optional[int] = Field(default=None, description="Unit of figures")  # 5
    share_capital: Optional[float] = Field(default=None, description="Share Capital")  # 6
    shares_warrants: Optional[float] = Field(default=None, description="Share Warrants")  # 7
    shares_application_money: Optional[float] = Field(default=None, description="Share Application Money")  # 8
    esop_outstanding: Optional[float] = Field(default=None, description="ESOP Outstanding")  # 9
    share_suspense: Optional[float] = Field(default=None, description="Share Capital Suspense Account")  # 10
    reserve: Optional[float] = Field(default=None, description="Total Reserves")  # 11
    shareholders_funds: Optional[float] = Field(default=None, description="Shareholder's Funds")  # 12
    deposits: Optional[float] = Field(default=None, description="Deposits")  # 13
    borrowings: Optional[float] = Field(default=None, description="Borrowings")  # 14
    other_liabilities: Optional[float] = Field(default=None, description="Other Liabilities & Provisions")  # 15
    total_sources_funds: Optional[float] = Field(default=None, description="Total Liabilities")  # 16
    cash_balances_rbi: Optional[float] = Field(default=None, description="Cash and balances with RBI")  # 17
    bank_call_money: Optional[float] = Field(default=None, description="Balances with banks and money at call")  # 18
    investments: Optional[float] = Field(default=None, description="Investments")  # 19
    advances: Optional[float] = Field(default=None, description="Advances")  # 20
    gross_block: Optional[float] = Field(default=None, description="Gross Block")  # 21
    acc_depreciation: Optional[float] = Field(default=None, description="Accumulated Depreciation")  # 22
    impairment_assets: Optional[float] = Field(default=None, description="Impairment of Assets")  # 23
    lease_adj: Optional[float] = Field(default=None, description="Lease Adjustment A/c")  # 24
    cwip: Optional[float] = Field(default=None, description="Capital Work in Progress")  # 25
    other_assets: Optional[float] = Field(default=None, description="Other Assets")  # 26
    total_applications: Optional[float] = Field(default=None, description="Total Assets")  # 27
    contingent_liabilities: Optional[float] = Field(default=None, description="Contingent Liabilities")  # 28
    bills_for_collection: Optional[float] = Field(default=None, description="Bills for Collection")  # 29
    ltrm_borrowing: Optional[float] = Field(default=None, description="Long-Term Borrowings")  # 30
    secured_loans: Optional[float] = Field(default=None, description="Secured Loans")  # 31
    unsecured_loans_nf: Optional[float] = Field(default=None, description="Unsecured Loans")  # 32
    net_def_tax_nf: Optional[float] = Field(default=None, description="Deferred Tax Assets and Liabilities")  # 33
    oth_lt_liab: Optional[float] = Field(default=None, description="Other Long Term Liabilities")  # 34
    lt_trade_payables: Optional[float] = Field(default=None, description="Long Term Trade Payables")  # 35
    lt_provisions: Optional[float] = Field(default=None, description="Long Term Provisions")  # 36
    total_nonliab: Optional[float] = Field(default=None, description="Total Non-Current Liabilities")  # 37
    trade_payables: Optional[float] = Field(default=None, description="Trade Payables")  # 38
    current_liabilities_nf: Optional[float] = Field(default=None, description="Other Current Liabilities")  # 39
    st_borrowings: Optional[float] = Field(default=None, description="Short Term Borrowings")  # 40
    provisions: Optional[float] = Field(default=None, description="Short Term Provisions")  # 41
    current_laib_prov_nf: Optional[float] = Field(default=None, description="Total Current Liabilities")  # 42
    total_sounces_funds_nf: Optional[float] = Field(default=None, description="Total Liabilities")  # 43
    loan_non_cuurent_assets: Optional[float] = Field(default=None, description="Loan Non-Current Assets")  # 44
    net_block: Optional[float] = Field(default=None, description="Net Block / Fixed Asset")  # 45
    cwip_nf: Optional[float] = Field(default=None, description="CWIP Non-Financial")  # 46
    intang_assetunderdev: Optional[float] = Field(default=None, description="Intangible assets under development")  # 47
    pre_operatieve_exps: Optional[float] = Field(default=None, description="Pre-operative Expenses pending")  # 48
    assetsintransit: Optional[float] = Field(default=None, description="Assets in Transit")  # 49
    investments_nf: Optional[float] = Field(default=None, description="Non-Current Investments")  # 50
    lt_loanadv: Optional[float] = Field(default=None, description="Long Term Loans & Advances")  # 51
    other_noc_assets: Optional[float] = Field(default=None, description="Other Non-Current Assets")  # 52
    total_nonassets: Optional[float] = Field(default=None, description="Total Non-Current Assets")  # 53
    current_investments: Optional[float] = Field(default=None, description="Current Investments")  # 54
    inventory: Optional[float] = Field(default=None, description="Inventories")  # 55
    debtors: Optional[float] = Field(default=None, description="Sundry Debtors")  # 56
    cash_bank: Optional[float] = Field(default=None, description="Cash and Bank")  # 57
    other_current_nf: Optional[float] = Field(default=None, description="Other Current Assets")  # 58
    loans_adv: Optional[float] = Field(default=None, description="Short Term Loans and Advances")  # 59
    currant_assets_nf: Optional[float] = Field(default=None, description="Total Current Assets")  # 60
    net_current_assets_nf: Optional[float] = Field(default=None, description="Net Current Assets incl. Investments")  # 61
    other_crnt_assets: Optional[float] = Field(default=None, description="Current Assets excl. Investments")  # 62
    misc_exps_not_wo: Optional[float] = Field(default=None, description="Miscellaneous Expenses not written off")  # 63
    total_applications_nf: Optional[float] = Field(default=None, description="Total Assets (New)")  # 64
    total_debt_inclst_nf: Optional[float] = Field(default=None, description="Total Debt incl. ST")  # 65
    book_nav_share: Optional[float] = Field(default=None, description="Book Value per Share")  # 66
    adj_bv: Optional[float] = Field(default=None, description="Adjusted Book Value")  # 67
    minority_interest: Optional[float] = Field(default=None, description="Minority Interest")  # 68
    equity_authorised: Optional[float] = Field(default=None, description="Equity - Authorised")  # 69
    equity_issued: Optional[float] = Field(default=None, description="Equity - Issued")  # 70
    equity_paidup: Optional[float] = Field(default=None, description="Equity - Paid Up")  # 71
    equity_forfeited: Optional[float] = Field(default=None, description="Equity - Forfeited")  # 72
    adj_equity: Optional[float] = Field(default=None, description="Adjusted Equity")  # 73
    preference_capital: Optional[float] = Field(default=None, description="Preference Capital")  # 74
    face_value: Optional[float] = Field(default=None, description="Face Value")  # 75
    share_premium: Optional[float] = Field(default=None, description="Securities Premium")  # 76
    capital_reserve: Optional[float] = Field(default=None, description="Capital Reserves")  # 77
    profit_loss: Optional[float] = Field(default=None, description="Profit & Loss Account Balances")  # 78
    general_reserve: Optional[float] = Field(default=None, description="General Reserves")  # 79
    reserve_excl_revaluation: Optional[float] = Field(default=None, description="Reserve excluding Revaluation")  # 80
    revaluation_reserve: Optional[float] = Field(default=None, description="Revaluation Reserve")  # 81
    demand_deposits: Optional[float] = Field(default=None, description="Demand Deposits")  # 82
    saving_deposits: Optional[float] = Field(default=None, description="Savings Deposits")  # 83
    term_fixed_deposits: Optional[float] = Field(default=None, description="Term/Fixed Deposits")  # 84
    current_deposits: Optional[float] = Field(default=None, description="Current Deposits")  # 85
    recurring_deposits: Optional[float] = Field(default=None, description="Recurring Deposits")  # 86
    borrowings_rbi: Optional[float] = Field(default=None, description="Borrowings from RBI")  # 87
    borrowings_banks: Optional[float] = Field(default=None, description="Borrowings from Banks")  # 88
    borrowings_goi: Optional[float] = Field(default=None, description="Borrowings from GOI")  # 89
    borrowings_fi: Optional[float] = Field(default=None, description="Borrowings from Financial Institutions")  # 90
    borrowings_bonds: Optional[float] = Field(default=None, description="Borrowings in Bonds/Debentures")  # 91
    borrowings_other: Optional[float] = Field(default=None, description="Borrowings - Other")  # 92
    borrowings_outof_india: Optional[float] = Field(default=None, description="Borrowings from Outside India")  # 93
    bills_payable: Optional[float] = Field(default=None, description="Bills Payable")  # 94
    inter_oofice_adjustments_liabilities: Optional[float] = Field(default=None, description="Inter-office Adjustments (Liabilities)")  # 95
    interest_accrued: Optional[float] = Field(default=None, description="Interest Accrued")  # 96
    proposed_dividend: Optional[float] = Field(default=None, description="Proposed Dividend")  # 97
    corporate_dividend_tax: Optional[float] = Field(default=None, description="Corporate Dividend Tax Payable")  # 98
    cash_rbi: Optional[float] = Field(default=None, description="Cash with RBI")  # 99
    cash_in_hand: Optional[float] = Field(default=None, description="Cash in Hand & Others")  # 100
    investments_india: Optional[float] = Field(default=None, description="Investments in India")  # 101
    goi_securities: Optional[float] = Field(default=None, description="GOI / State Government Securities")  # 102
    equity_shares_corporate: Optional[float] = Field(default=None, description="Equity Shares - Corporate")  # 103
    debentures_bonds: Optional[float] = Field(default=None, description="Debentures & Bonds")  # 104
    subsidiaries_joint_ventures: Optional[float] = Field(default=None, description="Subsidiaries / Joint Ventures / Associates")  # 105
    mutual_funds_insurance_units: Optional[float] = Field(default=None, description="Units - MF / Insurance / CP / PTC")  # 106
    investments_outside_india: Optional[float] = Field(default=None, description="Investments Outside India")  # 107
    goi_securities_outside_india: Optional[float] = Field(default=None, description="GOI Securities Outside India")  # 108
    subsidiaries_joint_ventures_outside_india: Optional[float] = Field(default=None, description="Foreign Subsidiaries / Joint Ventures")  # 109
    other_investments_outside_india: Optional[float] = Field(default=None, description="Other Foreign Investments")  # 110
    dimunition_investments: Optional[float] = Field(default=None, description="Provision for Diminution in Investment Value")  # 111
    bills_purchased_discounted: Optional[float] = Field(default=None, description="Bills Purchased & Discounted")  # 112
    cash_credit_overdraft: Optional[float] = Field(default=None, description="Cash Credit, Overdraft, Loans Repayable")  # 113
    term_loans: Optional[float] = Field(default=None, description="Term Loans")  # 114
    finance_lease_hire_purchase: Optional[float] = Field(default=None, description="Finance Lease and Hire Purchase Receivables")  # 115
    buildings: Optional[float] = Field(default=None, description="Premises / Buildings")  # 116
    assets_despose: Optional[float] = Field(default=None, description="Assets Given on Lease")  # 117
    other_fix_assets: Optional[float] = Field(default=None, description="Other Fixed Assets")  # 118
    inter_office_adjustments_assets: Optional[float] = Field(default=None, description="Inter-office Adjustment - Assets")  # 119
    interest_accrued_assets: Optional[float] = Field(default=None, description="Interest Accrued - Assets")  # 120
    advance_tax_tds: Optional[float] = Field(default=None, description="Advance Tax / TDS")  # 121
    stationery_stamps: Optional[float] = Field(default=None, description="Stationery & Stamps")  # 122
    non_banking_assets: Optional[float] = Field(default=None, description="Non-Banking Assets Acquired")  # 123
    deferred_tax_assets: Optional[float] = Field(default=None, description="Deferred Tax Asset")  # 124
    misc_expenditures_not_writtoff: Optional[float] = Field(default=None, description="Miscellaneous Expenditures Not Written Off")  # 125
    claims_not_acknowledge_debt: Optional[float] = Field(default=None, description="Claims Not Acknowledged as Debts")  # 126
    outstanding_forward_exchange_contract: Optional[float] = Field(default=None, description="Outstanding Forward Exchange Contracts")  # 127
    guarantees_constituents_india: Optional[float] = Field(default=None, description="Guarantees Given in India")  # 128
    guarantees_constituents_outside_india: Optional[float] = Field(default=None, description="Guarantees Given Outside India")  # 129
    acceptances_endorsements_obligations: Optional[float] = Field(default=None, description="Acceptances, Endorsements, Obligations")  # 130
    non_convertible_debenture: Optional[float] = Field(default=None, description="Non-Convertible Debentures")  # 131
    conv_debenture_bnds: Optional[float] = Field(default=None, description="Convertible Debentures & Bonds")  # 132
    packing_credit: Optional[float] = Field(default=None, description="Packing Credit - Bank")  # 133
    intercorpsecdeposits: Optional[float] = Field(default=None, description="Inter-Corporate and Security Deposits")  # 134
    term_loan_bank: Optional[float] = Field(default=None, description="Term Loans from Banks")  # 135
    term_loan_inst: Optional[float] = Field(default=None, description="Term Loans from Institutions")  # 136
    fixed_deposits: Optional[float] = Field(default=None, description="Fixed Deposits - Public")  # 137
    loans_susidiaries: Optional[float] = Field(default=None, description="Loans and Advances from Subsidiaries")  # 138
    inter_corp_deposits: Optional[float] = Field(default=None, description="Inter Corporate Deposits (Unsecured)")  # 139
    foreign_curr_convertible_notes: Optional[float] = Field(default=None, description="Foreign Currency Convertible Notes")  # 140
    foreign_curr_long_term_loans: Optional[float] = Field(default=None, description="Long Term Foreign Currency Loans")  # 141
    loans_bank: Optional[float] = Field(default=None, description="Loans from Banks")  # 142
    loans_govt: Optional[float] = Field(default=None, description="Loans from Government")  # 143
    loans_others: Optional[float] = Field(default=None, description="Loans from Others")  # 144
    def_tax_assets: Optional[float] = Field(default=None, description="Deferred Tax Assets")  # 145
    def_tax_liability: Optional[float] = Field(default=None, description="Deferred Tax Liability")  # 146
    sundry_crs: Optional[float] = Field(default=None, description="Sundry Creditors")  # 147
    acceptances: Optional[float] = Field(default=None, description="Acceptances")  # 148
    trade_paysubs: Optional[float] = Field(default=None, description="Due to Subsidiaries - Trade Payables")  # 149
    bank_od: Optional[float] = Field(default=None, description="Bank Overdraft / Short-term Credit")  # 150
    adv_customers: Optional[float] = Field(default=None, description="Advances received from customers")  # 151
    interest_not_due: Optional[float] = Field(default=None, description="Interest Accrued but Not Due")  # 152
    share_application: Optional[float] = Field(default=None, description="Share Application Money")  # 153
    cl_curmat_deb: Optional[float] = Field(default=None, description="Current Maturity of Debentures & Bonds")  # 154
    cl_curmat_others: Optional[float] = Field(default=None, description="Current Maturity - Others")  # 155
    loans: Optional[float] = Field(default=None, description="Loans")  # 156
    stbs_sec_loandemand: Optional[float] = Field(default=None, description="Secured ST Loans Repayable on Demand")  # 157
    stbs_wcap: Optional[float] = Field(default=None, description="Working Capital Loans - Secured")  # 158
    stbu_buycredit: Optional[float] = Field(default=None, description="Buyers Credits - Unsecured")  # 159
    stbu_commborr: Optional[float] = Field(default=None, description="Commercial Borrowings - Unsecured")  # 160
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 161
    proposed_divided: Optional[float] = Field(default=None, description="Proposed Equity Dividend")  # 162
    prov_corp_tax: Optional[float] = Field(default=None, description="Provision for Corporate Dividend Tax")  # 163
    prov_tax: Optional[float] = Field(default=None, description="Provision for Tax")  # 164
    prov_retirements: Optional[float] = Field(default=None, description="Provision for Post Retirement Benefits")  # 165
    prefence_dividend: Optional[float] = Field(default=None, description="Preference Dividend")  # 166
    long_term_investments_nf: Optional[float] = Field(default=None, description="Long Term Investments")  # 167
    long_term_quoted: Optional[float] = Field(default=None, description="Long Term Quoted Investments")  # 168
    long_term_unquoted: Optional[float] = Field(default=None, description="Long Term Unquoted Investments")  # 169
    current_quoted: Optional[float] = Field(default=None, description="Current Quoted Investments")  # 170
    raw_material: Optional[float] = Field(default=None, description="Raw Materials")  # 171
    wip: Optional[float] = Field(default=None, description="Work in Progress")  # 172
    fin_goods: Optional[float] = Field(default=None, description="Finished Goods")  # 173
    packing_materials_inventory: Optional[float] = Field(default=None, description="Packing Materials Inventory")  # 174
    store_spares: Optional[float] = Field(default=None, description="Stores and Spares")  # 175
    drs_six_months: Optional[float] = Field(default=None, description="Debtors > 6 months")  # 176
    drs_others: Optional[float] = Field(default=None, description="Other Debtors")  # 177
    cash_hand: Optional[float] = Field(default=None, description="Cash in Hand")  # 178
    bank_balance: Optional[float] = Field(default=None, description="Bank Balances")  # 179
    interest_investments: Optional[float] = Field(default=None, description="Interest on Investments")  # 180
    interest_debentures: Optional[float] = Field(default=None, description="Interest on Debentures")  # 181
    deposits_govt_other: Optional[float] = Field(default=None, description="Deposits with Government / Others")  # 182
    interest_accrued_due: Optional[float] = Field(default=None, description="Interest Accrued and Due on Loans")  # 183
    prepaid_exps: Optional[float] = Field(default=None, description="Prepaid Expenses")  # 184
    adv_cash_kind: Optional[float] = Field(default=None, description="Advances Recoverable in Cash or Kind")  # 185
    adv_tax: Optional[float] = Field(default=None, description="Advance Income Tax and TDS")  # 186
    due_director: Optional[float] = Field(default=None, description="Amounts Due from Directors")  # 187
    due_subsidiaries: Optional[float] = Field(default=None, description="Amounts Due from Subsidiaries")  # 188
    inter_corporate_deposits: Optional[float] = Field(default=None, description="Inter Corporate Deposits")  # 189
    non_current_assets : Optional[float] = Field(default=None, description="Inter Corporate Deposits")  # 190
    corporate_deposits: Optional[float] = Field(default=None, description="Corporate Deposits")  # 191
    



