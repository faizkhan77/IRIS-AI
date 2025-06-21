from typing import Optional
from sqlmodel import SQLModel, Field


class CompanyProfitLoss(SQLModel, table=True):
    __tablename__="company_finance_profitloss"

    fincode: int = Field(primary_key=True, description="Accord Company Code")  # 1
    year_end: int = Field(primary_key=True, description="Year End")  # 2
    type: str = Field(primary_key=True, max_length=1, description="Type of Balance Sheet: S = Standalone, C = Consolidated")  # 3
    no_months: Optional[int] = Field(default=None, description="No. of Months in Balance Sheet")  # 4
    unit: Optional[int] = Field(default=None, description="Unit of Figures")  # 5
    interest_earned: Optional[float] = Field(default=None, description="Interest Earned")  # 6
    other_income: Optional[float] = Field(default=None, description="Other Income")  # 7
    total_income: Optional[float] = Field(default=None, description="Total Income")  # 8
    interest: Optional[float] = Field(default=None, description="Interest")  # 9
    operating_expenses: Optional[float] = Field(default=None, description="Operating Expenses")  # 10
    provisions_contigency: Optional[float] = Field(default=None, description="Provisions and Contingencies")  # 11
    tax: Optional[float] = Field(default=None, description="Income Tax")  # 12
    total: Optional[float] = Field(default=None, description="Total Expenditure for Banks")  # 13
    profit_after_tax: Optional[float] = Field(default=None, description="Profit After Tax")  # 14
    extra_items: Optional[float] = Field(default=None, description="Extra Items")  # 15
    profit_brought_forward: Optional[float] = Field(default=None, description="Profit Balance B/F")  # 16
    minority_interest: Optional[float] = Field(default=None, description="Minority Interest")  # 17
    share_associate: Optional[float] = Field(default=None, description="Share of Associate")  # 18
    other_consitems: Optional[float] = Field(default=None, description="Other Consolidated Items")  # 19
    consolidated_netprofit: Optional[float] = Field(default=None, description="Consolidated Net Profit")  # 20
    adj_net_profit: Optional[float] = Field(default=None, description="Adjustments to PAT")  # 21
    extr_items: Optional[float] = Field(default=None, description="Extra Items")  # 22
    pnlbf: Optional[float] = Field(default=None, description="Profit brought forward")  # 23
    profit_availble_appr: Optional[float] = Field(default=None, description="Appropriations Available")  # 24
    appropriation: Optional[float] = Field(default=None, description="Appropriations")  # 25
    dividend_perc: Optional[float] = Field(default=None, description="Equity Dividend %")  # 26
    reported_eps: Optional[float] = Field(default=None, description="Reported EPS")  # 27
    gross_sales: Optional[float] = Field(default=None, description="Gross Sales")  # 28
    net_sales: Optional[float] = Field(default=None, description="Net Sales")  # 29
    inc_dec_inventory: Optional[float] = Field(default=None, description="Increase/Decrease in Stock")  # 30
    raw_matrs_consumed: Optional[float] = Field(default=None, description="Raw Material Consumed")  # 31
    employees: Optional[float] = Field(default=None, description="Employee Cost")  # 32
    other_mfg_exps: Optional[float] = Field(default=None, description="Other Manufacturing Expenses")  # 33
    sellg_admn_exps: Optional[float] = Field(default=None, description="Selling, Administration and Other Expenses")  # 34
    misc_exps: Optional[float] = Field(default=None, description="Miscellaneous Expenses")  # 35
    pre_op_exps: Optional[float] = Field(default=None, description="Less: Expenses Capitalised")  # 36
    total_expendiure: Optional[float] = Field(default=None, description="Total Expenditure")  # 37
    operating_profit: Optional[float] = Field(default=None, description="Operating Profit")  # 38
    depreciation: Optional[float] = Field(default=None, description="Depreciation")  # 39
    profit_before_exception: Optional[float] = Field(default=None, description="Profit Before Taxation & Exceptional Items")  # 40
    exceptionalincome_expenses: Optional[float] = Field(default=None, description="Exceptional Income / Expenses")  # 41
    profit_before_tax: Optional[float] = Field(default=None, description="Profit Before Tax")  # 42
    taxation: Optional[float] = Field(default=None, description="Provision for Tax")  # 43
    profit_available_appropriation: Optional[float] = Field(default=None, description="Profit Available for appropriations")  # 44
    total_gross_sales: Optional[float] = Field(default=None, description="Sales Turnover")  # 45
    excise: Optional[float] = Field(default=None, description="Less: Excise")  # 46
    power_fuel: Optional[float] = Field(default=None, description="Power & Fuel Cost")  # 47
    gross_profits: Optional[float] = Field(default=None, description="PBDT")  # 48
    inter_div_trf: Optional[float] = Field(default=None, description="Less: Inter divisional transfers")  # 49
    sales_return: Optional[float] = Field(default=None, description="Less: Sales Returns")  # 50
    sellg_exps: Optional[float] = Field(default=None, description="Selling and Distribution Expenses")  # 51
    cost_sw: Optional[float] = Field(default=None, description="Cost of Software development")  # 52
    provision_investment: Optional[float] = Field(default=None, description="Provision for investments")  # 53
    provision_advances: Optional[float] = Field(default=None, description="Provision for advances")  # 54
    provision_other: Optional[float] = Field(default=None, description="Others")  # 55
    curr_tax: Optional[float] = Field(default=None, description="Current Income Tax")  # 56
    def_tax: Optional[float] = Field(default=None, description="Deferred Tax")  # 57
    fringe_benefits: Optional[float] = Field(default=None, description="Fringe Benefit tax")  # 58
    wealth_tax: Optional[float] = Field(default=None, description="Wealth Tax")  # 59
    interest_advances_bills: Optional[float] = Field(default=None, description="Interest/Discount on advances/Bills")  # 60
    job_works: Optional[float] = Field(default=None, description="Job works")  # 61
    service_income: Optional[float] = Field(default=None, description="Service income")  # 62
    opening_rm: Optional[float] = Field(default=None, description="Opening Raw Materials")  # 63
    purchases_rm: Optional[float] = Field(default=None, description="Purchases Raw Materials")  # 64
    closing_rm: Optional[float] = Field(default=None, description="Closing Raw Materials")  # 65
    purchases_fg: Optional[float] = Field(default=None, description="Other Direct Purchases / Brought in cost")  # 66
    electricity: Optional[float] = Field(default=None, description="Electricity & Power")  # 67
    oils_fuels: Optional[float] = Field(default=None, description="Aircraft Fuel")  # 68
    coals: Optional[float] = Field(default=None, description="Coals etc")  # 69
    salaries: Optional[float] = Field(default=None, description="Salaries, Wages & Bonus")  # 70
    providend_fund_contri: Optional[float] = Field(default=None, description="Contributions to EPF & Pension Funds")  # 71
    staff_welfare: Optional[float] = Field(default=None, description="Workmen and Staff Welfare Expenses")  # 72
    sub_contract: Optional[float] = Field(default=None, description="sub contract")  # 73
    processing_charges: Optional[float] = Field(default=None, description="Processing Charges")  # 74
    repairs_maintenance: Optional[float] = Field(default=None, description="Repairs and Maintenance")  # 75
    upkeep_maintenance: Optional[float] = Field(default=None, description="UpKeep maintenance")  # 76
    rent_rates_taxes: Optional[float] = Field(default=None, description="Rent , Rates & Taxes")  # 77
    insurance: Optional[float] = Field(default=None, description="Insurance")  # 78
    priting_stationery: Optional[float] = Field(default=None, description="Printing and stationery")  # 79
    professional_charges: Optional[float] = Field(default=None, description="Professional and legal fees")  # 80
    travelling: Optional[float] = Field(default=None, description="Traveling and conveyance")  # 81
    advertising: Optional[float] = Field(default=None, description="Advertisement & Sales Promotion")  # 82
    commission_incentives: Optional[float] = Field(default=None, description="Sales Commissions & Incentives")  # 83
    freight_forwardings: Optional[float] = Field(default=None, description="Freight and Forwarding")  # 84
    handling_clearing: Optional[float] = Field(default=None, description="Handling and Clearing Charges")  # 85
    bad_debts: Optional[float] = Field(default=None, description="Bad debts /advances written off")  # 86
    prov_doubtfull_debts: Optional[float] = Field(default=None, description="Provision for doubtful debts")  # 87
    loss_fixed_assets: Optional[float] = Field(default=None, description="Loss on disposal of fixed assets(net)")  # 88
    loss_foreign_exchange: Optional[float] = Field(default=None, description="Loss on foreign exchange fluctuations")  # 89
    loss_sale_investment: Optional[float] = Field(default=None, description="Loss on sale of non-trade current investments")  # 90
    interest_income: Optional[float] = Field(default=None, description="Interest Received")  # 91
    dividend_income: Optional[float] = Field(default=None, description="Dividend Received")  # 92
    profit_fa: Optional[float] = Field(default=None, description="Profit on sale of Fixed Assets")  # 93
    profit_investment: Optional[float] = Field(default=None, description="Profits on sale of Investments")  # 94
    prov_written_back: Optional[float] = Field(default=None, description="Provision Written Back")  # 95
    foreign_exchange_gain: Optional[float] = Field(default=None, description="Foreign Exchange Gains")  # 96
    interest_deb: Optional[float] = Field(default=None, description="Interest on Debenture / Bonds")  # 97
    interest_term_loans: Optional[float] = Field(default=None, description="Interest on Term Loan")  # 98
    interest_fixed_deposits: Optional[float] = Field(default=None, description="Interest on Fixed deposits")  # 99
    bank_charges: Optional[float] = Field(default=None, description="Bank Charges etc")  # 100
    appropriation_general_reserve: Optional[float] = Field(default=None, description="General Reserves")  # 101
    proposed_equity_devided: Optional[float] = Field(default=None, description="Proposed Equity Dividend")  # 102
    corp_divd_tax: Optional[float] = Field(default=None, description="Corporate dividend tax")  # 103
    eps: Optional[float] = Field(default=None, description="Earnings Per Share")  # 104
    adj_eps: Optional[float] = Field(default=None, description="Adjusted EPS")  # 105
    interest_rbi: Optional[float] = Field(default=None, description="Interest on balances with RBI and other Interbank funds")  # 106
    interest_investment: Optional[float] = Field(default=None, description="Income on investments")  # 107
    income_jv_subs: Optional[float] = Field(default=None, description="Income earned from subsidiaries/joint venture")  # 108
    rent_income: Optional[float] = Field(default=None, description="Rent / Lease Income")  # 109
    interest_rbi_borrowings: Optional[float] = Field(default=None, description="Interest on RBI / inter-bank borrowings")  # 110
    interest_other: Optional[float] = Field(default=None, description="Other Interest")  # 111
    depreciation_leased_assets: Optional[float] = Field(default=None, description="Depreciation on leased assets")  # 112
    auditor_payment: Optional[float] = Field(default=None, description="Auditor's fees and expenses")  # 113
    telephone: Optional[float] = Field(default=None, description="Communication Expenses")  # 114
    repairs_other_admin: Optional[float] = Field(default=None, description="Repairs and Maintenance")  # 115
    statutory_reserve: Optional[float] = Field(default=None, description="Transfer to Statutory Reserve")  # 116
    appropriation_revenue_reserve: Optional[float] = Field(default=None, description="Appropriation to Revenue Reserve")  # 117
    other_appropriation: Optional[float] = Field(default=None, description="Appropriation to Other Reserves")  # 118
    sale_shares_units: Optional[float] = Field(default=None, description="Sale of Shares / Units")  # 119
    interest_earned_loan: Optional[float] = Field(default=None, description="Interest income")  # 120
    portfolio_mgt_income: Optional[float] = Field(default=None, description="Portfolio management services")  # 121
    dividend_earned: Optional[float] = Field(default=None, description="Dividend income")  # 122
    brokerage_commission: Optional[float] = Field(default=None, description="Brokerages & commissions")  # 123
    processing_fees: Optional[float] = Field(default=None, description="Processing fees and other charges")  # 124
    depository_charges: Optional[float] = Field(default=None, description="Depository Charges")  # 125
    security_transaction_tax: Optional[float] = Field(default=None, description="Security Transaction tax")  # 126
    software_technical_charges: Optional[float] = Field(default=None, description="Software & Technical expenses")  # 127
    provision_contigency: Optional[float] = Field(default=None, description="Provisions for contingencies")  # 128
    provision_npa: Optional[float] = Field(default=None, description="Provisions against NPAs")  # 129
    other_interest_income: Optional[float] = Field(default=None, description="Other Interest Income")  # 130
    commission: Optional[float] = Field(default=None, description="Other Commission")  # 131
    discounts: Optional[float] = Field(default=None, description="Discounts")  # 132
    other_investment_income: Optional[float] = Field(default=None, description="Income from investments")  # 133
    sales: Optional[float] = Field(default=None, description="Income from Medical Services")  # 134
    income_diagnostic: Optional[float] = Field(default=None, description="Income from Diagnostic centre")  # 135
    cash_discount: Optional[float] = Field(default=None, description="Less: Concession / Free Treatment")  # 136
    upkeep_service: Optional[float] = Field(default=None, description="House Keeping Expenses")  # 137
    consultant_changes: Optional[float] = Field(default=None, description="Consultant / Inhouse Fees")  # 138
    packing_materials: Optional[float] = Field(default=None, description="Packing Material Consumed")  # 139
    freight_outward: Optional[float] = Field(default=None, description="Freight outwards")  # 140
    room_restaurants: Optional[float] = Field(default=None, description="Rooms / Restaurant / Banquets")  # 141
    communication_income: Optional[float] = Field(default=None, description="Communication Services")  # 142
    foods_beverage_sales: Optional[float] = Field(default=None, description="Food & Beverages")  # 143
    linen_room_supplies: Optional[float] = Field(default=None, description="Linen & Room Supplies")  # 144
    catering_supplies: Optional[float] = Field(default=None, description="Catering Supplies")  # 145
    laundry_washing_expenses: Optional[float] = Field(default=None, description="Laundry & Washing Expenses")  # 146
    music_banquet_restaurants: Optional[float] = Field(default=None, description="Music, Banquets and Restaurants")  # 147
    packing_expenses: Optional[float] = Field(default=None, description="Packing expenses")  # 148
    sales_property_development: Optional[float] = Field(default=None, description="Revenue from property development")  # 149
    broadcasting_revenue: Optional[float] = Field(default=None, description="Broadcasting Revenue")  # 150
    advertisement_revenue: Optional[float] = Field(default=None, description="Advertising Revenue")  # 151
    licence_income: Optional[float] = Field(default=None, description="License income")  # 152
    subscription_income: Optional[float] = Field(default=None, description="Subscription income")  # 153
    contents_film_income: Optional[float] = Field(default=None, description="Income from content / Event Shows/ Films")  # 154
    program_production_exps: Optional[float] = Field(default=None, description="Program Production Expenses")  # 155
    telecasting_expenses: Optional[float] = Field(default=None, description="Telecasting Expenses")  # 156
    programs_films_right: Optional[float] = Field(default=None, description="Programs and Films rights")  # 157
    transmission_epc: Optional[float] = Field(default=None, description="Transmission EPC Business")  # 158
    wheeling_transmission: Optional[float] = Field(default=None, description="Wheeling & Transmission Charges recoverable")  # 159
    power_purchased: Optional[float] = Field(default=None, description="Cost of power purchased")  # 160
    power_project_cost: Optional[float] = Field(default=None, description="Power Project Expenses")  # 161
    wheeling_charges: Optional[float] = Field(default=None, description="Wheeling & Transmission Charges Payable")  # 162
    spare_consumed: Optional[float] = Field(default=None, description="Cost of Elastimold, Store & Spares Consumed")  # 163
    sub_contract_charges: Optional[float] = Field(default=None, description="Sub Contract Charges")  # 164
    development_rights: Optional[float] = Field(default=None, description="Sale of Development Rights")  # 165
    development_charges: Optional[float] = Field(default=None, description="Development Charges")  # 166
    income_investment_property: Optional[float] = Field(default=None, description="Income From Investment in Properties")  # 167
    development_rights_cost: Optional[float] = Field(default=None, description="Development Rights Cost")  # 168
    shipbuilding_income: Optional[float] = Field(default=None, description="Income from ship building & Repairs")  # 169
    charter_income: Optional[float] = Field(default=None, description="Charter Income")  # 170
    freight_income: Optional[float] = Field(default=None, description="Freight and Demurrage")  # 171
    stevedoreage_cargo_expenses: Optional[float] = Field(default=None, description="Stevedoring, Despatch and Cargo expenses")  # 172
    port_charges: Optional[float] = Field(default=None, description="Port, Light and canal Dues")  # 173
    sale_licenses: Optional[float] = Field(default=None, description="Sale of Equipments & licenses")  # 174
    traded_sw: Optional[float] = Field(default=None, description="Software Purchase")  # 175
    tech_fees: Optional[float] = Field(default=None, description="Technical sub-contractors")  # 176
    traing_exps: Optional[float] = Field(default=None, description="Training Expenses")  # 177
    software_licences: Optional[float] = Field(default=None, description="Software License cost")  # 178
    travels_sw: Optional[float] = Field(default=None, description="Travel Expenses")  # 179
    insurance_sw: Optional[float] = Field(default=None, description="Overseas Group Health Insurance")  # 180
    visa_charges: Optional[float] = Field(default=None, description="Visa & Other Charges")  # 181
    contract_support_sw: Optional[float] = Field(default=None, description="Post contract support services")  # 182
    rates_taxes: Optional[float] = Field(default=None, description="Rates & Taxes")  # 183
    sales_scrap: Optional[float] = Field(default=None, description="Excess Baggage & Cancellation Charges")  # 184
    export_benefits: Optional[float] = Field(default=None, description="Export Benefits")  # 185
    subsidy_incentives: Optional[float] = Field(default=None, description="Subsidy / Grants / Incentives")  # 186
    freight_inward: Optional[float] = Field(default=None, description="Landing, Parking and Navigation charges")  # 187
    hire_charges_mfg: Optional[float] = Field(default=None, description="Aircrafts / Engines Lease & Hire Charges")  # 188
    donation: Optional[float] = Field(default=None, description="Donations")  # 189
    interest_other_income: Optional[float] = Field(default=None, description="Others")  # 190
    commission: Optional[float] = Field(default=None, description="Commission, exchange and brokerage")  # 191
    power_fuel_cost: Optional[float] = Field(default=None, description="Cost of Fuel")  # 192
    royalty: Optional[float] = Field(default=None, description="License, Royalty and Spectrum Charges")  # 193
    project_expenses: Optional[float] = Field(default=None, description="Internet / Bandwidth and Port Charges")  # 194
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 195



class CompanyProfitLossCons(SQLModel, table=True):
    __tablename__='company_finance_profitloss_cons'

    fincode: int = Field(primary_key=True, description="Accord Company Code")  # 1
    year_end: int = Field(primary_key=True, description="Year End")  # 2
    type: str = Field(primary_key=True, max_length=1, description="Type of Balance Sheet: S = Standalone, C = Consolidated")  # 3
    no_months: Optional[int] = Field(default=None, description="No. of Months in Balance Sheet")  # 4
    unit: Optional[int] = Field(default=None, description="Unit of Figures")  # 5
    interest_earned: Optional[float] = Field(default=None, description="Interest Earned")  # 6
    other_income: Optional[float] = Field(default=None, description="Other Income")  # 7
    total_income: Optional[float] = Field(default=None, description="Total Income")  # 8
    interest: Optional[float] = Field(default=None, description="Interest")  # 9
    operating_expenses: Optional[float] = Field(default=None, description="Operating Expenses")  # 10
    provisions_contigency: Optional[float] = Field(default=None, description="Provisions and Contingencies")  # 11
    tax: Optional[float] = Field(default=None, description="Income Tax")  # 12
    total: Optional[float] = Field(default=None, description="Total Expenditure for Banks")  # 13
    profit_after_tax: Optional[float] = Field(default=None, description="Profit After Tax")  # 14
    extra_items: Optional[float] = Field(default=None, description="Extra Items")  # 15
    profit_brought_forward: Optional[float] = Field(default=None, description="Profit Balance B/F")  # 16
    minority_interest: Optional[float] = Field(default=None, description="Minority Interest")  # 17
    share_associate: Optional[float] = Field(default=None, description="Share of Associate")  # 18
    other_consitems: Optional[float] = Field(default=None, description="Other Consolidated Items")  # 19
    consolidated_netprofit: Optional[float] = Field(default=None, description="Consolidated Net Profit")  # 20
    adj_net_profit: Optional[float] = Field(default=None, description="Adjustments to PAT")  # 21
    extr_items: Optional[float] = Field(default=None, description="Extra Items")  # 22
    pnlbf: Optional[float] = Field(default=None, description="Profit brought forward")  # 23
    profit_availble_appr: Optional[float] = Field(default=None, description="Appropriations Available")  # 24
    appropriation: Optional[float] = Field(default=None, description="Appropriations")  # 25
    dividend_perc: Optional[float] = Field(default=None, description="Equity Dividend %")  # 26
    reported_eps: Optional[float] = Field(default=None, description="Reported EPS")  # 27
    gross_sales: Optional[float] = Field(default=None, description="Gross Sales")  # 28
    net_sales: Optional[float] = Field(default=None, description="Net Sales")  # 29
    inc_dec_inventory: Optional[float] = Field(default=None, description="Increase/Decrease in Stock")  # 30
    raw_matrs_consumed: Optional[float] = Field(default=None, description="Raw Material Consumed")  # 31
    employees: Optional[float] = Field(default=None, description="Employee Cost")  # 32
    other_mfg_exps: Optional[float] = Field(default=None, description="Other Manufacturing Expenses")  # 33
    sellg_admn_exps: Optional[float] = Field(default=None, description="Selling, Administration and Other Expenses")  # 34
    misc_exps: Optional[float] = Field(default=None, description="Miscellaneous Expenses")  # 35
    pre_op_exps: Optional[float] = Field(default=None, description="Less: Expenses Capitalised")  # 36
    total_expendiure: Optional[float] = Field(default=None, description="Total Expenditure")  # 37
    operating_profit: Optional[float] = Field(default=None, description="Operating Profit")  # 38
    depreciation: Optional[float] = Field(default=None, description="Depreciation")  # 39
    profit_before_exception: Optional[float] = Field(default=None, description="Profit Before Taxation & Exceptional Items")  # 40
    exceptionalincome_expenses: Optional[float] = Field(default=None, description="Exceptional Income / Expenses")  # 41
    profit_before_tax: Optional[float] = Field(default=None, description="Profit Before Tax")  # 42
    taxation: Optional[float] = Field(default=None, description="Provision for Tax")  # 43
    profit_available_appropriation: Optional[float] = Field(default=None, description="Profit Available for appropriations")  # 44
    total_gross_sales: Optional[float] = Field(default=None, description="Sales Turnover")  # 45
    excise: Optional[float] = Field(default=None, description="Less: Excise")  # 46
    power_fuel: Optional[float] = Field(default=None, description="Power & Fuel Cost")  # 47
    gross_profits: Optional[float] = Field(default=None, description="PBDT")  # 48
    inter_div_trf: Optional[float] = Field(default=None, description="Less: Inter divisional transfers")  # 49
    sales_return: Optional[float] = Field(default=None, description="Less: Sales Returns")  # 50
    sellg_exps: Optional[float] = Field(default=None, description="Selling and Distribution Expenses")  # 51
    cost_sw: Optional[float] = Field(default=None, description="Cost of Software development")  # 52
    provision_investment: Optional[float] = Field(default=None, description="Provision for investments")  # 53
    provision_advances: Optional[float] = Field(default=None, description="Provision for advances")  # 54
    provision_other: Optional[float] = Field(default=None, description="Others")  # 55
    curr_tax: Optional[float] = Field(default=None, description="Current Income Tax")  # 56
    def_tax: Optional[float] = Field(default=None, description="Deferred Tax")  # 57
    fringe_benefits: Optional[float] = Field(default=None, description="Fringe Benefit tax")  # 58
    wealth_tax: Optional[float] = Field(default=None, description="Wealth Tax")  # 59
    interest_advances_bills: Optional[float] = Field(default=None, description="Interest/Discount on advances/Bills")  # 60
    job_works: Optional[float] = Field(default=None, description="Job works")  # 61
    service_income: Optional[float] = Field(default=None, description="Service income")  # 62
    opening_rm: Optional[float] = Field(default=None, description="Opening Raw Materials")  # 63
    purchases_rm: Optional[float] = Field(default=None, description="Purchases Raw Materials")  # 64
    closing_rm: Optional[float] = Field(default=None, description="Closing Raw Materials")  # 65
    purchases_fg: Optional[float] = Field(default=None, description="Other Direct Purchases / Brought in cost")  # 66
    electricity: Optional[float] = Field(default=None, description="Electricity & Power")  # 67
    oils_fuels: Optional[float] = Field(default=None, description="Aircraft Fuel")  # 68
    coals: Optional[float] = Field(default=None, description="Coals etc")  # 69
    salaries: Optional[float] = Field(default=None, description="Salaries, Wages & Bonus")  # 70
    providend_fund_contri: Optional[float] = Field(default=None, description="Contributions to EPF & Pension Funds")  # 71
    staff_welfare: Optional[float] = Field(default=None, description="Workmen and Staff Welfare Expenses")  # 72
    sub_contract: Optional[float] = Field(default=None, description="sub contract")  # 73
    processing_charges: Optional[float] = Field(default=None, description="Processing Charges")  # 74
    repairs_maintenance: Optional[float] = Field(default=None, description="Repairs and Maintenance")  # 75
    upkeep_maintenance: Optional[float] = Field(default=None, description="UpKeep maintenance")  # 76
    rent_rates_taxes: Optional[float] = Field(default=None, description="Rent , Rates & Taxes")  # 77
    insurance: Optional[float] = Field(default=None, description="Insurance")  # 78
    priting_stationery: Optional[float] = Field(default=None, description="Printing and stationery")  # 79
    professional_charges: Optional[float] = Field(default=None, description="Professional and legal fees")  # 80
    travelling: Optional[float] = Field(default=None, description="Traveling and conveyance")  # 81
    advertising: Optional[float] = Field(default=None, description="Advertisement & Sales Promotion")  # 82
    commission_incentives: Optional[float] = Field(default=None, description="Sales Commissions & Incentives")  # 83
    freight_forwardings: Optional[float] = Field(default=None, description="Freight and Forwarding")  # 84
    handling_clearing: Optional[float] = Field(default=None, description="Handling and Clearing Charges")  # 85
    bad_debts: Optional[float] = Field(default=None, description="Bad debts /advances written off")  # 86
    prov_doubtfull_debts: Optional[float] = Field(default=None, description="Provision for doubtful debts")  # 87
    loss_fixed_assets: Optional[float] = Field(default=None, description="Loss on disposal of fixed assets(net)")  # 88
    loss_foreign_exchange: Optional[float] = Field(default=None, description="Loss on foreign exchange fluctuations")  # 89
    loss_sale_investment: Optional[float] = Field(default=None, description="Loss on sale of non-trade current investments")  # 90
    interest_income: Optional[float] = Field(default=None, description="Interest Received")  # 91
    dividend_income: Optional[float] = Field(default=None, description="Dividend Received")  # 92
    profit_fa: Optional[float] = Field(default=None, description="Profit on sale of Fixed Assets")  # 93
    profit_investment: Optional[float] = Field(default=None, description="Profits on sale of Investments")  # 94
    prov_written_back: Optional[float] = Field(default=None, description="Provision Written Back")  # 95
    foreign_exchange_gain: Optional[float] = Field(default=None, description="Foreign Exchange Gains")  # 96
    interest_deb: Optional[float] = Field(default=None, description="Interest on Debenture / Bonds")  # 97
    interest_term_loans: Optional[float] = Field(default=None, description="Interest on Term Loan")  # 98
    interest_fixed_deposits: Optional[float] = Field(default=None, description="Interest on Fixed deposits")  # 99
    bank_charges: Optional[float] = Field(default=None, description="Bank Charges etc")  # 100
    appropriation_general_reserve: Optional[float] = Field(default=None, description="General Reserves")  # 101
    proposed_equity_devided: Optional[float] = Field(default=None, description="Proposed Equity Dividend")  # 102
    corp_divd_tax: Optional[float] = Field(default=None, description="Corporate dividend tax")  # 103
    eps: Optional[float] = Field(default=None, description="Earnings Per Share")  # 104
    adj_eps: Optional[float] = Field(default=None, description="Adjusted EPS")  # 105
    interest_rbi: Optional[float] = Field(default=None, description="Interest on balances with RBI and other Interbank funds")  # 106
    interest_investment: Optional[float] = Field(default=None, description="Income on investments")  # 107
    income_jv_subs: Optional[float] = Field(default=None, description="Income earned from subsidiaries/joint venture")  # 108
    rent_income: Optional[float] = Field(default=None, description="Rent / Lease Income")  # 109
    interest_rbi_borrowings: Optional[float] = Field(default=None, description="Interest on RBI / inter-bank borrowings")  # 110
    interest_other: Optional[float] = Field(default=None, description="Other Interest")  # 111
    depreciation_leased_assets: Optional[float] = Field(default=None, description="Depreciation on leased assets")  # 112
    auditor_payment: Optional[float] = Field(default=None, description="Auditor's fees and expenses")  # 113
    telephone: Optional[float] = Field(default=None, description="Communication Expenses")  # 114
    repairs_other_admin: Optional[float] = Field(default=None, description="Repairs and Maintenance")  # 115
    statutory_reserve: Optional[float] = Field(default=None, description="Transfer to Statutory Reserve")  # 116
    appropriation_revenue_reserve: Optional[float] = Field(default=None, description="Appropriation to Revenue Reserve")  # 117
    other_appropriation: Optional[float] = Field(default=None, description="Appropriation to Other Reserves")  # 118
    sale_shares_units: Optional[float] = Field(default=None, description="Sale of Shares / Units")  # 119
    interest_earned_loan: Optional[float] = Field(default=None, description="Interest income")  # 120
    portfolio_mgt_income: Optional[float] = Field(default=None, description="Portfolio management services")  # 121
    dividend_earned: Optional[float] = Field(default=None, description="Dividend income")  # 122
    brokerage_commission: Optional[float] = Field(default=None, description="Brokerages & commissions")  # 123
    processing_fees: Optional[float] = Field(default=None, description="Processing fees and other charges")  # 124
    depository_charges: Optional[float] = Field(default=None, description="Depository Charges")  # 125
    security_transaction_tax: Optional[float] = Field(default=None, description="Security Transaction tax")  # 126
    software_technical_charges: Optional[float] = Field(default=None, description="Software & Technical expenses")  # 127
    provision_contigency: Optional[float] = Field(default=None, description="Provisions for contingencies")  # 128
    provision_npa: Optional[float] = Field(default=None, description="Provisions against NPAs")  # 129
    other_interest_income: Optional[float] = Field(default=None, description="Other Interest Income")  # 130
    commission: Optional[float] = Field(default=None, description="Other Commission")  # 131
    discounts: Optional[float] = Field(default=None, description="Discounts")  # 132
    other_investment_income: Optional[float] = Field(default=None, description="Income from investments")  # 133
    sales: Optional[float] = Field(default=None, description="Income from Medical Services")  # 134
    income_diagnostic: Optional[float] = Field(default=None, description="Income from Diagnostic centre")  # 135
    cash_discount: Optional[float] = Field(default=None, description="Less: Concession / Free Treatment")  # 136
    upkeep_service: Optional[float] = Field(default=None, description="House Keeping Expenses")  # 137
    consultant_changes: Optional[float] = Field(default=None, description="Consultant / Inhouse Fees")  # 138
    packing_materials: Optional[float] = Field(default=None, description="Packing Material Consumed")  # 139
    freight_outward: Optional[float] = Field(default=None, description="Freight outwards")  # 140
    room_restaurants: Optional[float] = Field(default=None, description="Rooms / Restaurant / Banquets")  # 141
    communication_income: Optional[float] = Field(default=None, description="Communication Services")  # 142
    foods_beverage_sales: Optional[float] = Field(default=None, description="Food & Beverages")  # 143
    linen_room_supplies: Optional[float] = Field(default=None, description="Linen & Room Supplies")  # 144
    catering_supplies: Optional[float] = Field(default=None, description="Catering Supplies")  # 145
    laundry_washing_expenses: Optional[float] = Field(default=None, description="Laundry & Washing Expenses")  # 146
    music_banquet_restaurants: Optional[float] = Field(default=None, description="Music, Banquets and Restaurants")  # 147
    packing_expenses: Optional[float] = Field(default=None, description="Packing expenses")  # 148
    sales_property_development: Optional[float] = Field(default=None, description="Revenue from property development")  # 149
    broadcasting_revenue: Optional[float] = Field(default=None, description="Broadcasting Revenue")  # 150
    advertisement_revenue: Optional[float] = Field(default=None, description="Advertising Revenue")  # 151
    licence_income: Optional[float] = Field(default=None, description="License income")  # 152
    subscription_income: Optional[float] = Field(default=None, description="Subscription income")  # 153
    contents_film_income: Optional[float] = Field(default=None, description="Income from content / Event Shows/ Films")  # 154
    program_production_exps: Optional[float] = Field(default=None, description="Program Production Expenses")  # 155
    telecasting_expenses: Optional[float] = Field(default=None, description="Telecasting Expenses")  # 156
    programs_films_right: Optional[float] = Field(default=None, description="Programs and Films rights")  # 157
    transmission_epc: Optional[float] = Field(default=None, description="Transmission EPC Business")  # 158
    wheeling_transmission: Optional[float] = Field(default=None, description="Wheeling & Transmission Charges recoverable")  # 159
    power_purchased: Optional[float] = Field(default=None, description="Cost of power purchased")  # 160
    power_project_cost: Optional[float] = Field(default=None, description="Power Project Expenses")  # 161
    wheeling_charges: Optional[float] = Field(default=None, description="Wheeling & Transmission Charges Payable")  # 162
    spare_consumed: Optional[float] = Field(default=None, description="Cost of Elastimold, Store & Spares Consumed")  # 163
    sub_contract_charges: Optional[float] = Field(default=None, description="Sub Contract Charges")  # 164
    development_rights: Optional[float] = Field(default=None, description="Sale of Development Rights")  # 165
    development_charges: Optional[float] = Field(default=None, description="Development Charges")  # 166
    income_investment_property: Optional[float] = Field(default=None, description="Income From Investment in Properties")  # 167
    development_rights_cost: Optional[float] = Field(default=None, description="Development Rights Cost")  # 168
    shipbuilding_income: Optional[float] = Field(default=None, description="Income from ship building & Repairs")  # 169
    charter_income: Optional[float] = Field(default=None, description="Charter Income")  # 170
    freight_income: Optional[float] = Field(default=None, description="Freight and Demurrage")  # 171
    stevedoreage_cargo_expenses: Optional[float] = Field(default=None, description="Stevedoring, Despatch and Cargo expenses")  # 172
    port_charges: Optional[float] = Field(default=None, description="Port, Light and canal Dues")  # 173
    sale_licenses: Optional[float] = Field(default=None, description="Sale of Equipments & licenses")  # 174
    traded_sw: Optional[float] = Field(default=None, description="Software Purchase")  # 175
    tech_fees: Optional[float] = Field(default=None, description="Technical sub-contractors")  # 176
    traing_exps: Optional[float] = Field(default=None, description="Training Expenses")  # 177
    software_licences: Optional[float] = Field(default=None, description="Software License cost")  # 178
    travels_sw: Optional[float] = Field(default=None, description="Travel Expenses")  # 179
    insurance_sw: Optional[float] = Field(default=None, description="Overseas Group Health Insurance")  # 180
    visa_charges: Optional[float] = Field(default=None, description="Visa & Other Charges")  # 181
    contract_support_sw: Optional[float] = Field(default=None, description="Post contract support services")  # 182
    rates_taxes: Optional[float] = Field(default=None, description="Rates & Taxes")  # 183
    sales_scrap: Optional[float] = Field(default=None, description="Excess Baggage & Cancellation Charges")  # 184
    export_benefits: Optional[float] = Field(default=None, description="Export Benefits")  # 185
    subsidy_incentives: Optional[float] = Field(default=None, description="Subsidy / Grants / Incentives")  # 186
    freight_inward: Optional[float] = Field(default=None, description="Landing, Parking and Navigation charges")  # 187
    hire_charges_mfg: Optional[float] = Field(default=None, description="Aircrafts / Engines Lease & Hire Charges")  # 188
    donation: Optional[float] = Field(default=None, description="Donations")  # 189
    interest_other_income: Optional[float] = Field(default=None, description="Others")  # 190
    commission: Optional[float] = Field(default=None, description="Commission, exchange and brokerage")  # 191
    power_fuel_cost: Optional[float] = Field(default=None, description="Cost of Fuel")  # 192
    royalty: Optional[float] = Field(default=None, description="License, Royalty and Spectrum Charges")  # 193
    project_expenses: Optional[float] = Field(default=None, description="Internet / Bandwidth and Port Charges")  # 194
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 195
