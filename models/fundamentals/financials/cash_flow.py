from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime


class CompanyCashflow(SQLModel, table=True):
    __tablename__ = "company_finance_cashflow"

    fincode: int = Field(primary_key=True, description="Accord Company Code")  # 1
    year_end: int = Field(primary_key=True, description="Year End")  # 2
    type: str = Field(primary_key=True, max_length=1, description="To indicate type of the balancesheet. S = Standalone, C = Consolidated")  # 3
    no_months: Optional[int] = Field(default=None, description="No. of Months balancesheet")  # 4
    unit: Optional[int] = Field(default=None, description="To indicate figures are in which unit")  # 5
    profit_bef_tax: Optional[float] = Field(default=None, description="Profit_bef_tax")  # 6
    adjustment_total: Optional[float] = Field(default=None, description="Adjustment_total")  # 7
    change_wc_total: Optional[float] = Field(default=None, description="Change_WC_total")  # 8
    tax_paid: Optional[float] = Field(default=None, description="Tax_paid")  # 9
    other_dir_exps: Optional[float] = Field(default=None, description="Other_Dir_exps")  # 10
    cash_from_operation: Optional[float] = Field(default=None, description="Cash_from_Operation")  # 11
    cash_from_investment: Optional[float] = Field(default=None, description="Cash_from_Investment")  # 12
    cash_from_financing: Optional[float] = Field(default=None, description="Cash_from_Financing")  # 13
    f_exch_diff: Optional[float] = Field(default=None, description="FExchDiff")  # 14
    net_cash_inflow_outflow: Optional[float] = Field(default=None, description="Net_Cash_inflow_outflow")  # 15
    opening_cash: Optional[float] = Field(default=None, description="Opening_cash")  # 16
    cash_amalgamation: Optional[float] = Field(default=None, description="Cash_amalgamation")  # 17
    cash_subsidiaries: Optional[float] = Field(default=None, description="Cash_subsidiaries")  # 18
    traslation_adj_subsidiaries: Optional[float] = Field(default=None, description="Traslation_adj_Subsidiaries")  # 19
    effect_foreign_exchange: Optional[float] = Field(default=None, description="Effect_foreign_exchange")  # 20
    closing_cash: Optional[float] = Field(default=None, description="Closing_cash")  # 21
    cashflow_after_wc: Optional[float] = Field(default=None, description="Cashflow_after_WC")  # 22
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 23


class CompanyCashflowCons(SQLModel, table=True):
    __tablename__ = "company_finance_cashflow_cons"

    fincode: int = Field(primary_key=True, description="Accord Company Code")  # 1
    year_end: int = Field(primary_key=True, description="Year End")  # 2
    type: str = Field(primary_key=True, max_length=1, description="To indicate type of the balancesheet. S = Standalone, C = Consolidated")  # 3
    no_months: Optional[int] = Field(default=None, description="No. of Months balancesheet")  # 4
    unit: Optional[int] = Field(default=None, description="To indicate figures are in which unit")  # 5
    profit_bef_tax: Optional[float] = Field(default=None, description="Profit_bef_tax")  # 6
    adjustment_total: Optional[float] = Field(default=None, description="Adjustment_total")  # 7
    change_wc_total: Optional[float] = Field(default=None, description="Change_WC_total")  # 8
    tax_paid: Optional[float] = Field(default=None, description="Tax_paid")  # 9
    other_dir_exps: Optional[float] = Field(default=None, description="Other_Dir_exps")  # 10
    cash_from_operation: Optional[float] = Field(default=None, description="Cash_from_Operation")  # 11
    cash_from_investment: Optional[float] = Field(default=None, description="Cash_from_Investment")  # 12
    cash_from_financing: Optional[float] = Field(default=None, description="Cash_from_Financing")  # 13
    f_exch_diff: Optional[float] = Field(default=None, description="FExchDiff")  # 14
    net_cash_inflow_outflow: Optional[float] = Field(default=None, description="Net_Cash_inflow_outflow")  # 15
    opening_cash: Optional[float] = Field(default=None, description="Opening_cash")  # 16
    cash_amalgamation: Optional[float] = Field(default=None, description="Cash_amalgamation")  # 17
    cash_subsidiaries: Optional[float] = Field(default=None, description="Cash_subsidiaries")  # 18
    traslation_adj_subsidiaries: Optional[float] = Field(default=None, description="Traslation_adj_Subsidiaries")  # 19
    effect_foreign_exchange: Optional[float] = Field(default=None, description="Effect_foreign_exchange")  # 20
    closing_cash: Optional[float] = Field(default=None, description="Closing_cash")  # 21
    cashflow_after_wc: Optional[float] = Field(default=None, description="Cashflow_after_WC")  # 22
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 23