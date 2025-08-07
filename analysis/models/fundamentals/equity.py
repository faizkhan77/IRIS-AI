from typing import Optional
from sqlalchemy import Column, BigInteger
from sqlmodel import SQLModel, Field
from datetime import datetime

class CompanyEquity(SQLModel, table=True):
    __tablename__ = "company_equity"

    fincode: int = Field(primary_key=True, description="Fincode")  # 1
    year_end: Optional[int] = Field(default=None, description="Year end")  # 2
    no_shs_subscribed: Optional[int] = Field(
        default=None,
        description="No of shares subscribed",
        sa_column=Column(BigInteger)  # <-- explicitly BIGINT
    )
    latest_equity: Optional[float] = Field(default=None, description="Latest Equity")  # 4
    latest_reserve: Optional[float] = Field(default=None, description="Latest Reserve")  # 5
    price: Optional[float] = Field(default=None, description="Price")  # 6
    mcap: Optional[float] = Field(default=None, description="Market Capitalization")  # 7
    dividend_yield: Optional[float] = Field(default=None, description="Dividend Yield")  # 8
    fv: Optional[float] = Field(default=None, description="Face Value")  # 9
    booknavpershare: Optional[float] = Field(default=None, description="Book Value per Share")  # 10
    ttm_yearend: Optional[int] = Field(default=None, description="TTM Year End")  # 11
    ttmeps: Optional[float] = Field(default=None, description="TTM EPS")  # 12
    ttmpe: Optional[float] = Field(default=None, description="TTM PE")  # 13
    price_sales: Optional[float] = Field(default=None, description="Price to Sales")  # 14
    ev_sales: Optional[float] = Field(default=None, description="EV to Sales")  # 15
    mcap_sales: Optional[float] = Field(default=None, description="Mcap to Sales")  # 16
    ev: Optional[float] = Field(default=None, description="Enterprise Value")  # 17
    price_bv: Optional[float] = Field(default=None, description="Price to Book Value")  # 18
    ev_ebitda: Optional[float] = Field(default=None, description="EV to EBITDA")  # 19
    ttmceps: Optional[float] = Field(default=None, description="TTM CEPS")  # 20
    price_ceps: Optional[float] = Field(default=None, description="Price to CEPS")  # 21
    price_date: Optional[datetime] = Field(default=None, description="Price Date")  # 22
    stk_exchange: Optional[str] = Field(default=None, max_length=25, description="Stock Exchange")  # 23
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 24



class CompanyEquityCons(SQLModel, table=True):
    __tablename__ = "company_equity_cons"

    fincode: int = Field(primary_key=True, description="Fincode")  # 1
    year_end: Optional[int] = Field(default=None, description="Year end")  # 2
    no_shs_subscribed: Optional[int] = Field(
        default=None,
        description="No of shares subscribed",
        sa_column=Column(BigInteger)  # <-- explicitly BIGINT
    ) # 3
    latest_equity: Optional[float] = Field(default=None, description="Latest Equity")  # 4
    latest_reserve: Optional[float] = Field(default=None, description="Latest Reserve")  # 5
    price: Optional[float] = Field(default=None, description="Price")  # 6
    mcap: Optional[float] = Field(default=None, description="Market Capitalization")  # 7
    dividend_yield: Optional[float] = Field(default=None, description="Dividend Yield")  # 8
    fv: Optional[float] = Field(default=None, description="Face Value")  # 9
    booknavpershare: Optional[float] = Field(default=None, description="Book Value per Share")  # 10
    ttm_yearend: Optional[int] = Field(default=None, description="TTM Year End")  # 11
    ttmeps: Optional[float] = Field(default=None, description="TTM EPS")  # 12
    ttmpe: Optional[float] = Field(default=None, description="TTM PE")  # 13
    price_sales: Optional[float] = Field(default=None, description="Price to Sales")  # 14
    ev_sales: Optional[float] = Field(default=None, description="EV to Sales")  # 15
    mcap_sales: Optional[float] = Field(default=None, description="Mcap to Sales")  # 16
    ev: Optional[float] = Field(default=None, description="Enterprise Value")  # 17
    price_bv: Optional[float] = Field(default=None, description="Price to Book Value")  # 18
    ev_ebitda: Optional[float] = Field(default=None, description="EV to EBITDA")  # 19
    ttmceps: Optional[float] = Field(default=None, description="TTM CEPS")  # 20
    price_ceps: Optional[float] = Field(default=None, description="Price to CEPS")  # 21
    price_date: Optional[datetime] = Field(default=None, description="Price Date")  # 22
    stk_exchange: Optional[str] = Field(default=None, max_length=25, description="Stock Exchange")  # 23
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 24


