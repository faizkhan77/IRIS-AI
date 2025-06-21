from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import BigInteger,Column

class MonthlyPriceBSE(SQLModel, table=True):
    __tablename__ = "monthly_price_bse"

    fincode: int = Field(primary_key=True, description="AFPL’s Company Code")
    scripcode: Optional[int] = Field(default=None, description="BSE Scrip Code")
    month: int = Field(primary_key=True, description="Month in digit (1-12)")
    year: int = Field(primary_key=True, description="Year")

    open: Optional[float] = Field(default=None, description="Open price of the month")
    high: Optional[float] = Field(default=None, description="High price of the month")
    low: Optional[float] = Field(default=None, description="Low price of the month")
    close: Optional[float] = Field(default=None, description="Close price of the month")
    volume: Optional[int] = Field(
        default=None, 
        sa_column=Column(BigInteger),  # <-- set as BIGINT explicitly
        description="Total Volume for the month"
    )
    value: Optional[float] = Field(default=None, description="Total Value for the month")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")

class MonthlyPriceNSE(SQLModel, table=True):
    __tablename__ = "monthly_price_nse"

    fincode: int = Field(primary_key=True, description="AFPL’s Company Code")
    symbol: Optional[str] = Field(default=None, max_length=25, description="NSE Script Code")
    month: int = Field(primary_key=True, description="Month in digit (1-12)")
    year: int = Field(primary_key=True, description="Year")

    open: Optional[float] = Field(default=None, description="Open price of the month")
    high: Optional[float] = Field(default=None, description="High price of the month")
    low: Optional[float] = Field(default=None, description="Low price of the month")
    close: Optional[float] = Field(default=None, description="Close price of the month")
    volume: Optional[int] = Field(
        default=None, 
        sa_column=Column(BigInteger),  # <-- set as BIGINT explicitly
        description="Total Volume for the month"
    )
    value: Optional[float] = Field(default=None, description="Total Value for the month")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")
