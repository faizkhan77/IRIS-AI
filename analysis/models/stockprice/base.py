from sqlmodel import Relationship, SQLModel, Field
from typing import List, Optional
from sqlalchemy import Column, PrimaryKeyConstraint
from datetime import datetime


class IndicesMaster(SQLModel, table=True):
    __tablename__ = "indices_master"
    index_code: int = Field(primary_key=True, description="Index Code")
    exchange: Optional[str] = Field(default=None, max_length=3, description="Exchange")
    index_name: Optional[str] = Field(default=None, max_length=30, description="Index Short Name")
    index_lname: Optional[str] = Field(default=None, max_length=50, description="Index Long Name")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")

    company_index_parts: List["CompanyIndexPart"] = Relationship(back_populates="index")


class CompanyIndexPart(SQLModel, table=True):
    __tablename__ = "company_index_part"
    fincode: int = Field(description="AFPL Company Code")
    scripcode: Optional[int] = Field(default=None, description="BSE Script Code")
    symbol: Optional[str] = Field(default=None, max_length=30, description="NSE Script Code")
    index_code: Optional[int] = Field(default=None, foreign_key="indices_master.index_code", description="Index Code")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")
    index: Optional[IndicesMaster] = Relationship(back_populates="company_index_parts")
    __table_args__ = (
        PrimaryKeyConstraint("fincode", "index_code"),
    )



# 1.2.1 BSE Intraday Stock Price
class BSEPrice(SQLModel, table=True):
    __tablename__ = "bse_stock_price"
    scripcode: int = Field(primary_key=True, description="BSE Scrip Code")
    updtime: datetime = Field(primary_key=True, description="Price Date and Time")
    open: Optional[float] = Field(default=None, description="Open Price")
    close: Optional[float] = Field(default=None, description="Last Traded Price")
    high: Optional[float] = Field(default=None, description="High Price")
    low: Optional[float] = Field(default=None, description="Low Price")
    bidprice: Optional[float] = Field(default=None, description="Best Buy/Bid Price")
    offerprice: Optional[float] = Field(default=None, description="Best Offer Price")
    bidqty: Optional[float] = Field(default=None, description="Best Buy/Bid Quantity")
    offerqty: Optional[float] = Field(default=None, description="Offer Quantity")
    volume: Optional[float] = Field(default=None, description="Volume Traded")
    value: Optional[float] = Field(default=None, description="Value Traded")


# 1.3.1 BSE Intraday Indices
class BSEIntradayIndex(SQLModel, table=True):
    __tablename__ = "bse_index_price"
    scripcode: int = Field(primary_key=True, description="Index Scrip Code (refer Indices Master)")
    updtime: datetime = Field(primary_key=True, description="Price Date and Time")
    open: Optional[float] = Field(default=None, description="Open Price")
    high: Optional[float] = Field(default=None, description="High Price")
    low: Optional[float] = Field(default=None, description="Low Price")
    close: Optional[float] = Field(default=None, description="Close Price")
    volume: Optional[float] = Field(default=None, description="Volume Traded")
    value: Optional[float] = Field(default=None, description="Value Traded")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")


# 1.4.1 BSE Daily Indices (EOD) (hst)
class BSEIndicesEOD(SQLModel, table=True):
    __tablename__ = "bse_indices_price_eod"
    scripcode: int = Field(primary_key=True, description="Index Scrip Code (refer Indices Master)")
    date: datetime = Field(primary_key=True, description="Price Date")
    open: Optional[float] = Field(default=None, description="Open Price")
    high: Optional[float] = Field(default=None, description="High Price")
    low: Optional[float] = Field(default=None, description="Low Price")
    close: Optional[float] = Field(default=None, description="Close Price")
    volume: Optional[float] = Field(default=None, description="Volume Traded")
    value: Optional[float] = Field(default=None, description="Value Traded")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")


# 1.5.1 BSE Adjusted Price
class BSEAdjustedPrice(SQLModel, table=True):
    __tablename__ = "bse_abjusted_price_eod"
    fincode: int = Field(primary_key=True, description="AFPL Company Code")
    scripcode: Optional[int] = Field(default=None, description="BSE Scrip Code")
    date: datetime = Field(primary_key=True, description="Price Date")
    open: Optional[float] = Field(default=None, description="Adjusted Open Price")
    high: Optional[float] = Field(default=None, description="Adjusted High Price")
    low: Optional[float] = Field(default=None, description="Adjusted Low Price")
    close: Optional[float] = Field(default=None, description="Adjusted Close Price")
    volume: Optional[float] = Field(default=None, description="Volume Traded")
    value: Optional[float] = Field(default=None, description="Value Traded")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")

