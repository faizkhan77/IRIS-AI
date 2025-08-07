from sqlmodel import SQLModel, Field,Relationship
from typing import Optional,List
from sqlalchemy import Column, Text

class CompanyMaster(SQLModel, table=True):
    __tablename__ = "company_master"

    fincode: int = Field(primary_key=True, description="AFPL's Company Code")
    scripcode: Optional[int] = Field(description="BSE Scrip Code")
    scrip_name: Optional[str] = Field(max_length=35, description="BSE Script Name")
    scrip_group: Optional[str] = Field(max_length=3, description="BSE Scrip Group")
    compname: Optional[str] = Field(max_length=255, description="Company Long Name")
    ind_code: Optional[int] = Field(description="AFPL's sector / Industry Code")
    industry: Optional[str] = Field(max_length=100, description="Industry name")
    hse_code: Optional[int] = Field(description="AFPL's Business House Code")
    house: Optional[str] = Field(max_length=50, description="HOUSE")
    symbol: Optional[str] = Field(max_length=20, description="NSE Script Code")
    series: Optional[str] = Field(max_length=2, description="NSE Scrip Series")
    isin: Optional[str] = Field(max_length=20, description="ISIN Number")
    s_name: Optional[str] = Field(max_length=100, description="Company Short Name")
    rformat: Optional[str] = Field(max_length=5, description="Result Format")
    fformat: Optional[str] = Field(max_length=5, description="Finance Format")
    chairman: Optional[str] = Field(max_length=50, description="Chairman")
    mdir: Optional[str] = Field(max_length=100, description="Managing Director")
    cosec: Optional[str] = Field(max_length=100, description="Company Secretary")
    inc_month: Optional[str] = Field(max_length=15, description="Incorporation Month")
    inc_year: Optional[str] = Field(max_length=4, description="Incorporated Year")
    fv: Optional[float] = Field(description="Face Value")
    status: Optional[str] = Field(max_length=15, description="Company Status Details, Active / In-active etc")
    sublisting: Optional[str] = Field(max_length=50, description="Sublisting")
    bse_scrip_id: Optional[str] = Field(max_length=100, description="BSE ScripID")
    securitytoken: Optional[int] = Field(description="NSE Security Token")
    cin: Optional[str] = Field(max_length=255, description="CIN")
    bse_sublisting: Optional[str] = Field(max_length=50, description="BSE Sublisting")
    nse_sublisting: Optional[str] = Field(max_length=50, description="NSE Sublisting")
    flag: Optional[str] = Field(max_length=1, description="Updation Flag")

class IndustryMaster(SQLModel, table=True):
    __tablename__ = "industry_master"
    ind_code: int = Field(primary_key=True, description="AccordFintech’s Industry Code")
    industry: str = Field(nullable=True,max_length=100, description="Industry Name")
    ind_shortname: str = Field(max_length=100, description="Industry Short Name")
    sector: str = Field(max_length=100, description="Sector")
    sector_code: int = Field(description="Sector Code")
    flag: Optional[str] = Field(max_length=1, description="Updation Flag")



class HouseMaster(SQLModel, table=True):
    __tablename__ = "house_master"
    house_code: int = Field(primary_key=True, description="AccordFintech’s Business House Code")
    house: str = Field(max_length=50, description="House Name")
    flag: Optional[str] = Field(max_length=1, description="Updation Flag")

class StockExchangeMaster(SQLModel, table=True):
    __tablename__ = "stock_exchange_master"

    stk_id: int = Field(primary_key=True)
    stk_name: str = Field(max_length=100)
    flag: Optional[str] = Field(max_length=1)

    company_listings: List["CompanyListings"] = Relationship(back_populates="stock_exchange")


class CompanyListings(SQLModel, table=True):
    __tablename__ = "company_listings"

    fincode: int = Field(primary_key=True)
    stk_id: int = Field(primary_key=True, foreign_key="stock_exchange_master.stk_id")
    flag: Optional[str] = Field(max_length=1)

    stock_exchange: Optional[StockExchangeMaster] = Relationship(back_populates="company_listings")


class CompanyAddress(SQLModel, table=True):
    __tablename__ = "company_address"  # 🏢 Comment: office address details

    fincode: int = Field(primary_key=True, description="AccordFintech’s Company Code")
    add1: Optional[str] = Field(max_length=500, description="Company Registered Address 1")
    add2: Optional[str] = Field(max_length=500, description="Company Registered Address 2")
    add3: Optional[str] = Field(max_length=500, description="Company Registered Address 3")
    city_name: Optional[str] = Field(max_length=50, description="City")
    pincode: Optional[str] = Field(max_length=20, description="PIN code")
    state_name: Optional[str] = Field(max_length=50, description="State")
    phone: Optional[str] = Field(max_length=500, description="Telephone Numbers")
    fax_no: Optional[str] = Field(max_length=150, description="Fax Number")
    website: Optional[str] = Field(max_length=150, description="Website")
    e_mail: Optional[str] = Field(max_length=150, description="Email Address")
    flag: Optional[str] = Field(max_length=1, description="Updation Flag")


class CompanyProfile(SQLModel, table=True):
    __tablename__ = "company_profile"
 
    fincode: int = Field(primary_key=True, description="AFPL Company Code")
    details: Optional[str] = Field(default=None,sa_column=Column(Text), description="Company Description")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")