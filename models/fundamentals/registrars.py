from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime



class RegistrarMaster(SQLModel, table=True):
    __tablename__ = "company_registrar_master"  # 🧾 Registrar master details

    registrar_no: int = Field(primary_key=True, description="Registrar Master Code")
    registrar_name: Optional[str] = Field(default=None, max_length=255, description="Registrar Name")
    reg_address: Optional[str] = Field(default=None, max_length=255, description="Address 1")
    registrar_address1: Optional[str] = Field(default=None, max_length=255, description="Address 2")
    registrar_address2: Optional[str] = Field(default=None, max_length=255, description="Address 3")
    registrar_address3: Optional[str] = Field(default=None, max_length=255, description="Address 4")
    registrar_phone: Optional[str] = Field(default=None, max_length=255, description="Telephone Number")
    registrar_fax: Optional[str] = Field(default=None, max_length=255, description="Fax Number")
    registrar_email: Optional[str] = Field(default=None, max_length=255, description="Email ID")
    registrar_website: Optional[str] = Field(default=None, max_length=255, description="Website URL")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")

    # Relationship
    company_links: List["CompanyRegistrar"] = Relationship(back_populates="registrar")



class CompanyRegistrar(SQLModel, table=True):
    __tablename__ = "company_registrar_data"  # 🔗 Links companies with registrars

    fincode: int = Field(primary_key=True, description="AccordFintech’s Company Code")
    registrar_no: int = Field(foreign_key="company_registrar_master.registrar_no", primary_key=True, description="Registrar Number (references registrar_master)")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")

    # Relationships
    registrar: Optional["RegistrarMaster"] = Relationship(back_populates="company_links")



class CompanyDirector(SQLModel, table=True):
    __tablename__ = "company_board_director"  # 👔 Company Board directors information

    fincode: int = Field(primary_key=True, foreign_key="company_master.fincode", description="AccordFintech’s Company Code")
    yrc: int = Field(primary_key=True, description="Financial Year End")
    serialno: int = Field(primary_key=True, description="Sequential Serial Number of the entry")
    dirtype_id: int = Field(primary_key=True, description="Designation ID")

    srno: Optional[int] = Field(default=None, description="Serial Number in which the data should be displayed")
    effect_date: Optional[datetime] = Field(default=None, description="Effective Date")
    dirname: Optional[str] = Field(default=None, max_length=255, description="Director’s Name")
    dirrem: Optional[float] = Field(default=None, description="Director's Remuneration")
    reported_dsg: Optional[str] = Field(default=None, max_length=255, description="Reported Designation")
    rem_unit: Optional[float] = Field(default=None, description="Remuneration Units")
    independent: Optional[str] = Field(default=None, max_length=100, description="Independent Director Indicator")
    director_id: Optional[int] = Field(default=None, description="Director Name Code")
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")
