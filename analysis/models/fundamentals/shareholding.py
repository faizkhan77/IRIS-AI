from typing import Optional
from sqlmodel import SQLModel, Field

class ShareholdingCategoryMaster(SQLModel, table=True):
    __tablename__ = "shareholding_category_master"

    shp_catid: int = Field(primary_key=True, description="Shareholders Category ID")  # 1
    shp_catname: Optional[str] = Field(default=None, max_length=255, description="Shareholders Category Name")  # 2
    sub_category: Optional[str] = Field(default=None, max_length=255, description="Shareholders Sub Category Name")  # 3
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 4



class CompanyShareholding(SQLModel, table=True):
    __tablename__ = "company_shareholders_details"

    fincode: int = Field(primary_key=True, description="Company Code")  # 1
    date_end: int = Field(primary_key=True, description="Date End (Year/Quarter/Month End)")  # 2
    srno: int = Field(primary_key=True, description="Serial Number")  # 3
    shp_catid: Optional[int] = Field(default=None, description="Shareholders Category ID")  # 4
    name: Optional[str] = Field(default=None, max_length=255, description="Name")  # 5
    percentage: Optional[float] = Field(default=None, description="Shareholding Percentage")  # 6
    no_of_shares: Optional[float] = Field(default=None, description="Number of Shares")  # 7
    pledge_encumbered_no_of_shares: Optional[float] = Field(default=None, description="Pledged / Encumbered Number of Shares")  # 8
    pledge_encumbered_percentage: Optional[float] = Field(default=None, description="Pledged / Encumbered Percentage")  # 9
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 10


class ShpSummary(SQLModel, table=True):
    __tablename__ = "company_shareholding_pattern"

    # id: Optional[int] = Field(default=None, primary_key=True)
    fincode: int = Field(primary_key=True, description="Company Code")  # 1
    date_end: int = Field(primary_key=True, description="Sharehoding As on Date")  # 2
    nh_ind_indivd: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Individuals / Hindu Undivided Family")  # 3
    ns_ind_indivd: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Individuals / Hindu Undivided Family")  # 4
    dp_ind_indivd: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Individuals / Hindu Undivided Family")  # 5
    tp_ind_indivd: Optional[float] = Field(default=None, description="% - Promoter - Indian - Individuals / Hindu Undivided Family")  # 6
    nh_ind_cgovt: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Central Government/State Government(s)")  # 7
    ns_ind_cgovt: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Central Government/State Government(s)")  # 8
    dp_ind_cgovt: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Central Government/State Government(s)")  # 9
    tp_ind_cgovt: Optional[float] = Field(default=None, description="% - Promoter - Indian - Central Government/State Government(s)")  # 10
    nh_ind_body_corp: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Bodies Corporate")  # 11
    ns_ind_body_corp: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Bodies Corporate")  # 12
    dp_ind_body_corp: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Bodies Corporate")  # 13
    tp_ind_body_corp: Optional[float] = Field(default=None, description="% - Promoter - Indian - Bodies Corporate")  # 14
    nh_ind_fi_bankcc: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Financial Institutions / Banks")  # 15
    ns_ind_fi_bankcc: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Financial Institutions / Banks")  # 16
    dp_ind_fi_bankcc: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Financial Institutions / Banks")  # 17
    tp_ind_fi_bankcc: Optional[float] = Field(default=None, description="% - Promoter - Indian - Financial Institutions / Banks")  # 18
    nh_ind_other: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Other")  # 19
    ns_ind_other: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Other")  # 20
    dp_ind_other: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Other")  # 21
    tp_ind_other: Optional[float] = Field(default=None, description="% - Promoter - Indian - Other")  # 22
    nh_ind_subtotal: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Sub Total")  # 23
    ns_ind_subtotal: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Sub Total")  # 24
    dp_ind_subtotal: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Sub Total")  # 25
    tp_ind_subtotal: Optional[float] = Field(default=None, description="% - Promoter - Indian - Sub Total")  # 26
    nh_f_nri_fn: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Foreign - Non-Residents Individuals / Foreign Individuals")  # 27
    ns_f_nri_fn: Optional[float] = Field(default=None, description="No of Shares - Promoter - Foreign - Non-Residents Individuals / Foreign Individuals")  # 28
    dp_f_nri_fn: Optional[float] = Field(default=None, description="Demat - Promoter - Foreign - Non-Residents Individuals / Foreign Individuals")  # 29
    tp_f_nri_fn: Optional[float] = Field(default=None, description="% - Promoter - Foreign - Non-Residents Individuals / Foreign Individuals")  # 30
    nh_f_body_corp: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Foreign - Bodies Corporate")  # 31
    ns_f_body_corp: Optional[float] = Field(default=None, description="No of Shares - Promoter - Foreign - Bodies Corporate")  # 32
    dp_f_body_corp: Optional[float] = Field(default=None, description="Demat - Promoter - Foreign - Bodies Corporate")  # 33
    tp_f_body_corp: Optional[float] = Field(default=None, description="% - Promoter - Foreign - Bodies Corporate")  # 34
    nh_f_institution: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Foreign - Institutions")  # 35
    ns_f_institution: Optional[float] = Field(default=None, description="No of Shares - Promoter - Foreign - Institutions")  # 36
    dp_f_institution: Optional[float] = Field(default=None, description="Demat - Promoter - Foreign - Institutions")  # 37
    tp_f_institution: Optional[float] = Field(default=None, description="% - Promoter - Foreign - Institutions")  # 38
    nh_f_other: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Foreign - Other")  # 39
    ns_f_other: Optional[float] = Field(default=None, description="No of Shares - Promoter - Foreign - Other")  # 40
    dp_f_other: Optional[float] = Field(default=None, description="Demat - Promoter - Foreign - Other")  # 41
    tp_f_other: Optional[float] = Field(default=None, description="% - Promoter - Foreign - Other")  # 42
    nh_f_subtotal: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Foreign - Sub Total")  # 43
    ns_f_subtotal: Optional[float] = Field(default=None, description="No of Shares - Promoter - Foreign - Sub Total")  # 44
    dp_f_subtotal: Optional[float] = Field(default=None, description="Demat - Promoter - Foreign - Sub Total")  # 45
    tp_f_subtotal: Optional[float] = Field(default=None, description="% - Promoter - Foreign - Sub Total")  # 46
    nh_f_total_promoter: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Total")  # 47
    ns_f_total_promoter: Optional[float] = Field(default=None, description="No of Shares - Promoter - Total")  # 48
    dp_f_total_promoter: Optional[float] = Field(default=None, description="Demat - Promoter - Total")  # 49
    tp_f_total_promoter: Optional[float] = Field(default=None, description="% - Promoter - Total")  # 50
    nh_in_mf_uti: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Mutual Funds / UTI")  # 51
    ns_in_mf_uti: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Mutual Funds / UTI")  # 52
    dp_in_mf_uti: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Mutual Funds / UTI")  # 53
    tp_in_mf_uti: Optional[float] = Field(default=None, description="% - Public - Institutions - Mutual Funds / UTI")  # 54
    nh_in_fi_banks: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Financial Institutions / Banks")  # 55
    ns_in_fi_banks: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Financial Institutions / Banks")  # 56
    dp_in_fi_banks: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Financial Institutions / Banks")  # 57
    tp_in_fi_banks: Optional[float] = Field(default=None, description="% - Public - Institutions - Financial Institutions / Banks")  # 58
    nh_in_insurance: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Insurance Companies")  # 59
    ns_in_insurance: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Insurance Companies")  # 60
    dp_in_insurance: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Insurance Companies")  # 61
    tp_in_insurance: Optional[float] = Field(default=None, description="% - Public - Institutions - Insurance Companies")  # 62
    nh_in_fii: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Foreign Institutional Investors")  # 63
    ns_in_fii: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Foreign Institutional Investors")  # 64
    dp_in_fii: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Foreign Institutional Investors")  # 65
    tp_in_fii: Optional[float] = Field(default=None, description="% - Public - Institutions - Foreign Institutional Investors")  # 66
    nh_in_ven_cap: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Venture Capital Funds")  # 67
    ns_in_ven_cap: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Venture Capital Funds")  # 68
    dp_in_ven_cap: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Venture Capital Funds")  # 69
    tp_in_ven_cap: Optional[float] = Field(default=None, description="% - Public - Institutions - Venture Capital Funds")  # 70
    nh_in_for_ven_cap: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Foreign Venture Capital Investors")  # 71
    ns_in_for_ven_cap: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Foreign Venture Capital Investors")  # 72
    dp_in_for_ven_cap: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Foreign Venture Capital Investors")  # 73
    tp_in_for_ven_cap: Optional[float] = Field(default=None, description="% - Public - Institutions - Foreign Venture Capital Investors")  # 74
    nh_in_cgovt: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Central Government / State Government(s)")  # 75
    # ns_in_cgovt: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Central Government / State Government(s)")  # 76
    dp_in_cgovt: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Central Government / State Government(s)")  # 77
    tp_in_cgovt: Optional[float] = Field(default=None, description="% - Public - Institutions - Central Government / State Government(s)")  # 78
    nh_in_for_fin_ins: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Foreign Financial Institutions / Banks")  # 79
    ns_in_for_fin_ins: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Foreign Financial Institutions / Banks")  # 80
    dp_in_for_fin_ins: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Foreign Financial Institutions / Banks")  # 81
    tp_in_for_fin_ins: Optional[float] = Field(default=None, description="% - Public - Institutions - Foreign Financial Institutions / Banks")  # 82
    nh_in_state_fin_corp: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - State Finance Corporation")  # 83
    ns_in_state_fin_corp: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - State Finance Corporation")  # 84
    dp_in_state_fin_corp: Optional[float] = Field(default=None, description="Demat - Public - Institutions - State Finance Corporation")  # 85
    tp_in_state_fin_corp: Optional[float] = Field(default=None, description="% - Public - Institutions - State Finance Corporation")  # 86
    nh_in_for_body: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Foreign Bodies DR")  # 87
    ns_in_for_body: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Foreign Bodies DR")  # 88
    dp_in_for_body: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Foreign Bodies DR")  # 89
    tp_in_for_body: Optional[float] = Field(default=None, description="% - Public - Institutions - Foreign Bodies DR")  # 90
    nh_in_other: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Other")  # 91
    ns_in_other: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Other")  # 92
    dp_in_other: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Other")  # 93
    tp_in_other: Optional[float] = Field(default=None, description="% - Public - Institutions - Other")  # 94
    nh_in_subtotal: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Sub Total")  # 95
    ns_in_subtotal: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Sub Total")  # 96
    dp_in_subtotal: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Sub Total")  # 97
    tp_in_subtotal: Optional[float] = Field(default=None, description="% - Public - Institutions - Sub Total")  # 98
    nh_nin_body_corp: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Bodies Corporate")  # 99
    ns_nin_body_corp: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Bodies Corporate")  # 100
    dp_nin_body_corp: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Bodies Corporate")  # 101
    tp_nin_body_corp: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Bodies Corporate")  # 102
    nh_nin_indivd: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Individuals")  # 103
    ns_nin_indivd: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Individuals")  # 104
    dp_nin_indivd: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Individuals")  # 105
    tp_nin_indivd: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Individuals")  # 106
    nh_nin_indivd_1lac: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Individual shareholders holding nominal share capital up to Rs. 1 lakh")  # 107
    ns_nin_indivd_1lac: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Individual shareholders holding nominal share capital up to Rs. 1 lakh")  # 108
    dp_nin_indivd_1lac: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Individual shareholders holding nominal share capital up to Rs. 1 lakh")  # 109
    tp_nin_indivd_1lac: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Individual shareholders holding nominal share capital up to Rs. 1 lakh")  # 110
    nh_nin_indivd_1lacmore: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Individual shareholders holding nominal share capital in excess of Rs. 1 lakh")  # 111
    ns_nin_indivd_1lacmore: Optional[float] = Field(default=None, description="No of Shares -Public - Non-Institutions - Individual shareholders holding nominal share capital in excess of Rs. 1 lakh")  # 112
    dp_nin_indivd_1lacmore: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Individual shareholders holding nominal share capital in excess of Rs. 1 lakh")  # 113
    tp_nin_indivd_1lacmore: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Individual shareholders holding nominal share capital in excess of Rs. 1 lakh")  # 114
    nh_nin_clear_memb: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Clearing Members")  # 115
    ns_nin_clear_memb: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Clearing Members")  # 116
    dp_nin_clear_memb: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Clearing Members")  # 117
    tp_nin_clear_memb: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Clearing Members")  # 118
    nh_nin_nri: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Non Resident Indians")  # 119
    ns_nin_nri: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Non Resident Indians")  # 120
    dp_nin_nri: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Non Resident Indians")  # 121
    tp_nin_nri: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Non Resident Indians")  # 122
    nh_nin_director: Optional[float] = Field(default=None, description="Sharehodreflectors - Public - Non-Institutions - Directors & their Relatives & Friends")  # 123
    ns_nin_director: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Directors & their Relatives & Friends")  # 124
    dp_nin_director: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Directors & their Relatives & Friends")  # 125
    tp_nin_director: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Directors & their Relatives & Friends")  # 126
    nh_nin_forn_coll: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Foreign Collaborators")  # 127
    ns_nin_forn_coll: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Foreign Collaborators")  # 128
    dp_nin_forn_coll: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Foreign Collaborators")  # 129
    tp_nin_forn_coll: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Foreign Collaborators")  # 130
    nh_nin_forn_mf: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Foreign Mutual Fund")  # 131
    ns_nin_forn_mf: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Foreign Mutual Fund")  # 132
    dp_nin_forn_mf: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Foreign Mutual Fund")  # 133
    tp_nin_forn_mf: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Foreign Mutual Fund")  # 134
    nh_nin_trusts: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Trusts")  # 135
    ns_nin_trusts: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Trusts")  # 136
    dp_nin_trusts: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Trusts")  # 137
    tp_nin_trusts: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Trusts")  # 138
    nh_nin_huf: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Hindu Undivided Families")  # 139
    ns_nin_huf: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Hindu Undivided Families")  # 140
    dp_nin_huf: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Hindu Undivided Families")  # 141
    tp_nin_huf: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Hindu Undivided Families")  # 142
    nh_nin_forn_corp_body: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Foreign Corporate Bodies")  # 143
    ns_nin_forn_corp_body: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Foreign Corporate Bodies")  # 144
    dp_nin_forn_corp_body: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Foreign Corporate Bodies")  # 145
    tp_nin_forn_corp_body: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Foreign Corporate Bodies")  # 146
    nh_nin_share_intransit: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Shares in transit")  # 147
    ns_nin_share_intransit: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Shares in transit")  # 148
    dp_nin_share_intransit: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Shares in transit")  # 149
    tp_nin_share_intransit: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Shares in transit")  # 150
    nh_nin_mkt_maker: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Market Maker")  # 151
    ns_nin_mkt_maker: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Market Maker")  # 152
    dp_nin_mkt_maker: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Market Maker")  # 153
    tp_nin_mkt_maker: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Market Maker")  # 154
    nh_nin_employees: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - ESOP/ESOS/ESPS")  # 155
    ns_nin_employees: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - ESOP/ESOS/ESPS")  # 156
    dp_nin_employees: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - ESOP/ESOS/ESPS")  # 157
    tp_nin_employees: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - ESOP/ESOS/ESPS")  # 158
    nh_nin_society: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Societies")  # 159
    ns_nin_society: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Societies")  # 160
    dp_nin_society: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Societies")  # 161
    tp_nin_society: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Societies")  # 162
    nh_nin_escrow: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Escrow Account")  # 163
    ns_nin_escrow: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Escrow Account")  # 164
    dp_nin_escrow: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Escrow Account")  # 165
    tp_nin_escrow: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Escrow Account")  # 166
    nh_nin_other: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Any Other")  # 167
    ns_nin_other: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Any Other")  # 168
    dp_nin_other: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Any Other")  # 169
    tp_nin_other: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Any Other")  # 170
    nh_nin_subtotal: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Sub Total")  # 171
    ns_nin_subtotal: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Sub Total")  # 172
    dp_nin_subtotal: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Sub Total")  # 173
    tp_nin_subtotal: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Sub Total")  # 174
    nh_total_public: Optional[float] = Field(default=None, description="Sharehoders - Public - Total")  # 175
    ns_total_public: Optional[float] = Field(default=None, description="No of Shares - Public - Total")  # 176
    dp_total_public: Optional[float] = Field(default=None, description="Demat - Public - Total")  # 177
    tp_total_public: Optional[float] = Field(default=None, description="% - Public - Total")  # 178
    nh_total_prom_public: Optional[float] = Field(default=None, description="Sharehoders - Total Promoter & Publics")  # 179
    ns_total_prom_public: Optional[float] = Field(default=None, description="No of Shares - Total Promoter & Publics")  # 180
    dp_total_prom_public: Optional[float] = Field(default=None, description="Demat - Total Promoter & Publics")  # 181
    tp_total_prom_public: Optional[float] = Field(default=None, description="% - Total Promoter & Publics")  # 182
    nh_custodian_drs: Optional[float] = Field(default=None, description="Sharehoders - Custodians")  # 183
    ns_custodian_drs: Optional[float] = Field(default=None, description="No of Shares - Custodians")  # 184
    dp_custodian_drs: Optional[float] = Field(default=None, description="Demat - Custodians")  # 185
    tp_custodian_drs: Optional[float] = Field(default=None, description="% - Custodians")  # 186
    nh_adr: Optional[float] = Field(default=None, description="Sharehoders - Custodians - American Depository Receipts")  # 187
    ns_adr: Optional[float] = Field(default=None, description="No of Shares - Custodians - American Depository Receipts")  # 188
    dp_adr: Optional[float] = Field(default=None, description="Demat - Custodians - American Depository Receipts")  # 189
    tp_adr: Optional[float] = Field(default=None, description="% - Custodians - American Depository Receipts")  # 190
    nh_gdr: Optional[float] = Field(default=None, description="Sharehoders - Custodians - GDRs")  # 191
    ns_gdr: Optional[float] = Field(default=None, description="No of Shares - Custodians - GDRs")  # 192
    dp_gdr: Optional[float] = Field(default=None, description="Demat - Custodians - GDRs")  # 193
    tp_gdr: Optional[float] = Field(default=None, description="% - Custodians - GDRs")  # 194
    nh_other: Optional[float] = Field(default=None, description="Sharehoders - Custodians - Other")  # 195
    ns_other: Optional[float] = Field(default=None, description="No of Shares - Custodians - Other")  # 196
    dp_other: Optional[float] = Field(default=None, description="Demat - Custodians - Other")  # 197
    tp_other: Optional[float] = Field(default=None, description="% - Custodians - Other")  # 198
    nh_grand_total: Optional[float] = Field(default=None, description="Sharehoders - Grand Total")  # 199
    ns_grand_total: Optional[float] = Field(default=None, description="No of Shares - Grand Total")  # 200
    dp_grand_total: Optional[float] = Field(default=None, description="Demat - Grand Total")  # 201
    tp_grand_total: Optional[float] = Field(default=None, description="% - Grand Total")  # 202
    nh_other_prt_firms: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Partnership Firms")  # 203
    ns_other_prt_firms: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Partnership Firms")  # 204
    dp_other_prt_firms: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Partnership Firms")  # 205
    tp_other_prt_firms: Optional[float] = Field(default=None, description="% - Promoter - Indian - Partnership Firms")  # 206
    nh_other_prom_group: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Promoter Group")  # 207
    ns_other_prom_group: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Promoter Group")  # 208
    dp_other_prom_group: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Promoter Group")  # 209
    tp_other_prom_group: Optional[float] = Field(default=None, description="% - Promoter - Indian - Promoter Group")  # 210
    nh_other_ewf: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Employees Welfare Fund")  # 211
    ns_other_ewf: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Employees Welfare Fund")  # 212
    dp_other_ewf: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Employees Welfare Fund")  # 213
    tp_other_ewf: Optional[float] = Field(default=None, description="% - Promoter - Indian - Employees Welfare Fund")  # 214
    nh_sasf: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Stressed Assets Stabilisation Fund")  # 215
    ns_sasf: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Stressed Assets Stabilisation Fund")  # 216
    dp_sasf: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Stressed Assets Stabilisation Fund")  # 217
    tp_sasf: Optional[float] = Field(default=None, description="% - Public - Institutions - Stressed Assets Stabilisation Fund")  # 218
    nh_ind_any_other_specfd: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Indian - Any Other")  # 219
    nh_in_any_other_specfd: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Any Other")  # 220
    ns_in_any_other_specfd: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Any Other")  # 221
    dp_in_any_other_specfd: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Any Other")  # 222
    tp_in_any_other_specfd: Optional[float] = Field(default=None, description="% - Public - Institutions - Any Other")  # 223
    nh_nin_any_other_specfd: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Any Other")  # 224
    ns_nin_any_other_specfd: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Any Other")  # 225
    dp_nin_any_other_specfd: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Any Other")  # 226
    tp_nin_any_other_specfd: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Any Other")  # 227
    nh_f_any_other_specfd: Optional[float] = Field(default=None, description="Sharehoders - Promoter - Foreign - Any Other")  # 228
    ns_f_any_other_specfd: Optional[float] = Field(default=None, description="No of Shares - Promoter - Foreign - Any Other")  # 229
    dp_f_any_other_specfd: Optional[float] = Field(default=None, description="Demat - Promoter - Foreign - Any Other")  # 230
    tp_f_any_other_specfd: Optional[float] = Field(default=None, description="% - Promoter - Foreign - Any Other")  # 231
    ns_ind_any_other_specfd: Optional[float] = Field(default=None, description="No of Shares - Promoter - Indian - Any Other")  # 232
    dp_ind_any_other_specfd: Optional[float] = Field(default=None, description="Demat - Promoter - Indian - Any Other")  # 233
    tp_ind_any_other_specfd: Optional[float] = Field(default=None, description="% - Promoter - Indian - Any Other")  # 234
    nsh_ind_subtotal: Optional[float] = Field(default=None, description="Indian Promoters")  # 235
    nsh_ind_indivd: Optional[float] = Field(default=None, description="Individuals / Hindu Undivided Family")  # 236
    nsh_ind_cgovt: Optional[float] = Field(default=None, description="Central Government/State Government(s)")  # 237
    nsh_ind_body_corp: Optional[float] = Field(default=None, description="Bodies Corporate")  # 238
    nsh_ind_fi_bankcc: Optional[float] = Field(default=None, description="Financial Institutions / Banks")  # 239
    nsh_other_prt_firms: Optional[float] = Field(default=None, description="Partnership Firms")  # 240
    nsh_other_prom_group: Optional[float] = Field(default=None, description="Promoter Group")  # 241
    nsh_other_ewf: Optional[float] = Field(default=None, description="Employees Welfare Fund")  # 242
    nsh_ind_other: Optional[float] = Field(default=None, description="Other")  # 243
    nsh_ind_any_other_specfd: Optional[float] = Field(default=None, description="Any Others (Specify)")  # 244
    nsh_f_subtotal: Optional[float] = Field(default=None, description="Foreign Promoters")  # 245
    nsh_f_nri_fn: Optional[float] = Field(default=None, description="Non-Residents Individuals / Foreign Individuals")  # 246
    nsh_f_body_corp: Optional[float] = Field(default=None, description="Bodies Corporate")  # 247
    nsh_f_institution: Optional[float] = Field(default=None, description="Institutions")  # 248
    nsh_f_other: Optional[float] = Field(default=None, description="Other")  # 249
    nsh_f_any_other_specfd: Optional[float] = Field(default=None, description="Any Others (Specify)")  # 250
    nsh_f_total_promoter: Optional[float] = Field(default=None, description="Total of Promoter and Promoter Group")  # 251
    nsh_in_subtotal: Optional[float] = Field(default=None, description="Institutions")  # 252
    nsh_in_mf_uti: Optional[float] = Field(default=None, description="Mutual Funds / UTI")  # 253
    nsh_in_fi_banks: Optional[float] = Field(default=None, description="Financial Institutions / Banks")  # 254
    nsh_in_insurance: Optional[float] = Field(default=None, description="Insurance Companies")  # 255
    nsh_in_fii: Optional[float] = Field(default=None, description="Foreign Institutional Investors")  # 256
    nsh_in_ven_cap: Optional[float] = Field(default=None, description="Venture Capital Funds")  # 257
    nsh_in_for_ven_cap: Optional[float] = Field(default=None, description="Foreign Venture Capital Investors")  # 258
    nsh_in_any_other_specfd: Optional[float] = Field(default=None, description="Any Others (Specify)")  # 259
    nsh_in_cgovt: Optional[float] = Field(default=None, description="Central Government / State Government(s)")  # 260
    nsh_in_for_fin_ins: Optional[float] = Field(default=None, description="Foreign Financial Institutions / Banks")  # 261
    nsh_sasf: Optional[float] = Field(default=None, description="Stressed Assets Stabilisation Fund")  # 262
    nsh_in_state_fin_corp: Optional[float] = Field(default=None, description="State Finance Corporation")  # 263
    nsh_in_for_body: Optional[float] = Field(default=None, description="Foreign Bodies DR")  # 264
    nsh_in_other: Optional[float] = Field(default=None, description="Other")  # 265
    nsh_nin_subtotal: Optional[float] = Field(default=None, description="Non-Institutions")  # 266
    nsh_nin_body_corp: Optional[float] = Field(default=None, description="Bodies Corporate")  # 267
    nsh_nin_indivd: Optional[float] = Field(default=None, description="Individuals")  # 268
    nsh_nin_indivd_1lac: Optional[float] = Field(default=None, description="Individual shareholders holding nominal share capital up to Rs. 1 lakh")  # 269
    nsh_nin_indivd_1lacmore: Optional[float] = Field(default=None, description="Individual shareholders holding nominal share capital in excess of Rs. 1 lakh")  # 270
    nsh_nin_any_other_specfd: Optional[float] = Field(default=None, description="Any Others (Specify)")  # 271
    nsh_nin_clear_memb: Optional[float] = Field(default=None, description="Clearing Members")  # 272
    nsh_nin_nri: Optional[float] = Field(default=None, description="Non Resident Indians")  # 273
    nsh_nin_director: Optional[float] = Field(default=None, description="Directors & their Relatives & Friends")  # 274
    nsh_nin_forn_coll: Optional[float] = Field(default=None, description="Foreign Collaborators")  # 275
    nsh_nin_forn_mf: Optional[float] = Field(default=None, description="Foreign Mutual Fund")  # 276
    nsh_nin_trusts: Optional[float] = Field(default=None, description="Trusts")  # 277
    nsh_nin_huf: Optional[float] = Field(default=None, description="Hindu Undivided Families")  # 278
    nsh_nin_forn_corp_body: Optional[float] = Field(default=None, description="Foreign Corporate Bodies")  # 279
    nsh_nin_share_intransit: Optional[float] = Field(default=None, description="Shares in transit")  # 280
    nsh_nin_mkt_maker: Optional[float] = Field(default=None, description="Market Maker")  # 281
    nsh_nin_employees: Optional[float] = Field(default=None, description="ESOP/ESOS/ESPS")  # 282
    nsh_nin_society: Optional[float] = Field(default=None, description="Societies")  # 283
    nsh_nin_escrow: Optional[float] = Field(default=None, description="Escrow Account")  # 284
    nsh_nin_other: Optional[float] = Field(default=None, description="Any Other")  # 285
    nsh_total_public: Optional[float] = Field(default=None, description="Total Public Shareholding")  # 286
    nsh_total_prom_public: Optional[float] = Field(default=None, description="Total of Promoter and Public Shareholding")  # 287
    nsh_custodian_drs: Optional[float] = Field(default=None, description="Shares held by Custodians and against which Depository Receipts have been issued")  # 288
    nsh_adr: Optional[float] = Field(default=None, description="ADRs")  # 289
    nsh_gdr: Optional[float] = Field(default=None, description="GDRs")  # 290
    nsh_other: Optional[float] = Field(default=None, description="Other")  # 291
    nsh_grand_total: Optional[float] = Field(default=None, description="Grand Total")  # 292
    psh_ind_subtotal: Optional[float] = Field(default=None, description="Indian Promoters %")  # 293
    psh_ind_indivd: Optional[float] = Field(default=None, description="Individuals / Hindu Undivided Family %")  # 294
    psh_ind_cgovt: Optional[float] = Field(default=None, description="Central Government/State Government(s) %")  # 295
    psh_ind_body_corp: Optional[float] = Field(default=None, description="Bodies Corporate %")  # 296
    psh_ind_fi_bankcc: Optional[float] = Field(default=None, description="Financial Institutions / Banks %")  # 297
    psh_other_prt_firms: Optional[float] = Field(default=None, description="Partnership Firms %")  # 298
    psh_other_prom_group: Optional[float] = Field(default=None, description="Promoter Group %")  # 299
    psh_other_ewf: Optional[float] = Field(default=None, description="Employees Welfare Fund %")  # 300
    psh_ind_other: Optional[float] = Field(default=None, description="Other %")  # 301
    psh_ind_any_other_specfd: Optional[float] = Field(default=None, description="Any Others (Specify) %")  # 302
    psh_f_subtotal: Optional[float] = Field(default=None, description="Foreign Promoters %")  # 303
    psh_f_nri_fn: Optional[float] = Field(default=None, description="Non-Residents Individuals / Foreign Individuals %")  # 304
    psh_f_body_corp: Optional[float] = Field(default=None, description="Bodies Corporate %")  # 305
    psh_f_institution: Optional[float] = Field(default=None, description="Institutions %")  # 306
    psh_f_other: Optional[float] = Field(default=None, description="Other %")  # 307
    psh_f_any_other_specfd: Optional[float] = Field(default=None, description="Any Others (Specify) %")  # 308
    psh_f_total_promoter: Optional[float] = Field(default=None, description="Total of Promoter and Promoter Group %")  # 309
    psh_in_subtotal: Optional[float] = Field(default=None, description="Institutions %")  # 310
    psh_in_mf_uti: Optional[float] = Field(default=None, description="Mutual Funds / UTI %")  # 311
    psh_in_fi_banks: Optional[float] = Field(default=None, description="Financial Institutions / Banks %")  # 312
    psh_in_insurance: Optional[float] = Field(default=None, description="Insurance Companies %")  # 313
    psh_in_fii: Optional[float] = Field(default=None, description="Foreign Institutional Investors %")  # 314
    psh_in_ven_cap: Optional[float] = Field(default=None, description="Venture Capital Funds %")  # 315
    psh_in_for_ven_cap: Optional[float] = Field(default=None, description="Foreign Venture Capital Investors %")  # 316
    psh_in_any_other_specfd: Optional[float] = Field(default=None, description="Any Others (Specify) %")  # 317
    psh_in_cgovt: Optional[float] = Field(default=None, description="Central Government / State Government(s) %")  # 318
    psh_in_for_fin_ins: Optional[float] = Field(default=None, description="Foreign Financial Institutions / Banks %")  # 319
    psh_sasf: Optional[float] = Field(default=None, description="Stressed Assets Stabilisation Fund %")  # 320
    psh_in_state_fin_corp: Optional[float] = Field(default=None, description="State Finance Corporation %")  # 321
    psh_in_for_body: Optional[float] = Field(default=None, description="Foreign Bodies DR %")  # 322
    psh_in_other: Optional[float] = Field(default=None, description="Other %")  # 323
    psh_nin_subtotal: Optional[float] = Field(default=None, description="Non-Institutions %")  # 324
    psh_nin_body_corp: Optional[float] = Field(default=None, description="Bodies Corporate %")  # 325
    psh_nin_indivd: Optional[float] = Field(default=None, description="Individuals %")  # 326
    psh_nin_indivd_1lac: Optional[float] = Field(default=None, description="Individual shareholders holding nominal share capital up to Rs. 1 lakh %")  # 327
    psh_nin_indivd_1lacmore: Optional[float] = Field(default=None, description="Individual shareholders holding nominal share capital in excess of Rs. 1 lakh %")  # 328
    psh_nin_any_other_specfd: Optional[float] = Field(default=None, description="Any Others (Specify) %")  # 329
    psh_nin_clear_memb: Optional[float] = Field(default=None, description="Clearing Members %")  # 330
    psh_nin_nri: Optional[float] = Field(default=None, description="Non Resident Indians %")  # 331
    psh_nin_director: Optional[float] = Field(default=None, description="Directors & their Relatives & Friends %")  # 332
    psh_nin_forn_coll: Optional[float] = Field(default=None, description="Foreign Collaborators %")  # 333
    psh_nin_forn_mf: Optional[float] = Field(default=None, description="Foreign Mutual Fund %")  # 334
    psh_nin_trusts: Optional[float] = Field(default=None, description="Trusts %")  # 335
    psh_nin_huf: Optional[float] = Field(default=None, description="Hindu Undivided Families %")  # 336
    psh_nin_forn_corp_body: Optional[float] = Field(default=None, description="Foreign Corporate Bodies %")  # 337
    psh_nin_share_intransit: Optional[float] = Field(default=None, description="Shares in transit %")  # 338
    psh_nin_mkt_maker: Optional[float] = Field(default=None, description="Market Maker %")  # 339
    psh_nin_employees: Optional[float] = Field(default=None, description="ESOP/ESOS/ESPS %")  # 340
    psh_nin_society: Optional[float] = Field(default=None, description="Societies %")  # 341
    psh_nin_escrow: Optional[float] = Field(default=None, description="Escrow Account %")  # 342
    psh_nin_other: Optional[float] = Field(default=None, description="Any Other %")  # 343
    psh_total_public: Optional[float] = Field(default=None, description="Total Public Shareholding %")  # 344
    psh_total_prom_public: Optional[float] = Field(default=None, description="Total of Promoter and Public Shareholding %")  # 345
    psh_custodian_drs: Optional[float] = Field(default=None, description="Shares held by Custodians and against which Depository Receipts have been issued %")  # 346
    psh_adr: Optional[float] = Field(default=None, description="ADRs %")  # 347
    psh_other: Optional[float] = Field(default=None, description="Other %")  # 348
    psh_grand_total: Optional[float] = Field(default=None, description="Grand Total %")  # 349
    nh_in_alt_inv_fund: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Alternate Investment Funds")  # 350
    ns_in_alt_inv_fund: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Alternate Investment Funds")  # 351
    dp_in_alt_inv_fund: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Alternate Investment Funds")  # 352
    tp_in_alt_inv_fund: Optional[float] = Field(default=None, description="% - Public - Institutions - Alternate Investment Funds")  # 353
    nh_in_foreign_port_inv: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Foreign Portfolio Investors")  # 354
    ns_in_foreign_port_inv: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Foreign Portfolio Investors")  # 355
    dp_in_foreign_port_inv: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Foreign Portfolio Investors")  # 356
    tp_in_foreign_port_inv: Optional[float] = Field(default=None, description="% - Public - Institutions - Foreign Portfolio Investors")  # 357
    nh_in_provi_pen_fund: Optional[float] = Field(default=None, description="Sharehoders - Public - Institutions - Provident Funds/ Pension Funds")  # 358
    ns_in_provi_pen_fund: Optional[float] = Field(default=None, description="No of Shares - Public - Institutions - Provident Funds/ Pension Funds")  # 359
    dp_in_provi_pen_fund: Optional[float] = Field(default=None, description="Demat - Public - Institutions - Provident Funds/ Pension Funds")  # 360
    tp_in_provi_pen_fund: Optional[float] = Field(default=None, description="% - Public - Institutions - Provident Funds/ Pension Funds")  # 361
    nh_nin_indivd_nbfc: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - NBFCs registered with RBI")  # 362
    ns_nin_indivd_nbfc: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - NBFCs registered with RBI")  # 363
    dp_nin_indivd_nbfc: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - NBFCs registered with RBI")  # 364
    tp_nin_indivd_nbfc: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - NBFCs registered with RBI")  # 365
    nh_nin_ind_emp_trust: Optional[float] = Field(default=None, description="Sharehoders - Public - Non-Institutions - Employee Trusts")  # 366
    ns_nin_ind_emp_trust: Optional[float] = Field(default=None, description="No of Shares - Public - Non-Institutions - Employee Trusts")  # 367
    dp_nin_ind_emp_trust: Optional[float] = Field(default=None, description="Demat - Public - Non-Institutions - Employee Trusts")  # 368
    tp_nin_ind_emp_trust: Optional[float] = Field(default=None, description="% - Public - Non-Institutions - Employee Trusts")  # 369
    flag: Optional[str] = Field(default=None, max_length=1, description="Updation Flag")  # 370