# artifacts/tables.py

all_table_info = """
--- DATABASE TABLE INFORMATION ---

**Table: `company_master`**
Purpose: Core details for all listed companies. Essential for identifying companies by name or symbol and getting their unique `fincode`.
Key Columns:
- `fincode` (Integer): Unique company identifier (AFPL's unique code). **Use this to link to all other company-specific tables.**
- `compname` (Varchar(255)): Company's full legal name (e.g., "Reliance Industries Ltd."). Use for `LIKE` searches.
- `s_name` (Varchar(100)): Company's short name.
- `symbol` (Varchar(20)): Company's NSE stock symbol.
- `scripcode` (Integer): Company's BSE scrip code.
- `industry` (Varchar(100)): Name of the company's industry.
- `sector` (Varchar(100)): Sector to which the industry belongs (derived via industry_master).
- `fv` (Float): Face value of the company's shares.

**Table: `company_equity`**
Purpose: Provides standalone equity details and key valuation metrics like market capitalization, P/E ratio, and book value. Usually contains the latest or year-end figures.
Key Columns:
- `fincode` (Integer): Links to `company_master`.
- `year_end` (Integer): Financial year end for the data.
- `mcap` (Float): **Market Capitalization of the company.** This is the primary source for market cap.
- `ttmpe` (Float): Trailing Twelve Months **Price-to-Earnings (P/E) ratio.**
- `ttmeps` (Float): Trailing Twelve Months **Earnings Per Share (EPS).**
- `booknavpershare` (Float): Book Value per Share (Book NAV per share).
- `latest_equity` (Float): Latest total equity.
- `price` (Float): Latest market price per share used for some calculations.
- `price_date` (Datetime): Date of the `price` column.
- `dividend_yield` (Float): Dividend Yield.
- `fv` (Float): Face Value.

**Table: `company_equity_cons`** (Consolidated version of `company_equity`)
Purpose: Provides consolidated equity details and key valuation metrics. Use this for a group-level view if available.
Key Columns: Same as `company_equity` but representing consolidated figures (e.g., `mcap`, `ttmpe`, `ttmeps`).
- `fincode` (Integer): Links to `company_master`.
- `mcap` (Float): **Consolidated Market Capitalization.**
- `ttmpe` (Float): **Consolidated Trailing P/E ratio.**

**Table: `company_finance_ratio`**
Purpose: Contains a comprehensive set of standalone financial ratios for profitability, efficiency, solvency, and valuation.
Key Columns:
- `fincode` (Integer): Links to `company_master`.
- `year_end` (Integer): Financial year end for the ratios.
- `type` (Varchar(1)): Specifies if standalone (S) or consolidated (C) - this table seems to be standalone based on name.
- `reported_eps` (Float): Reported Earnings Per Share.
- `adjusted_eps` (Float): Adjusted Earnings Per Share.
- `roe` (Float): **Return on Equity.**
- `roa` (Float): Return on Assets.
- `per` (Float): Price-to-Earnings ratio (check against `company_equity.ttmpe` for latest).
- `price_book` (Float): Price-to-Book Value ratio.
- `mcap_sales` (Float): Market Capitalization relative to Sales (a valuation ratio, not direct market cap).
- `total_debt_equity` (Float): Total Debt-to-Equity ratio.
- `current_ratio` (Float): Current Ratio (liquidity).
- `mcap_growth` (Float): Growth rate of market capitalization.

**Table: `company_finance_ratio_cons`** (Consolidated version of `company_finance_ratio`)
Purpose: Contains consolidated financial ratios. Use this for a group-level ratio analysis.
Key Columns: Same as `company_finance_ratio` but representing consolidated figures (e.g., `roe`, `per`).
- `fincode` (Integer): Links to `company_master`.
- `roe` (Float): **Consolidated Return on Equity.**

**Table: `company_finance_cashflow`**
Purpose: Details standalone cash flow from operating, investing, and financing activities.
Key Columns:
- `fincode` (Integer): Links to `company_master`.
- `year_end` (Integer): Year end of the reporting period.
- `type` (Varchar(1)): Indicates standalone (S) or consolidated (C) - this table is standalone.
- `net_cash_inflow_outflow` (Float): **Net cash inflow or outflow for the period (Total Net Cash Flow).**
- `cash_from_operation` (Float): Net cash generated from operating activities.
- `cash_from_investment` (Float): Net cash used in investing activities.
- `cash_from_financing` (Float): Net cash generated from financing activities.

**Table: `company_finance_cashflow_cons`** (Consolidated version)
Purpose: Details consolidated cash flow statements.
Key Columns: Same as `company_finance_cashflow` but for consolidated figures.
- `fincode` (Integer): Links to `company_master`.
- `net_cash_inflow_outflow` (Float): **Consolidated Net Cash Flow.**

**Table: `company_finance_profitloss`**
Purpose: Standalone profit and loss (income statement) details, including revenue, expenses, and net profit.
Key Columns:
- `fincode` (Integer): Links to `company_master`.
- `year_end` (Integer): Financial year end.
- `type` (Varchar(1)): Standalone (S) or consolidated (C) - this table is standalone.
- `net_sales` (Float): Net revenue from core operations.
- `total_income` (Float): Sum of operating and non-operating revenues.
- `operating_profit` (Float): Operating Profit.
- `profit_after_tax` (Float): Profit After Tax (PAT).
- `consolidated_netprofit` (Float): *This column might be misleading in a standalone table; usually `profit_after_tax` is the standalone net profit.*
- `reported_eps` (Float): Earnings Per Share.

**Table: `company_finance_profitloss_cons`** (Consolidated version)
Purpose: Consolidated profit and loss (income statement) details.
Key Columns: Same as `company_finance_profitloss` but for consolidated figures.
- `fincode` (Integer): Links to `company_master`.
- `consolidated_netprofit` (Float): **Consolidated Net Profit.**
- `reported_eps` (Float): Consolidated Earnings Per Share.

**Table: `company_shareholding_pattern`**
Purpose: Detailed breakdown of shareholding patterns (promoter, public, FII, DII, etc.) for companies.
Key Columns:
- `fincode` (Integer): Links to `company_master`.
- `date_end` (Integer): Date when the shareholding data was reported (YYYYMMDD format).
- `tp_ind_subtotal` (Float): Total percentage of shares held by Indian promoters.
- `tp_f_total_promoter` (Float): Total percentage of shares held by all (Indian + foreign) promoters.
- `ns_ind_subtotal` (Float): Total shares held by Indian promoters.
- (Many other columns for specific promoter/public sub-categories like `ns_in_fii` for FII shares, `tp_in_fii` for FII percentage) - *Refer to detailed schema for specific categories like "Promoters", "FII", "DII", "Public".*

**Table: `company_results`**
Purpose: Comprehensive company financial results (standalone) in IND-AS format, reported quarterly, half-yearly, and annually. Values in millions.
Key Columns:
- `fincode` (Integer): Links to `company_master`.
- `date_end` (Integer): End date of the reporting period (YYYYMMDD).
- `result_type` (Varchar(2)): Q (Quarterly), H (Half-yearly), A (Annual). Revised records (QR, HR, AR) preferred.
- `net_sales` (Float): Net revenue.
- `pat` (Float): Profit After Tax.
- `eps_basic` (Float): Basic Earnings Per Share.
- `prom_percent_of_shares` (Float): Percentage of shares held by public shareholders (Note: PDF says "public shareholders" but context of `prom_` usually means promoters - CLARIFY THIS if it's actually promoter percentage).
- `promoter_nos` (Integer): Number of shares held by promoters.

**Table: `company_results_cons`** (Consolidated version of `company_results`)
Purpose: Consolidated financial results.
Key Columns: Similar to `company_results` but for consolidated figures.
- `fincode` (Integer): Links to `company_master`.
- `pat` (Float): Consolidated Profit After Tax.

**Table: `bse_abjusted_price_eod` / `monthly_price_bse` / `monthly_price_nse`**
Purpose: Stock price data (daily or monthly). `bse_abjusted_price_eod` is for daily adjusted prices. These are generally NOT used for direct fundamental value lookups like market cap (which is in `company_equity`) but can be used for calculating returns or in conjunction with other data.
Key Columns (`bse_abjusted_price_eod`):
- `fincode` (Integer): Links to `company_master`.
- `date` (Date): Trading date.
- `close` (Float): Closing price.
- `volume` (BigInteger): Trading volume.

--- End of Database Table Information ---
"""