all_table_info = """

## Fundamental Data

| **Table Name**                      | **Model Class**               | **Short Description**                                               | **Related Tables & Relationships**                                          |
| ----------------------------------- | ----------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `company_master`                    | `CompanyMaster`               | Core company data: fincode, name, industry, symbol, listing status.scripcode | Links to almost all company-level tables via `fincode`.                     |
| `industry_master`                   | `IndustryMaster`              | Master list of industries and sectors.                              | `ind_code` → `company_master.ind_code`.                                     |
| `house_master`                      | `HouseMaster`                 | Master of business groups (houses).                                 | `house_code` → `company_master.hse_code`.                                   |
| `stock_exchange_master`             | `StockExchangeMaster`         | List of stock exchanges.                                            | `stk_id` → `company_listings.stk_id`.                                       |
| `company_listings`                  | `CompanyListings`             | Stock exchange listings for companies.                              | `fincode` → `company_master`, `stk_id` → `stock_exchange_master`.           |
| `company_address`                   | `CompanyAddress`              | Registered office address of the company.                           | `fincode` → `company_master`.                                               |
| `company_board_director`            | `CompanyDirector`             | Board of directors: names, designations, tenure, remuneration.      | `fincode` → `company_master`.                                               |
| `company_equity`                    | `CompanyEquity`               | Standalone equity structure.                                        | `fincode` → `company_master`.                                               |
| `company_equity_cons`               | `CompanyEquityCons`           | Consolidated equity details.                                        | `fincode` → `company_master`.                                               |
| `company_finance_balancesheet`      | `CompanyBalanceSheet`         | Standalone balance sheet data.                                      | `fincode` → `company_master`.                                               |
| `company_finance_balancesheet_cons` | `CompanyBalanceSheetCons`     | Consolidated balance sheet.                                         | `fincode` → `company_master`.                                               |
| `company_finance_cashflow`          | `CompanyCashflow`             | Standalone cash flow statement.                                     | `fincode` → `company_master`.                                               |
| `company_finance_cashflow_cons`     | `CompanyCashflowCons`         | Consolidated cash flow data.                                        | `fincode` → `company_master`.                                               |
| `company_finance_profitloss`        | `CompanyProfitLoss`           | Standalone profit & loss data.                                      | `fincode` → `company_master`.                                               |
| `company_finance_profitloss_cons`   | `CompanyProfitLossCons`       | Consolidated P\&L data.                                             | `fincode` → `company_master`.                                               |
| `company_finance_ratio`             | `CompanyFinanceRatio`         | Standalone financial ratios (EPS, ROE, etc.).                       | `fincode` → `company_master`.                                               |
| `company_finance_ratio_cons`        | `CompanyFinanceRatioCons`     | Consolidated financial ratios.                                      | `fincode` → `company_master`.                                               |
| `company_results`                   | `FinancialResult`             | Standalone IND-AS quarterly/annual results.                         | `fincode` → `company_master`.                                               |
| `company_results_cons`              | `FinancialResultCons`         | Consolidated version of company results.                            | `fincode` → `company_master`.                                               |
| `company_shareholders_details`      | `CompanyShareholdersDetails`  | Shareholder details including shareholding and reporting period.    | `fincode` → `company_master`, `shp_catid` → `shareholding_category_master`. |
| `company_shareholding_pattern`      | `ShpSummary`                  | Shareholding summary (promoters, FIIs, etc.).                       | `fincode` → `company_master`.                                               |
| `company_registrar_master`          | `RegistrarMaster`             | Master data of registrars.                                          | `registrar_no` → `company_registrar_data`.                                  |
| `company_registrar_data`            | `CompanyRegistrar`            | Links companies to their registrars.                                | `fincode` → `company_master`, `registrar_no` → `company_registrar_master`.  |
| `company_profile`                   | `CompanyProfile`              | Brief company profile including executives.                         | `fincode` → `company_master`.                                               |
| `monthly_price_bse`                 | `MonthlyPriceBSE`             | Monthly prices of BSE-listed stocks.                                | `fincode` → `company_master`.                                               |
| `monthly_price_nse`                 | `MonthlyPriceNSE`             | Monthly prices of NSE-listed stocks.                                | `fincode` → `company_master`.                                               |
| `shareholding_category_master`      | `ShareholdingCategoryMaster`  | Categories like Promoter, Public, Institutions, etc.                | `shp_catid` → `company_shareholders_details`.                               |



## StockPrices Data


| `bse_abjusted_price_eod`            | `BSEAdjustedPriceEOD`         | Historical adjusted price data (BSE).                               | `fincode` → `company_master`.                                               |
| `bse_indices_price_eod`             | `BSEIndicesEOD`               | Daily EOD prices of BSE indices.                                    | `index_code` → `indices_master`.                                            |
| `company_index_part`                | `CompanyIndexPart`            | Links companies to the indices they belong to.                      | `fincode` → `company_master`, `index_code` → `indices_master`.              |
| `indices_master`                    | `IndicesMaster`               | Master list of indices.                                             | `index_code` → `bse_indices_price_eod`, `company_index_part`.               |
"""