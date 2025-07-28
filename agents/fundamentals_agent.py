# fundamentals_agent.py
import asyncio
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv 
import json
from sqlalchemy import text
import traceback
from sqlalchemy.ext.asyncio import create_async_engine
from db_config import ASYNC_DATABASE_URL
from .screener_strategies import SCREENER_STRATEGIES, COLUMN_TO_TABLE_MAP

# --- RAG INTEGRATION: Import the NEW intelligent context provider ---
from agents.rag_context_provider import get_intelligent_context
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any


# Custom module imports
from model_config import groq_llm,groq_llm_fast
from db_config import async_engine


load_dotenv()

# --- Pydantic Models for Structured LLM Outputs ---
class IdentifiedEntity(BaseModel):
    entity_name: str = Field(description="The name of the company/stock symbol identified.")


class KnownSchema(BaseModel):
    table: str = Field(description="The known table containing the required column.")
    column: str = Field(description="The known column that directly answers the query.")

class ExtractionOutput(BaseModel):
    is_database_required: bool = Field(description="Is database access required to answer the question?")
    is_recommendation_query: bool = Field(False, description="Set to true ONLY for recommendation or screener queries like 'suggest stocks', 'top 5 companies based on...', 'recommend me...'.")
    is_health_checklist_query: bool = Field(False, description="Set to true ONLY for broad, subjective questions like 'Is [company] strong?', 'Is it a good buy?', or 'fundamental analysis of [company]'.")
    is_multi_metric_query: bool = Field(False, description="Set to true if the user is asking for more than one specific metric for a company (e.g., 'P/E and Market Cap of...').")
    is_shareholding_query: bool = Field(False, description="Set to true ONLY for questions about shareholding patterns, promoters, FII, DII, or public holdings.")  
    entities: List[IdentifiedEntity] = Field(description="List of all unique financial entities found in the query.")
    tasks: Dict[str, Any] = Field(description="A dictionary mapping each entity_name (or 'General Query') to a concise description of the data needed for it.")
    known_schema_lookups: Optional[Dict[str, KnownSchema]] = Field(None, description="If a task can be answered by a single known column from the Golden Schema, map the task to its table and column here.")

class StrategySelection(BaseModel):
    strategy_key: str = Field(description="The chosen strategy key, e.g., 'QUALITY_INVESTING'.")
    reasoning: str = Field(description="A brief explanation for why this strategy was chosen.")

class GeneralizedQuery(BaseModel):
    generalized_question: str = Field(description="A generic question for RAG context retrieval, with specific entity names removed.")

class SQLQuery(BaseModel):
    # We make the query field optional at first, as we will populate it in the validator.
    query: Optional[str] = None
    
    # This validator runs after the initial fields are processed.
    @model_validator(mode='before')
    @classmethod
    def check_and_assign_query(cls, data: Any) -> Any:
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise ValueError('SQLQuery input must be a dictionary')
            
        # Check for 'sql' key and assign to 'query'
        if 'sql' in data:
            data['query'] = data.pop('sql')
        
        # If 'query' is still not in data after checking for 'sql', then it's a validation error.
        if 'query' not in data or not data['query']:
             raise ValueError("A 'query' or 'sql' key with a valid SQL string is required.")
             
        return data

# --- Agent State ---
class FundamentalsAgentState(TypedDict):
    question: str
    extraction: Optional[ExtractionOutput]
    selected_strategy_key: Optional[str]
    screener_results: Optional[Dict[str, Any]]
    generalized_tasks: Optional[Dict[str, str]]
    rag_contexts: Optional[Dict[str, str]]
    sql_queries: Optional[Dict[str, str]]
    query_results: Optional[Dict[str, str]]
    chart_data: Optional[Dict[str, List[Dict[str, Any]]]]
    error_message: Optional[str]
    final_answer: Dict[str, Any]

# --- Prompts (Overhauled for Robustness) ---

# This "Golden Schema" is a permanent guide for the LLM, reducing hallucination.
GOLDEN_SCHEMA_CONTEXT = """
    /*
    -- Golden Schema & High-Level Database Guide (v2) --

    1.  **Core Company Information Table:**
        -   Table: `company_master`
        -   Key Columns: `fincode` (Primary Key), `compname`, `scripcode`, `industry`.

    2.  **Date Column Rules (MANDATORY):**
        -   Use `year_end` for annual financial data in: `company_equity`, `company_finance_profitloss`, `company_finance_ratio`, `company_finance_cashflow`, `company_finance_cashflow_cons`.
        -   Use `date_end` for quarterly/periodic data in: `company_results`, `company_shareholding_pattern`.
        -   Use `date` for daily price data in: `bse_abjusted_price_eod`.

    3.  **Key Data Points Mapping (CRITICAL - Use this to find the correct table for a metric):**

        **Valuation & Size:**
        - Market Cap (`mcap`): in `company_equity`
        - Enterprise Value (`ev`): in `company_equity`
        - P/E Ratio (`ttmpe`): in `company_equity`
        - Price to Book Value (`price_bv`): in `company_equity`
        - Price to Sales (`price_sales`): in `company_equity`
        - EV to EBITDA (`ev_ebitda`): in `company_equity`
        - Dividend Yield (`dividend_yield`): in `company_equity`

        **Profitability & Performance:**
        - Net Sales / Revenue (`net_sales`): in `company_finance_profitloss`
        - Total Income (`total_income`): in `company_finance_profitloss`
        - Operating Profit (`operating_profit`): in `company_finance_profitloss`
        - Net Profit / Profit After Tax (`profit_after_tax`): in `company_finance_profitloss`
        - Earnings Per Share / EPS (`ttmeps`): in `company_equity`

        **Ownership:**
        - Promoter Shareholding (`tp_f_total_promoter`): in `company_shareholding_pattern`

        **Book Value & Other:**
        - Book Value per Share (`booknavpershare`): in `company_equity`
        - Face Value (`fv`): in `company_equity`

    4.  **Daily Pricing Data:**
        - For `open`, `high`, `low`, `close`, `volume`, you MUST use the `bse_abjusted_price_eod` table.

    5. **JOIN:**
        - Alwats use `fincode` column to join any tables.
    */
"""

# --- FIX: Incorporate Health Checklist identification into your original prompt ---
EXTRACT_PROMPT = """
    You are an expert financial data extraction assistant. Your most important task is to perform a Pre-Flight Check and categorize the user's intent.

    --- Golden Schema ---
    {golden_schema}
    ---
    
    **Intent Categorization Rules (in order of priority):**

    1. **Recommendation & Screener Query (`is_recommendation_query`):** HIGHEST PRIORITY. Check if the user is asking for a recommendation, suggestion, or a list of stocks based on criteria.
        - **Examples:** "recommend me some value stocks", "top 5 companies by market cap", "best stocks for beginners", "stocks with high ROCE and low debt"
        - If YES, you MUST set `is_recommendation_query` to `true`. This overrides all other categories.
        
    2. **Fundamental Health Check (`is_health_checklist_query`):** First, check if the user is asking a broad, subjective question about a company's overall health, strength, or a general "buy" recommendation.
        - **Examples:** "Is Reliance strong?", "Should I buy ABB India?", "fundamental analysis of Ambalal Sarabhai"
        - If YES, you MUST set `is_health_checklist_query` to `true`. This is the highest priority and overrides all other rules.

    2.  **Simple Lookup (`known_schema_lookups`):** If it is NOT a health check, then check if the question can be answered by a single known column in the Golden Schema.
        - **Examples:** "List all distinct industries", "What are the scrip codes?"
        - If YES, you MUST populate the `known_schema_lookups` field.

    3.  **Specific Metric Query (Default):** If it is neither of the above, it is a query for a specific metric.
        - **Example:** "Market cap of company ABC?"

    2.  **Multi-Metric Query (`is_multi_metric_query`):** If it's NOT a health check, check if the user is asking for **TWO OR MORE** specific metrics for a single company.
        - **Examples:** "What is the x, y and z of X?", "Give me the x, y, and y for X"
        - If YES, you MUST set `is_multi_metric_query` to `true`. The `tasks` for that entity MUST be a LIST of the individual metrics requested.

    4. **Shareholding Pattern Query (`is_shareholding_query`):** First, check if the question is specifically about the shareholding structure.
        - **Examples:** "shareholding pattern of Reliance", "Who are the major shareholders?", "promoter holding in ABB", "FII DII holding"
        - If YES, you MUST set `is_shareholding_query` to `true`. This is the highest priority.


    **Final JSON Output Rules:**
    - `is_database_required`: Set to true if financial data is needed.
    - `entities`: A list of all distinct company names or stock symbols.
    - `tasks`: A dictionary mapping each entity's name to a concise task description.

    --- EXAMPLES ---
    User Question: "Is Reliance Industries strong?"
    Your Output: {{"is_database_required": true, "is_health_checklist_query": true,"is_shareholding_query": false,"is_multi_metric_query": false, "entities": [{{"entity_name": "Reliance Industries"}}], "tasks": {{"Reliance Industries": "Fundamental health analysis for Reliance Industries"}}, "known_schema_lookups": null}}

    User Question: "List all distinct industries"
    Your Output: {{"is_database_required": true, "is_health_checklist_query": false,"is_shareholding_query": false, "is_multi_metric_query": false, "entities": [], "tasks": {{"General Query": "List all distinct industries from the database"}}, "known_schema_lookups": {{"General Query": {{"table": "company_master", "column": "industry"}}}}}}

    User Question: "What is the market cap of Company ABC?"
    Your Output: {{"is_database_required": true, "is_health_checklist_query": false,"is_shareholding_query": false,"is_multi_metric_query": false, "entities": [{{"entity_name": "Company ABC"}}], "tasks": {{"Company ABC": "market capitalization of Company ABC"}}, "known_schema_lookups": null}}

    User Question: "What is the x and y of Company X?"
    Your Output: {{"is_database_required": true, "is_health_checklist_query": false, "is_shareholding_query": false, "is_multi_metric_query": true, "entities": [{{"entity_name": "Company ABC"}}], "tasks": {{"Company ABC": ["market capitalization", "P/E ratio"]}}, "known_schema_lookups": null}}

    User Question: "What is the shareholding pattern for Reliance Industries?"
    Your Output: {{"is_database_required": true, "is_health_checklist_query": false, "is_shareholding_query": true, "is_multi_metric_query": false, "entities": [{{"entity_name": "Reliance Industries"}}], "tasks": {{"Reliance Industries": "Detailed shareholding pattern for Reliance Industries"}}, "known_schema_lookups": null}}

    ---
    User Question: {question}
    Output ONLY the valid JSON object.
"""

GENERALIZE_TASK_PROMPT = """
    You are a query transformation assistant. Rephrase a specific user request into a generic question for a RAG system. Remove all specific entity names.
    **If the user's task implies getting the 'latest' or 'most recent' data, you MUST include keywords like 'latest current date year'.**
    **If the user's task implies ranking (e.g., "top 5"), you MUST also include 'latest date year' to ensure the ranking is on current data.**

    --- EXAMPLES ---
    Specific Task: "latest market capitalization of a company"
    Your Output: {{"generalized_question": "tables and columns for latest current market capitalization mcap date year"}}

    Specific Task: "top 5 companies by market capitalization"
    Your Output: {{"generalized_question": "tables and columns for top 5 companies by latest market capitalization mcap date year"}}
    ---
    Specific Task: {specific_task}
    Your Output:
"""

# --- FINAL FIX: Incorporating the strict Golden Schema rule into your existing prompt ---
WRITE_QUERY_PROMPT = """
    You are a master SQL writer for a MySQL database. Your job is to write a single, syntactically correct SQL query. You must follow a strict thought process.

    --- SCHEMA CONTEXT ---
    -- **PRIMARY SOURCE OF TRUTH: Golden Schema** --
    -- This schema is ALWAYS correct. I will use it to determine which table contains a specific metric and what the correct date column is for that table.
    {golden_schema}

    -- **SECONDARY SOURCE: Dynamic RAG Schema** --
    -- This provides additional context. I will use it for column names, but if it conflicts with the Golden Schema, I will IGNORE the RAG context and trust the Golden Schema.
    {rag_context}
    --- END SCHEMA CONTEXT ---

    --- THOUGHT PROCESS (You MUST follow this sequence): ---

    1.  **Analyze the Task & Consult Golden Schema FIRST:**
        - The user wants: "{current_task}".
        - I will immediately look at the `Golden Schema` to identify the correct table and date column for the requested metrics.
        - **Example:** If the task is "latest closing price", I see in the Golden Schema that `close` is in `bse_abjusted_price_eod` and its date column is `date`. I will use ONLY this table and this date column, ignoring any conflicting suggestions from the RAG context.
        - The Golden Schema is my ultimate authority.

    2.  **CRITICAL DATE RULE:** I MUST use the exact date column mentioned in the Golden Schema for any given table.
        - For `company_equity` or `company_equity_cons`, the date column is `year_end`. I will NEVER use `date` for this table.
        - For `company_results` or `company_results_cons`, the date column is `date_end`.
        - For `bse_abjusted_price_eod`, the date column is `date`.

    3.  **CRITICAL JOIN RULE:** When joining tables, I MUST qualify all columns with their table alias (e.g., `cm.fincode`, `ce.mcap`) to prevent "ambiguous column" errors.

    4.  **Efficient Joins:** I will only JOIN tables if their columns are explicitly required by the RAG context OR if they are needed to fulfill the Golden Schema's instructions for a metric.

    5.  **Entity Filtering:** For a specific company like '{entity_name}', I MUST add a WHERE clause to filter by `fincode` using a subquery on `company_master`: `WHERE fincode = (SELECT fincode FROM company_master WHERE compname LIKE '%{entity_name}%' ORDER BY LENGTH(compname) ASC LIMIT 1)`.

    6.  **Ranking Queries:** For a 'General Query' (like "top 5"), I MUST NOT filter by a specific company name. I will `ORDER BY` the relevant metric and use `LIMIT`.

    7.  **"Latest" Data:** For a single entity, I will `ORDER BY [correct_date_column] DESC LIMIT 1`. For rankings, this is more complex and may require a subquery to find the max date per company.

    --- TASK ---
    Original Question: "{original_question}"
    Current Task: "{current_task}"
    Entity Name (if applicable): "{entity_name}"

    Now, following my rigorous thought process and trusting the Golden Schema above all else, I will write the SQL query as a single JSON object.
"""

ANSWER_COMPOSER_PROMPT = """
    You are IRIS, a sharp and confident financial analyst AI. Your goal is to synthesize raw data into a final, user-facing response formatted perfectly as Markdown.

    --- Data Collected ---
    Query Results (JSON format): {results_summary}
    ---

    **--- Part 1: Your Thought Process (for content) ---**
    1.  **Analyze Intent:** First, I will understand the user's original question: "{question}".
    2.  **Extract Key Data:** I will read the JSON `results_summary` to find all the core data points and verdicts.
    3. **Recommendation Logic:** If the results contain a `strategy_name`, this is a stock recommendation.
        -   First, state which strategy was used and why (using the `strategy_name` and `reasoning`).
        -   Then, present the `stocks` in a clean markdown table.
        -   Provide a brief concluding summary.
    4. **Synthesize Answer:** I will craft a direct, conversational answer that uses the data to address the user's specific intent. I will be concise and use simple language.
    5.  **Shareholding Logic:** If the results contain keys like 'Promoters', 'FIIs', 'DIIs', this is a shareholding breakdown. I MUST present this clearly in a bulleted list.
    6.  **Health Checklist Logic:** If the results contain multiple metrics like `market_cap`, `pe_ratio`, etc., this is a full "Fundamental Analysis". I MUST provide a multi-part answer: a summary, a detailed breakdown, and a final verdict.
    7.  **Single Metric Logic:** If the results contain only one or two metrics (e.g., just `mcap`), I will provide a simple, direct 1-2 sentence answer.
    8. **Comparison Logic:** If the user asked to "compare" and the JSON has data for multiple entities, I MUST present the data in a comparison format, ideally a markdown table.
    9.  **Handle Errors:** If a query for a company resulted in an error, I will state it simply. Example: "I couldn't retrieve data for Reliance Industries due to a data availability issue."


    **Part 2: CRITICAL RULES:**
    1.  **Be Direct:** Do NOT use fluff like "Here is the answer". Start the response directly.
    2.  **Indian Numbering:** Format large numbers using 'Lakhs' and 'Crores'.
    3.  **Use Full Names:** Use full company names from the results.
    4.  **Handle Lists:** If the result is a list (e.g., top 5 companies), format it as a clean, bulleted list.
    5.  **Handle Errors:** If a query for a company resulted in an error, state it simply. Example: "I couldn't retrieve the market cap for Reliance Industries due to a data availability issue."
    6.  **Combine Results:** If there are results for multiple companies, combine them into one natural sentence. Example: "The market cap for Reliance Industries is 19 Lakh Crores, while ABB India's latest sales are 31,595 Crores."

    7.  **NEW -> Synthesize Health Checklist:** If the query results contain multiple fundamental metrics (mcap, net_sales, pe_ratio, etc.), this is a health check. Do not just list the numbers. Synthesize them into a 2-3 sentence summary and conclude with a **Hold, Buy, or Sell** recommendation based on the data.
        - *Example:* "Reliance Industries shows strong fundamentals. It has a significant market cap of 19 Lakh Crores and healthy net sales. However, its P/E ratio is quite high, suggesting the price may be expensive. Based on these factors, the fundamental outlook is a **Hold**."

    --- PERFECT Comparison Example ---
    *If the user asked "Compare PE and market cap of X and Y"*
    Here is a comparison of the requested metrics for **X** and **Y**:

    | Metric              | X                | Y                |
    | ------------------- | ---------------- | ---------------- |
    | example1            | **25.5**         | **30.1**         |
    | example2            | **1,50,000 Cr.** | **95,000 Cr.**   |

    Now, applying BOTH your thought process and formatting rules, transform the internal JSON data into the final, perfect, user-facing markdown response.
"""

# --- Agent Nodes ---

async def extract_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Extract Entities & Tasks (with Pre-Flight Check)---")
    prompt = EXTRACT_PROMPT.format(question=state["question"], golden_schema=GOLDEN_SCHEMA_CONTEXT)
    structured_llm = groq_llm_fast.with_structured_output(ExtractionOutput, method="json_mode")
    try:
        result = await structured_llm.ainvoke(prompt)
        print(f"Extraction Result: {result.model_dump_json(indent=2)}")
        return {"extraction": result}
    except Exception as e:
        return {"error_message": f"Failed to understand the query's structure. Error: {e}"}

async def select_strategy_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Select Screener Strategy---")
    strategy_descriptions = "\n".join([f"- **{key}**: {details['name']} - {details['description']}" for key, details in SCREENER_STRATEGIES.items()])
    prompt = ChatPromptTemplate.from_template("Select the best strategy for the user's query.\n\n**Strategies:**\n{strategies}\n\n**User's Query:** \"{question}\"\n\nChoose the single best strategy key and provide your reasoning.")
    strategy_chain = prompt | groq_llm_fast.with_structured_output(StrategySelection)
    try:
        selection = await strategy_chain.ainvoke({"strategies": strategy_descriptions, "question": state["question"]})
        print(f"Strategy Selected: {selection.strategy_key}, Reason: {selection.reasoning}")
        return {"selected_strategy_key": selection.strategy_key, "query_results": {"recommendation": json.dumps({"reasoning": selection.reasoning})}}
    except Exception as e:
        return {"error_message": f"I had trouble determining a strategy: {e}"}


async def execute_screener_node(state: FundamentalsAgentState) -> Dict:
    """
    Deterministically builds and executes a SQL query based on the selected strategy.
    """
    print("---NODE: Execute Screener---")
    strategy_key = state.get("selected_strategy_key")
    if not strategy_key or strategy_key not in SCREENER_STRATEGIES:
        return {"error_message": "An invalid or no strategy was selected."}

    strategy = SCREENER_STRATEGIES[strategy_key]
    print(f"Executing strategy: {strategy['name']}")
    
    # --- MORE ROBUST SQL BUILDER USING THE IMPORTED MAP ---
    table_aliases = {"company_equity": "ce", "company_finance_ratio": "cfr"}
    
    select_clauses, join_clauses, where_clauses = set(), set(), []
    required_tables = set()

    # Determine required tables and build clauses
    for col, (op, val) in strategy["rules"].items():
        table = COLUMN_TO_TABLE_MAP.get(col)
        if not table:
            print(f"Warning: Column '{col}' not found in COLUMN_TO_TABLE_MAP. Skipping.")
            continue
        
        required_tables.add(table)
        alias = table_aliases[table]
        
        select_clauses.add(f"{alias}.{col}")
        where_clauses.append(f"{alias}.{col} {op} {val}")

    # Build join statements and latest-data clauses
    for table in required_tables:
        alias = table_aliases[table]
        join_clauses.add(f"JOIN {table} {alias} ON cm.fincode = {alias}.fincode")
        where_clauses.append(f"{alias}.year_end = (SELECT MAX(year_end) FROM {table} WHERE fincode = cm.fincode)")

    select_str = ", ".join(list(select_clauses))
    join_str = " ".join(list(join_clauses))
    where_str = " AND ".join(where_clauses)
    
    # Ensure there's always an ORDER BY, defaulting to mcap if company_equity is joined
    order_by_clause = "ORDER BY ce.mcap DESC" if "company_equity" in required_tables else ""

    full_query = f"SELECT cm.compname, {select_str} FROM company_master cm {join_str} WHERE {where_str} {order_by_clause} LIMIT 10"

    print(f"Generated Screener SQL: {full_query}")
    
    async_engine_instance = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    try:
        async with async_engine_instance.connect() as connection:
            result = await connection.execute(text(full_query))
            rows = result.mappings().all()
        
        existing_results = json.loads(state["query_results"]["recommendation"])
        existing_results.update({
            "strategy_name": strategy["name"],
            "stocks": [dict(row) for row in rows]
        })
        return {"query_results": {"recommendation": json.dumps(existing_results, default=str)}}
    except Exception as e:
        traceback.print_exc()
        return {"error_message": f"I encountered an error while searching for stocks with the selected strategy. It's possible the required data is not available for this combination. Error: {e!r}"}
    finally:
        await async_engine_instance.dispose()

async def execute_shareholding_pattern_node(state: FundamentalsAgentState) -> Dict:
    """
    Executes a comprehensive, pre-defined SQL query to get a detailed
    shareholding pattern, then performs all necessary calculations.
    """
    print("---NODE: Execute Shareholding Pattern (Hard-coded Tool)---")
    
    entities = state["extraction"].entities
    if not entities:
        return {"error_message": "A company name is required for shareholding analysis."}

    # This single, expanded query fetches all necessary raw percentage fields for a detailed breakdown.
    shareholding_query = text("""
        SELECT
            tp_f_total_promoter,
            tp_in_fii,
            tp_in_subtotal,
            tp_in_cgovt,
            tp_total_public,
            nh_grand_total,
            tp_in_mf_uti,
            tp_in_insurance,
            tp_in_fi_banks,
            tp_nin_indivd,
            tp_nin_body_corp,
            tp_nin_subtotal
        FROM company_shareholding_pattern
        WHERE fincode = :fincode
        ORDER BY date_end DESC
        LIMIT 1
    """)

    async_engine_instance = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    query_results = {}
    
    try:
        async with async_engine_instance.connect() as connection:
            for entity in entities:
                entity_name = entity.entity_name
                print(f"Executing detailed shareholding query for: {entity_name}")
                
                fincode_query = text("SELECT fincode FROM company_master WHERE compname LIKE :pattern LIMIT 1")
                fincode_result = await connection.execute(fincode_query, {"pattern": f"%{entity_name}%"})
                fincode = fincode_result.scalar_one_or_none()

                if not fincode:
                    result_str = json.dumps([{"error": f"Company '{entity_name}' not found."}])
                else:
                    result = await connection.execute(shareholding_query, {"fincode": fincode})
                    raw_data = result.mappings().one_or_none()
                    
                    if not raw_data:
                         result_str = json.dumps([{"error": f"No shareholding data found for '{entity_name}'."}])
                    else:
                        # Get all raw values, defaulting to 0 if None
                        promoters_perc = raw_data.get('tp_f_total_promoter') or 0
                        fii_perc = raw_data.get('tp_in_fii') or 0
                        total_inst_perc = raw_data.get('tp_in_subtotal') or 0
                        govt_perc = raw_data.get('tp_in_cgovt') or 0
                        total_public_perc = raw_data.get('tp_total_public') or 0
                        
                        # DII sub-components
                        mf_perc = raw_data.get('tp_in_mf_uti') or 0
                        ins_perc = raw_data.get('tp_in_insurance') or 0
                        bnk_perc = raw_data.get('tp_in_fi_banks') or 0

                        # Public Non-Institutional sub-components
                        retail_perc = raw_data.get('tp_nin_indivd') or 0
                        corp_bodies_perc = raw_data.get('tp_nin_body_corp') or 0
                        total_non_inst_perc = raw_data.get('tp_nin_subtotal') or 0

                        # --- Perform Comprehensive Calculations ---
                        # DIIs
                        total_dii_perc = total_inst_perc - fii_perc
                        other_dii_perc = total_dii_perc - mf_perc - ins_perc - bnk_perc

                        # Public
                        other_public_non_inst_perc = total_non_inst_perc - retail_perc - corp_bodies_perc

                        # Final calculated data dictionary for the composer node
                        calculated_data = {
                            "Promoters": round(promoters_perc, 2),
                            "Foreign Institutions (FIIs)": round(fii_perc, 2),
                            "Domestic Institutions (DIIs)": round(total_dii_perc, 2),
                            "DII - Mutual Funds": round(mf_perc, 2),
                            "DII - Insurance Co.": round(ins_perc, 2),
                            "DII - Banks & Fin. Inst.": round(bnk_perc, 2),
                            "DII - Others": round(other_dii_perc, 2),
                            "Government": round(govt_perc, 2),
                            "Public": round(total_public_perc - total_inst_perc - govt_perc, 2),
                            "Public - Individuals (Retail)": round(retail_perc, 2),
                            "Public - Corporate Bodies": round(corp_bodies_perc, 2),
                            "Public - Others": round(other_public_non_inst_perc, 2),
                            "Total Shareholders": raw_data.get('nh_grand_total') or 0
                        }
                        result_str = json.dumps([calculated_data])
                
                print(f"Result for '{entity_name}': {result_str}")
                query_results[entity_name] = result_str
        
        return {"query_results": query_results}

    except Exception as e:
        error_msg = f"Error executing shareholding query: {e!r}"
        print(error_msg)
        traceback.print_exc()
        return {"error_message": error_msg}
    finally:
        await async_engine_instance.dispose()



async def execute_multi_metric_node(state: FundamentalsAgentState) -> Dict:
    """
    Handles any query asking for one-or-more metrics for one-or-more companies.
    It flattens all tasks, runs them in parallel, and re-groups the results.
    """
    print("---NODE: Execute Multi-Entity, Multi-Metric Parallel Queries---")
    
    original_extraction = state["extraction"]
    
    # --- STEP 1: Flatten the tasks from all entities into a single list ---
    # The key will be "Entity Name - Metric Name" to keep them unique
    flattened_tasks = {}
    for entity_name, metrics in original_extraction.tasks.items():
        if isinstance(metrics, list):
            for metric in metrics:
                task_key = f"{entity_name} - {metric}"
                flattened_tasks[task_key] = f"latest {metric} for {entity_name}"
        else: # Handles cases where a single metric is not in a list
            task_key = f"{entity_name} - {metrics}"
            flattened_tasks[task_key] = f"latest {metrics} for {entity_name}"

    print(f"Flattened tasks for processing: {flattened_tasks}")

    # Create a temporary extraction object for the sub-graph nodes
    sub_extraction_object = ExtractionOutput(
        is_database_required=True,
        is_health_checklist_query=False,
        is_multi_metric_query=False, # We are now treating them as individual tasks
        is_shareholding_query=False,
        entities=original_extraction.entities, # Pass all entities for context
        tasks=flattened_tasks,
        known_schema_lookups={}
    )
    
    # --- STEP 2: Run the full RAG -> SQL -> Execute pipeline on the flattened tasks ---
    rag_input_state = {"extraction": sub_extraction_object}
    rag_contexts_result = await parallel_generalize_and_rag_node(rag_input_state)
    
    sql_writer_input_state = {
        "question": state["question"],
        "extraction": sub_extraction_object,
        "rag_contexts": rag_contexts_result["rag_contexts"]
    }
    sql_queries_result = await parallel_write_queries_node(sql_writer_input_state)
    
    execute_queries_input_state = {"sql_queries": sql_queries_result["sql_queries"]}
    query_results_result = await parallel_execute_queries_node(execute_queries_input_state)

    # --- STEP 3: Re-group the results by the original entity name ---
    final_query_results = {}
    for task_key, json_str_result in query_results_result["query_results"].items():
        try:
            # Split "Entity Name - Metric Name" to get the parts
            entity_name, metric_name = task_key.split(" - ", 1)
            
            # Initialize the dictionary for the entity if it doesn't exist
            if entity_name not in final_query_results:
                final_query_results[entity_name] = {}
                
            data = json.loads(json_str_result)
            if data and isinstance(data, list) and data[0] is not None and len(data[0]) > 0:
                # Get the first value from the first row of the result
                metric_value = list(data[0].values())[0]
                final_query_results[entity_name][metric_name] = metric_value
            else:
                final_query_results[entity_name][metric_name] = "N/A"
        except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError) as e:
            # Handle cases where splitting or parsing fails
            print(f"Error processing result for task '{task_key}': {e}")
            entity_name, metric_name = task_key.split(" - ", 1)
            if entity_name not in final_query_results:
                final_query_results[entity_name] = {}
            final_query_results[entity_name][metric_name] = "Error parsing result"

    # Convert the inner dictionaries to JSON strings for the composer node
    # Final structure: {"Reliance Industries": "[{...}]", "ABB India": "[{...}]"}
    final_results_for_composer = {
        entity: json.dumps([data]) for entity, data in final_query_results.items()
    }
    
    return {"query_results": final_results_for_composer}

# --- FINAL FIX: Replace the entire execute_health_checklist_node function with this one ---
async def execute_health_checklist_node(state: FundamentalsAgentState) -> Dict:
    """
    Executes a pre-defined, robust SQL query to get a comprehensive view of a
    company's fundamental health, using correlated subqueries for reliability.
    This version includes robust state handling and logging.
    """
    print("---NODE: Execute Fundamental Health Checklist (Hard-coded Tool)---")
    
    entities = state["extraction"].entities
    if not entities:
        return {"error_message": "A company name is required for a fundamental health check."}

    health_check_query_template = text("""
        SELECT
            (SELECT compname FROM company_master WHERE fincode = :fincode) AS compname,
            (SELECT close FROM bse_abjusted_price_eod WHERE fincode = :fincode ORDER BY date DESC LIMIT 1) AS latest_close_price,
            (SELECT mcap FROM company_equity WHERE fincode = :fincode ORDER BY year_end DESC LIMIT 1) AS market_cap,
            (SELECT ttmpe FROM company_equity WHERE fincode = :fincode ORDER BY year_end DESC LIMIT 1) AS pe_ratio,
            (SELECT price_bv FROM company_equity WHERE fincode = :fincode ORDER BY year_end DESC LIMIT 1) AS price_to_book_value,
            (SELECT ev_ebitda FROM company_equity WHERE fincode = :fincode ORDER BY year_end DESC LIMIT 1) AS ev_to_ebitda,
            (SELECT ttmeps FROM company_equity WHERE fincode = :fincode ORDER BY year_end DESC LIMIT 1) AS earnings_per_share,
            (SELECT net_sales FROM company_finance_profitloss WHERE fincode = :fincode ORDER BY year_end DESC LIMIT 1) AS net_sales,
            (SELECT profit_after_tax FROM company_finance_profitloss WHERE fincode = :fincode ORDER BY year_end DESC LIMIT 1) AS profit_after_tax,
            (SELECT operating_profit FROM company_finance_profitloss WHERE fincode = :fincode ORDER BY year_end DESC LIMIT 1) AS operating_profit,
            (SELECT tp_f_total_promoter FROM company_shareholding_pattern WHERE fincode = :fincode ORDER BY date_end DESC LIMIT 1) AS promoter_shareholding
    """)

    async_engine_instance = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    query_results = {}
    
    try:
        async with async_engine_instance.connect() as connection:
            for entity in entities:
                entity_name = entity.entity_name
                print(f"Executing comprehensive health check for: {entity_name}")
                
                fincode_query = text("SELECT fincode FROM company_master WHERE compname LIKE :company_name_pattern LIMIT 1")
                fincode_result = await connection.execute(fincode_query, {"company_name_pattern": f"%{entity_name}%"})
                fincode = fincode_result.scalar_one_or_none()

                if not fincode:
                    print(f"Could not find fincode for {entity_name}")
                    result_str = json.dumps([{"error": f"Company '{entity_name}' not found."}])
                else:
                    print(f"Found fincode {fincode} for {entity_name}. Executing main query.")
                    result = await connection.execute(health_check_query_template, {"fincode": fincode})
                    rows = result.mappings().all()
                    result_str = json.dumps([dict(row) for row in rows], default=str) if rows else "[]"
                
                print(f"Result for '{entity_name}': {result_str[:400]}...")
                query_results[entity_name] = result_str
        
        # This is the crucial part: return the dictionary to update the graph's state.
        return {"query_results": query_results}

    except Exception as e:
        error_msg = f"Error executing health check query: {e!r}"
        print(error_msg)
        return {"error_message": error_msg}
    finally:
        print("--- Disposing of temporary database engine ---")
        await async_engine_instance.dispose()


async def get_price_history_for_chart_node(state: FundamentalsAgentState) -> Dict:
    """
    Fetches the last year of daily price history for each entity.
    This data is specifically for rendering a UI chart on the frontend.
    """
    print("---NODE: Get Price History for UI Chart---")
    entities = state["extraction"].entities
    if not entities:
        # This node should only run if there are entities, but as a safeguard:
        return {}

    # Fetches the last year of trading data (approx. 252 days)
    price_history_query = text("""
        SELECT
            date, `open`, high, low, `close`, volume
        FROM bse_abjusted_price_eod
        WHERE fincode = :fincode
        ORDER BY date DESC
        LIMIT 252
    """)

    async_engine_instance = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    chart_data_results = {}

    try:
        async with async_engine_instance.connect() as connection:
            for entity in entities:
                entity_name = entity.entity_name
                print(f"Fetching price history for: {entity_name}")

                fincode_query = text("SELECT fincode FROM company_master WHERE compname LIKE :company_name_pattern LIMIT 1")
                fincode_result = await connection.execute(fincode_query, {"company_name_pattern": f"%{entity_name}%"})
                fincode = fincode_result.scalar_one_or_none()

                if fincode:
                    result = await connection.execute(price_history_query, {"fincode": fincode})
                    # Convert rows to a list of dicts, format date to string for JSON
                    rows = [
                        {**row, "date": row["date"].isoformat()}
                        for row in result.mappings().all()
                    ]
                    # The data should be chronological for charting
                    chart_data_results[entity_name] = rows[::-1]
                else:
                    chart_data_results[entity_name] = [] # No data found

        return {"chart_data": chart_data_results}

    except Exception as e:
        error_msg = f"Error fetching price history data: {e!r}"
        print(error_msg)
        # We can return an empty dict or add to an error log in the state
        return {"chart_data": {}}
    finally:
        await async_engine_instance.dispose()


async def parallel_generalize_and_rag_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Parallel Generalize & RAG---")
    tasks = state["extraction"].tasks
    known_lookups = state["extraction"].known_schema_lookups or {}

    async def _generalize_and_fetch(task_key: str, specific_task: str):
        if task_key in known_lookups:
            known_info = known_lookups[task_key]
            print(f"Pre-Flight Check successful for task '{specific_task}'. Bypassing RAG.")
            perfect_context = f"Table: {known_info.table}\nRequired Columns:\n- {known_info.column}: The column that directly answers the query."
            return task_key, "N/A (Known Schema)", perfect_context
        
        print(f"Pre-Flight Check failed for task '{specific_task}'. Proceeding with RAG.")
        try:
            gen_prompt = GENERALIZE_TASK_PROMPT.format(specific_task=specific_task)
            structured_llm = groq_llm_fast.with_structured_output(GeneralizedQuery)
            gen_result = await structured_llm.ainvoke(gen_prompt)
            generalized_question = gen_result.generalized_question
            print(f"Task '{specific_task}' ==> Generalized: '{generalized_question}'")
            rag_context = await get_intelligent_context(generalized_question)
            return task_key, generalized_question, rag_context
        except Exception as e:
            print(f"Error processing task '{task_key}': {e}")
            return task_key, None, f"Error retrieving context for task: {specific_task}"

    coroutines = [_generalize_and_fetch(key, task) for key, task in tasks.items()]
    results = await asyncio.gather(*coroutines)
    generalized_tasks = {key: gen_q for key, gen_q, _ in results if gen_q}
    rag_contexts = {key: rag_c for key, _, rag_c in results if rag_c}
    return {"generalized_tasks": generalized_tasks, "rag_contexts": rag_contexts}

# The rest of the nodes and graph definition remain the same as the previous correct version
async def parallel_write_queries_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Parallel Write SQL Queries---")
    tasks = state["extraction"].tasks
    rag_contexts = state["rag_contexts"]
    question = state["question"]
    async def _write_one_query(task_key: str, task_desc: str):
        rag_context = rag_contexts.get(task_key, "No specific context found.")
        prompt = WRITE_QUERY_PROMPT.format(
            original_question=question,
            current_task=task_desc,
            entity_name=task_key,
            golden_schema=GOLDEN_SCHEMA_CONTEXT,
            rag_context=rag_context
        )
        structured_llm = groq_llm.with_structured_output(SQLQuery, method="json_mode")
        try:
            result = await structured_llm.ainvoke(prompt)
            print(f"Generated SQL for '{task_key}': {result.query}")
            return task_key, result.query
        except Exception as e:
            error_msg = f"Error generating SQL for task '{task_desc}': {e}"
            print(error_msg)
            return task_key, f"/* {error_msg} */ SELECT 'Error: Could not generate a valid SQL query.';"
            
    coroutines = [_write_one_query(key, task) for key, task in tasks.items()]
    results = await asyncio.gather(*coroutines)
    return {"sql_queries": dict(results)}


async def parallel_execute_queries_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Parallel Execute SQL Queries---")
    
    # R-NOTE: The engine is now created *inside* the function. This is the core fix.
    # It ensures the engine and its connections belong to the currently active event loop.
    async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    
    queries = state["sql_queries"]

    async def _execute_one_query(task_key: str, query: str):
        if "Error" in query:
            return task_key, f'{{"error": "Query generation failed.", "details": "{json.dumps(query)}"}}'
        
        try:
            # Use the locally created engine. This connection is now safe.
            async with async_engine.connect() as connection:
                result = await connection.execute(text(query))
                # Use mappings() for direct dictionary conversion, which is cleaner.
                rows = result.mappings().all()
            
            result_str = json.dumps([dict(row) for row in rows], default=str) if rows else "[]"
            print(f"Result for '{task_key}': {result_str[:250]}...")
            return task_key, result_str
        except Exception as e:
            error_msg = f"Error executing SQL for task '{task_key}': {e!r}"
            print(error_msg)
            # Ensure the error detail is a valid JSON string
            return task_key, f'{{"error": "Failed to execute query.", "details": "{json.dumps(error_msg)}"}}'

    # The rest of the logic can stay the same
    coroutines = [_execute_one_query(key, q) for key, q in queries.items()]
    
    try:
        results = await asyncio.gather(*coroutines)
        return {"query_results": dict(results)}
    finally:
        # R-NOTE: This is CRITICAL. We must dispose of the engine and its pool
        # before the temporary event loop from asyncio.run() closes.
        # This prevents the "Event loop is closed" errors during garbage collection.
        print("--- Disposing of temporary database engine ---")
        await async_engine.dispose()


async def compose_final_answer_node(state: FundamentalsAgentState) -> Dict:
    """
    Assembles the final structured response. It generates the text summary using an LLM
    and then intelligently packages it with any relevant UI components based on the data shape.
    """
    print("---NODE: Compose Final Structured Answer---")
    
    if state.get("error_message"):
        return {"final_answer": {"text_response": state["error_message"], "ui_components": []}}

    # --- NEW LOGIC TO PREPARE UI COMPONENTS ---
    ui_components = []
    query_results = state.get("query_results", {})
    
    # Check for chart data from the health check path
    chart_data = state.get("chart_data")
    if chart_data:
        for entity_name, data_points in chart_data.items():
            if data_points:
                ui_components.append({
                    "type": "stock_price_chart",
                    "title": f"{entity_name} Price History",
                    "data": data_points
                })


    # --- NEW: Check for shareholding data and create a pie chart ---
    if not ui_components and query_results:
        first_result_key = next(iter(query_results))
        try:
            first_result_data = json.loads(query_results[first_result_key])
            # Check for the unique 'Promoters' key to identify shareholding data
            if first_result_data and "Promoters" in first_result_data[0]:
                print("--- Shareholding data detected. Creating Pie Chart. ---")
                shareholding_data = first_result_data[0]
                # We only chart the top-level categories for a clean pie chart
                pie_chart_data = [
                    {"name": "Promoters", "value": shareholding_data.get("Promoters", 0)},
                    {"name": "FIIs", "value": shareholding_data.get("Foreign Institutions (FIIs)", 0)},
                    {"name": "DIIs", "value": shareholding_data.get("Domestic Institutions (DIIs)", 0)},
                    {"name": "Government", "value": shareholding_data.get("Government", 0)},
                    {"name": "Public", "value": shareholding_data.get("Public", 0)},
                ]
                # Filter out zero-value slices
                pie_chart_data = [item for item in pie_chart_data if item["value"] > 0]
                
                ui_components.append({
                    "type": "pie_chart",
                    "title": f"Shareholding Pattern for {first_result_key}",
                    "data": pie_chart_data
                })
        except (json.JSONDecodeError, IndexError, KeyError):
            pass # Data wasn't shareholding data, so we do nothing.

    # Check for data suitable for a ranking bar chart
    if not ui_components and "General Query" in query_results:
        try:
            ranking_data = json.loads(query_results["General Query"])
            # Check if it's a list of dictionaries with at least 2 keys (a name and a value)
            if isinstance(ranking_data, list) and all(isinstance(i, dict) and len(i.keys()) >= 2 for i in ranking_data):
                print("--- Data suitable for Ranking Bar Chart detected ---")
                # Identify the label (compname) and the value (the other key)
                keys = list(ranking_data[0].keys())
                label_key = 'compname' if 'compname' in keys else keys[0]
                value_key = next((k for k in keys if k != label_key), keys[1])

                ui_components.append({
                    "type": "ranking_bar_chart",
                    "title": f"Top {len(ranking_data)} by {value_key.replace('_', ' ').title()}",
                    "data": ranking_data,
                    "labelKey": label_key,
                    "valueKey": value_key,
                })
        except (json.JSONDecodeError, IndexError):
            print("Could not parse General Query result for bar chart.")

    # --- Generate the text-based summary using the LLM (existing logic) ---
    results_summary = json.dumps(query_results) if query_results else "No database results."
    
    prompt = ANSWER_COMPOSER_PROMPT.format(
        question=state["question"],
        results_summary=results_summary
    )
    llm_response = await groq_llm.ainvoke(prompt)
    text_answer = llm_response.content
    
    # --- Combine into a single final answer object ---
    final_answer_object = {
        "text_response": text_answer,
        "ui_components": ui_components
    }

    return {"final_answer": final_answer_object}

def route_after_extraction(state: FundamentalsAgentState) -> str:
    """
    Decides the path after the initial query extraction.
    - If it's a health check, go to the hard-coded tool.
    - If multi-metric -> multi_metric_path (NEW)
    - If DB is not needed, go straight to the answer.
    - Otherwise, go to the RAG path.
    """
    if state.get("error_message"): return "end"
    
    extraction = state["extraction"]

    if extraction.is_recommendation_query:
        print("Routing to: Recommendation Path")
        return "recommendation_path"
    if extraction.is_shareholding_query:
        print("Routing to: Shareholding Pattern Tool")
        return "shareholding_path"
    if extraction.is_health_checklist_query:
        print("Routing to: Health Checklist Tool")
        return "health_check_path"
    if extraction.is_multi_metric_query:
        print("Routing to: Multi-Metric Parallel Path")
        return "multi_metric_path"
    if extraction.is_database_required:
        print("Routing to: RAG Path for Specific Metrics")
        return "rag_path"
    else:
        print("Routing to: General Path (No DB required)")
        return "general_path"

graph_builder = StateGraph(FundamentalsAgentState)

# Add all nodes, including our new one
graph_builder.add_node("extract", extract_node)
graph_builder.add_node("select_strategy", select_strategy_node)
graph_builder.add_node("execute_screener", execute_screener_node)
graph_builder.add_node("execute_shareholding_pattern", execute_shareholding_pattern_node)
graph_builder.add_node("generalize_and_rag", parallel_generalize_and_rag_node)
graph_builder.add_node("write_queries", parallel_write_queries_node)
graph_builder.add_node("execute_queries", parallel_execute_queries_node)
graph_builder.add_node("execute_health_checklist", execute_health_checklist_node)
# --- NEW NODE ADDED TO GRAPH ---
graph_builder.add_node("get_price_history", get_price_history_for_chart_node)
graph_builder.add_node("compose_answer", compose_final_answer_node)

graph_builder.add_node("execute_multi_metric", execute_multi_metric_node)

# EDGES
graph_builder.set_entry_point("extract")

graph_builder.add_conditional_edges(
    "extract",
    route_after_extraction,
    {
        "recommendation_path": "select_strategy", 
        "shareholding_path": "execute_shareholding_pattern", # NEW ROUTE
        "health_check_path": "execute_health_checklist", 
        "multi_metric_path": "execute_multi_metric", 
        "rag_path": "generalize_and_rag", 
        "general_path": "compose_answer", 
        "end": END
    }
)

graph_builder.add_edge("select_strategy", "execute_screener")
graph_builder.add_edge("execute_screener", "compose_answer")

graph_builder.add_edge("execute_shareholding_pattern", "compose_answer")

# --- MODIFIED EDGES FOR THE HEALTH CHECK PATH ---
# After fetching health metrics, now fetch the price history for the chart
graph_builder.add_edge("execute_health_checklist", "get_price_history")

graph_builder.add_edge("execute_multi_metric", "compose_answer")

# After fetching chart data, compose the final structured answer
graph_builder.add_edge("get_price_history", "compose_answer")


# Edges for the standard RAG path remain the same
graph_builder.add_edge("generalize_and_rag", "write_queries")
graph_builder.add_edge("write_queries", "execute_queries")
graph_builder.add_edge("execute_queries", "compose_answer")

# Final edge to the end
graph_builder.add_edge("compose_answer", END)

app = graph_builder.compile()



# --- Main Execution for Testing ---
if __name__ == "__main__":
    async def run_funda_agent_test():
        test_queries = [
            "What are the market capitalization, current price, high/low, dividend yield, P/E ratio, book value, return on capital employed (ROCE), and return on equity (ROE) of Reliance Industries?",
            "What is the shareholding pattern of Reliance Industries?",
            "Show me the promoter vs public holding for ABB India",
            # "Is Reliance strong based on Fundamental Analysis?",
            # "What is the market cap of Reliance Industries and the latest sales for ABB India?",
            # "What is the business description of Ambalal Sarabhai?",
            # "What are the top 5 companies by market cap?",
            # "Compare the market cap of Reliance Industries and Ambalal Sarabhai",
            # "Scripcode and fincode of ABB India",
            # "What is the latest promoter shareholding percentage for Reliance Industries?",
            # "List all distinct industries",
            # "What are the top 5 companies by market capitalization?"       

        ]
        for q_text in test_queries:
            print(f"\n\n{'='*20} TESTING: \"{q_text}\" {'='*20}")
            inputs = {"question": q_text}
            try:
                final_state = await app.ainvoke(inputs)
                print(f"\n>>> FINAL ANSWER:\n{final_state.get('final_answer', 'No answer found.')}")
            except Exception as e:
                print(f"\nError during agent execution: {e}")
                import traceback
                traceback.print_exc()

    asyncio.run(run_funda_agent_test())