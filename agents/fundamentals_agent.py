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
from .industry_mapper import INDUSTRY_LIST

# --- RAG INTEGRATION: Import the NEW intelligent context provider ---
from agents.rag_context_provider import get_intelligent_context
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any

from .technicals_agent import app as technicals_app_instance
from .sentiment_agent import app as sentiment_app_instance

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
    is_performance_analysis_query: bool = Field(False, description="Set true for time-based performance analysis.")
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

# NEW: Pydantic model for the agent to extract its own parameters
class PerformanceAnalysisParams(BaseModel):
    time_period_years: Optional[int] = Field(None, description="The time period in years, if mentioned.")
    sector: Optional[str] = Field(None, description="The industry sector, if mentioned.")

class SupervisorDecision(BaseModel):
    route: str
    reasoning: str
    time_period_years: Optional[int] = None
    sector: Optional[str] = None

# --- Agent State ---
class FundamentalsAgentState(TypedDict):
    question: str
    supervisor_decision: SupervisorDecision
    mapped_sector: Optional[str]
    performance_params: Optional[PerformanceAnalysisParams]
    extraction: Optional[ExtractionOutput]
    selected_strategy_key: Optional[str]
    screener_results: Optional[Dict[str, Any]]
    fundamental_screener_results: Optional[List[Dict]]
    performance_analysis_results: Optional[List[Dict]]
    generalized_tasks: Optional[Dict[str, str]]
    rag_contexts: Optional[Dict[str, str]]
    sql_queries: Optional[Dict[str, str]]
    query_results: Optional[Dict[str, str]]
    chart_data: Optional[Dict[str, List[Dict[str, Any]]]]
    error_message: Optional[str]
    final_answer: Dict[str, Any]

class MappedIndustry(BaseModel):
    mapped_industry_name: Optional[str] = Field(None, description="The single best matching industry name from the provided list, or null if no confident match is found.")

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

EXTRACT_PROMPT = """
    You are an expert financial data extraction assistant. Your most important task is to perform a Pre-Flight Check and categorize the user's intent based on the rules below.

    --- Golden Schema ---
    {golden_schema}
    ---
    
    **Intent Categorization Rules (in strict order of priority):**

    1.  **Performance Analysis (`is_performance_analysis_query`): HIGHEST PRIORITY.**
        - Does the query ask for "best performing", "most consistent", "top stocks over time", etc., AND explicitly mention a **time period** (e.g., "last 5 years", "since 2020")?
        - If YES, you MUST set `is_performance_analysis_query` to `true`. This overrides all other rules. Do NOT extract entities or tasks for this route.

    2.  **Recommendation / Screener (`is_recommendation_query`):**
        - If it's NOT a performance query, does it ask for a recommendation, suggestion, or a list of stocks based on criteria (e.g., "recommend value stocks", "top 5 by P/E")?
        - If YES, set `is_recommendation_query` to `true`.

    3.  **Shareholding (`is_shareholding_query`):**
        - If not the above, does it ask specifically about shareholding structure (promoter, FII, DII, etc.)?
        - If YES, set `is_shareholding_query` to `true`.

    4.  **Health Checklist (`is_health_checklist_query`): HIGHEST PRIORITY.**
        -   Does the query ask for "fundamental analysis of", "fundamentals of", "health of", "is [company] strong?", or to "compare the fundamentals of" one or more companies?
        -   These phrases are a DIRECT command to use the comprehensive health checklist tool.
        -   If YES, you MUST set `is_health_checklist_query` to `true`. This overrides all other rules. You MUST also extract all company names into the `entities` list.

    5.  **Multi-Metric Query (`is_multi_metric_query`):**
        -   Is the user asking for a **LIST or RANKING** of stocks based on a **single, specific, current metric** (e.g., "top 5 by X", "companies with highest X")?
        -   If YES, you **MUST** set `is_multi_metric_query` to `false`. This is not a multi-metric query.
        -   The key in the `tasks` dictionary **MUST** be `"General Query"`.
        -   The value for the task **MUST** be a single, descriptive **STRING**.

    6. **Simple Ranking Query:**
        -   Is the user asking for a **LIST or RANKING** of stocks based on a **single, specific, current metric** (e.g., "top y by X", "companies with highest X")?
        -   If YES, you MUST set `is_multi_metric_query` to `false`.
        -   The key in the `tasks` dictionary MUST be `"General Query"`.
        -   The value for the task MUST be a single, descriptive **STRING**.

    7.  **Specific Metric / Simple Lookup (Default):**
        - If NONE of the above match, it is a simple query for a single metric.
        - You should extract the `entities` and `tasks`.
        - If possible, populate `known_schema_lookups`.

    **Final JSON Output Rules:**
    - `is_database_required`: Set to true if financial data is needed.
    - For all routes EXCEPT `performance_analysis` and `recommendation`, you should extract `entities` and `tasks`. For those two routes, you can leave them as empty lists/dicts.

    --- EXAMPLES ---
    User Question: "What were the top performing IT stocks over the last 3 years?"
    Your Output: {{"is_database_required": true, "is_performance_analysis_query": true, "is_recommendation_query": false, "is_health_checklist_query": false, "is_multi_metric_query": false, "is_shareholding_query": false, "entities": [], "tasks": {{}}, "known_schema_lookups": null}}

    User Question: "Recommend some good value stocks."
    Your Output: {{"is_database_required": true, "is_performance_analysis_query": false, "is_recommendation_query": true, "is_health_checklist_query": false, "is_multi_metric_query": false, "is_shareholding_query": false, "entities": [], "tasks": {{}}, "known_schema_lookups": null}}
    
    User Question: "What is the X and Y of Company ABC?"
    Your Output: {{"is_database_required": true, "is_performance_analysis_query": false, "is_recommendation_query": false, "is_health_checklist_query": false, "is_multi_metric_query": true, "is_shareholding_query": false, "entities": [{{"entity_name": "Company ABC"}}], "tasks": {{"Company ABC": ["X", "Y"]}}, "known_schema_lookups": null}}

    User Question: "What are the top y companies by X?"
    Your Output: {{"is_database_required": true, "is_recommendation_query": false, "is_performance_analysis_query": false, "is_health_checklist_query": false, "is_multi_metric_query": false, "is_shareholding_query": false, "entities": [], "tasks": {{"General Query": "top y companies by X"}}, "known_schema_lookups": null}}

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

WRITE_QUERY_PROMPT = """
    You are a master SQL writer for a MySQL database. Your job is to write a single, syntactically correct SQL query. You must follow a strict thought process.

    --- SCHEMA CONTEXT ---
    -- **PRIMARY SOURCE OF TRUTH: Golden Schema** --
    {golden_schema}

    -- **SECONDARY SOURCE: Dynamic RAG Schema** --
    {rag_context}
    --- END SCHEMA CONTEXT ---

    --- THOUGHT PROCESS (You MUST follow this sequence): ---

    1.  **Analyze the Task & Consult Golden Schema FIRST:**
        - The user wants: "{current_task}".
        - I will immediately look at the `Golden Schema` to identify the correct table for the requested metrics. The Golden Schema is my ultimate authority.

    2.  **CRITICAL DATE RULE:** I MUST use the exact date column mentioned in the Golden Schema for any given table (`year_end` for `company_equity`, `date_end` for `company_shareholding_pattern`, etc.).

    3.  **CRITICAL JOIN RULE:** When joining tables, I MUST qualify all columns with their table alias (e.g., `cm.fincode`, `ce.mcap`).

    4.  **HANDLE ENTITY TYPES (THIS IS THE FIX):**
        -   If `Entity Name` is a specific company (e.g., 'HDFC Bank', 'Reliance Industries'), I MUST add a WHERE clause to filter by `compname` on the `company_master` table.
        -   If `Entity Name` is a special key like 'General Query' OR if it describes a sector (e.g., 'bank sector'), this is a RANKING query. I MUST NOT filter by a specific company name. Instead, if a sector is mentioned in the task, I will add a `WHERE cm.industry LIKE '%[sector]%'` clause.
        -   I will then ORDER BY the relevant metric and apply a LIMIT if specified in the task.

    5.  **"Latest" Data:** For rankings, I must construct a query that finds the latest data for EACH company, often using a subquery or a window function. A simple `ORDER BY year_end DESC LIMIT 1` is wrong for rankings. The correct way is often to join with a subquery like `JOIN (SELECT fincode, MAX(year_end) as max_year FROM company_equity GROUP BY fincode) latest ON ce.fincode = latest.fincode AND ce.year_end = latest.max_year`.

    --- TASK ---
    Original Question: "{original_question}"
    Current Task: "{current_task}"
    Entity Name (if applicable): "{entity_name}"

    NOTE: Whenever trying to match something always use wildcards and LIKE clause instead of direct matching (e.g xyz LIKE "%xyz%")

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
    3.  **Performance Analysis Logic (CRITICAL):** If the results contain a `methodology` and a list of `stocks`, this is a Performance Analysis query. I MUST follow these steps:
        -   State the methodology clearly.
        -   Create a comprehensive markdown table. The table header MUST include these exact columns in this order: `Company Name`, `Performance Score`, `Funda Score`, `Tech Score`, `Avg ROCE (%)`, `Price CAGR (%)`, `Volatility (%)`, and `Sharpe Ratio`.
        -   Populate the table with the data from the `stocks` list.
        -   Write a concluding "Summary" section that highlights the top 1-2 stocks and mentions their key scores (Performance, Funda, and Tech) to justify the ranking.
    4. **Recommendation Logic:** If the results contain a `strategy_name`, this is a stock recommendation.
        -   First, state which strategy was used and why (using the `strategy_name` and `reasoning`).
        -   Then, present the `stocks` in a clean markdown table.
        -   Provide a brief concluding summary.
    5. **Synthesize Answer:** I will craft a direct, conversational answer that uses the data to address the user's specific intent. I will be concise and use simple language.
    6.  **Shareholding Logic:** If the results contain keys like 'Promoters', 'FIIs', 'DIIs', this is a shareholding breakdown. I MUST present this clearly in a bulleted list.
    7.  **Health Checklist Logic:** If the results contain multiple metrics like `market_cap`, `pe_ratio`, etc., this is a full "Fundamental Analysis". I MUST provide a multi-part answer: a summary, a detailed breakdown, and a final verdict.
    8.  **Single Metric Logic:** If the results contain only one or two metrics (e.g., just `mcap`), I will provide a simple, direct 1-2 sentence answer.
    9. **Comparison Logic:** If the user asked to "compare" and the JSON has data for multiple entities, I MUST present the data in a comparison format, ideally a markdown table.
    10.  **Handle Errors:** If a query for a company resulted in an error, I will state it simply. Example: "I couldn't retrieve data for Reliance Industries due to a data availability issue."


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

INDUSTRY_MAPPER_PROMPT = """
    Your task is to be a precise data mapper. Given a user's potentially vague or misspelled "sector" input, you must find the single best match from the provided list of "Canonical Industries".

    **Canonical Industries List:**
    {industry_list}

    **User's Sector Input:** "{user_sector_input}"

    **Rules:**
    1.  **Exact Match Priority:** If the user's input exactly matches an item in the list (case-insensitive), return that item.
    2.  **Abbreviation Mapping:** Map common abbreviations (e.g., "IT" -> "IT - Software", "NBFC" -> "Finance - NBFC").
    3.  **Best Semantic Match:** If no exact match, find the closest semantic fit. For "Pharma", the best match is "Pharmaceuticals & Drugs". For "Real Estate", the best is "Construction - Real Estate".
    4.  **Handle Ambiguity:** If the user says "Finance", a good default is "Finance - Others", but if they say "Housing Finance", the specific "Finance - Housing" is better.
    5.  **No Confident Match:** If the input is completely unrelated (e.g., "Food Delivery") and has no clear match in the list, you MUST return null.

    Output ONLY the valid JSON object.
"""

# --- Agent Nodes ---

async def map_industry_and_params_node(state: FundamentalsAgentState) -> Dict:
    """
    Extracts time period and sector from the query and maps the sector to a canonical name.
    """
    print(f"---NODE: Map Industry & Extract Params---")
    
    # LLM call to extract parameters
    param_extractor_prompt = f"From the user query '{state['question']}', extract the time period in years and the industry sector."
    extractor_llm = groq_llm_fast.with_structured_output(PerformanceAnalysisParams)
    params = await extractor_llm.ainvoke(param_extractor_prompt)
    
    # LLM call to map the extracted sector
    user_sector = params.sector
    if not user_sector:
        print("No sector provided, skipping mapping.")
        return {"performance_params": params.model_dump()}

    mapper_prompt = INDUSTRY_MAPPER_PROMPT.format(
        industry_list=json.dumps(INDUSTRY_LIST),
        user_sector_input=user_sector
    )
    mapper_chain = groq_llm_fast.with_structured_output(MappedIndustry)
    mapped_result = await mapper_chain.ainvoke(mapper_prompt)
    mapped_name = mapped_result.mapped_industry_name
    
    print(f"Mapped sector '{user_sector}' to '{mapped_name}'")
    return {"performance_params": params.model_dump(), "mapped_sector": mapped_name}

def entry_point_node(state: FundamentalsAgentState) -> Dict:
    """
    Initializes the state from the input payload, ensuring the supervisor's
    decision is correctly passed into the graph's state.
    """
    print("---NODE: Fundamentals Agent Entry Point---")
    # This is the crucial step. It takes the `supervisor_decision` from the initial
    # input `state` and ensures it's part of the dictionary that gets passed along.
    if "supervisor_decision" not in state:
        raise ValueError("Supervisor decision must be provided in the input payload.")
    
    return {
        "question": state["question"],
        "supervisor_decision": state["supervisor_decision"]
    }

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
    
    # --- SQL Builder Logic (this part is correct and remains unchanged) ---
    table_aliases = {"company_equity": "ce", "company_finance_ratio": "cfr"}
    select_clauses, join_clauses, where_clauses = set(), set(), []
    required_tables = set()

    for col, (op, val) in strategy["rules"].items():
        table = COLUMN_TO_TABLE_MAP.get(col)
        if not table:
            print(f"Warning: Column '{col}' not found in COLUMN_TO_TABLE_MAP. Skipping.")
            continue
        required_tables.add(table)
        alias = table_aliases[table]
        select_clauses.add(f"cm.{col}" if col == 'compname' else f"{alias}.{col}") # Qualify all columns
        where_clauses.append(f"{alias}.{col} {op} {val}")

    for table in required_tables:
        alias = table_aliases[table]
        join_clauses.add(f"JOIN {table} {alias} ON cm.fincode = {alias}.fincode")
        where_clauses.append(f"{alias}.year_end = (SELECT MAX(year_end) FROM {table} WHERE fincode = cm.fincode)")

    # Ensure compname is always selected for the table
    select_clauses.add("cm.compname")
    select_str = ", ".join(list(select_clauses))
    join_str = " ".join(list(join_clauses))
    where_str = " AND ".join(where_clauses)
    order_by_clause = "ORDER BY ce.mcap DESC" if "company_equity" in required_tables else ""
    full_query = f"SELECT {select_str} FROM company_master cm {join_str} WHERE {where_str} {order_by_clause} LIMIT 10"

    print(f"Generated Screener SQL: {full_query}")
    
    async_engine_instance = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    try:
        async with async_engine_instance.connect() as connection:
            result = await connection.execute(text(full_query))
            rows = result.mappings().all()
        
        # --- THIS IS THE FIX ---
        # 1. Get the reasoning from the previous step.
        reasoning = json.loads(state["query_results"]["recommendation"]).get("reasoning", "N/A")

        # 2. Build a new, clean result dictionary.
        final_screener_payload = {
            "reasoning": reasoning,
            "strategy_name": strategy["name"],
            "stocks": [dict(row) for row in rows]
        }

        # 3. Return the correct variable and use a consistent key ("recommendation").
        return {"query_results": {"recommendation": json.dumps(final_screener_payload, default=str)}}
        # --- END OF FIX ---

    except Exception as e:
        traceback.print_exc()
        return {"error_message": f"I encountered an error while searching for stocks. Error: {e!r}"}
    finally:
        await async_engine_instance.dispose()

# --- NEW Nodes for Performance Analysis Path ---
async def fundamental_filter_and_score_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Fundamental Filter & Score---")
    params = state["performance_params"]
    time_period = params.get("time_period_years", 3)
    sector = state.get("mapped_sector")
    
    # This complex query calculates CAGR and filters for fundamentally sound companies
    # NOTE: CAGR is approximated here. A real implementation might use a DB function.
    query_string = f"""
            SELECT
            cm.fincode, 
            cm.compname,
            cm.industry, 
            ce.mcap,
            cfr.net_sales_growth AS latest_sales_growth,
            cfr.pat_growth AS latest_pat_growth,
                (
                    SELECT AVG(sub_cfr.roce)
                    FROM company_finance_ratio AS sub_cfr
                    WHERE sub_cfr.fincode = cm.fincode
                    AND sub_cfr.year_end >= (YEAR(CURDATE()) - :time_period)
                ) AS avg_roce
            FROM company_master AS cm
            JOIN company_equity AS ce ON cm.fincode = ce.fincode
            JOIN company_finance_ratio AS cfr ON cm.fincode = cfr.fincode
            WHERE
                ce.year_end = (SELECT MAX(year_end) FROM company_equity WHERE fincode = cm.fincode)
                AND cfr.year_end = (SELECT MAX(year_end) FROM company_finance_ratio WHERE fincode = cm.fincode)
                AND ce.mcap > 5000
                AND cfr.pat_growth > 0
                { "AND cm.industry = :sector" if sector else "" }
            LIMIT 100;
        """
    query = text(query_string)
    params = {"time_period": time_period}
    if sector:
        print(f"Applying exact sector filter: '{sector}'")
        params["sector"] = sector
    
    async_engine = create_async_engine(ASYNC_DATABASE_URL)
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(query, params)
            rows = [dict(row) for row in result.mappings().all()]
        
        if not rows:
            return {"error_message": "Could not find any stocks matching the initial fundamental criteria."}
            
        print(f"Fundamental filter found {len(rows)} candidate stocks.")
        return {"fundamental_screener_results": rows}
    except Exception as e:
        traceback.print_exc()
        return {"error_message": f"DB error during fundamental filtering: {e!r}"}
    finally:
        await async_engine.dispose()


async def parallel_technical_analysis_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Parallel Technical Analysis (with Fincode Forwarding)---")
    candidates = state["fundamental_screener_results"]
    time_period_years = state["performance_params"].get("time_period_years", 3)
    
    async def analyze_technicals(stock: Dict):
        compname = stock["compname"]
        fincode = stock["fincode"] # We already have the exact fincode here.
        
        # --- THIS IS THE FIX ---
        # We now pass the pre-resolved fincode directly to the technicals agent.
        tech_payload = {
            "question": f"analyze performance of {compname} over {time_period_years} years",
            "supervisor_decision": {
                "time_period_years": time_period_years
            },
            "target_fincode": fincode, # Pass the fincode
            "stock_identifier": compname, # Pass the name for context
            "return_structured_data": True
        }
        # --- END OF FIX ---
        
        tech_result = await technicals_app_instance.ainvoke(tech_payload)
        tech_analysis = tech_result.get("final_answer", {})
        
        # If the technical agent had an error, make sure it's captured
        if not tech_analysis or "error" in tech_analysis:
             return {**stock, "technical_analysis": {"error": tech_analysis.get("error", "Unknown technical analysis error")}}

        return {**stock, "technical_analysis": tech_analysis}

    tasks = [analyze_technicals(s) for s in candidates]
    results = await asyncio.gather(*tasks)
    
    return {"performance_analysis_results": results}


async def aggregate_performance_scores_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Aggregate Performance Scores (Final with Funda Score)---")
    results = state["performance_analysis_results"]
    scored_stocks = []

    for stock in results:
        tech_analysis = stock.get("technical_analysis", {})
        
        avg_roce = stock.get("avg_roce") or 0
        roce_score = min(10, max(0, avg_roce / 4.0))
        funda_score = roce_score

        if "error" in tech_analysis:
            print(f"Warning: Technical analysis for '{stock['compname']}' failed with error: {tech_analysis['error']}")
            tech_score = "N/A"
            price_cagr, volatility, sharpe_ratio = "N/A", "N/A", "N/A"
            final_overall_score = round(funda_score, 2)
        else:
            tech_score = tech_analysis.get("technical_performance_score", 0)
            price_cagr = tech_analysis.get("price_cagr_perc", "N/A")
            volatility = tech_analysis.get("annualized_volatility_perc", "N/A")
            sharpe_ratio = tech_analysis.get("sharpe_ratio", "N/A")
            final_overall_score = (0.5 * funda_score) + (0.5 * tech_score)

        # --- THIS IS THE FIX: Added "Funda Score" column ---
        scored_stocks.append({
            "Company Name": stock["compname"],
            "Performance Score": round(final_overall_score, 2) if isinstance(final_overall_score, (int, float)) else final_overall_score,
            "Funda Score": round(funda_score, 2), # <-- ADDED THIS LINE
            "Tech Score": round(tech_score, 2) if isinstance(tech_score, (int, float)) else tech_score,
            "Avg ROCE (%)": round(avg_roce, 2),
            "Price CAGR (%)": price_cagr,
            "Volatility (%)": volatility,
            "Sharpe Ratio": sharpe_ratio,
        })
        # --- END OF FIX ---

    sorted_stocks = sorted(
        scored_stocks, 
        key=lambda x: x["Performance Score"] if isinstance(x["Performance Score"], (int, float)) else -1, 
        reverse=True
    )[:5]
    
    methodology = (
        "The overall Performance Score is a weighted average of Fundamental Quality (ROCE) "
        "and a comprehensive Technical Score (combining historical price trends and current indicator signals). "
        "If Technical Score is 'N/A', it means data was unavailable for that stock."
    )
    final_result = {"stocks": sorted_stocks, "methodology": methodology}
    
    return {"query_results": {"performance_analysis": json.dumps(final_result)}}


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

    if extraction.is_performance_analysis_query:
        return "performance_analysis_path"
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
    if "General Query" in tasks:
        print("Routing to: Ranking Query Path (RAG via Guardrail)")
        return "rag_path"
    else:
        print("Routing to: General Path (No DB required)")
        return "general_path"

graph_builder = StateGraph(FundamentalsAgentState)

# Add all nodes, including our new one
graph_builder.add_node("entry_point", entry_point_node)
graph_builder.add_node("extract", extract_node)
graph_builder.add_node("map_industry_and_params", map_industry_and_params_node)
graph_builder.add_node("select_strategy", select_strategy_node)
graph_builder.add_node("execute_screener", execute_screener_node)
graph_builder.add_node("fundamental_filter_and_score", fundamental_filter_and_score_node)
graph_builder.add_node("parallel_technical_analysis", parallel_technical_analysis_node)
graph_builder.add_node("aggregate_performance_scores", aggregate_performance_scores_node)
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
graph_builder.set_entry_point("entry_point")
graph_builder.add_edge("entry_point", "extract")

graph_builder.add_conditional_edges(
    "extract",
    route_after_extraction,
    {
       "performance_analysis_path": "map_industry_and_params",

        "recommendation_path": "map_industry_and_params",
        "shareholding_path": "execute_shareholding_pattern", # NEW ROUTE
        "health_check_path": "execute_health_checklist", 
        "multi_metric_path": "execute_multi_metric", 
        "rag_path": "generalize_and_rag", 
        "general_path": "compose_answer", 
        "end": END
    }
)

# --- NEW: Edges after the mapping step ---
graph_builder.add_conditional_edges("map_industry_and_params", 
    # A simple router to direct flow after mapping
    lambda state: state["supervisor_decision"]["route"],
    {
        "performance_analysis": "fundamental_filter_and_score",
        "recommendation": "select_strategy"
    }
)

graph_builder.add_edge("fundamental_filter_and_score", "parallel_technical_analysis")
graph_builder.add_edge("parallel_technical_analysis", "aggregate_performance_scores")
graph_builder.add_edge("aggregate_performance_scores", "compose_answer")

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