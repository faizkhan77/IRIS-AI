# agents/technicals_agent.py
import asyncio
import json
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, List as TypingList, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import text
import numpy as np
from db_config import async_engine
from model_config import groq_llm, groq_llm_fast
from sqlalchemy.ext.asyncio import create_async_engine 
from db_config import ASYNC_DATABASE_URL
import re

from tools.technical_indicators import (
    INDICATOR_TOOLS_MAP,
    DEFAULT_INDICATORS_BY_CATEGORY,
    AGGREGATION_DEFAULT_INDICATORS,
    aggregate_signals
)

load_dotenv()


def _convert_numpy_to_python(data: Any) -> Any:
    """Recursively converts numpy types in a nested structure to native Python types."""
    if isinstance(data, dict):
        return {k: _convert_numpy_to_python(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_numpy_to_python(i) for i in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data)
    return data

# --- Pydantic Models for Structured Outputs ---
class ExtractedTechDetailsV3(BaseModel):
    is_stock_market_related: bool = Field(..., description="True if the query is about stocks, finance, or technical analysis.")
    is_calculation_requested: bool = Field(..., description="True if a calculation for a specific stock is requested or clearly implied.")
    indicator_names: Optional[TypingList[str]] = Field(None, description="A list of specific, lowercase, space-removed indicator names if EXPLICITLY mentioned.")
    stock_identifier: Optional[str] = Field(None, description="The stock name or symbol if a calculation is requested.")
    # --- NEW: Field to capture time period ---
    time_period_years: Optional[int] = Field(None, description="The time period in years if mentioned (e.g., '5 years', 'last year').")
    is_general_outlook_query: bool = Field(False, description="True for broad buy/sell/hold questions without specific indicators.")
    is_historical_performance_query: bool = Field(False, description="True for questions about 'best performing', 'most consistent', or performance over a time period.")
    implied_category: Optional[str] = Field(None, description="If a category is implied (e.g., 'is it volatile?'), infer one from: 'momentum', 'trend', 'volatility', 'strength'.")


# --- TypedDict for State ---
class TechnicalAgentState(TypedDict):
    question: str
    supervisor_decision: Optional[Dict]
    return_structured_data: bool
    stock_identifier: Optional[str] # Add this field
    company_name: Optional[str] 
    extraction_details: Optional[ExtractedTechDetailsV3]
    time_period_days: Optional[int] # NEW
    target_fincode: Optional[int]
    price_data: Optional[TypingList[Dict[str, Any]]]
    indicators_to_run: TypingList[str]
    indicator_results: Optional[TypingList[Dict[str, Any]]]
    aggregation_result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    final_answer: Optional[str]


# --- Prompts ---

EXTRACT_QUERY_DETAILS_PROMPT_V3 = """
    You are an expert financial analyst. Your job is to meticulously analyze the user's question and extract key details into a JSON object matching the `ExtractedTechDetailsV3` schema.

    **CRITICAL RULES:**


    1. **Combined Analysis (HIGHEST PRIORITY):**
        - If the user asks for a broad "technical analysis" or "technical performance" of a specific stock, you MUST set BOTH `is_historical_performance_query` AND `is_general_outlook_query` to `true`. This signals a request for a complete, combined analysis.
        - Example: "Technical analysis of AAPL" -> `is_historical_performance_query: true`, `is_general_outlook_query: true`
        - Example: "Technical performance of TSLA over 3 years" -> `is_historical_performance_query: true`, `is_general_outlook_query: true`, `time_period_years: 3` 
    2. **Time Period (`time_period_years`):** If the user mentions a duration like "last 5 years", "over 3 years", "last year", you MUST extract the number of years.
    2.  **Historical Performance (`is_historical_performance_query`):** If the query asks about performance over a time period (e.g., "best performing stock in the last 5 years"), you MUST set `is_historical_performance_query` to `true`. This is very important.
    3.  **`stock_identifier` is MANDATORY if a calculation is needed.** If the user mentions a company (e.g., "Reliance", "Aegis Logistics"), you MUST extract it.
    4.  **`is_calculation_requested` MUST be `true`** if the user asks any question about a specific stock that requires data (e.g., "is it volatile?", "is it overbought?", "is it a good buy?").
    5.  **`indicator_names`:** Extract ONLY explicitly named indicators (e.g., RSI, MACD). Do NOT invent indicators from words like "overbought" or "volatile".
    6.  **`implied_category`:** Use this for questions about a technical *concept* where no specific indicator is named.
        - "is it overbought?" -> `implied_category`: 'momentum'
        - "is it volatile?" -> `implied_category`: 'volatility'
        - "is it strong?" -> `implied_category`: 'strength'
    7.  **`is_general_outlook_query`:** Set to `true` for broad, open-ended analysis questions.
        - "is [company] a good buy right now?" -> `is_general_outlook_query`: true
        - "technical analysis of [company]" -> `is_general_outlook_query`: true
        - "is [company] strong?" -> `is_general_outlook_query`: true

    **--- EXAMPLES (Study these carefully) ---**

    - **User Question:** "Technical performance of X over X years"
    - **Your Output:** {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_names": null, "stock_identifier": "X", "time_period_years": X, "is_general_outlook_query": true, "is_historical_performance_query": true, "implied_category": null}}

    - **User Question:** "Technical analysis of X"
    - **Your Output:** {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_names": null, "stock_identifier": "X", "time_period_years": null, "is_general_outlook_query": true, "is_historical_performance_query": true, "implied_category": null}}

    - **User Question:** "Should I buy based on RSI and MACD?"
    - **Your Output:** {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_names": ["rsi", "macd"], "stock_identifier": "stock", "time_period_years": null, "is_general_outlook_query": false, "is_historical_performance_query": false, "implied_category": null}}

    - **User Question:** "Is X overbought?"
    - **Your Output:** {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_names": null, "stock_identifier": "X", "time_period_years": null, "is_general_outlook_query": false, "is_historical_performance_query": false, "implied_category": "momentum"}}

    - **User Question:** "Should I buy xyz based on RSI, MACD, and Supertrend?"
    - **Your Output:** {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_names": ["rsi", "macd", "supertrend"], "stock_identifier": "Aegis Logistics","time_period_years": null, "is_general_outlook_query": false, "implied_category": null}}

    - **User Question:** "What is an EMA?"
    - **Your Output:** {{"is_stock_market_related": true, "is_calculation_requested": false, "indicator_names": ["ema"], "stock_identifier": null, "time_period_years": null,"is_general_outlook_query": false, "implied_category": null}}

    ---
    User Question: {question}

    Provide ONLY the valid JSON output.
"""

ANSWER_COMPOSER_PROMPT_V3 = """
    You are IRIS, a financial assistant providing clear, simple technical analysis insights.
    Your tone is helpful and direct. Your audience is non-technical.

    **User's Question:** "{question}"

     --- Analysis Data Provided ---
    - Stock Analyzed: {{stock_identifier}}
    - Aggregated Result Data (JSON):{{aggregation_result}}
    - Individual Indicator Results Data (JSON): {{indicator_results}}
    - Error Message (if any): {{error_message}}
    ---

    **Instructions (in order of priority):**

    1.  **If an `Error Message` exists:** State the problem gracefully.
        - Example: "I'm sorry, I couldn't find a unique stock matching '{{stock_identifier}}'. Could you please provide a more specific name or symbol?"


    2.  **If BOTH historical data and an `Aggregated Result` are present:** This is a comprehensive analysis. You MUST present both parts using the following markdown structure.
        - **Step 1:** Extract the `historical_performance_score` result from the `Individual Indicator Results`. Extract the `Aggregated Result` data.
        - **Step 2:** Generate the response using this exact format:
        ```markdown
        ## Comprehensive Technical Analysis for {{stock_identifier}}

        ### Current Indicator Outlook
        Based on a real-time aggregation of key indicators (like RSI, MACD), the current outlook is a **'{{overall_verdict}}'** with a composite score of **{{composite_score}}**. This suggests a {{overall_verdict}} short-term sentiment.

        ### Historical Performance
        Over the long term, the stock has a **Technical Performance Score of {{technical_performance_score}}/10**. This score reflects its performance over the analyzed period and is based on these key metrics:
        - **Price CAGR:** {{price_cagr_perc}}%
        - **Annualized Volatility:** {{annualized_volatility_perc}}%
        - **Sharpe Ratio (Risk-Adjusted Return):** {{sharpe_ratio}}

        ### Summary
        In summary, while the stock's long-term historical performance score is {{technical_performance_score}}/10, the current indicators are pointing towards a {{overall_verdict}} signal.
        ```

    3.  **If ONLY an `Aggregated Result` exists:** This is for a simple "buy/sell" query.
        - "The overall technical outlook for {{stock_identifier}} is currently a **'{{overall_verdict}}'**, with a composite score of **{{composite_score}}**. This is based on signals from key indicators like RSI and MACD."

    4.  **If ONLY a `historical_performance_score` result exists:**
        - "The **Technical Performance Score for {{stock_identifier}} is {{technical_performance_score}}/10**. This is based on historical data including a Price CAGR of **{{price_cagr_perc}}%**, Volatility of **{{annualized_volatility_perc}}%**, and a Sharpe Ratio of **{{sharpe_ratio}}**."

    5.  **If there is a single `Individual Indicator Result` (but no aggregation):**
        - You MUST analyze the `Individual Indicator Results` data.
        - Extract the specific values (e.g., `macd_line`, `rsi_value`).
        - Extract the `signal` (e.g., "buy", "sell").
        - Explain the result and the signal in simple terms.
        - **Example:** "Based on the latest data for {{stock_identifier}}, the **MACD** indicator is showing a **Buy** signal. The MACD line is at {{macd_line}} and the signal line is at {{signal_line}}."
        - **Example:** "The **RSI** for {{stock_identifier}} is {{rsi}}, which suggests the stock is in a **Neutral** zone, not overbought or oversold."

    6.  **If it's a general question (no calculation):**
        - Provide a simple, 2-sentence definition of the indicator(s) asked about.
        - Example: "The Relative Strength Index, or RSI, is a tool traders use to see if a stock might be overbought or oversold."

    **CRITICAL RULE:** Do NOT use jargon or complex phrasing. Be concise and direct.
"""


async def entry_point_node(state: TechnicalAgentState) -> Dict:
    """
    NEW ENTRY POINT: This node simply ensures the input state is correctly formed.
    The 'return_structured_data' flag is often passed in the initial payload
    but might not be in the TypedDict's default. This node normalizes it.
    """
    print("---NODE: Technical Agent Entry Point---")
    # Set the flag to False if it's not explicitly passed as True.
    # This ensures it always has a boolean value in the state.
    return {"return_structured_data": state.get("return_structured_data", False)}


async def extract_and_prepare_node(state: TechnicalAgentState) -> Dict:
    """
    Extracts details from the query. For combined analysis, it prepares all required indicators.
    Defaults to a 1-year timeframe if not specified for historical parts.
    """
    print("---NODE: Extract & Prepare (Unified Analysis)---")
    
    # High-priority path for fundamentals_agent remains unchanged
    if state.get("target_fincode") and state.get("stock_identifier"):
        # This logic is for the multi-stock performance path and is correct.
        print(f"--- Fincode {state['target_fincode']} was provided directly. ---")
        details = ExtractedTechDetailsV3(
            is_stock_market_related=True, is_calculation_requested=True,
            stock_identifier=state["stock_identifier"], is_historical_performance_query=True
        )
        time_period_days = None
        if state.get("supervisor_decision") and state["supervisor_decision"].get("time_period_years"):
            time_period_years = state["supervisor_decision"]["time_period_years"]
            time_period_days = time_period_years * 365
        return {
            "indicators_to_run": ["historical_performance_score"],
            "extraction_details": details, "time_period_days": time_period_days
        }

    # Standard path for direct user queries
    print("--- Performing LLM extraction for direct query. ---")
    prompt = EXTRACT_QUERY_DETAILS_PROMPT_V3.format(question=state["question"])
    structured_llm = groq_llm_fast.with_structured_output(ExtractedTechDetailsV3, method="json_mode")
    try:
        details = await structured_llm.ainvoke(prompt)
        print(f"LLM Extraction: {details.model_dump_json(indent=2)}")
        
        if not details.is_calculation_requested:
            return {"indicators_to_run": details.indicator_names or [], "extraction_details": details}
        if not details.stock_identifier:
            return {"error_message": "Please specify a stock name.", "extraction_details": details}

        indicators_to_run = []
        time_period_days = None
        
        # --- THIS IS THE NEW UNIFIED LOGIC ---
        # If a full analysis is requested (historical + general)
        if details.is_historical_performance_query and details.is_general_outlook_query:
            years = details.time_period_years if details.time_period_years is not None else 1
            print(f"Combined analysis query detected. Timeframe: {years} year(s).")
            # Run BOTH historical score and the default indicators for aggregation
            indicators_to_run = ["historical_performance_score"] + AGGREGATION_DEFAULT_INDICATORS
            time_period_days = years * 365
        # --- END NEW LOGIC ---
        else: # Handle specific, non-combined queries
            if details.indicator_names:
                indicators_to_run = [name for name in details.indicator_names if name in INDICATOR_TOOLS_MAP]
            elif details.is_general_outlook_query: # Just aggregation, no history
                indicators_to_run = AGGREGATION_DEFAULT_INDICATORS
            elif details.implied_category:
                indicator = DEFAULT_INDICATORS_BY_CATEGORY.get(details.implied_category)
                if indicator: indicators_to_run = [indicator]
        
        if not indicators_to_run:
            return {"error_message": "I couldn't determine which analysis to perform.", "extraction_details": details}
        
        return {"indicators_to_run": indicators_to_run, "extraction_details": details, "time_period_days": time_period_days}
    except Exception as e:
        return {"error_message": f"I had trouble understanding your request: {e}"}

async def resolve_fincode_node(state: Dict) -> Dict:
    print("---NODE: Resolve Fincode---")
    async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    try:
        stock_id = state["extraction_details"].stock_identifier
        
        # This new query is more robust. It prioritizes exact matches, then handles partial matches.
        # It handles "Reliance" by finding "Reliance Industries Ltd." as the best match.
        fincode_lookup_query = text("""
            SELECT fincode FROM accord_base_live.company_master 
            WHERE 
                compname = :exact_id OR 
                symbol = :exact_id OR
                compname LIKE :like_id
            ORDER BY 
                CASE 
                    WHEN symbol = :exact_id THEN 0
                    WHEN compname = :exact_id THEN 1
                    WHEN compname LIKE :like_id_start THEN 2
                    ELSE 3
                END, 
                LENGTH(compname) ASC 
            LIMIT 1;
        """)

        params = {
            "exact_id": stock_id,
            "like_id": f"%{stock_id}%",
            "like_id_start": f"{stock_id}%",
        }

        async with async_engine.connect() as connection:
            result = await connection.execute(fincode_lookup_query, params)
            fincode = result.scalar_one_or_none()
        
        if fincode:
            print(f"Resolved Fincode for '{stock_id}': {fincode}")
            return {"target_fincode": fincode}
        else:
            return {"error_message": f"Could not find a unique stock corresponding to '{stock_id}'."}
            
    except Exception as e:
        print(f"Database error during fincode resolution: {e}")
        return {"error_message": f"A database error occurred while looking up the stock."}
    finally:
        if 'async_engine' in locals():
            print("--- Disposing of temporary database engine (fincode) ---")
            await async_engine.dispose()


async def fetch_price_data_node(state: TechnicalAgentState) -> Dict:
    print("---NODE: Fetch Price Data (Time-Aware)---")
    async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    try:
        fincode = state["target_fincode"]
        time_period_days = state.get("time_period_days")

        if time_period_days:
            # Fetch data for a specific historical period
            print(f"Fetching data for the last {time_period_days} days.")
            price_query = text("""
                SELECT date, open, high, low, close, volume 
                FROM bse_abjusted_price_eod 
                WHERE fincode = :fincode AND date >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
                ORDER BY date ASC;
            """)
            params = {"fincode": fincode, "days": time_period_days}
        else:
            # Default to fetching the last 300 data points for point-in-time analysis
            print("Fetching latest 300 data points for point-in-time analysis.")
            price_query = text("""
                SELECT date, open, high, low, close, volume 
                FROM bse_abjusted_price_eod 
                WHERE fincode = :fincode ORDER BY date DESC LIMIT 300;
            """)
            params = {"fincode": fincode}

        async with async_engine.connect() as connection:
            result = await connection.execute(price_query, params)
            price_data = [dict(row) for row in result.mappings().all()]

        if not price_data:
            return {"error_message": f"No price data found for the stock (fincode: {fincode})."}
        
        # If we used DESC LIMIT, we need to reverse for calculations
        if not time_period_days:
            price_data.reverse()
        
        print(f"Fetched {len(price_data)} price records.")
        return {"price_data": price_data}
    except Exception as e:
        return {"error_message": f"A database error occurred while fetching price data: {e}"}
    finally:
        if 'async_engine' in locals(): await async_engine.dispose()

async def calculate_indicators_node(state: TechnicalAgentState) -> Dict:
    """Runs all requested indicator calculations in parallel for efficiency."""
    print("---NODE: Calculate Indicators (Parallel)---")
    indicators_to_run = state["indicators_to_run"]
    price_data = state["price_data"]
    
    async def run_calculation(indicator_name):
        """Helper to run a synchronous calculation in a separate thread."""
        loop = asyncio.get_running_loop()
        tool_function = INDICATOR_TOOLS_MAP[indicator_name]
        # Use asyncio.to_thread to run the sync, CPU-bound function without blocking the event loop
        result = await loop.run_in_executor(None, tool_function, price_data)
        return indicator_name, result

    tasks = [run_calculation(name) for name in indicators_to_run]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_results = []
    for i, res in enumerate(results):
        indicator_name = indicators_to_run[i]
        if isinstance(res, Exception):
            print(f"Error calculating {indicator_name}: {res}")
            # Store error message in a structured way
            final_results.append({'indicator': indicator_name, 'result': {'error': f"Calculation failed: {res}"}})
        else:
            _, calc_result = res
            clean_result = _convert_numpy_to_python(calc_result)
            final_results.append({'indicator': indicator_name, 'result': clean_result})
            
    print(f"Indicator Results: {final_results}")
    return {"indicator_results": final_results}

def aggregate_results_node(state: TechnicalAgentState) -> Dict:
    """Aggregates multiple indicator results into a single verdict."""
    print("---NODE: Aggregate Results---")
    indicator_results = state["indicator_results"]
    valid_results = [(res['indicator'], res['result']) for res in indicator_results if 'error' not in res.get('result', {})]

    if len(valid_results) < 2:
        return {} # Not enough data to aggregate
        
    aggregation = aggregate_signals(valid_results)
    print(f"Aggregation Result: {aggregation}")
    return {"aggregation_result": aggregation}

# ... (all other code in technicals_agent.py remains the same) ...

async def compose_final_answer_node(state: TechnicalAgentState) -> Dict:
    """
    Composes the final answer, now with robust data unpacking for the prompt.
    """
    print("---NODE: Compose Final Answer (with Data Unpacking)---")

    # Data Mode for agent-to-agent communication remains the same
    if state.get("return_structured_data") is True:
        # ... (no changes needed here) ...
        print("--- Data Mode detected. Returning structured JSON. ---")
        final_data = {}
        if state.get("aggregation_result"):
            final_data = state.get("aggregation_result")
        
        hist_perf = next((item.get('result') for item in state.get("indicator_results", []) if item.get('indicator') == 'historical_performance_score'), None)
        if hist_perf:
            final_data.update(hist_perf)
        
        if not final_data:
             final_data = {"individual_results": state.get("indicator_results", [])}

        return {"final_answer": final_data}


    print("--- User-Facing Mode detected. Composing formatted text. ---")
    
    # --- THIS IS THE FIX ---
    # We will now pass the data as clean JSON strings to the prompt.
    # This avoids any KeyError and lets the LLM do the unpacking.
    
    # Prepare individual results
    indicator_results_dict = {}
    for item in state.get("indicator_results", []):
        indicator_name = item.get('indicator')
        result_data = item.get('result')
        if indicator_name and result_data:
            indicator_results_dict[indicator_name] = result_data

    # Prepare final arguments for the prompt template
    format_args = {
        "question": state.get("question", ""),
        "stock_identifier": state.get("extraction_details").stock_identifier if state.get("extraction_details") else "the stock",
        "error_message": state.get("error_message", "Not applicable."),
        # Pass the raw dictionaries, converted to JSON strings
        "aggregation_result": json.dumps(state.get("aggregation_result", {})),
        "indicator_results": json.dumps(indicator_results_dict),
    }

    print(f"Final prompt arguments: {json.dumps(format_args, indent=2)}")

    # Use .format(), as all keys are guaranteed to be in format_args now.
    prompt = ANSWER_COMPOSER_PROMPT_V3.format(**format_args)
    
    try:
        response = await groq_llm.ainvoke(prompt)
        final_answer = response.content.strip()
        print(f"Composed Final Answer (Text): {final_answer}")
        return {"final_answer": final_answer}
    except Exception as e:
        print(f"CRITICAL ERROR in compose_final_answer_node: {e}")
        import traceback
        traceback.print_exc()
        return {"final_answer": f"I encountered a problem formulating the response: {e}"}

def route_after_preparation(state: TechnicalAgentState) -> str:
    """Router to decide the next step after preparation."""
    if state.get("error_message"): return "compose_answer"
    
    # --- THIS IS THE FIX ---
    # If we already have a fincode, go straight to fetching prices.
    if state.get("target_fincode"):
        print("--- ROUTING: Fincode present, skipping resolution. ---")
        return "fetch_prices"
    # --- END OF FIX ---
    
    if state["extraction_details"].is_calculation_requested: 
        return "resolve_fincode"
        
    return "compose_answer"

def route_after_db_steps(state: TechnicalAgentState) -> str:
    if state.get("error_message"): return "compose_answer"
    if state.get("target_fincode") and not state.get("price_data"): return "fetch_prices"
    if state.get("price_data"): return "calculate"
    return "compose_answer"

def route_after_calculation(state: TechnicalAgentState) -> str:
    """
    Intelligently decides whether to aggregate results based on the initial user intent.
    """
    extraction_details = state.get("extraction_details")
    
    # If the user asked a general "buy/sell" question, we should aggregate.
    if extraction_details and extraction_details.is_general_outlook_query:
        print("--- ROUTING: General outlook query detected. Proceeding to aggregate results. ---")
        return "aggregate"
    else:
        # For all other cases (specific indicators, historical analysis), go directly to the composer.
        print("--- ROUTING: Specific indicator or historical query. Skipping aggregation. ---")
        return "compose_answer"


# --- Build the Graph ---
graph_builder = StateGraph(TechnicalAgentState)

graph_builder.add_node("entry_point", entry_point_node) # NEW ENTRY
graph_builder.add_node("prepare", extract_and_prepare_node)
graph_builder.add_node("resolve_fincode", resolve_fincode_node)
graph_builder.add_node("fetch_prices", fetch_price_data_node)
graph_builder.add_node("calculate", calculate_indicators_node)
graph_builder.add_node("aggregate", aggregate_results_node) # New node
graph_builder.add_node("compose_answer", compose_final_answer_node)

graph_builder.set_entry_point("entry_point")
graph_builder.add_edge("entry_point", "prepare")
# --- MODIFIED EDGES ---
graph_builder.add_conditional_edges("prepare", route_after_preparation)
graph_builder.add_conditional_edges("resolve_fincode", route_after_db_steps, {
    "fetch_prices": "fetch_prices", "calculate": "calculate", "compose_answer": "compose_answer"
})
graph_builder.add_conditional_edges("fetch_prices", route_after_db_steps, {
    "calculate": "calculate", "compose_answer": "compose_answer"
})

# Use the new, smarter router after calculation
graph_builder.add_conditional_edges("calculate", route_after_calculation, {
    "aggregate": "aggregate",
    "compose_answer": "compose_answer"
})

# Final edges
graph_builder.add_edge("aggregate", "compose_answer")
graph_builder.add_edge("compose_answer", END)

app = graph_builder.compile()


# --- Main Execution for Testing ---
if __name__ == "__main__":
    async def run_tech_agent_test():
        test_queries = [
          
        # "What is RSI?", 
        "Is Reliance overbought?", 
        "How volatile is Aegis Logistics now?", 
        # "stochastic for Ambalal Sarabhai", 
        "Is Reliance strong?", 
        "Should i buy Ambalal Sarabhai?",
        "Should I buy reliance based on Bollinger Bands",
        "Should I buy reliance based on RSI?", 
        "Should I buy reliance based on MACD?",
        "Should I buy Aegis Logistics based on RSI, MACD, and Supertrend?", # Explicit multi-indicator
        "Is Reliance a good buy right now?", # General outlook, should trigger default aggregation
        "Should I buy Aegis Logistics based on RSI, MACD, bollingerbands and Supertrend?", # Explicit multi-indicator
        "Technical analysis of ABB India",
       
        ]
        for query in test_queries:
            print(f"\n\n{'='*20} TESTING: \"{query}\" {'='*20}")
            inputs = {"question": query}
            try:
                # Use astream_events for more detailed logging during tests
                async for event in app.astream(inputs):
                    if "event" in event and event["event"] == "on_chain_end":
                         if "output" in event["data"]:
                             final_state = event["data"]["output"]
                             print(f"\n>>> FINAL ANSWER: {final_state.get('final_answer', 'No answer found.')}")
            except Exception as e:
                print(f"\nError during agent execution: {e}")
                import traceback
                traceback.print_exc()

    asyncio.run(run_tech_agent_test())