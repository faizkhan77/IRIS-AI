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
from model_config import groq_llm
from sqlalchemy.ext.asyncio import create_async_engine 
from db_config import ASYNC_DATABASE_URL

from tools.technical_indicators import (
    INDICATOR_TOOLS_MAP,
    DEFAULT_INDICATORS_BY_CATEGORY,
    AGGREGATION_DEFAULT_INDICATORS, # For generic "buy/sell" questions
    aggregate_signals # The new aggregation function
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

class ExtractedTechDetailsV2(BaseModel):
    """Schema for detailed extraction from technical queries, supporting multiple indicators."""
    is_stock_market_related: bool = Field(..., description="True if the query is about stocks, finance, or technical analysis.")
    is_calculation_requested: bool = Field(..., description="True if a calculation for a specific stock is requested or clearly implied.")
    indicator_names: Optional[TypingList[str]] = Field(None, description="A list of specific, lowercase, space-removed indicator names if EXPLICITLY mentioned (e.g., ['rsi', 'macd']). Null or empty list otherwise.")
    stock_identifier: Optional[str] = Field(None, description="The stock name or symbol if a calculation is requested. Null if not.")
    is_general_outlook_query: bool = Field(False, description="True if the user asks for a general buy/sell/hold opinion without naming indicators (e.g., 'should I buy stock abc?', 'is xyz a good investment now?').")
    implied_category: Optional[str] = Field(None, description="If a single category is implied without a specific indicator name (e.g., 'is it volatile?'), infer the category. Choose ONE from: 'momentum', 'trend', 'volatility', 'strength'. Null otherwise.")

# --- TypedDict for State ---
class TechnicalAgentState(TypedDict):
    question: str
    extraction_details: Optional[ExtractedTechDetailsV2]
    target_fincode: Optional[int]
    price_data: Optional[TypingList[Dict[str, Any]]]
    indicators_to_run: TypingList[str]
    indicator_results: Optional[TypingList[Dict[str, Any]]]
    aggregation_result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    final_answer: Optional[str]


# --- Prompts ---


EXTRACT_QUERY_DETAILS_PROMPT_V2 = """
    You are an expert financial analyst. Analyze the user's question about stock markets and technical indicators.
    Your goal is to extract specific details to determine the best course of action.
    Your output MUST be a single, valid JSON object matching the ExtractedTechDetailsV2 schema.

    Schema:
    {{
    "is_stock_market_related": "boolean",
    "is_calculation_requested": "boolean",
    "indicator_names": "Optional[List[string]]",
    "stock_identifier": "Optional[string]",
    "is_general_outlook_query": "boolean",
    "implied_category": "Optional[string]"
    }}

    Rules:
    1. `is_stock_market_related`: Is the query about stocks, finance, or indicators?
    2. `is_calculation_requested`: Is a calculation for a stock requested (e.g., "RSI for AAPL", "is GOOGL overbought?")? This requires a `stock_identifier`.
    3. `indicator_names`: If indicators like "RSI", "MACD" are explicitly named, provide a list of their lowercase, space-removed versions (e.g., ["rsi", "bollingerbands"]). If one is named, provide a list with one item. Otherwise, null or empty.
    4. `stock_identifier`: If `is_calculation_requested` is true, what is the stock's name/symbol? Otherwise, null.
    5. `is_general_outlook_query`: Set to true ONLY for broad questions like "Should I buy stock abc?" or "Is company xyz a good buy now?" which imply a need for an aggregated opinion. This requires a `stock_identifier`.
    6. `implied_category`: ONLY if a single category is implied (e.g., "is it volatile?", "is the trend strong?") but no specific indicator is named. Choose ONE from: 'momentum', 'trend', 'volatility', 'strength'. Otherwise, null.

    User Question: {question}

    Examples:
    - "What is the RSI and MACD for company ABC?": {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_names": ["rsi", "macd"], "stock_identifier": "company ABC", "is_general_outlook_query": false, "implied_category": null}}
    - "Is stock XYZ a good buy now?": {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_names": null, "stock_identifier": "stock XYZ", "is_general_outlook_query": true, "implied_category": null}}
    - "Is company XYZ overbought?": {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_names": null, "stock_identifier": "company XYZ", "is_general_outlook_query": false, "implied_category": "momentum"}}
    - "What is an Exponential Moving Average?": {{"is_stock_market_related": true, "is_calculation_requested": false, "indicator_names": ["ema"], "stock_identifier": null, "is_general_outlook_query": false, "implied_category": null}}
    - "Who invented the light bulb?": {{"is_stock_market_related": false, "is_calculation_requested": false, "indicator_names": null, "stock_identifier": null, "is_general_outlook_query": false, "implied_category": null}}

    Provide ONLY the JSON output.
"""

ANSWER_COMPOSER_PROMPT_V2 = """
    You are IRIS, a helpful and concise financial assistant. Your task is to compose the final answer based on the information provided.

    User's Original Question: "{question}"

    --- Provided Context ---
    - Stock Analyzed: {stock_identifier}
    - Individual Indicator Results: {indicator_results}
    - Aggregated Result (JSON object or 'Not applicable.'): {aggregation_result}
    - Error Message: {error_message}
    ---

    Instructions:
    1.  **Priority 1: Handle Errors.** If an `Error Message` is present, your entire response MUST be that error message, phrased politely. Example: "I couldn't complete the request because [Error Message]".

    2.  **Priority 2: Use Aggregated Result.** If `Aggregated Result` is a JSON object (i.e., not the string 'Not applicable.'), this is the most important information.
        - Start with the final verdict. Use the `overall_verdict` and `composite_score` from the JSON. Example: "Based on an aggregation of multiple indicators, the overall signal for {stock_identifier} is 'Buy' with a composite score of 1.25."
        - Then, briefly summarize the individual signals that led to this conclusion from the `breakdown` section of the JSON.
        - Conclude with a standard disclaimer about consulting other factors.

    3.  **Priority 3: Use Single Indicator Result.** If there is no aggregated result, but there is one `Individual Indicator Result` (that isn't an error).
        - Summarize that single result. Example: "The RSI for {stock_identifier} is 25.5, which suggests a 'buy' signal. In simple terms, this indicates the stock may be oversold. It is essential to consider other factors before making a decision."

    4.  **Priority 4: Handle General Questions.** If there are no calculations or errors (e.g., user asked "What is RSI?"), provide a concise, 2-3 sentence explanation of the indicator(s) mentioned in the original question.

    Keep your answer clear, concise, and professional.
"""



# --- Agent Nodes (Async, Optimized & Enhanced for Aggregation) ---

async def extract_and_prepare_node(state: TechnicalAgentState) -> Dict:
    """Extracts details, validates them, and determines which indicators to run."""
    print("---NODE: Extract & Prepare---")
    # Using the fixed prompt name from our previous step
    prompt = EXTRACT_QUERY_DETAILS_PROMPT_V2.format(question=state["question"])
    structured_llm = groq_llm.with_structured_output(ExtractedTechDetailsV2, method="json_mode")

    try:
        details = await structured_llm.ainvoke(prompt)
        print(f"LLM Extraction: {details.model_dump_json(indent=2)}")

        if not details.is_stock_market_related:
            return {"error_message": "That question is outside my current scope of stock market analysis."}

        if not details.is_calculation_requested:
            if details.indicator_names and all(name in INDICATOR_TOOLS_MAP for name in details.indicator_names):
                return {"indicators_to_run": details.indicator_names, "extraction_details": details}
            else:
                return {"error_message": "I can explain indicators, but I couldn't identify a valid one in your question."}

        if not details.stock_identifier:
            return {"error_message": "To perform a calculation, please specify a stock name or symbol."}

        indicators_to_run = []
        
        # --- LOGIC FIX: Re-prioritized the checks ---
        # PRIORITY 1: Always handle explicitly mentioned indicators first.
        # This correctly handles "Should I buy X based on Y indicator?".
        if details.indicator_names:
            indicators_to_run = [name for name in details.indicator_names if name in INDICATOR_TOOLS_MAP]
            if not indicators_to_run:
                return {"error_message": f"None of the specified indicators are supported for calculation."}
            print(f"Explicit indicators found: {indicators_to_run}")

        # PRIORITY 2: If no explicit indicators, check for a general outlook query.
        elif details.is_general_outlook_query:
            indicators_to_run = AGGREGATION_DEFAULT_INDICATORS
            print(f"General outlook query detected. Using default indicators: {indicators_to_run}")

        # PRIORITY 3: If neither of the above, check for an implied category.
        elif details.implied_category:
            indicator = DEFAULT_INDICATORS_BY_CATEGORY.get(details.implied_category)
            if indicator:
                indicators_to_run = [indicator]
                print(f"Inferred category '{details.implied_category}', using default indicator '{indicator}'.")
            else:
                 return {"error_message": f"Could not determine an indicator for the category '{details.implied_category}'."}
        
        if not indicators_to_run:
            return {"error_message": f"Please specify one or more technical indicators (e.g., RSI, MACD) for {details.stock_identifier}."}

        return {"indicators_to_run": indicators_to_run, "extraction_details": details}

    except Exception as e:
        print(f"Critical error in extraction: {e}")
        return {"error_message": "I had trouble understanding your request. Could you please rephrase?"}


async def resolve_fincode_node(state: Dict) -> Dict: # Using Dict for broader compatibility
    """Safely and accurately resolves a stock identifier to a fincode using parameterized queries."""
    print("---NODE: Resolve Fincode---")
    
    # R-NOTE: The engine is now created *inside* the function, bound to the correct event loop.
    async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    
    try:
        stock_id = state["extraction_details"].stock_identifier
        
        fincode_lookup_query = text("""
            SELECT fincode FROM accord_base_live.company_master 
            WHERE compname LIKE :like_id OR symbol = :exact_id OR fincode = :num_id OR scripcode = :num_id_str
            ORDER BY 
                CASE 
                    WHEN symbol = :exact_id THEN 0
                    WHEN compname = :exact_id THEN 1
                    WHEN compname LIKE :like_id_exact THEN 2
                    ELSE 3
                END, 
                LENGTH(compname) ASC 
            LIMIT 1;
        """)

        try:
            num_id = int(stock_id)
        except (ValueError, TypeError):
            num_id = -1

        params = {
            "like_id": f"%{stock_id}%",
            "like_id_exact": f"{stock_id}%",
            "exact_id": stock_id,
            "num_id": num_id,
            "num_id_str": str(num_id)
        }

        # The connection is now guaranteed to be on the correct loop
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
        # R-NOTE: CRITICAL - Dispose of the engine to clean up connections before the loop closes.
        if 'async_engine' in locals():
            print("--- Disposing of temporary database engine (fincode) ---")
            await async_engine.dispose()


# THIS IS THE NEW, CORRECTED VERSION of fetch_price_data_node
async def fetch_price_data_node(state: Dict) -> Dict: # Using Dict for broader compatibility
    """Asynchronously fetches price data using a parameterized query."""
    print("---NODE: Fetch Price Data---")
    
    # R-NOTE: Create a local engine instance for this function call as well.
    async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_recycle=3600)
    
    try:
        fincode = state["target_fincode"]
        
        price_query = text("""
            SELECT date, open, high, low, close, volume 
            FROM accord_base_live.bse_abjusted_price_eod 
            WHERE fincode = :fincode ORDER BY date DESC LIMIT 300;
        """)

        # This connection is also safe now.
        async with async_engine.connect() as connection:
            result = await connection.execute(price_query, {"fincode": fincode})
            # Use mappings() for clean dictionary conversion
            price_data = [dict(row) for row in result.mappings().all()]

        if not price_data:
            return {"error_message": f"No price data found for the specified stock (fincode: {fincode})."}
        
        price_data.reverse() # Reverse to have oldest data first for calculations
        print(f"Fetched {len(price_data)} price records.")
        return {"price_data": price_data}
    except Exception as e:
        print(f"Database error fetching prices: {e}")
        return {"error_message": "A database error occurred while fetching price data."}
    finally:
        # R-NOTE: CRITICAL - Dispose of this engine as well.
        if 'async_engine' in locals():
            print("--- Disposing of temporary database engine (price) ---")
            await async_engine.dispose()

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
    
    # We only need to aggregate if there are multiple valid results
    valid_results = [
        (res['indicator'], res['result']) for res in indicator_results 
        if res and 'result' in res and 'error' not in res['result']
    ]

    if len(valid_results) < 2:
        print("Not enough valid results to aggregate. Skipping.")
        return {} # Return empty dict, no change to state
        
    print(f"Aggregating {len(valid_results)} indicator signals...")
    # Assume '1d' for now, this could be extracted in the future
    aggregation = aggregate_signals(valid_results, time_period='1d')
    
    print(f"Aggregation Result: {aggregation}")
    return {"aggregation_result": aggregation}


async def compose_final_answer_node(state: TechnicalAgentState) -> Dict:
    """Composes the final answer, handling single, multiple, and aggregated results."""
    print("---NODE: Compose Final Answer---")

    # Prepare individual results for the prompt
    results_for_prompt = "Not applicable."
    if state.get("indicator_results"):
        simplified_results = []
        for res in state["indicator_results"]:
            indicator_name = res.get('indicator', 'unknown')
            result_data = res.get('result', {})
            if 'error' in result_data:
                simplified_results.append(f"{indicator_name}: Error - {result_data['error']}")
            else:
                signal = result_data.get('signal', 'N/A')
                key_val_str = ""
                if 'rsi' in result_data: key_val_str = f"value={result_data['rsi']}"
                elif 'macd_line' in result_data: key_val_str = f"macd={result_data['macd_line']:.2f}"
                elif 'supertrend_value' in result_data: key_val_str = f"value={result_data['supertrend_value']}"
                simplified_results.append(f"{indicator_name}: signal='{signal}' ({key_val_str})")
        results_for_prompt = "; ".join(simplified_results)
    
    # Safely prepare aggregation result for the prompt
    agg_result = state.get("aggregation_result")
    agg_result_for_prompt = json.dumps(agg_result, indent=2) if isinstance(agg_result, dict) else "Not applicable."
    
    format_args = {
        "question": state["question"],
        "stock_identifier": state.get("extraction_details").stock_identifier if state.get("extraction_details") else "the stock",
        "indicator_results": results_for_prompt,
        "aggregation_result": agg_result_for_prompt,
        "error_message": state.get("error_message", "Not applicable."),
    }
    
    # Use the new, fixed prompt
    prompt = ANSWER_COMPOSER_PROMPT_V2.format(**format_args)
    
    try:
        response = await groq_llm.ainvoke(prompt)
        final_answer = response.content.strip()
        print(f"Composed Final Answer: {final_answer}")
        return {"final_answer": final_answer}
    except Exception as e:
        print(f"Error in final answer composition: {e}")
        return {"final_answer": "I'm sorry, I encountered a problem while formulating the final response."}

# --- Graph Conditional Edges ---

def route_after_preparation(state: TechnicalAgentState) -> str:
    if state.get("error_message"): return "compose_answer"
    if state["extraction_details"].is_calculation_requested: return "resolve_fincode"
    return "compose_answer" 

def route_after_db_steps(state: TechnicalAgentState) -> str:
    if state.get("error_message"): return "compose_answer"
    if state.get("target_fincode") and not state.get("price_data"): return "fetch_prices"
    if state.get("price_data"): return "calculate"
    return "compose_answer"

def route_after_calculation(state: TechnicalAgentState) -> str:
    # If more than one indicator was run, proceed to aggregation. Otherwise, compose answer.
    if len(state.get("indicators_to_run", [])) > 1:
        return "aggregate"
    return "compose_answer"


# --- Build the Graph ---
graph_builder = StateGraph(TechnicalAgentState)

graph_builder.add_node("prepare", extract_and_prepare_node)
graph_builder.add_node("resolve_fincode", resolve_fincode_node)
graph_builder.add_node("fetch_prices", fetch_price_data_node)
graph_builder.add_node("calculate", calculate_indicators_node)
graph_builder.add_node("aggregate", aggregate_results_node) # New node
graph_builder.add_node("compose_answer", compose_final_answer_node)

graph_builder.set_entry_point("prepare")

graph_builder.add_conditional_edges("prepare", route_after_preparation)
graph_builder.add_conditional_edges("resolve_fincode", route_after_db_steps)
graph_builder.add_conditional_edges("fetch_prices", route_after_db_steps)
graph_builder.add_conditional_edges("calculate", route_after_calculation) # New routing logic
graph_builder.add_edge("aggregate", "compose_answer")
graph_builder.add_edge("compose_answer", END)

app = graph_builder.compile()


# --- Main Execution for Testing ---
if __name__ == "__main__":
    async def run_tech_agent_test():
        test_queries = [
          
        "What is RSI?", 
        "Is Reliance overbought?", 
        "How volatile is Aegis Logistics now?", 
        "stochastic for Ambalal Sarabhai", 
        "Is Reliance strong?", 
        "Should i buy Ambalal Sarabhai?",
        "Should I buy reliance based on Bollinger Bands",
        "Should I buy reliance based on RSI?", 
        "Should I buy reliance based on MACD?",
        "Should I buy Aegis Logistics based on RSI, MACD, and Supertrend?", # Explicit multi-indicator
        "Is Reliance a good buy right now?", # General outlook, should trigger default aggregation
       
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