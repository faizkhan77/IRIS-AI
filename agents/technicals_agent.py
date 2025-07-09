# agents/technicals_agent.py
import asyncio
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, List as TypingList, Dict, Any, Optional
from pydantic import BaseModel, Field  # OPTIMIZED: Correct Pydantic import
from dotenv import load_dotenv
from sqlalchemy import text
import numpy as np
from db_config import async_engine
from model_config import groq_llm
from tools.technical_indicators import INDICATOR_TOOLS_MAP, DEFAULT_INDICATORS_BY_CATEGORY

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

class ExtractedTechDetails(BaseModel):
    """Schema for the LLM's initial, detailed extraction for technical queries."""
    is_stock_market_related: bool = Field(..., description="True if the query is about stocks, finance, or technical analysis.")
    is_calculation_requested: bool = Field(..., description="True if a calculation for a specific stock is requested or clearly implied.")
    indicator_name: Optional[str] = Field(None, description="The specific, lowercase, space-removed indicator name if EXPLICITLY mentioned (e.g., 'rsi', 'bollingerbands'). Null otherwise.")
    stock_identifier: Optional[str] = Field(None, description="The stock name or symbol if a calculation is requested. Null if not.")
    implied_category: Optional[str] = Field(None, description="If no indicator is named but a calculation is requested, infer the category. Choose ONE from: 'momentum', 'trend', 'volatility', 'strength', 'general_outlook'. Null otherwise.")

# --- TypedDict for State ---
class TechnicalAgentState(TypedDict):
    question: str
    extraction_details: Optional[ExtractedTechDetails]
    target_fincode: Optional[int]
    price_data: Optional[TypingList[Dict[str, Any]]]
    indicator_to_run: Optional[str]
    indicator_result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    final_answer: Optional[str]


# --- Prompts ---

EXTRACT_QUERY_DETAILS_PROMPT = """
You are an expert financial analyst. Analyze the user's question about stock markets and technical indicators.
Your goal is to extract specific details to determine the best course of action.
Your output MUST be a single, valid JSON object matching the ExtractedTechDetails schema.

Schema:
{{
  "is_stock_market_related": "boolean",
  "is_calculation_requested": "boolean",
  "indicator_name": "Optional[string]",
  "stock_identifier": "Optional[string]",
  "implied_category": "Optional[string]"
}}

Rules:
1. `is_stock_market_related`: Is the question about stocks, finance, or indicators?
2. `is_calculation_requested`: Is a calculation for a stock requested (e.g., "RSI for AAPL", "is GOOGL overbought?")? This requires a stock identifier.
3. `indicator_name`: If an indicator like "RSI" or "MACD" is explicitly named, provide its lowercase, space-removed version (e.g., "bollingerbands"). Otherwise, null.
4. `stock_identifier`: If `is_calculation_requested` is true, what is the stock's name/symbol? Otherwise, null.
5. `implied_category`: ONLY if `is_calculation_requested` is true AND `indicator_name` is null, infer the query's category. Choose ONE from: 'momentum', 'trend', 'volatility', 'strength', 'general_outlook'. Otherwise, null.

User Question: {question}

Examples:
- "What is the RSI for Reliance Industries?": {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_name": "rsi", "stock_identifier": "Reliance Industries", "implied_category": null}}
- "Is Apple overbought?": {{"is_stock_market_related": true, "is_calculation_requested": true, "indicator_name": null, "stock_identifier": "Apple", "implied_category": "momentum"}}
- "What is SMA?": {{"is_stock_market_related": true, "is_calculation_requested": false, "indicator_name": "sma", "stock_identifier": null, "implied_category": null}}
- "Who invented the television?": {{"is_stock_market_related": false, "is_calculation_requested": false, "indicator_name": null, "stock_identifier": null, "implied_category": null}}

Provide ONLY the JSON output.
"""

ANSWER_COMPOSER_PROMPT = """
You are IRIS, a helpful and concise financial assistant. The user asked a question about technical analysis.
Your task is to compose the final answer based on the information provided below.

User's Original Question: "{question}"

--- Provided Context ---
- Stock Analyzed: {stock_identifier}
- Indicator Used: {indicator_name}
- Calculation Result: {indicator_result}
- Error Message: {error_message}
---

Instructions:
1.  **If an `Error Message` is present**, your entire response should be that error message, phrased clearly and politely. Example: "I couldn't complete the request because: [Error Message]".
2.  **If `Calculation Result` is "Not applicable."**, this means it was a general question. Provide a concise, 2-3 sentence explanation of the indicator requested.
3.  **If `Calculation Result` is present and not an error**, craft a natural language answer summarizing it. State the key values and the final 'signal'.
    - Example for RSI: "The 14-day RSI for Apple is 65.3, which suggests a 'neutral' signal, approaching overbought territory."
    - Example for MACD: "For Microsoft, the MACD Line is 1.25 and the Signal Line is 1.10. The current signal is 'buy', indicating bullish momentum."
4.  **If the query was out of scope (based on the error message)**, use the error message directly.

Keep your answer direct, professional, and to the point.
"""

# --- Agent Nodes (Async & Optimized) ---

async def extract_and_prepare_node(state: TechnicalAgentState) -> Dict:
    """Extracts details, validates them, and prepares the next step in one go."""
    print("---NODE: Extract & Prepare---")
    prompt = EXTRACT_QUERY_DETAILS_PROMPT.format(question=state["question"])
    structured_llm = groq_llm.with_structured_output(ExtractedTechDetails, method="json_mode")

    try:
        details = await structured_llm.ainvoke(prompt)
        print(f"LLM Extraction: {details.model_dump_json(indent=2)}")

        if not details.is_stock_market_related:
            return {"error_message": "That question is outside my current scope of stock market analysis."}

        if not details.is_calculation_requested:
            if details.indicator_name and details.indicator_name in INDICATOR_TOOLS_MAP:
                return {"indicator_to_run": details.indicator_name, "extraction_details": details}
            else:
                return {"error_message": "I can explain indicators, but I couldn't identify a valid one in your question."}
        
        if not details.stock_identifier:
            return {"error_message": "To perform a calculation, please specify a stock name or symbol."}

        indicator_to_run = details.indicator_name
        if not indicator_to_run:
            if details.implied_category and details.implied_category in DEFAULT_INDICATORS_BY_CATEGORY:
                indicator_to_run = DEFAULT_INDICATORS_BY_CATEGORY[details.implied_category]
                print(f"Inferred category '{details.implied_category}', using default indicator '{indicator_to_run}'.")
            else:
                return {"error_message": f"Please specify a technical indicator (e.g., RSI, MACD) for {details.stock_identifier}."}

        if indicator_to_run not in INDICATOR_TOOLS_MAP:
            return {"error_message": f"The indicator '{indicator_to_run}' is not supported for calculation."}

        return {"indicator_to_run": indicator_to_run, "extraction_details": details}

    except Exception as e:
        print(f"Critical error in extraction: {e}")
        return {"error_message": "I had trouble understanding your request. Could you please rephrase?"}

async def resolve_fincode_node(state: TechnicalAgentState) -> Dict:
    """Safely and accurately resolves a stock identifier to a fincode using parameterized queries."""
    print("---NODE: Resolve Fincode---")
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

    try:
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
        return {"error_message": f"A database error occurred while looking up '{stock_id}'."}

async def fetch_price_data_node(state: TechnicalAgentState) -> Dict:
    """Asynchronously fetches price data using a parameterized query."""
    print("---NODE: Fetch Price Data---")
    fincode = state["target_fincode"]
    
    price_query = text("""
        SELECT date, open, high, low, close, volume 
        FROM accord_base_live.bse_abjusted_price_eod 
        WHERE fincode = :fincode ORDER BY date DESC LIMIT 300;
    """)

    try:
        async with async_engine.connect() as connection:
            result = await connection.execute(price_query, {"fincode": fincode})
            price_data = [dict(row._mapping) for row in result.all()]

        if not price_data:
            return {"error_message": f"No price data found for the specified stock (fincode: {fincode})."}
        
        price_data.reverse()
        print(f"Fetched {len(price_data)} price records.")
        return {"price_data": price_data}
    except Exception as e:
        print(f"Database error fetching prices: {e}")
        return {"error_message": "A database error occurred while fetching price data."}

def calculate_indicator_node(state: TechnicalAgentState) -> Dict:
    """Synchronous, CPU-bound node for technical calculations."""
    print("---NODE: Calculate Indicator---")
    indicator = state["indicator_to_run"]
    price_data = state["price_data"]
    
    tool_function = INDICATOR_TOOLS_MAP[indicator]
    print(f"Invoking {indicator} tool...")
    
    try:
        result = tool_function(price_data)
        if isinstance(result, dict) and result.get("error"):
            return {"error_message": result["error"]}
        
        # <<--- APPLY THE FIX HERE ---
        # Convert numpy types to native python types before returning
        clean_result = _convert_numpy_to_python(result)
        
        print(f"Indicator Result ({indicator}): {clean_result}")
        return {"indicator_result": clean_result}
    except Exception as e:
        print(f"Critical error during {indicator} calculation: {e}")
        return {"error_message": f"An unexpected error occurred during the {indicator} calculation."}

async def compose_final_answer_node(state: TechnicalAgentState) -> Dict:
    """A single node to compose the final answer from all available context."""
    print("---NODE: Compose Final Answer---")
    
    format_args = {
        "question": state["question"],
        "stock_identifier": state.get("extraction_details").stock_identifier if state.get("extraction_details") else "the stock",
        "indicator_name": state.get("indicator_to_run", "the requested indicator"),
        "indicator_result": state.get("indicator_result", "Not applicable."),
        "error_message": state.get("error_message", None), # Pass None if no error
    }
    
    prompt = ANSWER_COMPOSER_PROMPT.format(**format_args)
    
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
    """Routes based on the output of the preparation node."""
    if state.get("error_message"):
        return "compose_answer" # Go directly to composer to format the error
    if state["extraction_details"].is_calculation_requested:
        return "resolve_fincode"
    else:
        return "compose_answer" # For general explanations

def route_after_db_steps(state: TechnicalAgentState) -> str:
    """Routes after any DB step; if an error occurred, go to composer."""
    if state.get("error_message"):
        return "compose_answer"
    # Determine next logical step if no error
    if state.get("target_fincode") and not state.get("price_data"):
        return "fetch_prices"
    if state.get("price_data"):
        return "calculate"
    # Fallback if state is inconsistent
    return "compose_answer"

# --- Build the Graph ---
graph_builder = StateGraph(TechnicalAgentState)

graph_builder.add_node("prepare", extract_and_prepare_node)
graph_builder.add_node("resolve_fincode", resolve_fincode_node)
graph_builder.add_node("fetch_prices", fetch_price_data_node)
graph_builder.add_node("calculate", calculate_indicator_node)
graph_builder.add_node("compose_answer", compose_final_answer_node)

graph_builder.set_entry_point("prepare")

graph_builder.add_conditional_edges("prepare", route_after_preparation)
graph_builder.add_conditional_edges("resolve_fincode", route_after_db_steps)
graph_builder.add_conditional_edges("fetch_prices", route_after_db_steps)
graph_builder.add_edge("calculate", "compose_answer")
graph_builder.add_edge("compose_answer", END)

app = graph_builder.compile()

# --- Main Execution for Testing ---
if __name__ == "__main__":
    async def run_tech_agent_test():
        test_queries = [
           "hi there", # General OOD / Greeting
        "What is RSI?", # General Explanation
        "Is Reliance overbought?", # Vague -> Momentum -> RSI for Reliance
        "What's the trend for INFY stock?", # Vague -> Trend -> Supertrend for INFY
        "How volatile is Aegis Logistics now?", # Vague -> Volatility -> ATR
        "Technical analysis for TCS.", # Vague -> General Outlook -> MACD
        "Calculate the MACD for fincode 500209", # Explicit
        "VWAP for non_existent_stock_xyz", # Fincode resolution failure
        "Supertrend for fincode 9999999", # No data failure
        "What is the Parabolic SAR for fincode 3?", # Explicit
        "stochastic for hdfc bank", # Explicit
        "Who invented the radio?", # OOD
        "Is L&T strong?", # Vague -> Strength -> ADX
        "Should I buy reliance based on bollinger bands?" # Explicit (though phrasing is decision-oriented)
        ]
        for query in test_queries:
            print(f"\n\n--- TESTING: \"{query}\" ---")
            inputs = {"question": query}
            try:
                final_state = await app.ainvoke(inputs)
                print(f"\n>>> FINAL ANSWER: {final_state.get('final_answer', 'No answer found.')}")
            except Exception as e:
                print(f"\nError during agent execution: {e}")
                import traceback
                traceback.print_exc()

    asyncio.run(run_tech_agent_test())