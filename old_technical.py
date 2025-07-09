# agents/technicals_agent.py
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional
import ast
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import text
import numpy as np

from model_config import groq_llm
from db_config import engine
# Import DEFAULT_INDICATORS_BY_CATEGORY from technical_indicators
from tools.technical_indicators import INDICATOR_TOOLS_MAP, DEFAULT_INDICATORS_BY_CATEGORY # Ensure this import is correct

load_dotenv()

db = SQLDatabase(engine=engine)

# --- Define Pydantic Models for Structured Outputs ---

class LLMInitialExtraction(BaseModel):
    """Schema for the LLM's initial, detailed extraction for technical queries."""
    is_stock_market_related: bool = Field(..., description="True if the user's question is related to stock markets, finance, or technical indicators. False otherwise.")
    is_db_required: bool = Field(..., description="True if a calculation for a specific stock is mentioned or clearly implied. False for general questions or if crucial info (like stock name for calculation) is missing.")
    
    explicit_indicator_name: Optional[str] = Field(None, description="Lowercase, space-removed indicator name if EXPLICITLY mentioned by the user (e.g., \"rsi\", \"macd\"). Null if not explicitly stated.")
    
    # New field for inferred category for vague queries
    implied_indicator_category: Optional[str] = Field(None, 
        description="IF `explicit_indicator_name` IS NULL AND `is_db_required` IS TRUE, infer the category of the financial question. Choose ONLY from: 'momentum', 'trend', 'volatility', 'strength', 'general_outlook'. Null if an explicit indicator is given, if not a DB query, or if category is unidentifiable.")
    
    stock_identifier: Optional[str] = Field(None, description="Original stock identifier (name, symbol). Null if is_db_required is false or no stock is mentioned for calculation.")

class TechnicalContextOutput(BaseModel):
    """Structured output for the first node, containing extracted and processed context."""
    is_stock_market_related: bool
    is_db_required: bool
    indicator_name: Optional[str] # Final indicator_name to use (explicit or default from category)
    stock_identifier: Optional[str]
    default_indicator_reason: Optional[str] = None # For user feedback if a default was chosen


# --- Define TypedDict for State ---
class TechnicalAgentState(TypedDict):
    question: str
    extraction_output: Optional[TechnicalContextOutput] # Will hold TechnicalContextOutput
    target_fincode: Optional[int]
    price_data: Optional[TypingList[Dict[str, Any]]]
    # Indicator result now includes value and signal from technical_indicators.py
    indicator_result: Optional[Dict[str, Any]] 
    draft_answer: Optional[str]
    final_answer: Optional[str]


# --- Prompts ---

EXTRACT_QUERY_DETAILS_PROMPT = """
You are an expert financial analyst. Analyze the user's question about stock markets and technical indicators.
Your goal is to extract specific details to determine the best course of action: either a general explanation or a specific calculation.
Your output MUST be a single, valid JSON object matching the LLMInitialExtraction schema.

LLMInitialExtraction Schema:
{{
  "is_stock_market_related": "boolean",
  "is_db_required": "boolean",
  "explicit_indicator_name": "Optional[string]",
  "implied_indicator_category": "Optional[string]",
  "stock_identifier": "Optional[string]"
}}

Extraction Rules:
1.  `is_stock_market_related`: (boolean) True if the question is about stocks, finance, or technical indicators. False otherwise.
2.  `is_db_required`: (boolean) True ONLY IF a specific stock calculation is clearly requested or implied AND a stock identifier is present. False for general questions (e.g., "What is RSI?"), or if a stock identifier is missing for a calculation-type question.
3.  `explicit_indicator_name`: (string, lowercase, no spaces, or null)
    -   The technical indicator name if EXPLICITLY mentioned (e.g., "RSI", "Bollinger Bands" -> "rsi", "bollingerbands").
    -   If not explicitly mentioned, this MUST BE NULL.
4.  `implied_indicator_category`: (string, or null)
    -   This field is ONLY considered IF `explicit_indicator_name` IS NULL AND `is_db_required` IS TRUE.
    -   Infer the category of the question if it's vague but implies a calculation for a stock. Choose ONE from:
        - 'momentum': For questions about overbought/oversold conditions, rate of price change (e.g., "is Tesla overbought?", "momentum of AAPL?").
        - 'trend': For questions about price direction (e.g., "is MSFT trending upwards?", "what's the current trend for GOOGL?").
        - 'volatility': For questions about price fluctuation range/risk (e.g., "how volatile is NVDA recently?").
        - 'strength': For questions about the strength/conviction of a trend (e.g., "how strong is the uptrend in AMZN stock?").
        - 'general_outlook': If a stock is named for analysis but no clear indicator or specific category is implied (e.g., "technical view on Microsoft").
    -   If `explicit_indicator_name` is NOT NULL, or if `is_db_required` is FALSE, this MUST BE NULL.
5.  `stock_identifier`: (string or null)
    -   If `is_db_required` is true, this is the stock symbol or company name (e.g., "Reliance Industries", "INFY", "500209").
    -   If `is_db_required` is false (e.g. general question, or stock name missing for calc), this MUST BE NULL.

User Question: {question}

Examples:
User Question: "What is the RSI for Reliance Industries?"
Output: {{ "is_stock_market_related": true, "is_db_required": true, "explicit_indicator_name": "rsi", "implied_indicator_category": null, "stock_identifier": "Reliance Industries" }}

User Question: "Is Apple overbought right now?"
Output: {{ "is_stock_market_related": true, "is_db_required": true, "explicit_indicator_name": null, "implied_indicator_category": "momentum", "stock_identifier": "Apple" }}

User Question: "What's the trend for INFY?"
Output: {{ "is_stock_market_related": true, "is_db_required": true, "explicit_indicator_name": null, "implied_indicator_category": "trend", "stock_identifier": "INFY" }}

User Question: "What is SMA?"
Output: {{ "is_stock_market_related": true, "is_db_required": false, "explicit_indicator_name": "sma", "implied_indicator_category": null, "stock_identifier": null }}

User Question: "Technical analysis for Adani Green."
Output: {{ "is_stock_market_related": true, "is_db_required": true, "explicit_indicator_name": null, "implied_indicator_category": "general_outlook", "stock_identifier": "Adani Green" }}

User Question: "Who invented the television?"
Output: {{ "is_stock_market_related": false, "is_db_required": false, "explicit_indicator_name": null, "implied_indicator_category": null, "stock_identifier": null }}

Provide ONLY the JSON output.
"""

GENERAL_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are IRIS, a helpful financial assistant. The user asked about a technical indicator."),
    ("user", "User question: {question}\nIndicator identified: {indicator_name_display}\nPlease provide a clear and concise explanation of what {indicator_name_display} is, how it's generally used, and any common interpretations. Do not refer to specific stocks. Explain it simply, as if to a beginner. Keep it brief (2-3 sentences).")
])

DB_ANSWER_DRAFT_PROMPT = """
You are IRIS, a financial assistant.
User asked: "{question}"
Analysis for: Stock (user input: '{stock_identifier}', resolved to fincode: {target_fincode})
Indicator: {indicator_name_display} {default_indicator_reason_text}
Calculation Result or Error (as a string for your reference): {indicator_result_str}

Provide a concise, natural language answer based on the 'Calculation Result or Error'.
Your response should clearly state the main indicator value(s) and the derived 'signal'.

- If the calculation was successful (no 'error' key in the result):
    - State the primary indicator value(s). For example, if RSI was calculated, "The 14-day RSI for {stock_identifier} is {rsi_value}." If MACD, "For {stock_identifier}, the MACD Line is {macd_line_value}, Signal Line is {signal_line_value}, and Histogram is {histogram_value}."
    - Then, state the signal. For example, "This suggests a '{calculated_signal}' signal." or "The current signal is '{calculated_signal}'."
    - If a 'note' was part of the calculation result, include it briefly if relevant to the signal.

- If 'Calculation Result or Error' indicates an error (e.g., contains an 'error' key):
  Explain this error clearly: "I couldn't calculate the {indicator_name_display} for {stock_identifier} because: {error_message_from_result}."

Be helpful and informative. Keep the answer straightforward and to the point.

Example of a successful RSI calculation response:
"The 14-day RSI for Reliance is 56.58. This currently suggests a 'neutral' signal."

Example of a successful MACD calculation response:
"For Infosys, the MACD Line is 0.25, Signal Line is 0.18, and Histogram is 0.07. The current MACD signal is 'buy'."

Example of an error response:
"I couldn't calculate the RSI for UnknownStock because: Could not find a unique fincode for stock: 'UnknownStock'."
"""

NORMALIZE_GENERAL_EXPLANATION_PROMPT = """
You are IRIS, a helpful financial assistant.
The user asked: "{question}"
Review the draft explanation below and ensure it's clear, natural, polite, and directly answers the question in simple terms.
Make your tone helpful and informative. Ensure it is easy for a beginner to understand. Remove any conversational filler. Keep it concise (2-3 sentences).

Draft Explanation: {draft_answer}

Refined Final Explanation:
"""

VALIDATE_DB_ANSWER_PROMPT = """
You are IRIS. Refine the following draft answer to be clear, natural, and directly address the user's original question: "{question}".
The analysis concerned: stock '{stock_identifier}' (fincode: '{target_fincode}'), using indicator {indicator_name_display} {default_indicator_reason_text}.
Make your tone helpful and informative. Do not mention internal calculation steps unless essential for an error message.
Ensure the answer correctly reflects the calculated data and the derived signal from the draft.
If the draft indicates an error, the refined answer should clearly state the error.
Remove conversational filler. Keep it concise and to the point.

Draft Answer: {draft_answer}

Refined Final Answer:
"""

# --- Agent Nodes ---

def extract_query_details_node(state: TechnicalAgentState):
    print("---NODE: Extract Query Details (Technical Enhanced)---")
    prompt_str = EXTRACT_QUERY_DETAILS_PROMPT.format(question=state["question"])
    structured_llm = groq_llm.with_structured_output(LLMInitialExtraction, method="json_mode")
    
    final_indicator_name = None
    default_reason = None
    final_stock_identifier = None
    final_is_db_required = False
    final_is_stock_market_related = False

    try:
        llm_output: LLMInitialExtraction = structured_llm.invoke(prompt_str)
        print(f"LLM Raw Extraction: {llm_output}")

        final_is_stock_market_related = llm_output.is_stock_market_related
        
        if not final_is_stock_market_related:
            # If not stock market related, all other flags become False/None
            final_is_db_required = False
            final_stock_identifier = None
            final_indicator_name = None
            default_reason = "The question does not seem related to stock markets or technical analysis."
        else:
            # Stock market related, now determine indicator and if DB is needed
            final_is_db_required = llm_output.is_db_required # Trust LLM's initial assessment
            final_stock_identifier = llm_output.stock_identifier if final_is_db_required else None

            if llm_output.explicit_indicator_name:
                final_indicator_name = llm_output.explicit_indicator_name.lower().replace(" ", "").replace("_", "")
                if final_indicator_name not in INDICATOR_TOOLS_MAP:
                    default_reason = f"The indicator '{final_indicator_name}' you mentioned is not one I can calculate. I can explain it generally."
                    # We can still attempt general explanation, so keep indicator_name. But DB calc is not possible.
                    final_is_db_required = False 
                    final_stock_identifier = None # No calculation, so no stock needed for DB
            elif final_is_db_required and llm_output.implied_indicator_category:
                category = llm_output.implied_indicator_category.lower()
                if category in DEFAULT_INDICATORS_BY_CATEGORY:
                    final_indicator_name = DEFAULT_INDICATORS_BY_CATEGORY[category]
                    default_reason = f"(using {final_indicator_name.upper()} as it's common for '{category}' analysis)"
                    print(f"Vague query: using default indicator '{final_indicator_name}' for category '{category}'.")
                else: # Category identified but no default mapping, or invalid category
                    default_reason = f"I understood you're asking about '{category}' for {final_stock_identifier}, but couldn't pick a default indicator. Please specify one."
                    final_is_db_required = False # Cannot proceed with DB if no specific indicator derived
            elif final_is_db_required and not final_indicator_name: # DB required but no explicit or default indicator
                 default_reason = f"To perform a calculation for {final_stock_identifier}, please specify a technical indicator (e.g., RSI, MACD)."
                 final_is_db_required = False 
            
            # If, after all logic, db is required but no stock_identifier, make db_required false
            if final_is_db_required and not final_stock_identifier:
                default_reason = (default_reason + " " if default_reason else "") + "Also, a stock name is needed for calculation."
                final_is_db_required = False

        final_extraction_output = TechnicalContextOutput(
            is_stock_market_related=final_is_stock_market_related,
            is_db_required=final_is_db_required,
            indicator_name=final_indicator_name,
            stock_identifier=final_stock_identifier if final_is_db_required else None, # Ensure stock_id is None if not DB path
            default_indicator_reason=default_reason
        )
        
        print(f"Processed Extraction Result: {final_extraction_output}")
        return {"extraction_output": final_extraction_output}

    except Exception as e:
        print(f"Error in extract_query_details_node: {e}. LLM output may have been malformed.")
        # Fallback for critical LLM failure
        error_reason = "I had trouble understanding your request. Could you please rephrase?"
        # Basic check for greetings even on LLM failure
        q_lower = state["question"].lower()
        if any(g in q_lower for g in ["hi", "hello", "how are you"]):
            error_reason = "Hello! I'm IRIS. How can I help with your stock analysis today?"
            # Assume stock-related for greetings, but no DB unless specified
            final_is_stock_market_related = True 

        fallback_extraction = TechnicalContextOutput(
            is_stock_market_related=final_is_stock_market_related, 
            is_db_required=False, 
            indicator_name=None, 
            stock_identifier=None,
            default_indicator_reason=error_reason
        )
        print(f"Error Extraction Fallback: {fallback_extraction}")
        return {"extraction_output": fallback_extraction}

def resolve_fincode_node(state: TechnicalAgentState):
    print("---NODE: Resolve Fincode---")
    extraction = state["extraction_output"]
    
    if extraction is None: # Should be caught by router
        error_msg = "Internal error: Extraction output is missing."
        return {"target_fincode": None, "indicator_result": {"error": error_msg, "signal": "error"}}

    if not extraction.stock_identifier: # Should be caught by router if db_path chosen
        error_msg = extraction.default_indicator_reason or "Stock identifier not provided for fincode resolution."
        return {"target_fincode": None, "indicator_result": {"error": error_msg, "signal": "error"}}

    user_identifier = extraction.stock_identifier
    resolved_fincode_val = None
    error_message = None
    fincode_lookup_query = ""

    try:
        num_identifier = int(user_identifier)
        fincode_lookup_query = f"SELECT fincode FROM accord_base_live.company_master WHERE fincode = {num_identifier} OR scripcode = '{str(num_identifier)}' LIMIT 1"
    except ValueError:
        safe_user_identifier = user_identifier.replace("'", "''")
        fincode_lookup_query = f"SELECT fincode FROM accord_base_live.company_master WHERE compname LIKE '%{safe_user_identifier}%' OR symbol = '{safe_user_identifier}' ORDER BY CASE WHEN compname = '{safe_user_identifier}' THEN 0 WHEN symbol = '{safe_user_identifier}' THEN 1 ELSE 2 END, LENGTH(compname) ASC LIMIT 1;"
    
    print(f"Fincode Lookup SQL: {fincode_lookup_query}")
    db_tool = QuerySQLDatabaseTool(db=db)

    try:
        db_result_str = db_tool.invoke(fincode_lookup_query)
        eval_result = ast.literal_eval(db_result_str)
        if isinstance(eval_result, list) and len(eval_result) > 0:
            first_row = eval_result[0]
            if isinstance(first_row, tuple) and len(first_row) > 0:
                resolved_fincode_val = int(first_row[0])
        
        if resolved_fincode_val is not None:
            print(f"Resolved Fincode: {resolved_fincode_val}")
        else:
            error_message = f"Could not find a unique fincode for stock: '{user_identifier}'. Please try a more specific name/symbol."
            
    except Exception as e:
        print(f"Error during fincode resolution DB call/parsing: {e}")
        error_message = f"Database error or parsing issue for '{user_identifier}'. Details: {str(e)[:150]}"
        
    if error_message:
        return {"target_fincode": None, "indicator_result": {"error": error_message, "signal": "error"}}
    return {"target_fincode": resolved_fincode_val}


def fetch_price_data_node(state: TechnicalAgentState):
    print("---NODE: Fetch Price Data---")
    target_fincode = state["target_fincode"]

    if state.get("indicator_result", {}).get("error"):
        return {} 

    if target_fincode is None:
        return {"price_data": None, "indicator_result": {"error": "Fincode not resolved, cannot fetch prices.", "signal": "error"}}

    price_query_str = f"""
    SELECT date, open AS open, high AS high, low AS low, close AS close, volume 
    FROM accord_base_live.bse_abjusted_price_eod 
    WHERE fincode = :fincode_param ORDER BY date DESC LIMIT 300;
    """
    
    price_data_list_of_dicts = []
    try:
        with engine.connect() as connection:
            result = connection.execute(text(price_query_str), {"fincode_param": target_fincode})
            column_names = list(result.keys())
            for row in result:
                row_dict = dict(zip(column_names, row))
                if 'date' in row_dict and hasattr(row_dict['date'], 'isoformat'):
                    row_dict['date'] = row_dict['date'].isoformat()
                elif 'date' in row_dict and row_dict['date'] is not None:
                    row_dict['date'] = str(row_dict['date'])
                price_data_list_of_dicts.append(row_dict)

        if not price_data_list_of_dicts:
            return {"price_data": None, "indicator_result": {"error": f"No price data found for fincode {target_fincode}.", "signal": "error"}}
        price_data_list_of_dicts.reverse()
        print(f"Fetched {len(price_data_list_of_dicts)} price records for fincode {target_fincode}.")
        return {"price_data": price_data_list_of_dicts}
    except Exception as e:
        error_detail = f"Database error fetching prices for fincode {target_fincode}. Details: {str(e)[:150]}"
        return {"price_data": None, "indicator_result": {"error": error_detail, "signal": "error"}}

def convert_numpy_to_python(data: Any) -> Any:
    if isinstance(data, dict): return {k: convert_numpy_to_python(v) for k, v in data.items()}
    if isinstance(data, list): return [convert_numpy_to_python(i) for i in data]
    if isinstance(data, np.integer): return int(data)
    if isinstance(data, np.floating): return float(data)
    if isinstance(data, np.ndarray): return data.tolist()
    if isinstance(data, np.bool_): return bool(data)
    return data

def calculate_indicator_node(state: TechnicalAgentState):
    print("---NODE: Calculate Indicator---")
    extraction = state["extraction_output"]
    price_data = state.get("price_data")

    if state.get("indicator_result", {}).get("error"): return {}

    if not extraction or not extraction.indicator_name: # Should be caught by routing
        return {"indicator_result": {"error": "Internal error: Indicator name missing.", "signal": "error"}}
    
    indicator_name = extraction.indicator_name
    if indicator_name not in INDICATOR_TOOLS_MAP:
        error_msg = extraction.default_indicator_reason or f"Unsupported indicator: '{indicator_name}'."
        return {"indicator_result": {"error": error_msg, "signal": "error"}}

    if not price_data: # Should be caught by routing after fetch_prices
        return {"indicator_result": {"error": "No price data for calculation.", "signal": "error"}}

    tool_function = INDICATOR_TOOLS_MAP[indicator_name]
    print(f"Invoking {indicator_name} tool...")
    try:
        raw_result = tool_function(price_data) 
        # Ensure 'signal' key exists, default to 'error' if calculation itself failed and returned error string
        if isinstance(raw_result, dict) and "error" in raw_result and "signal" not in raw_result:
            raw_result["signal"] = "error"
        elif isinstance(raw_result, dict) and "signal" not in raw_result : # Calculation succeeded but indicator func didn't add signal
            raw_result["signal"] = "neutral" # Default signal if not provided by calc func

        python_native_result = convert_numpy_to_python(raw_result)
        print(f"Indicator Result ({indicator_name}): {python_native_result}")
        return {"indicator_result": python_native_result}
    except Exception as e:
        print(f"Error calculating {indicator_name}: {e}")
        return {"indicator_result": {"error": f"Error during {indicator_name} calculation: {str(e)}", "signal": "error"}}

def generate_db_answer_draft_node(state: TechnicalAgentState):
    print("---NODE: Generate DB Answer Draft (Technical)---")
    extraction = state["extraction_output"]
    indicator_result = state.get("indicator_result", {}) 
    question = state["question"]
    target_fincode = state.get("target_fincode")

    if not extraction:
        return {"draft_answer": "IRIS: Internal error (missing extraction details)."}

    default_reason_text = extraction.default_indicator_reason or ""
    
    indicator_name_display = extraction.indicator_name.upper() if extraction.indicator_name else "the indicator"
    stock_id_display = extraction.stock_identifier if extraction.stock_identifier else "the stock"

    # Prepare values for the prompt's .format()
    # These must match placeholders that are *actually* in the prompt string, not just in examples.
    format_args = {
        "question": question,
        "stock_identifier": stock_id_display,
        "target_fincode": target_fincode if target_fincode is not None else "N/A",
        "indicator_name_display": indicator_name_display,
        "default_indicator_reason_text": default_reason_text,
        "indicator_result_str": str(indicator_result), # LLM sees the whole result as a string
    }

    # Add specific values if available, otherwise LLM infers from indicator_result_str
    if isinstance(indicator_result, dict):
        format_args["error_message_from_result"] = indicator_result.get("error", "an unknown issue occurred.")
        format_args["calculated_signal"] = indicator_result.get("signal", "neutral") # Default if not present
        
        # For common indicators, provide their primary values directly
        if extraction.indicator_name == "rsi" and "rsi" in indicator_result:
            format_args["rsi_value"] = indicator_result["rsi"]
        elif extraction.indicator_name == "macd":
            format_args["macd_line_value"] = indicator_result.get("macd_line", "N/A")
            format_args["signal_line_value"] = indicator_result.get("signal_line", "N/A")
            format_args["histogram_value"] = indicator_result.get("histogram", "N/A")
        # Add more for other indicators if you want to explicitly pass their values
        # Otherwise, the LLM will need to parse them from indicator_result_str based on the examples.
    else: # indicator_result is not a dict, likely an error string already
        format_args["error_message_from_result"] = str(indicator_result)
        format_args["calculated_signal"] = "error"


    # Fill any remaining specific value placeholders with "N/A" if not set,
    # so .format() doesn't break, though LLM should primarily use indicator_result_str
    # and the general instructions.
    # The prompt is now designed to let LLM extract from indicator_result_str mostly.
    potential_value_keys = ["rsi_value", "macd_line_value", "signal_line_value", "histogram_value"]
    for key in potential_value_keys:
        format_args.setdefault(key, "N/A")


    try:
        # The prompt string itself doesn't use rsi_key, signal_key anymore.
        # It instructs the LLM on how to formulate the sentence using values from indicator_result_str.
        final_prompt_str = DB_ANSWER_DRAFT_PROMPT.format(**format_args)
        response = groq_llm.invoke(final_prompt_str)
        return {"draft_answer": response.content}
    except KeyError as ke:
        print(f"KeyError during DB_ANSWER_DRAFT_PROMPT formatting: {ke}. This means a placeholder is still in the prompt string that's not in format_args.")
        print(f"Available format_args: {format_args.keys()}")
        return {"draft_answer": f"IRIS: Internal error formatting the response (KeyError: {ke})."}
    except Exception as e:
        print(f"Error in generate_db_answer_draft_node LLM call: {e}")
        error_part = indicator_result.get('error', 'processing data') if isinstance(indicator_result, dict) else str(indicator_result)
        fallback = f"IRIS: For {stock_id_display}, I had trouble generating the full analysis for {indicator_name_display}. Problem: {error_part} (LLM draft error)."
        return {"draft_answer": fallback}

def generate_general_answer_draft_node(state: TechnicalAgentState):
    print("---NODE: Generate General Answer Draft (Technical)---")
    extraction = state["extraction_output"]
    question = state["question"]

    if not extraction or not extraction.indicator_name:
        err_detail = (extraction.default_indicator_reason if extraction else "") or f"I couldn't identify the indicator in '{question}'. Please specify."
        return {"draft_answer": f"I can explain indicators, but {err_detail}"}

    prompt = GENERAL_EXPLANATION_PROMPT.invoke({
        "question": question, "indicator_name_display": extraction.indicator_name.upper()
    })
    try:
        return {"draft_answer": groq_llm.invoke(prompt).content}
    except Exception as e:
        return {"draft_answer": f"IRIS: Error explaining {extraction.indicator_name.upper()}: {e}"}


def generate_out_of_domain_answer_node(state: TechnicalAgentState):
    print("---NODE: Generate Out-of-Domain Answer (Technical)---")
    extraction = state.get("extraction_output")
    answer = extraction.default_indicator_reason if extraction and extraction.default_indicator_reason and not extraction.is_stock_market_related \
             else "I'm IRIS, your financial markets assistant. I specialize in stocks and technical analysis, but that question is outside my current scope."
    if any(g in state["question"].lower() for g in ["hello", "hi", "how are you"]):
        answer = "Hello! I'm IRIS, ready for your stock market and technical indicator questions."
    return {"draft_answer": answer}


def finalize_answer_node(state: TechnicalAgentState):
    print("---NODE: Finalize Answer (Technical)---")
    extraction = state["extraction_output"]
    draft_answer = state.get("draft_answer", "Apologies, I hit an issue generating a response.")
    question = state["question"]
    
    canned_error_keywords = [ "outside my current scope", "couldn't identify the indicator", "Cannot perform calculation", "internal error", "had trouble understanding" ]
    if any(keyword in draft_answer for keyword in canned_error_keywords):
        return {"final_answer": draft_answer} # Pass through pre-canned errors/OOD

    default_reason_text = ""
    if extraction and extraction.default_indicator_reason and \
       ("using " in extraction.default_indicator_reason or "common for" in extraction.default_indicator_reason): # Only include if it's about *why* a default was used
            default_reason_text = extraction.default_indicator_reason


    if extraction and extraction.is_stock_market_related:
        if extraction.is_db_required:
            prompt_template_str = VALIDATE_DB_ANSWER_PROMPT
            format_args = {
                "question": question,
                "stock_identifier": extraction.stock_identifier or "N/A",
                "target_fincode": state.get("target_fincode", "N/A"),
                "indicator_name_display": (extraction.indicator_name.upper() if extraction.indicator_name else "indicator"),
                "default_indicator_reason_text": default_reason_text,
                "draft_answer": draft_answer
            }
        else: # General explanation
            prompt_template_str = NORMALIZE_GENERAL_EXPLANATION_PROMPT
            format_args = {"question": question, "draft_answer": draft_answer}
    else: # Fallback for OOD if not caught
        prompt_template_str = NORMALIZE_GENERAL_EXPLANATION_PROMPT
        format_args = {"question": question, "draft_answer": draft_answer}

    try:
        final_prompt = ChatPromptTemplate.from_template(prompt_template_str).format_prompt(**format_args)
        return {"final_answer": groq_llm.invoke(final_prompt).content}
    except Exception as e:
        print(f"Error in finalize_answer_node LLM (Technical): {e}")
        return {"final_answer": draft_answer }


# --- Graph Conditional Edges ---
def route_after_extraction(state: TechnicalAgentState):
    extraction = state["extraction_output"]
    if not extraction: 
        state["draft_answer"] = "Internal error: Could not process query details." # For finalize_error_directly
        return "finalize_error_directly" 

    if not extraction.is_stock_market_related:
        return "out_of_domain_answer_path"

    if extraction.is_db_required: # Implies stock_id and indicator_name should be present
        if extraction.stock_identifier and extraction.indicator_name and extraction.indicator_name in INDICATOR_TOOLS_MAP:
            return "db_resolve_fincode"
        else: # DB was desired, but crucial info missing or indicator unsupported for calc
            # extraction.default_indicator_reason should contain why it's not proceeding to DB
            state["draft_answer"] = extraction.default_indicator_reason or "Cannot perform calculation due to missing stock or unsupported indicator."
            return "finalize_error_directly" # Go to finalize to present this message
    else: # General explanation path (is_stock_market_related=True, is_db_required=False)
        if extraction.indicator_name: 
            return "general_explanation_draft_path"
        else: # Stock market related, but no indicator identified for explanation
            state["draft_answer"] = extraction.default_indicator_reason or f"Please specify which indicator you'd like to know about for '{state['question'][:30]}...'."
            return "finalize_error_directly"


def route_after_fincode_resolution(state: TechnicalAgentState):
    if state.get("indicator_result", {}).get("error"): # Error from resolve_fincode_node
        return "generate_db_answer_draft" 
    if state.get("target_fincode") is not None:
        return "fetch_price_data"
    else: # Fincode None, but no error explicitly set (should be rare if resolve_fincode is robust)
        stock_id = state['extraction_output'].stock_identifier if state.get('extraction_output') else 'the stock'
        state["indicator_result"] = {"error": f"Could not find fincode for '{stock_id}'.", "signal": "error"}
        return "generate_db_answer_draft"

def route_after_price_fetch(state: TechnicalAgentState):
    if state.get("indicator_result", {}).get("error"): # Error from fetch_price_data_node
        return "generate_db_answer_draft"
    if state.get("price_data"):
        return "calculate_indicator"
    else: # No price data and no error set
        state["indicator_result"] = {"error": "Price data unavailable after fetch attempt.", "signal": "error"}
        return "generate_db_answer_draft"

# --- Build the Graph ---
graph_builder_tech = StateGraph(TechnicalAgentState)

graph_builder_tech.add_node("extract_query_details", extract_query_details_node)
graph_builder_tech.add_node("resolve_fincode", resolve_fincode_node)
graph_builder_tech.add_node("fetch_prices", fetch_price_data_node)
graph_builder_tech.add_node("calculate_indicator", calculate_indicator_node)
graph_builder_tech.add_node("generate_db_answer_draft", generate_db_answer_draft_node)
graph_builder_tech.add_node("generate_general_explanation_draft", generate_general_answer_draft_node)
graph_builder_tech.add_node("generate_out_of_domain_answer", generate_out_of_domain_answer_node)
graph_builder_tech.add_node("finalize_answer", finalize_answer_node) # Single finalizer

graph_builder_tech.set_entry_point("extract_query_details")

graph_builder_tech.add_conditional_edges(
    "extract_query_details",
    route_after_extraction,
    {
        "out_of_domain_answer_path": "generate_out_of_domain_answer",
        "db_resolve_fincode": "resolve_fincode",
        "general_explanation_draft_path": "generate_general_explanation_draft",
        "finalize_error_directly": "finalize_answer" # Critical errors or non-calculable paths from extraction
    }
)
graph_builder_tech.add_conditional_edges("resolve_fincode", route_after_fincode_resolution,
    {"fetch_price_data": "fetch_prices", "generate_db_answer_draft": "generate_db_answer_draft"} )
graph_builder_tech.add_conditional_edges("fetch_prices", route_after_price_fetch,
    {"calculate_indicator": "calculate_indicator", "generate_db_answer_draft": "generate_db_answer_draft"})

graph_builder_tech.add_edge("calculate_indicator", "generate_db_answer_draft")
graph_builder_tech.add_edge("generate_db_answer_draft", "finalize_answer")
graph_builder_tech.add_edge("generate_general_explanation_draft", "finalize_answer")
graph_builder_tech.add_edge("generate_out_of_domain_answer", "finalize_answer") # All paths lead to one finalizer
graph_builder_tech.add_edge("finalize_answer", END)

app = graph_builder_tech.compile()


if __name__ == "__main__":
    print("Technical Analysis Agent (Enhanced for Vague Queries & Signals) Compiled.")
    test_queries_tech = [
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

    for q_text in test_queries_tech:
        print(f"\n\n--- TESTING TECHNICAL QUERY: \"{q_text}\" ---")
        inputs = {"question": q_text}
        try:
            config = {"recursion_limit": 20} 
            for event_map in app.stream(inputs, config=config, stream_mode="values"):
                last_changed_node = list(event_map.keys())[-1]
                print(f"\n  State after node ~'{last_changed_node}':")
                if "extraction_output" in event_map and event_map["extraction_output"]:
                    print(f"    Extraction: {event_map['extraction_output']}")
                if "target_fincode" in event_map and event_map["target_fincode"] is not None:
                    print(f"    Target Fincode: {event_map['target_fincode']}")
                if "indicator_result" in event_map and event_map["indicator_result"]:
                    print(f"    Indicator Result: {event_map['indicator_result']}")
                # if "price_data" in event_map and event_map["price_data"]: # Can be verbose
                #     print(f"    Price Data Count: {len(event_map['price_data']) if event_map['price_data'] else 'None/Empty'}")
                if "draft_answer" in event_map and event_map["draft_answer"]:
                    print(f"    Draft Answer: {event_map['draft_answer']}")
                if "final_answer" in event_map and event_map["final_answer"]:
                    print(f"    >>> FINAL ANSWER: {event_map['final_answer']}")
        except Exception as e:
            print(f"Error invoking graph for \"{q_text}\": {e}")
            import traceback
            traceback.print_exc()
        print("---------------------------------------\n")

    try:
        print("\nAttempting to generate technical_agent_graph_vague_queries.png...")
        img_data = app.get_graph().draw_mermaid_png()
        with open("technical_agent_graph_vague_queries.png", "wb") as f:
            f.write(img_data)
        print("Graph saved to technical_agent_graph_vague_queries.png")
    except Exception as e:
        print(f"Could not generate graph visualization: {e}")