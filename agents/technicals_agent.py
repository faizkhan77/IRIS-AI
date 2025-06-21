# agents/technicals_agent.py
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional
import ast # For parsing price data string
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool # Still useful for executing fixed queries
from langchain_core.prompts import ChatPromptTemplate # Keep for other LLM calls
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import text
# Custom module imports
from model_config import groq_llm
from db_config import engine # Your SQLAlchemy engine
from tools.technical_indicators import INDICATOR_TOOLS_MAP

load_dotenv()

db = SQLDatabase(engine=engine) # Langchain SQLDatabase utility

# --- Define Pydantic Models for Structured Outputs ---

class LLMInitialExtraction(BaseModel):
    """Schema for the LLM's initial, basic extraction."""
    is_stock_market_related: bool = Field(..., description="True if the user's question is related to stock markets, finance, or technical indicators. False otherwise.")
    is_db_required: bool = Field(..., description="True if calculation for a specific stock is needed. False for general questions.")
    indicator_name: Optional[str] = Field(None, description="Lowercase, space-removed indicator name, e.g., \"rsi\", \"macd\". Null if not clearly identified.")
    stock_identifier: Optional[str] = Field(None, description="Original stock identifier. Null if is_db_required is false.")

class TechnicalContextOutput(BaseModel):
    """Structured output for the first node, containing extracted context."""
    is_stock_market_related: bool
    is_db_required: bool
    indicator_name: Optional[str] # Normalized: lowercase, no spaces
    stock_identifier: Optional[str] # Original from user or cleaned

# SQLQueryOutput is no longer needed as we'll hardcode queries

# --- Define TypedDict for State ---
class TechnicalAgentState(TypedDict):
    question: str
    extraction_output: Optional[TechnicalContextOutput]
    target_fincode: Optional[int]
    # Removed price_data_query as it's fixed now
    price_data: Optional[TypingList[Dict[str, Any]]]
    indicator_result: Optional[Dict[str, Any]]
    draft_answer: Optional[str]
    final_answer: Optional[str]


# --- Prompts ---

EXTRACT_QUERY_DETAILS_PROMPT = """
You are an expert financial analyst. Analyze the user's question about technical indicators.
Extract the following three pieces of information.
Your output MUST be a single, valid JSON object matching the LLMInitialExtraction schema, containing ONLY: `is_stock_market_related`, `is_db_required`, `indicator_name`, and `stock_identifier`.

1.  is_stock_market_related: (boolean) True if the user's question is about stock markets, companies stock, or technical indicators. False for any other topics (e.g., history, science, general knowledge).
1.  is_db_required: (boolean) True if calculation for a specific stock is needed. False for general questions (e.g., "What is RSI?").
2.  indicator_name: (string, lowercase or null) The name of the technical indicator. Standardize common names (e.g., "Bollinger Bands" becomes "bollingerbands", "RSI" becomes "rsi"). If a general question doesn't clearly state one specific indicator, this can be null.
3.  stock_identifier: (string or null) If `is_db_required` is true, this is the stock symbol, scripcode, fincode, or company name mentioned by the user (e.g., "Reliance Industries", "INFY", "500209"). If `is_db_required` is false, this MUST BE NULL.

User Question: {question}

Examples (showing ONLY the four fields you should generate):
Output: {{ "is_stock_market_related": true, "is_db_required": true, "indicator_name": "rsi", "stock_identifier": "Reliance Industries" }}

User Question: "What is SMA?"
Output: {{ "is_stock_market_related": true, "is_db_required": false, "indicator_name": "sma", "stock_identifier": null }}

User Question: "Tell me about moving averages."
Output: {{ "is_stock_market_related": true, "is_db_required": false, "indicator_name": null, "stock_identifier": null }}

User Question: "Who invented the television?"
Output: {{ "is_stock_market_related": false, "is_db_required": false, "indicator_name": null, "stock_identifier": null }}

User Question: "What's the weather like today?"
Output: {{ "is_stock_market_related": false, "is_db_required": false, "indicator_name": null, "stock_identifier": null }}

Provide only the JSON output. Do not include any other text or explanations.
"""
# FINCODE_LOOKUP_SQL_PROMPT_TEMPLATE and PRICE_DATA_SQL_PROMPT_TEMPLATE removed

GENERAL_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are IQUN-AI, a helpful financial assistant. The user asked about a technical indicator."),
    ("user", "User question: {question}\nIndicator identified: {indicator_name_display}\nPlease provide a clear and concise explanation of what {indicator_name_display} is, how it's generally used, and any common interpretations. Do not refer to specific stocks. Explain it simply, as if to a beginner.")
])

DB_ANSWER_DRAFT_PROMPT = """
You are IQUN-AI, a financial assistant.
User asked: "{question}"
Analysis for: Stock (user input: '{stock_identifier}', resolved to fincode: {target_fincode})
Indicator: {indicator_name_display} (calculated using its standard default parameters)
Calculation Result or Error: {indicator_result_str}

Provide a concise, natural language answer based on the 'Calculation Result or Error'.
- If 'Calculation Result or Error' contains specific indicator values (e.g., a dictionary like '{{"rsi": 65.7}}' or '{{"macd_line": 0.1, "signal_line": 0.2, "histogram": -0.1}}'), present these values clearly.
  For example:
    - "The standard 14-day RSI for {stock_identifier} is [value from rsi key]."
    - "For {stock_identifier}, using default periods, the MACD lines are: MACD Line: [value from macd_line key], Signal Line: [value from signal_line key], and Histogram: [value from histogram key]."
  Adapt your presentation based on the actual keys and values in '{indicator_result_str}'. If appropriate, mention the common default period for the indicator (e.g., "14-day RSI", "20-period EMA").
- If 'Calculation Result or Error' contains an 'error' key (e.g., a dictionary like '{{"error": "Insufficient data."}}'), explain this error clearly and politely.
  For example: "I couldn't calculate the {indicator_name_display} for {stock_identifier} because: [the error message from the 'error' key]."

Be helpful and informative.
"""

NORMALIZE_GENERAL_EXPLANATION_PROMPT = """
You are IQUN-AI, a helpful financial assistant.
The user asked: "{question}"
Review the draft explanation below and ensure it's clear, natural, polite, and directly answers the question in simple terms.
Make your tone helpful and informative. Ensure it is easy for a beginner to understand. Remove any conversational filler or self-references that are not part of the core answer.

Draft Explanation: {draft_answer}

Refined Final Explanation:
"""

VALIDATE_DB_ANSWER_PROMPT = """
You are IQUN-AI. Refine the following draft answer to be clear, natural, and directly address the user's original question: "{question}".
The analysis concerned: user input '{stock_identifier}', resolved to fincode '{target_fincode}'.
The {indicator_name_display} was calculated using its standard default parameters.
Make your tone helpful and informative. Do not mention internal calculation steps or SQL queries unless crucial for understanding an error reported in the draft answer.
Ensure the answer correctly reflects the data provided in the draft. If the draft indicates an error, the refined answer should clearly state the error.
Remove any conversational filler or self-references that are not part of the core answer.

Draft Answer: {draft_answer}

Refined Final Answer:
"""

# --- Agent Nodes ---

def extract_query_details_node(state: TechnicalAgentState):
    print("---NODE: Extract Query Details---")
    prompt_str = EXTRACT_QUERY_DETAILS_PROMPT.format(question=state["question"])
    structured_llm = groq_llm.with_structured_output(LLMInitialExtraction)
    
    try:
        llm_output: LLMInitialExtraction = structured_llm.invoke(prompt_str)

        indicator_name_normalized = None
        if llm_output.is_stock_market_related and llm_output.indicator_name:
            indicator_name_normalized = llm_output.indicator_name.lower().replace(" ", "").replace("_", "")

        stock_identifier_processed = None
        if llm_output.is_stock_market_related and llm_output.is_db_required and llm_output.stock_identifier:
            stock_identifier_processed = llm_output.stock_identifier
        
        # If not stock market related, override other flags for clarity
        is_db_required_final = llm_output.is_db_required
        if not llm_output.is_stock_market_related:
            is_db_required_final = False
            indicator_name_normalized = None
            stock_identifier_processed = None


        final_extraction = TechnicalContextOutput(
            is_stock_market_related=llm_output.is_stock_market_related,
            is_db_required=is_db_required_final,
            indicator_name=indicator_name_normalized,
            stock_identifier=stock_identifier_processed,
        )
        
        print(f"Extraction Result: {final_extraction}")
        return {"extraction_output": final_extraction}

    except Exception as e:
        print(f"Error in extract_query_details_node: {e}")
        # Fallback assumes it might be an in-domain question if LLM fails,
        # but with high uncertainty. Could also default to out-of-domain.
        # For now, let's keep the previous fallback for indicator identification if LLM fails.
        error_extraction = TechnicalContextOutput(
            is_stock_market_related=False, # Assume true on LLM error, let downstream handle
            is_db_required=False, 
            indicator_name=None, 
            stock_identifier=None
        )
        question_lower = state["question"].lower()
        if "what is" in question_lower or "explain" in question_lower or "define" in question_lower:
            for ind_key_map in INDICATOR_TOOLS_MAP.keys():
                if ind_key_map in question_lower.replace(" ", "").replace("_",""):
                    error_extraction.indicator_name = ind_key_map
                    break
        print(f"Error Extraction Fallback (assumed in-domain): {error_extraction}")
        return {"extraction_output": error_extraction}

def resolve_fincode_node(state: TechnicalAgentState):
    print("---NODE: Resolve Fincode---")
    extraction = state["extraction_output"]
    
    if not extraction or not extraction.stock_identifier:
        error_msg = "Internal error: Fincode resolution called without stock identifier."
        print(error_msg)
        return {"target_fincode": None, "indicator_result": {"error": error_msg}}

    user_identifier = extraction.stock_identifier
    resolved_fincode_val = None
    error_message = None
    fincode_lookup_query = ""

    # Determine query based on identifier type
    # Using 'accord_base_live' as your schema name from the example
    try:
        # Attempt to convert to int to check if it's potentially a fincode or scripcode
        num_identifier = int(user_identifier)
        # Prioritize fincode, then scripcode for numeric identifiers
        fincode_lookup_query = f"""
        SELECT fincode, scripcode, compname FROM accord_base_live.company_master 
        WHERE fincode = {num_identifier} 
        UNION ALL
        SELECT fincode, scripcode, compname FROM accord_base_live.company_master 
        WHERE scripcode = '{str(num_identifier)}' AND fincode != {num_identifier} 
        LIMIT 1;
        """
        # The UNION approach might be complex; a simpler priority might be better or multiple queries.
        # Let's simplify to one priority query if numeric
        fincode_lookup_query = f"SELECT fincode, scripcode, compname FROM accord_base_live.company_master WHERE fincode = {num_identifier} OR scripcode = '{str(num_identifier)}' LIMIT 1"

    except ValueError:
        # Identifier is not purely numeric, so it's a company name or symbol
        # Escape single quotes in user_identifier for SQL safety
        safe_user_identifier = user_identifier.replace("'", "''")
        fincode_lookup_query = f"SELECT fincode, scripcode, compname FROM accord_base_live.company_master WHERE compname LIKE '%{safe_user_identifier}%' OR symbol = '{safe_user_identifier}' ORDER BY CASE WHEN symbol = '{safe_user_identifier}' THEN 0 ELSE 1 END, LENGTH(compname) ASC LIMIT 1;"
        # ORDER BY helps prioritize symbol match, then shorter company name match.

    print(f"Fincode Lookup SQL: {fincode_lookup_query}")
    db_tool = QuerySQLDatabaseTool(db=db)

    try:
        db_result_str = db_tool.invoke(fincode_lookup_query)
        print(f"Fincode Lookup DB Result (string): {db_result_str}")
        eval_result = ast.literal_eval(db_result_str)
        if isinstance(eval_result, list) and len(eval_result) > 0:
            first_row = eval_result[0]
            if isinstance(first_row, tuple) and len(first_row) > 0:
                resolved_fincode_val = int(first_row[0])
            elif isinstance(first_row, dict) and 'fincode' in first_row: # Should not happen with direct SQL
                resolved_fincode_val = int(first_row['fincode'])
        
        if resolved_fincode_val is not None:
            print(f"Resolved Fincode: {resolved_fincode_val}")
        else:
            error_message = f"Could not find a unique fincode for stock identifier: '{user_identifier}'."
            print(error_message)
            
    except Exception as e:
        print(f"Error during fincode resolution DB execution or parsing: {e}")
        # Check if the error is from SQL execution (contains "pymysql.err")
        if "pymysql.err" in str(e) or "syntax error" in str(e).lower():
            error_message = f"Database error while looking up fincode for '{user_identifier}'. Query: {fincode_lookup_query}. Details: {str(e)[:200]}"
        else:
            error_message = f"Failed to parse database response or other error for stock '{user_identifier}'. Raw: {str(db_result_str)[:100] if 'db_result_str' in locals() else 'N/A'}. Error: {e}"
        
    if error_message:
        return {"target_fincode": None, "indicator_result": {"error": error_message}}
    return {"target_fincode": resolved_fincode_val}


def fetch_price_data_node(state: TechnicalAgentState):
    print("---NODE: Fetch Price Data---")
    target_fincode = state["target_fincode"]

    if target_fincode is None:
        if not state.get("indicator_result", {}).get("error"):
            return {"price_data": None, "indicator_result": {"error": "Fincode not resolved, cannot fetch price data."}}
        return {}

    # SQL query for price data
    price_query_str = f"""
    SELECT date, open AS open, high AS high, low AS low, close AS close, volume 
    FROM accord_base_live.bse_abjusted_price_eod 
    WHERE fincode = :fincode_param
    ORDER BY date DESC 
    LIMIT 300;
    """
    # Using named parameter :fincode_param for safety

    print(f"Price Data SQL (to be executed via SQLAlchemy): {price_query_str.format(fincode_param=target_fincode)}") # For logging
    
    price_data_list_of_dicts = []
    
    try:
        with engine.connect() as connection:
            result = connection.execute(text(price_query_str), {"fincode_param": target_fincode})
            # Convert SQLAlchemy Row objects to dictionaries
            # The ._asdict() method is available if your rows are KeyedTuples (common)
            # Or iterate through keys explicitly if needed.
            # result.keys() gives you the column names as selected (or aliased)
            column_names = list(result.keys())
            for row in result:
                row_dict = dict(zip(column_names, row))
                
                # IMPORTANT: Convert date/datetime objects to ISO string format
                if 'date' in row_dict and hasattr(row_dict['date'], 'isoformat'):
                    row_dict['date'] = row_dict['date'].isoformat()
                elif 'date' in row_dict and row_dict['date'] is not None: # If not None and not datetime object (e.g. already string)
                    row_dict['date'] = str(row_dict['date'])
                
                price_data_list_of_dicts.append(row_dict)

        if not price_data_list_of_dicts:
            return {"price_data": None, "indicator_result": {"error": f"No price data found for fincode {target_fincode}."}}

        # Reverse the list to have dates in ascending order for indicators
        price_data_list_of_dicts.reverse()
            
        print(f"Fetched {len(price_data_list_of_dicts)} price records for fincode {target_fincode} via SQLAlchemy.")
        return {"price_data": price_data_list_of_dicts}

    except Exception as e:
        print(f"Error fetching price data for fincode {target_fincode} via SQLAlchemy: {e}")
        # More specific error for DB issues
        error_detail = f"Database error during price data retrieval for fincode {target_fincode}. Details: {str(e)[:200]}" \
                       if "pymysql.err" in str(e) or "syntax error" in str(e).lower() \
                       else f"Error processing price data for fincode {target_fincode}: {e}"
        return {"price_data": None, "indicator_result": {"error": error_detail}}


def calculate_indicator_node(state: TechnicalAgentState):
    print("---NODE: Calculate Indicator---")
    extraction = state["extraction_output"]
    price_data = state.get("price_data")

    if state.get("indicator_result", {}).get("error"):
        print(f"Passing through existing error: {state['indicator_result']['error']}")
        return {}

    if not extraction or not extraction.indicator_name:
        return {"indicator_result": {"error": "Internal error: Indicator name missing for calculation."}}
    
    indicator_name = extraction.indicator_name
    if indicator_name not in INDICATOR_TOOLS_MAP:
        return {"indicator_result": {"error": f"Unsupported indicator: '{indicator_name}'."}}

    if not price_data:
        return {"indicator_result": {"error": "No price data available for calculation."}}

    tool_function = INDICATOR_TOOLS_MAP[indicator_name]
    print(f"Invoking {indicator_name} tool with its hardcoded default parameters.")
    try:
        # Pass the list of dicts directly to the indicator function
        result = tool_function(price_data) 
        print(f"Indicator Result ({indicator_name}): {result}")
        return {"indicator_result": result}
    except Exception as e:
        print(f"Error calculating {indicator_name}: {e}")
        import traceback
        traceback.print_exc() # For detailed debugging of indicator calculation errors
        return {"indicator_result": {"error": f"Error during {indicator_name} calculation: {str(e)}"}}


def generate_db_answer_draft_node(state: TechnicalAgentState):
    print("---NODE: Generate DB Answer Draft---")
    extraction = state["extraction_output"]
    indicator_result = state.get("indicator_result", {}) 
    question = state["question"]
    target_fincode = state.get("target_fincode") # This is now directly in state

    if not extraction:
        return {"draft_answer": "IQUN-AI: I'm sorry, there was an internal issue (missing query details)."}

    if "error" in indicator_result and indicator_result["error"]:
        error_detail = indicator_result["error"]
        stock_id_for_error = extraction.stock_identifier if extraction and extraction.stock_identifier else 'the stock'
        draft = f"I'm IQUN-AI. I encountered an issue processing your request for '{stock_id_for_error}': {error_detail}."
        if target_fincode is not None: # Add fincode if it was resolved but subsequent step failed
            draft += f" (Fincode considered: {target_fincode})."
        return {"draft_answer": draft}

    indicator_name_display = extraction.indicator_name.upper() if extraction.indicator_name else "the indicator"
    stock_id_display = extraction.stock_identifier if extraction.stock_identifier else "the specified stock"
    fincode_display = target_fincode if target_fincode is not None else "N/A"
    
    prompt_str = DB_ANSWER_DRAFT_PROMPT.format(
        question=question,
        stock_identifier=stock_id_display,
        target_fincode=fincode_display,
        indicator_name_display=indicator_name_display,
        indicator_result_str=str(indicator_result) 
    )
    try:
        response = groq_llm.invoke(prompt_str)
        response_content = response.content if hasattr(response, 'content') else str(response)
        return {"draft_answer": response_content}
    except Exception as e:
        print(f"Error in generate_db_answer_draft_node LLM call: {e}")
        return {"draft_answer": f"IQUN-AI: I have the technical data ({indicator_result}), but I encountered an issue phrasing the answer: {e}"}

def generate_general_answer_draft_node(state: TechnicalAgentState):
    print("---NODE: Generate General Answer Draft---")
    extraction = state["extraction_output"]
    question = state["question"]

    if not extraction or not extraction.indicator_name:
        draft_err = f"I can explain technical indicators, but I couldn't determine which one you're asking about from your question: '{question}'. Please specify clearly."
        return {"draft_answer": draft_err}

    indicator_name_display = extraction.indicator_name.upper()
    prompt = GENERAL_EXPLANATION_PROMPT.invoke({
        "question": question, "indicator_name_display": indicator_name_display
    })
    try:
        response = groq_llm.invoke(prompt)
        return {"draft_answer": response.content if hasattr(response, 'content') else str(response)}
    except Exception as e:
        print(f"Error in generate_general_answer_draft_node LLM call: {e}")
        return {"draft_answer": f"I'm IQUN-AI. I can usually explain {indicator_name_display}, but I'm having trouble phrasing the explanation: {e}"}

def generate_out_of_domain_answer_node(state: TechnicalAgentState):
    print("---NODE: Generate Out-of-Domain Answer---")
    # This node is for questions identified as not stock market related.
    # The final_answer can be set directly here, or we can use a simple prompt for consistency.
    # For now, let's set it directly.
    answer = "I'm IQUN-AI, your financial markets assistant. I'm trained to help with questions about stocks, technical indicators, and financial analysis. I'm afraid I don't have information on topics outside of this domain."
    if "hello" in state["question"].lower() or "hi" in state["question"].lower() or "how are you" in state["question"].lower():
        answer = "Hello! I'm IQUN-AI, ready to help with your stock market and technical indicator questions. How can I assist you today?"
    
    print(f"Out-of-domain answer: {answer}")
    # Setting draft_answer so finalize_answer can potentially refine it if needed,
    # or we could directly set final_answer. Let's use draft_answer for consistency with flow.
    return {"draft_answer": answer}

def finalize_answer_node(state: TechnicalAgentState):
    print("---NODE: Finalize Answer---")
    extraction = state["extraction_output"] # Might be None if very early error
    draft_answer = state.get("draft_answer", "I'm sorry, I encountered an issue and couldn't generate a response.")
    question = state["question"]
    
    # If the draft_answer is already the out-of-domain message,
    # we might not need further LLM refinement unless we want to ensure politeness.
    # For simplicity, if it's already an out-of-domain or direct error message from routing,
    # we can pass it through.
    if "I'm afraid I don't have information on topics outside of this domain." in draft_answer or \
       "I can explain indicators, but I couldn't identify which one" in draft_answer or \
       "Cannot perform calculation:" in draft_answer or \
       "internal error occurred" in draft_answer: # Check for our canned error/OOD messages
        print(f"Final Answer (Direct Passthrough): {draft_answer}")
        return {"final_answer": draft_answer}

    if extraction and extraction.is_stock_market_related:
        if extraction.is_db_required:
            prompt_template = ChatPromptTemplate.from_template(VALIDATE_DB_ANSWER_PROMPT)
            format_args = {
                "question": question,
                "stock_identifier": extraction.stock_identifier if extraction.stock_identifier else "N/A",
                "target_fincode": state.get("target_fincode", "N/A"),
                "indicator_name_display": extraction.indicator_name.upper() if extraction.indicator_name else "the indicator",
                "draft_answer": draft_answer
            }
        else: # In-domain general explanation of an indicator
            prompt_template = ChatPromptTemplate.from_template(NORMALIZE_GENERAL_EXPLANATION_PROMPT)
            format_args = {"question": question, "draft_answer": draft_answer}
    else:
        # This case implies an out-of-domain question that somehow didn't get caught by the direct passthrough.
        # Or, if we want to *always* refine even the out-of-domain message.
        # For now, this path might not be hit if direct passthrough above is effective.
        # If it is hit, it means draft_answer is something else, maybe from a failed general explanation.
        print(f"Finalizing an answer that seems out-of-domain or unclassified: {draft_answer}")
        # Default to a generic refinement or pass through.
        # To be safe, let's assume it was intended for general normalization if it reaches here.
        prompt_template = ChatPromptTemplate.from_template(NORMALIZE_GENERAL_EXPLANATION_PROMPT)
        format_args = {"question": question, "draft_answer": draft_answer}


    try:
        response = groq_llm.invoke(prompt_template.format_prompt(**format_args))
        final_ans = response.content if hasattr(response, 'content') else str(response)
        print(f"Final Answer (Refined): {final_ans}")
        return {"final_answer": final_ans}
    except Exception as e:
        print(f"Error in finalize_answer_node LLM call: {e}")
        return {"final_answer": draft_answer }

# --- Graph Conditional Edges ---
def route_after_extraction(state: TechnicalAgentState):
    extraction = state["extraction_output"]
    if not extraction: 
        print("CRITICAL: No extraction_output in state for routing!")
        state["draft_answer"] = "I'm sorry, an internal error occurred (missing query details)."
        return "finalize_error_directly" 

    if not extraction.is_stock_market_related:
        print("Routing to: Out-of-Domain Answer")
        return "out_of_domain_answer_path"

    # At this point, question is stock_market_related
    if extraction.is_db_required:
        if extraction.stock_identifier and extraction.indicator_name:
            print("Routing to: Resolve Fincode (DB Path)")
            return "db_resolve_fincode"
        else:
            error_msg = "Cannot perform calculation: "
            if not extraction.stock_identifier: error_msg += "No stock was specified for the calculation. "
            if not extraction.indicator_name: error_msg += "No specific indicator was mentioned for calculation. "
            state["draft_answer"] = f"I'm IQUN-AI. {error_msg.strip()} Please clarify your request for a calculation."
            return "finalize_error_directly"
    else: # General explanation of an indicator (is_stock_market_related=True, is_db_required=False)
        if extraction.indicator_name: 
            print("Routing to: General Indicator Explanation Draft")
            return "general_explanation_draft_path"
        else:
            # Stock market related, but no specific indicator mentioned for general explanation.
            # e.g., "Tell me about technical analysis"
            state["draft_answer"] = f"I can explain specific technical indicators (like RSI, MACD). Your question '{state['question']}' seems finance-related, but please specify which indicator you'd like to know more about."
            return "finalize_error_directly"


def route_after_fincode_resolution(state: TechnicalAgentState):
    if state.get("indicator_result", {}).get("error"): # Error from resolve_fincode_node
        print(f"Routing to: Generate DB Answer (to report fincode resolution error)")
        return "generate_db_answer_draft" 

    if state.get("target_fincode") is not None:
        print("Routing to: Fetch Price Data")
        return "fetch_price_data"
    else: 
        # Fincode is None, and no error was explicitly set in indicator_result by resolve_fincode_node
        # This implies fincode lookup returned no results.
        stock_id = state['extraction_output'].stock_identifier if state.get('extraction_output') else 'the stock'
        error_msg = f"I couldn't find a matching stock for '{stock_id}'. Please check the name/symbol/fincode or try a more specific identifier."
        print(f"Routing to: Generate DB Answer (Error: {error_msg})")
        state["indicator_result"] = {"error": error_msg} # Ensure error is set for db_answer_draft
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
graph_builder_tech.add_node("finalize_answer", finalize_answer_node)

graph_builder_tech.set_entry_point("extract_query_details")

graph_builder_tech.add_conditional_edges(
    "extract_query_details",
    route_after_extraction,
    {
        "out_of_domain_answer_path": "generate_out_of_domain_answer",
        "db_resolve_fincode": "resolve_fincode",
        "general_explanation_draft_path": "generate_general_explanation_draft",
        "finalize_error_directly": "finalize_answer" 
    }
)

# DB Path
graph_builder_tech.add_conditional_edges(
    "resolve_fincode",
    route_after_fincode_resolution,
    {
        "fetch_price_data": "fetch_prices",
        "generate_db_answer_draft": "generate_db_answer_draft" 
    }
)
graph_builder_tech.add_edge("fetch_prices", "calculate_indicator") 
graph_builder_tech.add_edge("calculate_indicator", "generate_db_answer_draft")
graph_builder_tech.add_edge("generate_db_answer_draft", "finalize_answer")

# General Indicator Explanation Path
graph_builder_tech.add_edge("generate_general_explanation_draft", "finalize_answer")

# Out-of-Domain Path
graph_builder_tech.add_edge("generate_out_of_domain_answer", "finalize_answer") # OOD answers also go to finalize

graph_builder_tech.add_edge("finalize_answer", END)

app_technical = graph_builder_tech.compile()



# --- Main Execution for Testing ---
if __name__ == "__main__":
    print("Technical Analysis Agent Compiled. Ready for testing.")
    test_queries_tech = [
        # "What is RSI?",
        # "Explain Bollinger Bands.",
        # "Calculate the RSI for Reliance Industries Ltd.",
        # "What is the MACD for fincode 500209?",
        # "ATR for INFY",
        # "VWAP for non_existent_stock_xyz",
        # "Tell me about trends",
        # "Calculate Supertrend for fincode 9999999" # Fincode likely without data
        "hi",
        "how are you?",
        "what is RSI of reliance?",
        "bollinger bands of Aegis Logistics",
        "market cap of reliance",
        "Who invented the lightbulb?",
        "What's the capital of France?",
        "How are you today?"
    ]

    for q_text in test_queries_tech:
        print(f"\n\n--- TESTING TECHNICAL QUERY: {q_text} ---")
        inputs = {"question": q_text}
        try:
            config = {"recursion_limit": 15} # Reset to 15, should be enough for simpler flow
            for event in app_technical.stream(inputs, config=config):
                for node_name, state_update_dict in event.items():
                    print(f"--- Event from Node: {node_name} ---")
                    if isinstance(state_update_dict, dict):
                        if "extraction_output" in state_update_dict:
                             print(f"    Extraction Output: {state_update_dict['extraction_output']}")
                        if "target_fincode" in state_update_dict:
                             print(f"    Target Fincode: {state_update_dict['target_fincode']}")
                        if "indicator_result" in state_update_dict:
                             print(f"    Indicator Result: {state_update_dict['indicator_result']}")
                        if "price_data" in state_update_dict:
                             print(f"    Price Data Count: {len(state_update_dict['price_data']) if state_update_dict['price_data'] else 'None or Empty'}")
                        if "draft_answer" in state_update_dict:
                             print(f"    Draft Answer: {state_update_dict['draft_answer']}")
                        if "final_answer" in state_update_dict:
                             print(f"    FINAL ANSWER: {state_update_dict['final_answer']}")
                if END in event:
                    print("--- End of Graph Execution ---")
        except Exception as e:
            print(f"Error invoking graph for question '{q_text}': {e}")
            import traceback
            traceback.print_exc()
        print("---------------------------------------\n")

    try:
        print("\nAttempting to generate technical_agent_graph.png...")
        img_data = app_technical.get_graph().draw_mermaid_png()
        with open("technical_agent_graph.png", "wb") as f:
            f.write(img_data)
        print("Graph saved to technical_agent_graph.png")
    except Exception as e:
        print(f"Could not generate graph visualization: {e}")