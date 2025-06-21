# fundamentals_agent.py
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated, List as TypingList # Renamed to avoid conflict with List from typing
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os

# Custom module imports (ensure these paths are correct for your project structure)
from model_config import groq_llm # This should now use the model like llama-3.3-70b-versatile
from db_config import engine
from artifacts.tables import all_table_info
from tools.table_info import get_column_info # Relies on FILEMAPPING from utils.mapping

# For graph visualization if run directly
from IPython.display import Image, display


load_dotenv()

db = SQLDatabase(engine=engine)

# --- Define TypedDicts for State and Outputs ---
class ExtractOutput(TypedDict):
    """Output of the table extraction step."""
    is_required_database: Annotated[bool, ..., "Is required call database to get context for response"]
    tables_required: Annotated[TypingList[str], ...,"List of required table names to get database for question "]

class AgentState(TypedDict): # Renamed to AgentState for clarity if used elsewhere
    question: str
    extract: ExtractOutput
    columns_context: str # Detailed schema for relevant tables
    query: str
    result: str # SQL execution result
    answer: str # Initial natural language answer from DB path or general response
    final_answer: str # Polished final answer

class QueryOutput(TypedDict):
    """Generated SQL query by the LLM."""
    query: Annotated[str, "Syntactically valid SQL query."]


# --- Prompts ---

# Prompt for extracting necessary tables (from your main.py)
EXTRACT_TABLES_PROMPT_TEMPLATE = """
You are a helpful assistant that analyzes natural language questions to determine whether accessing a database is necessary to answer them. You must extract two pieces of information:

1. is_required_database: A boolean value.
- Return true if answering the question requires data from the database from below table info context.
- Return false if the question can be answered without querying the database (e.g., it's a general or definitional question).

2. tables_required: A list of strings.
- Include the names of all tables that must be accessed in order to answer the question accurately.
- Don't add uncessory tables names strictly just add necessory tables
- Return an empty list if is_required_database is false.

You must only include table names mentioned in the question or strongly implied.

here is user question :
Question: {question}

Here is table info:
{all_table_info}
"""

# System prompt for SQL query generation (from your main.py, ensuring dialect is handled)
WRITE_QUERY_SYSTEM_MESSAGE = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables (general overview):
{table_info}

Detailed columns context for relevant tables (use this primarily for query construction):
{columns_context}

Additionally, when necessary, include appropriate JOINS between multiple tables based on their relationships to retrieve the required data.
also add limit when it necessory.
"""
USER_PROMPT_FOR_QUERY = "Question: {input}"
query_prompt_template = ChatPromptTemplate.from_messages(
    [("system", WRITE_QUERY_SYSTEM_MESSAGE), ("user", USER_PROMPT_FOR_QUERY)]
)

# Prompt for general response when DB is not needed (from your main.py)
GENERAL_RESPONSE_SYSTEM_PROMPT = """
---

### 🗃️ Data Available:

You have access to the following categories of data:

#### 🔍 Technical Data:
- Daily and monthly stock prices (`bse_stock_price`, `bse_adjusted_price_eod`, `monthly_price_bse`, `monthly_price_nse`)
- Index price data (`bse_index_price`, `bse_indices_price_eod`)
- Index metadata (`indices_master`, `company_index_part`)

#### 🔍 Upcoming Feature:
- You can Calculate Technical and Fundamental Indicaters and help to pick better profitable stocks

#### 📊 Fundamental Data:
- Financial statements: Balance sheet, cash flow, profit/loss (`company_finance_*`)
- Financial ratios (`company_finance_ratio`, `company_finance_ratio_cons`)
- Equity structure and changes (`company_equity`, `company_equity_cons`)
- Results and earnings (`company_results`, `company_results_cons`)
- Shareholding patterns and detailed shareholders (`company_shareholding_pattern`, `company_shareholders_details`)
- Company overview (`company_master`, `company_profile`, `company_address`, `company_board_director`)
- Legal and registrar details (`company_registrar_data`, `company_registrar_master`)
- Sector and industry info (`industry_master`, `shareholding_category_master`, `stock_exchange_master`, `house_master`)
---

You are IQUN-AI. Respond to the user's question naturally, keeping in mind the data capabilities listed above if relevant.
If the question is a general greeting or unrelated to financial data, provide a polite, general response.
"""

# Prompt for generating an answer from SQL results (from your main.py)
GENERATE_ANSWER_DB_PATH_PROMPT_TEMPLATE = """
Given the following user question, corresponding SQL query,
and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer:
"""

# Prompt for final validation/normalization (for DB path, from your main.py's validate_and_response)
VALIDATE_DB_RESPONSE_PROMPT_TEMPLATE = """
You are a helpful assistant Named IQUN-AI. Based on the question and the SQL result provided (context below),
improve the draft answer to be clear, natural, and directly respond to the question.
Make your tone natural, not generic.
Do not mention the SQL query or any internal technical details.

Question: {question}
SQL Result (for context, not to be exposed): {sql_result}
Response to Improve: {draft_answer}

Final Answer:
"""

# Prompt for final normalization (for non-DB path, from your main.py's normalize_response)
NORMALIZE_GENERAL_RESPONSE_PROMPT_TEMPLATE = """
You are a helpful assistant Named IQUN-AI.
The user asked a question that was answered directly without database access.
Review the draft answer and ensure it's clear, natural, and polite.
Make your tone natural, not generic.

Question: {question}
Response to Improve: {draft_answer}

Final Answer:
"""


# --- Agent Nodes (adapted from your main.py and previous fundamentals_agent.py) ---

def extract_from_input_node(state: AgentState):
    """Get Info and Understand Question"""
    print("---NODE: Extract Info from Input---")
    prompt_str = EXTRACT_TABLES_PROMPT_TEMPLATE.format(
        question=state["question"], all_table_info=all_table_info
    )
    structured_llm = groq_llm.with_structured_output(ExtractOutput)
    try:
        result = structured_llm.invoke(prompt_str)
        print(f"Extraction Result: {result}")
        return {"extract": result}
    except Exception as e:
        print(f"Error in extract_from_input_node: {e}")
        return {"extract": {"is_required_database": False, "tables_required": []}} # Fallback

def general_response_node(state: AgentState):
    """Give Questions of answer where no need to database context"""
    print("---NODE: General Response---")
    # This node now directly generates the 'answer' field which will be passed to normalize_general_response_node
    messages = [
        SystemMessage(content=GENERAL_RESPONSE_SYSTEM_PROMPT), # all_table_info is part of this system prompt
        HumanMessage(content=state["question"])
    ]
    try:
        response_content = groq_llm.invoke(messages).content
        print(f"General Response (Draft): {response_content}")
        return {"answer": response_content} # This is the draft answer
    except Exception as e:
        print(f"Error in general_response_node: {e}")
        return {"answer": "I'm sorry, I encountered an issue trying to process your request."}

def get_columns_context_node(state: AgentState):
    """Fetches detailed column information for the required tables."""
    print("---NODE: Get Columns Context---")
    required_tables = state["extract"].get("tables_required", [])
    if not required_tables:
        print("No tables required, providing generic column context.")
        return {"columns_context": "No specific tables were identified as necessary for this query."}
    try:
        context = get_column_info.invoke({"table_names": required_tables})
        print(f"Columns Context for {required_tables} generated.")
        return {"columns_context": context}
    except Exception as e:
        print(f"Error in get_columns_context_node: {e}")
        return {"columns_context": f"Error fetching column details: {e}. Using general table info."}

def write_query_node(state: AgentState):
    """Generate SQL query to fetch information."""
    print("---NODE: Write SQL Query---")
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect, # Added dialect here
            "columns_context": state["columns_context"],
            "top_k": 10,
            "table_info": all_table_info, # General overview
            "input": state["question"],
        }
    )
    # print(f"Query Prompt: {prompt}") # For debugging the full prompt
    structured_llm = groq_llm.with_structured_output(QueryOutput)
    try:
        result = structured_llm.invoke(prompt)
        print(f"Generated SQL Query: {result['query']}")
        return {"query": result["query"]}
    except Exception as e:
        print(f"Error in write_query_node: {e}")
        return {"query": "SELECT 'Error generating query due to LLM failure.'"}


def execute_query_node(state: AgentState):
    """Execute SQL query."""
    print("---NODE: Execute SQL Query---")
    query_to_execute = state["query"]
    if not query_to_execute or "Error generating query" in query_to_execute:
        print(f"Skipping execution for invalid query: {query_to_execute}")
        return {"result": "Error: Invalid or no SQL query provided for execution."}
    
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    try:
        result = execute_query_tool.invoke(query_to_execute)
        print(f"SQL Execution Result (Snippet): {str(result)[:200]}...")
        return {"result": str(result)}
    except Exception as e:
        print(f"Error executing SQL query '{query_to_execute}': {e}")
        return {"result": f"Error executing query: {e}. Query was: {query_to_execute}"}


def generate_answer_db_path_node(state: AgentState):
    """Answer question using retrieved information as context (DB path)."""
    print("---NODE: Generate Answer (DB Path)---")
    if "Error executing query" in state["result"] or "Error: Invalid or no SQL query" in state["result"]:
        # Formulate a user-friendly message if SQL execution failed
        error_message = f"I'm IQUN-AI. I tried to find the information for '{state['question']}', but encountered a problem accessing the data. "
        error_message += "This might be due to an issue with the data source or the way the information was requested. Please try rephrasing or ask again later."
        print(f"DB Path Error Message: {error_message}")
        return {"answer": error_message} # This draft error answer goes to validate_db_response_node

    prompt_str = GENERATE_ANSWER_DB_PATH_PROMPT_TEMPLATE.format(
        question=state["question"], query=state["query"], result=state["result"]
    )
    try:
        response_content = groq_llm.invoke(prompt_str).content
        print(f"Generated Answer (DB Path - Draft): {response_content}")
        return {"answer": response_content} # This is the draft answer
    except Exception as e:
        print(f"Error in generate_answer_db_path_node: {e}")
        return {"answer": "I'm IQUN-AI. I retrieved some data but had trouble putting together an answer. Could you try rephrasing?"}


def validate_db_response_node(state: AgentState):
    """Final response generation for DB path."""
    print("---NODE: Validate DB Response---")
    prompt_str = VALIDATE_DB_RESPONSE_PROMPT_TEMPLATE.format(
        question=state["question"],
        sql_result=state["result"], # provide SQL result for context during refinement
        draft_answer=state["answer"]
    )
    try:
        response_content = groq_llm.invoke(prompt_str).content
        print(f"Final Validated Answer (DB Path): {response_content}")
        return {"final_answer": response_content}
    except Exception as e:
        print(f"Error in validate_db_response_node: {e}")
        return {"final_answer": state.get("answer", "An unexpected error occurred while finalizing the answer.")} # Fallback to draft


def normalize_general_response_node(state: AgentState):
    """Final response generation for non-DB path."""
    print("---NODE: Normalize General Response---")
    prompt_str = NORMALIZE_GENERAL_RESPONSE_PROMPT_TEMPLATE.format(
        question=state["question"],
        draft_answer=state["answer"]
    )
    try:
        response_content = groq_llm.invoke(prompt_str).content
        print(f"Final Normalized Answer (General Path): {response_content}")
        return {"final_answer": response_content}
    except Exception as e:
        print(f"Error in normalize_general_response_node: {e}")
        return {"final_answer": state.get("answer", "An unexpected error occurred.")} # Fallback to draft

# --- Graph Conditional Edges ---
def question_router(state: AgentState):
    """Routes based on whether a DB query is needed."""
    print(f"---ROUTER: DB Query Needed? --- Extract: {state['extract']}")
    if state["extract"].get("is_required_database") and state["extract"].get("tables_required"):
        print("Routing to: Get Columns Context (DB Path)")
        return "database_path"
    else:
        print("Routing to: General Response (Non-DB Path)")
        return "general_path"

# --- Build the Graph ---
graph_builder = StateGraph(AgentState)

# Add nodes (using more descriptive names from your working main.py style)
graph_builder.add_node("extract_from_query", extract_from_input_node)
graph_builder.add_node("db_columns_context", get_columns_context_node)
graph_builder.add_node("general_response_draft", general_response_node) # Draft generation
graph_builder.add_node("write_sql_query", write_query_node)
graph_builder.add_node("execute_sql_query", execute_query_node)
graph_builder.add_node("generate_db_answer_draft", generate_answer_db_path_node) # Draft generation
graph_builder.add_node("validate_db_final_response", validate_db_response_node) # Final for DB
graph_builder.add_node("normalize_general_final_response", normalize_general_response_node) # Final for general

# Set entry point
graph_builder.add_edge(START, "extract_from_query")

# Conditional routing after extraction
graph_builder.add_conditional_edges(
    "extract_from_query",
    question_router,
    {
        "database_path": "db_columns_context",
        "general_path": "general_response_draft"
    }
)

# DB Path
graph_builder.add_edge("db_columns_context", "write_sql_query")
graph_builder.add_edge("write_sql_query", "execute_sql_query")
graph_builder.add_edge("execute_sql_query", "generate_db_answer_draft")
graph_builder.add_edge("generate_db_answer_draft", "validate_db_final_response")
graph_builder.add_edge("validate_db_final_response", END)

# General Path
graph_builder.add_edge("general_response_draft", "normalize_general_final_response")
graph_builder.add_edge("normalize_general_final_response", END)

# Compile the graph
app = graph_builder.compile()


# --- Main Execution for Testing (if this file is run directly) ---
if __name__ == "__main__":
    print("Fundamentals Agent Graph (Refactored) Compiled. Ready for testing.")

    # test_queries = [
    #     "Hello there, IQUN-AI!",
    #     "what is scripcode for reliance industries?",
    #     "What is the market cap of TCS?", # Assuming TCS is in your DB
    #     "Tell me about your capabilities.",
    #     "List all distinct industries from company_master.", # More specific
    #      "what is the latest PAT for fincode 3?" # uses fincode
    # ]

    # for q_text in test_queries:
    #     print(f"\n\n--- TESTING QUESTION: {q_text} ---")
    #     inputs = {"question": q_text}
    #     try:
    #         # Stream events to see the flow
    #         for event in app.stream(inputs, stream_mode="values"):
    #             # event is the full state dict at each step
    #             print(f"\nState after step {event.get('__end__', list(event.keys())[-1])}:") # Try to get last key if no __end__
    #             # For brevity, print only relevant parts or new/changed parts
    #             if "final_answer" in event and event["final_answer"]:
    #                 print(f"  FINAL ANSWER: {event['final_answer']}")
    #             elif "answer" in event and event["answer"]:
    #                  print(f"  Draft Answer: {event['answer'][:200]}...")
    #             elif "query" in event and event["query"]:
    #                 print(f"  Generated Query: {event['query']}")
    #             elif "result" in event and event["result"]:
    #                 print(f"  Query Result: {str(event['result'])[:200]}...")
    #             elif "extract" in event and event["extract"]:
    #                 print(f"  Extraction: {event['extract']}")


            # Get the final accumulated state if needed (though streaming shows intermediate states)
            # final_state_result = app.invoke(inputs)
            # print(f"\n--- FINAL ACCUMULATED STATE for '{q_text}' ---")
            # print(final_state_result.get("final_answer", "No final answer in accumulated state."))
    
    user_input = input("Enter your queries: ")
    input = {"question": user_input}

    for event in app.stream(input, stream_mode="values"):
        # event is the full state dict at each step
            print(f"\nState after step {event.get('__end__', list(event.keys())[-1])}:") # Try to get last key if no __end__
            # For brevity, print only relevant parts or new/changed parts
            if "final_answer" in event and event["final_answer"]:
                print(f"  FINAL ANSWER: {event['final_answer']}")
            # elif "answer" in event and event["answer"]:
            #         print(f"  Draft Answer: {event['answer'][:200]}...")
            # elif "query" in event and event["query"]:
            #     print(f"  Generated Query: {event['query']}")
            # elif "result" in event and event["result"]:
            #     print(f"  Query Result: {str(event['result'])[:200]}...")
            # elif "extract" in event and event["extract"]:
            #     print(f"  Extraction: {event['extract']}")
            

        # except Exception as e:
        #     print(f"Error invoking graph for question '{q_text}': {e}")
        #     import traceback
        #     traceback.print_exc()
        # print("---------------------------------------\n")

    try:
        print("\nAttempting to generate graph visualization...")
        img_data = app.get_graph().draw_mermaid_png()
        with open("fundamentals_agent_refactored_graph.png", "wb") as f:
            f.write(img_data)
        print("Graph saved to fundamentals_agent_refactored_graph.png")
        # display(Image(img_data)) # Uncomment if in a Jupyter environment
    except Exception as e:
        print(f"Could not generate graph visualization: {e}. Ensure graphviz/pygraphviz are installed.")