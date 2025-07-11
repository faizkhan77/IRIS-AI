# agents/rag_context_provider.py
import pickle
from collections import defaultdict
import colorama
from colorama import Fore, Style
from typing import List

# --- Import LLM and Pydantic dependencies ---
from pydantic import BaseModel, Field
from model_config import groq_llm

colorama.init(autoreset=True)

# --- Load Pre-built Vector Store ---
VECTOR_STORE_PATH = "context/rag_vector_store.pkl"
_vectorstore = None
_rag_initialized = False

print("Loading pre-built RAG vector store...")
try:
    with open(VECTOR_STORE_PATH, "rb") as f:
        _vectorstore = pickle.load(f)
    print("RAG vector store loaded successfully.")
    _rag_initialized = True
except FileNotFoundError:
    print(f"{Fore.RED}FATAL ERROR: RAG store not found at '{VECTOR_STORE_PATH}'. Run 'build_rag_store.py' first.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}FATAL ERROR: Failed to load RAG store: {e}{Style.RESET_ALL}")

# --- Pydantic Models (Simplified and Cleaned) ---
# No more aliases. We will enforce the structure in the prompt itself.
class ExpandedColumn(BaseModel):
    name: str = Field(description="The name of the column.")
    reason: str = Field(description="A brief reason why this column is necessary for the query.")

class ExpandedTable(BaseModel):
    table_name: str = Field(description="The name of the database table.")
    columns: List[ExpandedColumn] = Field(description="A list of all columns from this table required to answer the user's query.")

class ExpandedContext(BaseModel):
    tables: List[ExpandedTable] = Field(description="A list of tables with their necessary columns.")

# --- CONTEXT EXPANSION PROMPT (Overhauled for Explicit Formatting) ---
CONTEXT_EXPANSION_PROMPT = """
You are an expert SQL Query Planner. Your goal is to analyze an initial, limited set of database schema information and expand it to include ALL columns necessary to answer a user's question.

Your output MUST be a single, valid JSON object. Follow this EXACT structure:
{{
  "tables": [
    {{
      "table_name": "example_table_one",
      "columns": [
        {{
          "name": "example_column_a",
          "reason": "Reason why this column is needed."
        }},
        {{
          "name": "example_column_b",
          "reason": "Another reason."
        }}
      ]
    }}
  ]
}}
---
Now, analyze the user's question and the initial context below.

User's Question: "{user_question}"

--- Initial Schema Context (from a keyword search) ---
{initial_context}
---

Based on the user's question, think step-by-step. What columns are *really* needed?
1.  **Identify Entities:** You must include a column to identify the company (e.g., `compname`).
2.  **Handle Time:** If the user asks for "latest", "historical", etc., you MUST include a date or year column (e.g., `year`, `date_end`).
3.  **Handle Ordering/Ranking:** If the user asks for "top 5", "highest", etc., you need the column they are ranking by (e.g., `mcap`).
4.  **Include All Mentioned Metrics:** Ensure every metric the user asked for is present.

Provide the complete and final list of all required tables and columns in the valid JSON format shown in the example structure above.
"""

# --- The main function remains the same, but will now use the new prompt ---
async def get_intelligent_context(task_query: str) -> str:
    """
    Performs a two-step "Retrieve and Expand" process to get complete, actionable schema context.
    """
    if not _rag_initialized or not _vectorstore:
        return "Error: RAG context provider is not initialized. Please run the build script."

    # --- 1. RETRIEVE ---
    retrieved_docs = _vectorstore.similarity_search(task_query, k=15)
    
    if not retrieved_docs:
        print(f"{Fore.YELLOW}---RAG: No initial columns found for task: '{task_query}'---{Style.RESET_ALL}")
        return ""

    grouped_by_table = defaultdict(list)
    for doc in retrieved_docs:
        table = doc.metadata['table_name']
        column_info = f"- {doc.metadata['column_name']}: {doc.metadata['description']}"
        if column_info not in grouped_by_table[table]:
            grouped_by_table[table].append(column_info)
    
    initial_context_parts = []
    for table_name, columns in grouped_by_table.items():
        initial_context_parts.append(f"Table: {table_name}\n" + "\n".join(columns))
    initial_context_str = "\n\n".join(initial_context_parts)

    print(Style.BRIGHT + Fore.BLUE + f"--- RAG: Step 1 (Retrieve) - Initial Context Found ---" + Style.RESET_ALL)
    print(Fore.BLUE + initial_context_str)
    print(Style.BRIGHT + Fore.BLUE + "----------------------------------------------------" + Style.RESET_ALL)

    # --- 2. EXPAND ---
    print(Style.BRIGHT + Fore.MAGENTA + "--- RAG: Step 2 (Expand) - Asking LLM to plan query components ---" + Style.RESET_ALL)
    
    expansion_prompt_formatted = CONTEXT_EXPANSION_PROMPT.format(
        user_question=task_query,
        initial_context=initial_context_str
    )
    
    structured_llm = groq_llm.with_structured_output(ExpandedContext, method="json_mode")
    
    try:
        expanded_context_obj = await structured_llm.ainvoke(expansion_prompt_formatted)
    except Exception as e:
        print(f"{Fore.RED}Error during LLM expansion step: {e}")
        print(f"{Fore.YELLOW}Falling back to initial retrieved context.")
        return initial_context_str

    # --- 3. FORMAT ---
    final_formatted_parts = []
    for table in expanded_context_obj.tables:
        table_part = f"Table: {table.table_name}\nRequired Columns:\n"
        columns_part = "\n".join([f"- {col.name}: {col.reason}" for col in table.columns])
        final_formatted_parts.append(table_part + columns_part)

    final_context_str = "\n\n".join(final_formatted_parts)

    print(Style.BRIGHT + Fore.CYAN + f"--- RAG: Final Intelligent Context ---" + Style.RESET_ALL)
    print(Fore.CYAN + final_context_str)
    print(Style.BRIGHT + Fore.CYAN + "------------------------------------" + Style.RESET_ALL)

    return final_context_str