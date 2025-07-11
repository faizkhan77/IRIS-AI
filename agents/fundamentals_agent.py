# fundamentals_agent.py
import asyncio
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
from sqlalchemy import text

# --- RAG INTEGRATION: Import the NEW intelligent context provider ---
from agents.rag_context_provider import get_intelligent_context

# Custom module imports
from model_config import groq_llm
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
    entities: List[IdentifiedEntity] = Field(description="List of all unique financial entities found in the query.")
    tasks: Dict[str, str] = Field(description="A dictionary mapping each entity_name (or 'General Query') to a concise description of the data needed for it.")
    known_schema_lookups: Optional[Dict[str, KnownSchema]] = Field(None, description="If a task can be answered by a single known column from the Golden Schema, map the task to its table and column here.")

class GeneralizedQuery(BaseModel):
    generalized_question: str = Field(description="A generic question for RAG context retrieval, with specific entity names removed.")

class SQLQuery(BaseModel):
    query: str = Field(description="A single, syntactically valid SQL query.")


# --- Agent State ---
class FundamentalsAgentState(TypedDict):
    question: str
    extraction: Optional[ExtractionOutput]
    generalized_tasks: Optional[Dict[str, str]]
    rag_contexts: Optional[Dict[str, str]]
    sql_queries: Optional[Dict[str, str]]
    query_results: Optional[Dict[str, str]]
    error_message: Optional[str]
    final_answer: str

# --- Prompts (Overhauled for Robustness) ---

# This "Golden Schema" is a permanent guide for the LLM, reducing hallucination.
GOLDEN_SCHEMA_CONTEXT = """
    /*
    -- Golden Schema & High-Level Database Guide --
    1.  **Core Company Information:**
        -   Table: `company_master`
        -   Columns: `fincode` (primary key), `compname` (full company name), `scripcode`, `industry` (industry classification).
    2.  **Finding "Latest" Data:**
        -   `company_results` and `company_shareholding_pattern`: use `date_end`.
        -   Most others (e.g., `company_equity`): use `year_end`.
    3.  **Key Data Points:**
        -   Market Cap: `mcap` in `company_equity`.
        -   Sales/Revenue: `net_sales` or `total_income` in `company_results`.
        -   Promoter Shareholding: The definitive column is `tp_f_total_promoter` in `company_shareholding_pattern`.
    */
"""


EXTRACT_PROMPT = """
    You are an expert financial data extraction assistant. Your most important task is to perform a Pre-Flight Check.

    --- Golden Schema ---
    {golden_schema}
    ---

    First, analyze the user's question. Can it be answered by a single known column in the Golden Schema?
    - "List all distinct industries" -> YES, `industry` from `company_master`.
    - "What are the scrip codes?" -> YES, `scripcode` from `company_master`.
    - "Market cap of company ABC?" -> NO, this requires a join and filter.

    Now, process the user's question and produce a single JSON object matching the 'ExtractionOutput' schema.

    Rules:
    1.  **Pre-Flight Check (MANDATORY):** If the check above is YES, you MUST populate the `known_schema_lookups` field. The key should be "General Query" and the value should be the table and column. If the check is NO, `known_schema_lookups` MUST be null.
    2.  `is_database_required`: Set to true if financial data is needed.
    3.  `entities`: A list of all distinct company names or stock symbols.
    4.  `tasks`: A dictionary mapping each entity's name to a concise task description. For non-entity questions, use the key "General Query".

    --- EXAMPLES ---
    User Question: "List all distinct industries"
    Your Output: {{"is_database_required": true, "entities": [], "tasks": {{"General Query": "List all distinct industries from the database"}}, "known_schema_lookups": {{"General Query": {{"table": "company_master", "column": "industry"}}}}}}

    User Question: "What is the market cap of Company ABC?"
    Your Output: {{"is_database_required": true, "entities": [{{"entity_name": "Company ABC"}}], "tasks": {{"Company ABC": "market capitalization of Company ABC"}}, "known_schema_lookups": null}}
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
    You are a master SQL writer for a MySQL database. Your job is to write a single, syntactically correct SQL query.

    --- THOUGHT PROCESS (You MUST follow this) ---
    1.  **Prioritize Golden Schema:** I will first read the Golden Schema. If there is a conflict, the Golden Schema is ALWAYS correct. For "Promoter Shareholding", I MUST use `psh_f_total_promoter`.

    2.  **Efficient Joins:** I will only `JOIN` with `company_master` if I need the `compname`. If the primary data (like `psh_f_total_promoter`) and the `fincode` for filtering exist in the same table, I will query that table directly to keep the query simple and efficient.

    3.  **Entity Identification & Filtering:**
        - If the `entity_name` is specific (e.g., 'Reliance Industries'), I MUST find its `fincode` using a subquery: `WHERE fincode = (SELECT fincode FROM company_master WHERE compname LIKE '%{entity_name}%' ORDER BY LENGTH(compname) ASC LIMIT 1)`.

    4.  **Handle Ranking Queries:** If the `entity_name` is 'General Query' (like for a "top 5" ranking), I MUST NOT add a `WHERE` clause to filter by company name.

    5.  **Handle "Latest" Data for Rankings:** For a ranking query, I need the latest data for ALL companies. I will join on a subquery that finds the latest `year_end` for each `fincode`, for example: `JOIN (SELECT fincode, MAX(year_end) as max_year FROM company_equity GROUP BY fincode) latest ON ce.fincode = latest.fincode AND ce.year_end = latest.max_year`.

    6.  **Handle "Latest" Data for a Single Entity:** For one company, I will use `ORDER BY [date_column] DESC LIMIT 1`.

    7.  **Verify & Formulate:** I will build the query using ONLY columns from the schemas and prefix ambiguous columns with their table alias.

    --- SCHEMA CONTEXT ---
    -- Golden Schema & High-Level Database Guide --
    {golden_schema}
    -- Dynamic RAG Schema (Columns specific to this task) --
    {rag_context}
    --- END SCHEMA CONTEXT ---

    --- TASK ---
    Original Question: "{original_question}"
    Current Task: "{current_task}"
    Entity Name (if applicable): "{entity_name}"

    Now, following my thought process and all rules, I will write the SQL query as a single JSON object.
"""


ANSWER_COMPOSER_PROMPT = """
    You are IRIS, a financial assistant. Synthesize a single, cohesive, and natural language answer based on the user's question and the data provided.

    --- Data Collected ---
    Query Results (JSON format): {results_summary}
    ---

    **CRITICAL FORMATTING RULES:**
    1.  **Direct Answer:** Do NOT write any introductory phrases like "Here is the answer:". Start the response directly.
    2.  **Indian Numbering System:** All large numerical values MUST be formatted for an Indian audience using 'Lakhs' and 'Crores'. For example, 15,000,000 should be '1.5 Crores' and 500,000 should be '5 Lakhs'. Do not use 'millions' or 'billions'.
    3.  **Full Company Names:** Always use the full company name (e.g., the `compname` value) from the results, not just the short name from the user's question.

    **CONTENT RULES:**
    - **Handle Multiple Results:** If the `Query Results` JSON has multiple top-level keys (e.g., one for "Reliance Industries" and one for "ABB India"), you must combine the information for each key into a single, flowing answer.
    - If the result is a list (like a list of industries), format it as a clean, bulleted list.
    - If results are available, use them to directly answer the user's question.
    - If a query result contains an error, explain the problem clearly and politely.
"""

# --- Agent Nodes ---

async def extract_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Extract Entities & Tasks (with Pre-Flight Check)---")
    prompt = EXTRACT_PROMPT.format(question=state["question"], golden_schema=GOLDEN_SCHEMA_CONTEXT)
    structured_llm = groq_llm.with_structured_output(ExtractionOutput, method="json_mode")
    try:
        result = await structured_llm.ainvoke(prompt)
        print(f"Extraction Result: {result.model_dump_json(indent=2)}")
        return {"extraction": result}
    except Exception as e:
        return {"error_message": f"Failed to understand the query's structure. Error: {e}"}

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
            structured_llm = groq_llm.with_structured_output(GeneralizedQuery)
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
    queries = state["sql_queries"]
    async def _execute_one_query(task_key: str, query: str):
        if "Error" in query:
            return task_key, query
        try:
            async with async_engine.connect() as connection:
                result = await connection.execute(text(query))
                rows = result.fetchall()
            result_str = json.dumps([dict(row._mapping) for row in rows], default=str) if rows else "[]"
            print(f"Result for '{task_key}': {result_str[:250]}...")
            return task_key, result_str
        except Exception as e:
            error_msg = f"Error executing SQL for task '{task_key}': {str(e).strip()}"
            print(error_msg)
            return task_key, f'{{"error": "Failed to execute query.", "details": "{json.dumps(error_msg)}"}}'
            
    coroutines = [_execute_one_query(key, q) for key, q in queries.items()]
    results = await asyncio.gather(*coroutines)
    return {"query_results": dict(results)}

async def compose_final_answer_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Compose Final Answer---")
    if state.get("error_message"):
        return {"final_answer": state["error_message"]}
    
    query_results = state.get("query_results", {})
    
    results_summary = "No database results were retrieved."
    if query_results:
        results_summary = json.dumps(query_results, indent=2)

    prompt = ANSWER_COMPOSER_PROMPT.format(
        question=state["question"],
        results_summary=results_summary
    )
    response = await groq_llm.ainvoke(prompt)
    return {"final_answer": response.content}

# --- Graph Definition ---
def route_after_extraction(state: FundamentalsAgentState) -> str:
    if state.get("error_message"):
        return "end"
    if state["extraction"].is_database_required:
        return "db_path"
    else:
        return "general_path"
        
graph_builder = StateGraph(FundamentalsAgentState)
graph_builder.add_node("extract", extract_node)
graph_builder.add_node("generalize_and_rag", parallel_generalize_and_rag_node)
graph_builder.add_node("write_queries", parallel_write_queries_node)
graph_builder.add_node("execute_queries", parallel_execute_queries_node)
graph_builder.add_node("compose_answer", compose_final_answer_node)

graph_builder.set_entry_point("extract")
graph_builder.add_conditional_edges("extract", route_after_extraction, {
    "db_path": "generalize_and_rag", 
    "general_path": "compose_answer", 
    "end": END
})
graph_builder.add_edge("generalize_and_rag", "write_queries")
graph_builder.add_edge("write_queries", "execute_queries")
graph_builder.add_edge("execute_queries", "compose_answer")
graph_builder.add_edge("compose_answer", END)
app = graph_builder.compile()



# --- Main Execution for Testing ---
if __name__ == "__main__":
    async def run_funda_agent_test():
        test_queries = [
            "What is the market cap of Reliance Industries and the latest sales for ABB India?",
            "What is the business description of Ambalal Sarabhai?",
            "What are the top 5 companies by market cap?",
            "Compare the market cap of Reliance Industries and Ambalal Sarabhai",
            "Scripcode and fincode of ABB India",
            "What is the latest promoter shareholding percentage for Reliance Industries?",
            "List all distinct industries",

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