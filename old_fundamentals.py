# fundamentals_agent.py
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated, List as TypingList, Union, Dict
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os
import json

# --- RAG INTEGRATION: Import the function that provides dynamic context ---
from agents.rag_context_provider import get_rag_based_context

# Custom module imports (ensure these paths are correct)
from model_config import groq_llm
from db_config import async_engine

# For graph visualization
from IPython.display import Image, display

load_dotenv()

class IdentifiedEntity(BaseModel):
    entity_name: str = Field(description="The name of the company/stock symbol identified.")

class ExtractionOutput(BaseModel):
    is_database_required: bool = Field(description="Is database access required to answer the question?")
    entities: List[IdentifiedEntity] = Field(description="List of all unique financial entities found in the query.")
    tasks: Dict[str, str] = Field(description="A dictionary mapping each entity_name (or 'General Query') to a concise description of the data needed for it.")

class GeneralizedQuery(BaseModel):
    generalized_question: str = Field(description="A generic question for RAG context retrieval, with specific entity names removed.")

class SQLQuery(BaseModel):
    query: str = Field(description="A single, syntactically valid SQL query.")


# --- Agent State ---
class FundamentalsAgentState(TypedDict):
    question: str
    extraction: Optional[ExtractionOutput]
    # Store results of parallel operations
    generalized_tasks: Optional[Dict[str, str]]
    rag_contexts: Optional[Dict[str, str]]
    sql_queries: Optional[Dict[str, str]]
    query_results: Optional[Dict[str, str]]
    error_message: Optional[str]
    final_answer: str

# --- Prompts ---
# Using f-strings for cleaner, multi-line prompts
EXTRACT_PROMPT = """
You are an expert financial data extraction assistant. Analyze the user's question to identify all financial entities and describe the task for each.

User Question: {question}

Your task is to produce a single JSON object matching the 'ExtractionOutput' schema.
- `is_database_required`: Set to true if financial data from a database is needed.
- `entities`: A list of all distinct company names or stock symbols.
- `tasks`: A dictionary mapping each entity's name to a concise task description. For non-entity questions, use the key "General Query".

--- EXAMPLES ---
User Question: "What is the market cap of Reliance and latest sales of Infosys?"
Your Output:
{{
  "is_database_required": true,
  "entities": [{{"entity_name": "Reliance"}}, {{"entity_name": "Infosys"}}],
  "tasks": {{
      "Reliance": "market capitalization of Reliance",
      "Infosys": "latest sales figures for Infosys"
  }}
}}

User Question: "List all distinct industries"
Your Output:
{{
  "is_database_required": true,
  "entities": [],
  "tasks": {{ "General Query": "List all distinct industries from the database" }}
}}

User Question: "Hello, how are you?"
Your Output:
{{
  "is_database_required": false,
  "entities": [],
  "tasks": {{ "General Query": "General greeting" }}
}}
---

Output ONLY the valid JSON object.
"""

GENERALIZE_TASK_PROMPT = """
You are a query transformation assistant. Rephrase a specific user request into a generic question suitable for a RAG system to find relevant database schema. Remove all specific entity names.

Produce a JSON object matching the `GeneralizedQuery` schema.

--- EXAMPLES ---
Specific Task: "market cap of Reliance Industries"
Your Output: {{"generalized_question": "which tables and columns are needed to calculate a company's market capitalization?"}}

Specific Task: "latest promoter holding for Aegis Logistics"
Your Output: {{"generalized_question": "how to find the latest promoter holding percentage for a company?"}}
---

Specific Task: {specific_task}
Your Output:
"""

WRITE_QUERY_PROMPT = """
You are an SQL query writing expert. Given a task, a user question, and relevant database schema, create ONE syntactically correct SQL query.

Original User Question: "{original_question}"
Current Task: "{current_task}"
Entity Name (if applicable): "{entity_name}"

--- SCHEMA CONTEXT (Your ONLY source for table/column names) ---
{rag_context}
--- END SCHEMA CONTEXT ---

CRITICAL INSTRUCTION: If an `entity_name` is provided (and not 'General Query'), you MUST use a subquery to find its `fincode` from `company_master`.
Example for `entity_name = "SomeCompany"`:
`... WHERE fincode = (SELECT fincode FROM company_master WHERE compname LIKE '%SomeCompany%' ORDER BY LENGTH(compname) ASC LIMIT 1)`

Output MUST be a single, valid JSON object matching the `SQLQuery` schema.
"""

ANSWER_COMPOSER_PROMPT = """
You are IRIS, a financial assistant. Synthesize a single, cohesive, and natural language answer based on the user's question and the data provided.

Original Question: {question}

--- Data Collected ---
{results_summary}
---

- If the data contains an error message, explain the problem clearly and politely.
- Otherwise, directly answer all parts of the user's question using the provided data.
- Be concise and do not mention SQL or how the data was retrieved.
"""

# --- Agent Nodes (Async & Parallelized) ---

async def extract_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Extract Entities & Tasks---")
    prompt = EXTRACT_PROMPT.format(question=state["question"])
    structured_llm = groq_llm.with_structured_output(ExtractionOutput)
    try:
        result = await structured_llm.ainvoke(prompt)
        print(f"Extraction Result: {result.model_dump_json(indent=2)}")
        return {"extraction": result}
    except Exception as e:
        return {"error_message": f"Failed to understand the query's structure. Error: {e}"}

async def parallel_generalize_and_rag_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Parallel Generalize & RAG---")
    tasks = state["extraction"].tasks
    
    async def _generalize_and_fetch(task_key: str, specific_task: str):
        try:
            # 1. Generalize task
            gen_prompt = GENERALIZE_TASK_PROMPT.format(specific_task=specific_task)
            structured_llm = groq_llm.with_structured_output(GeneralizedQuery)
            gen_result = await structured_llm.ainvoke(gen_prompt)
            generalized_question = gen_result.generalized_question
            print(f"Task '{specific_task}' ==> Generalized: '{generalized_question}'")
            
            # 2. Fetch RAG context with generalized question
            rag_context = get_rag_based_context(generalized_question)
            return task_key, generalized_question, rag_context
        except Exception as e:
            print(f"Error processing task '{task_key}': {e}")
            return task_key, None, f"Error retrieving context for task: {specific_task}"

    # Run all generalization and RAG fetches in parallel
    coroutines = [_generalize_and_fetch(key, task) for key, task in tasks.items()]
    results = await asyncio.gather(*coroutines)
    
    generalized_tasks = {key: gen_q for key, gen_q, _ in results if gen_q}
    rag_contexts = {key: rag_c for key, _, rag_c in results if rag_c}
    
    return {"generalized_tasks": generalized_tasks, "rag_contexts": rag_contexts}


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
            rag_context=rag_context
        )
        structured_llm = groq_llm.with_structured_output(SQLQuery)
        try:
            result = await structured_llm.ainvoke(prompt)
            print(f"Generated SQL for '{task_key}': {result.query}")
            return task_key, result.query
        except Exception as e:
            error_msg = f"Error generating SQL for task '{task_desc}': {e}"
            print(error_msg)
            return task_key, f"/* {error_msg} */ SELECT 'Error generating query';"
            
    coroutines = [_write_one_query(key, task) for key, task in tasks.items()]
    results = await asyncio.gather(*coroutines)
    
    sql_queries = {key: query for key, query in results}
    return {"sql_queries": sql_queries}


async def parallel_execute_queries_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Parallel Execute SQL Queries---")
    queries = state["sql_queries"]
    
    async def _execute_one_query(task_key: str, query: str):
        if "Error generating query" in query:
            return task_key, query # Pass through generation errors
        try:
            async with async_engine.connect() as connection:
                result = await connection.execute(text(query))
                rows = result.fetchall()
            # Convert result to a more readable string format
            result_str = json.dumps([dict(row._mapping) for row in rows], indent=2)
            print(f"Result for '{task_key}': {result_str[:200]}...")
            return task_key, result_str
        except Exception as e:
            error_msg = f"Error executing SQL for '{task_key}': {e}"
            print(error_msg)
            return task_key, error_msg

    coroutines = [_execute_one_query(key, q) for key, q in queries.items()]
    results = await asyncio.gather(*coroutines)
    
    query_results = {key: res for key, res in results}
    return {"query_results": query_results}


async def compose_final_answer_node(state: FundamentalsAgentState) -> Dict:
    print("---NODE: Compose Final Answer---")
    if state.get("error_message"):
        return {"final_answer": state["error_message"]}
        
    results = state.get("query_results", {})
    results_summary = "\n".join([f"- For '{key}':\n{res}" for key, res in results.items()])
    
    if not results_summary: # Handle non-DB path
        return {"final_answer": "Hello! How can I assist you today?"}

    prompt = ANSWER_COMPOSER_PROMPT.format(
        question=state["question"],
        results_summary=results_summary
    )
    response = await groq_llm.ainvoke(prompt)
    return {"final_answer": response.content}

# --- Graph Definition ---

def route_after_extraction(state: FundamentalsAgentState) -> str:
    if state.get("error_message"):
        return "end" # End if extraction failed
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
graph_builder.add_conditional_edges(
    "extract",
    route_after_extraction,
    {
        "db_path": "generalize_and_rag",
        "general_path": "compose_answer",
        "end": END
    }
)
graph_builder.add_edge("generalize_and_rag", "write_queries")
graph_builder.add_edge("write_queries", "execute_queries")
graph_builder.add_edge("execute_queries", "compose_answer")
graph_builder.add_edge("compose_answer", END)

app = graph_builder.compile()

# --- Main Execution for Testing ---
if __name__ == "__main__":
    async def run_funda_agent_test():
        test_queries = [
            "Hello there",
            "What is the market cap of Reliance Industries and the latest sales for Infosys?",
            "List all distinct industries",
            "Description of Ambalal Sarabhai"
        ]
        for q_text in test_queries:
            print(f"\n\n--- TESTING: \"{q_text}\" ---")
            inputs = {"question": q_text}
            try:
                final_state = await app.ainvoke(inputs)
                print(f"\n>>> FINAL ANSWER:\n{final_state.get('final_answer', 'No answer found.')}")
            except Exception as e:
                print(f"\nError during agent execution: {e}")
                import traceback
                traceback.print_exc()

    asyncio.run(run_funda_agent_test())