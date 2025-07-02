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
from db_config import engine

# For graph visualization
from IPython.display import Image, display

load_dotenv()

db = SQLDatabase(engine=engine)

# TypedDicts are the same as before, no changes needed here.
class IdentifiedEntityInfo(TypedDict):
    entity_name: Annotated[str, "The name of the company/stock symbol identified."]

class ExtractionOutput(TypedDict):
    is_required_database: Annotated[bool, "Is database access required?"]
    entities_info: Annotated[TypingList[IdentifiedEntityInfo], "List of entities found in the query."]
    entity_task_description: Annotated[Dict[str, str], "A dictionary mapping entity_name to a brief description of what data is needed for it. Use 'General Query' for non-entity questions."]

class GeneralizedQueryOutput(TypedDict):
    """Output for the generalization step."""
    generalized_question: str

class AgentState(TypedDict):
    question: str
    extract: ExtractionOutput
    generalized_tasks: Dict[str, str]
    columns_context: str
    queries_meta: TypingList[Dict[str, Union[str, bool]]]
    results: TypingList[str]
    answer: str
    final_answer: str

class QueryOutput(TypedDict):
    query: Annotated[str, "Syntactically valid SQL query."]

# Prompts are also the same as before, they are already well-defined.
EXTRACT_MULTI_ENTITY_PROMPT_TEMPLATE = """
You are an expert financial data extraction assistant. Your job is to analyze the user's question to identify financial entities and describe the task for each. You DO NOT need to identify database tables.

User Question: {question}

Your task is to produce a JSON object adhering to the 'ExtractionOutput' schema.
1.  `is_required_database`: Set to `true` if financial data is needed.
2.  `entities_info`: Identify ALL distinct company names or stock symbols.
3.  `entity_task_description`: For each `entity_name`, provide a concise description of the information the user wants for that entity. If the query is general and has no specific entity, you must use "General Query" as the key.

Schema for 'ExtractionOutput':
{{
  "is_required_database": "boolean",
  "entities_info": [
    {{
      "entity_name": "string (e.g., 'Reliance Industries', 'TCS')"
    }}
  ],
  "entity_task_description": {{
      "entity_name_1": "string (e.g., 'market capitalization')",
      "entity_name_2": "string (e.g., 'latest sales figures')",
      "General Query": "string (e.g., 'list all distinct industries')"
  }}
}}

--- EXAMPLES ---
User Question: "What is the market cap of Reliance and latest sales of Infosys?"
Your Output:
{{
  "is_required_database": true,
  "entities_info": [
    {{"entity_name": "Reliance"}},
    {{"entity_name": "Infosys"}}
  ],
  "entity_task_description": {{
      "Reliance": "market capitalization of Reliance",
      "Infosys": "latest sales figures for Infosys"
  }}
}}

User Question: "List all distinct industries"
Your Output:
{{
    "is_required_database": true,
    "entities_info": [],
    "entity_task_description": {{
        "General Query": "List all distinct industries from the database"
    }}
}}
---

Output ONLY the valid JSON object. Do not include any other text or explanations.
"""

WRITE_QUERY_SYSTEM_MESSAGE = """
You are an SQL query writing expert. Given an input question, a specific task description, and a highly relevant `columns_context` (from a RAG system), create ONE syntactically correct {dialect} query.

Original User Question (for overall context): "{original_question}"
Current Task Description (focus for THIS query): "{current_task_description}"
Entity Name (if applicable, for subquery): "{current_entity_name}"

--- CRITICAL INSTRUCTION: COLUMN NAMES SOURCE OF TRUTH ---
Use `columns_context` (detailed schema for relevant tables provided by a RAG system) AS THE ABSOLUTE AND ONLY SOURCE OF TRUTH FOR TABLE AND COLUMN NAMES.
Do NOT invent column names. PRIORITIZE `columns_context`. If a required column is not in the context, do your best with what is provided.
--- END CRITICAL INSTRUCTION ---

DETAILED SCHEMA FOR RELEVANT TABLES (YOUR PRIMARY GUIDE):
{columns_context}

--- CRITICAL INSTRUCTIONS FOR COMPANY NAME LOOKUPS ---
If `current_entity_name` is not 'General Query', you MUST use a subquery to find its `fincode` from `company_master`.
Example for `current_entity_name = "SomeCompany"`:
`... WHERE fincode = (SELECT fincode FROM company_master WHERE compname LIKE '%SomeCompany%' ORDER BY LENGTH(compname) ASC LIMIT 1)`
--- END OF CRITICAL INSTRUCTIONS ---

Output MUST be a single, valid JSON object matching the `QueryOutput` schema:
`{{ "query": "string (The generated SQL query.)" }}`
Do NOT include any other text, explanations, or markdown. Just the JSON.
"""
GENERAL_RESPONSE_SYSTEM_PROMPT = "You are IRIS. Respond to the user's question naturally. If the question is a general greeting or unrelated to financial data, provide a polite, general response."
GENERATE_MULTI_ANSWER_DB_PATH_PROMPT_TEMPLATE = "Given the user's original question and a list of SQL queries executed along with their results, synthesize a single, cohesive, and natural language answer.\nMake sure to address all parts of the user's original question. Be concise and direct.\n\nOriginal Question: {question}\n\nExecuted Queries and Their Results:\n{queries_and_results_formatted_str}\n\nConsolidated Answer:"
VALIDATE_DB_RESPONSE_PROMPT_TEMPLATE = "You are IRIS. Based on the question and the SQL results, improve the draft answer to be clear, natural, and directly respond to all aspects of the question.\nEnsure a natural, non-generic tone. Do not mention SQL queries or technical details. Be concise.\n\nOriginal Question: {question}\nSQL Results (for context only):\n{sql_results_raw_str}\nDraft Response to Improve: {draft_answer}\n\nRefined Final Answer:"
NORMALIZE_GENERAL_RESPONSE_PROMPT_TEMPLATE = "You are a helpful assistant Named IRIS. Review the draft answer to the user's non-database question and ensure it's clear, natural, and polite.\n\nQuestion: {question}\nResponse to Improve: {draft_answer}\n\nFinal Answer:"


# NEW PROMPT for the new generalization node
GENERALIZE_TASK_PROMPT_TEMPLATE = """
You are a query transformation assistant. Your job is to rephrase a specific user request about a company into a generic question about finding the right database schema.
- REMOVE all specific company names, stock symbols, or personal identifiers.
- Focus ONLY on the financial or data concept being asked for.
- The output should be a question phrased to help a vector search find relevant table and column descriptions.

Produce a JSON object matching the `GeneralizedQueryOutput` schema: `{{"generalized_question": "string"}}`

--- EXAMPLES ---
Specific Task: "market cap of Reliance Industries"
Your Output:
{{"generalized_question": "which table or columns are needed to calculate a company's market capitalization?"}}

Specific Task: "latest promoter holding for Aegis Logistics"
Your Output:
{{"generalized_question": "how to find the latest promoter holding percentage for a company?"}}

Specific Task: "Description of Ambalal Sarabhai"
Your Output:
{{"generalized_question": "which table and column contains the business description of a company?"}}

Specific Task: "list all distinct industries"
Your Output:
{{"generalized_question": "how to list all distinct industries from the database?"}}
---

Specific Task: {specific_task}
Your Output:
"""


# --- Agent Nodes (Refactored `get_columns_context_node`) ---

def extract_from_input_node(state: AgentState):
    print("---NODE: Extract Entities & Tasks from Input---")
    prompt_str = EXTRACT_MULTI_ENTITY_PROMPT_TEMPLATE.format(question=state["question"])
    structured_llm = groq_llm.with_structured_output(ExtractionOutput, method="json_mode")
    try:
        result = structured_llm.invoke(prompt_str)
        print(f"Extraction Result: {json.dumps(result, indent=2)}")
        return {"extract": result}
    except Exception as e:
        print(f"Error in extract_from_input_node: {e}. Defaulting to non-DB path.")
        return {"extract": {"is_required_database": False, "entities_info": [], "entity_task_description": {}}}

def general_response_node(state: AgentState):
    print("---NODE: General Response---")
    messages = [SystemMessage(content=GENERAL_RESPONSE_SYSTEM_PROMPT), HumanMessage(content=state["question"])]
    response_content = groq_llm.invoke(messages).content
    return {"answer": response_content}


def generalize_tasks_for_rag_node(state: AgentState):
    print("---NODE: Generalize Tasks for RAG---")
    tasks_to_run = state["extract"].get("entity_task_description", {})
    generalized_tasks = {}
    
    generalizer_llm = groq_llm.with_structured_output(GeneralizedQueryOutput, method="json_mode")

    for entity_name, specific_task in tasks_to_run.items():
        try:
            prompt_str = GENERALIZE_TASK_PROMPT_TEMPLATE.format(specific_task=specific_task)
            response = generalizer_llm.invoke(prompt_str)
            gen_q = response.get("generalized_question", "")
            if gen_q:
                generalized_tasks[entity_name] = gen_q
                print(f"Original: '{specific_task}'  ==>  Generalized: '{gen_q}'")
            else:
                # Fallback to the original if generalization fails
                generalized_tasks[entity_name] = specific_task
                print(f"Warning: Could not generalize task for '{entity_name}', using original.")

        except Exception as e:
            print(f"Error generalizing task for '{entity_name}': {e}. Using original task.")
            generalized_tasks[entity_name] = specific_task
            
    return {"generalized_tasks": generalized_tasks}

# --- THIS IS THE REFACTORED RAG INTEGRATION NODE ---
# --- THIS NODE IS NOW UPDATED to use the generalized tasks ---
def get_columns_context_node(state: AgentState):
    print("---NODE: Get Columns Context (RAG-Powered)---")
    # It now uses the output from the new generalization node
    generalized_tasks = state.get("generalized_tasks", {})
    if not generalized_tasks:
        return {"columns_context": "No tasks were identified or generalized for the RAG."}

    all_contexts = set()
    for task_description in generalized_tasks.values():
        # The query to the RAG is now the clean, generic question
        context_for_task = get_rag_based_context(task_description)
        if context_for_task and "Error" not in context_for_task:
            all_contexts.add(context_for_task)

    if not all_contexts:
        final_context = "RAG provider could not find relevant table context for the tasks."
    else:
        final_context = "\n\n".join(sorted(list(all_contexts))) # sorted for deterministic output
    
    print("\n---COMBINED CONTEXT FOR SQL WRITER---")
    print(final_context)
    print("-------------------------------------\n")
    
    return {"columns_context": final_context}

def write_queries_node(state: AgentState):
    print("---NODE: Write SQL Queries (Multi-Task)---")
    # ... (code is unchanged)
    extract_output = state["extract"]
    columns_context = state["columns_context"]
    original_question = state["question"]
    queries_meta_list = []
    
    # IMPORTANT: We still iterate over the ORIGINAL tasks for the query writer prompt
    # The `columns_context` is now just richer.
    tasks_to_run = extract_output.get("entity_task_description", {})
    # ... rest of the function is unchanged
    def _generate_one_query(task_desc: str, entity_name_for_query: str):
        prompt_obj = ChatPromptTemplate.from_messages([("system", WRITE_QUERY_SYSTEM_MESSAGE)]).invoke({
            "dialect": db.dialect, "original_question": original_question, "current_task_description": task_desc,
            "current_entity_name": entity_name_for_query, "columns_context": columns_context,
        })
        try:
            query_writer_llm = groq_llm.with_structured_output(QueryOutput, method="json_mode")
            query_output = query_writer_llm.invoke(prompt_obj)
            sql_query = query_output.get("query")
            if sql_query and isinstance(sql_query, str): return sql_query.strip(), False
            return f"LLM failed to provide valid SQL for task: {task_desc}", True
        except Exception as e: return f"LLM invocation failed for task '{task_desc}': {e}", True
    if not tasks_to_run:
        print("Warning: No tasks identified to generate queries for.")
        return {"queries_meta": []}
    for entity_or_general, task_description in tasks_to_run.items():
        query_str, is_err = _generate_one_query(task_description, entity_or_general)
        if not is_err: print(f"Generated SQL for task '{task_description}': {query_str}")
        else: print(f"Failed to generate SQL for task '{task_description}': {query_str}")
        queries_meta_list.append({"query_string": query_str, "entity_name": entity_or_general, "is_error_placeholder": is_err})
    return {"queries_meta": queries_meta_list}

def execute_queries_node(state: AgentState):
    print("---NODE: Execute SQL Queries---")
    # ... (code is unchanged)
    queries_meta_list = state.get("queries_meta", [])
    execution_results = []
    if not queries_meta_list: return {"results": ["No queries were provided for execution."]}
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    for item_meta in queries_meta_list:
        query_to_run = item_meta["query_string"]
        if item_meta["is_error_placeholder"]:
            execution_results.append(query_to_run)
            continue
        try:
            db_result = execute_query_tool.invoke(query_to_run)
            execution_results.append(str(db_result))
        except Exception as e:
            error_msg = f"Error executing query for {item_meta['entity_name']}: {e}"
            execution_results.append(error_msg)
    return {"results": execution_results}

def generate_answer_db_path_node(state: AgentState):
    print("---NODE: Generate Answer (DB Path)---")
    queries_meta = state.get("queries_meta", [])
    results = state.get("results", [])
    
    if not results or not queries_meta:
        return {"answer": "I'm sorry, I couldn't process your request as no results were generated."}

    q_and_r_formatted_list = []
    for i, res_str in enumerate(results):
        if i < len(queries_meta):
            entity_name = queries_meta[i]['entity_name']
            q_and_r_formatted_list.append(f"--- For: {entity_name} ---\nResult: {res_str}\n---")
    
    prompt_str = GENERATE_MULTI_ANSWER_DB_PATH_PROMPT_TEMPLATE.format(
        question=state["question"], 
        queries_and_results_formatted_str="\n\n".join(q_and_r_formatted_list)
    )
    response_content = groq_llm.invoke(prompt_str).content
    return {"answer": response_content}

def validate_db_response_node(state: AgentState):
    print("---NODE: Validate DB Response---")
    prompt_str = VALIDATE_DB_RESPONSE_PROMPT_TEMPLATE.format(
        question=state["question"],
        sql_results_raw_str="\n".join(state.get("results", [])),
        draft_answer=state.get("answer", "")
    )
    response_content = groq_llm.invoke(prompt_str).content
    return {"final_answer": response_content}

def normalize_general_response_node(state: AgentState):
    print("---NODE: Normalize General Response---")
    prompt_str = NORMALIZE_GENERAL_RESPONSE_PROMPT_TEMPLATE.format(
        question=state["question"],
        draft_answer=state.get("answer", "")
    )
    response_content = groq_llm.invoke(prompt_str).content
    return {"final_answer": response_content}


# --- Graph Definition (Workflow Preserved) ---
def question_router(state: AgentState):
    print("---ROUTER: DB or General?---")
    extract_output = state["extract"]
    if extract_output.get("is_required_database"):
        print("Routing to: DB Path")
        return "database_path"
    else:
        print("Routing to: General Path")
        return "general_path"

graph_builder = StateGraph(AgentState)
graph_builder.add_node("extract_from_query", extract_from_input_node)
graph_builder.add_node("generalize_tasks_for_rag", generalize_tasks_for_rag_node) # New node
graph_builder.add_node("db_columns_context", get_columns_context_node)
graph_builder.add_node("general_response_draft", general_response_node)
graph_builder.add_node("write_sql_queries", write_queries_node)
graph_builder.add_node("execute_sql_queries", execute_queries_node)
graph_builder.add_node("generate_db_answer_draft", generate_answer_db_path_node)
graph_builder.add_node("validate_db_final_response", validate_db_response_node)
graph_builder.add_node("normalize_general_final_response", normalize_general_response_node)

# Define the graph's edges and control flow
graph_builder.add_edge(START, "extract_from_query")
graph_builder.add_conditional_edges(
    "extract_from_query",
    question_router,
    {
        # The database path now goes to our NEW generalization node first
        "database_path": "generalize_tasks_for_rag", 
        "general_path": "general_response_draft"
    }
)

# This is the new flow: generalize -> get context -> write query
graph_builder.add_edge("generalize_tasks_for_rag", "db_columns_context")
graph_builder.add_edge("db_columns_context", "write_sql_queries")
graph_builder.add_edge("write_sql_queries", "execute_sql_queries")
#... rest of the db path
graph_builder.add_edge("execute_sql_queries", "generate_db_answer_draft")
graph_builder.add_edge("generate_db_answer_draft", "validate_db_final_response")
graph_builder.add_edge("validate_db_final_response", END)

# The general path remains the same
graph_builder.add_edge("general_response_draft", "normalize_general_final_response")
graph_builder.add_edge("normalize_general_final_response", END)

app = graph_builder.compile()

# --- Main Execution for Testing ---
if __name__ == "__main__":
    print("Fundamentals Agent Graph (Lean RAG Integration) Compiled.")
    # The same test queries as before
    test_queries = [
        "Hello there",
        "List all distinct industries",
        "What is the market cap of Reliance Industries?",
        "What is the market cap of Reliance Industries and Aegis Logistics",
        "What are the top 5 companies by market cap?",
        "Compare the market cap of Reliance Industries and Ambalal Sarabhai",
        "Description of Ambalal Sarabhai"
        
    ]
    for q_text in test_queries:
        print(f"\n\n--- TESTING FUNDAMENTALS QUESTION: {q_text} ---\n")
        inputs = {"question": q_text}
        try:
            for event_map in app.stream(inputs, stream_mode="values", config={"recursion_limit": 25}):
                if "final_answer" in event_map and event_map.get("final_answer"):
                    print(f"\nFINAL ANSWER: {event_map['final_answer']}")
        except Exception as e:
            print(f"Error invoking graph for question '{q_text}': {e}")
            import traceback
            traceback.print_exc()
        print("\n---------------------------------------\n")