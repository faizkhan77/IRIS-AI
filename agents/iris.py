# iris.py
import os
import json
import uuid
import asyncio
import traceback
from enum import Enum
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from model_config import groq_llm
from .fundamentals_agent import app as fundamentals_app_instance
from .sentiment_agent import app as sentiment_app_instance
from .technicals_agent import app as technical_app_instance
import db_ltm

load_dotenv()

# --- Configuration ---
MAX_HISTORY_TURNS = 5
MAX_HISTORY_MESSAGES_FOR_PROMPT = MAX_HISTORY_TURNS * 2

class IrisRoute(str, Enum):
    FUNDAMENTALS = "fundamentals"
    SENTIMENT = "sentiment"
    TECHNICALS = "technicals"
    CROSS_AGENT_REASONING = "cross_agent_reasoning"
    IN_DOMAIN_GENERAL = "in_domain_general"
    OUT_OF_DOMAIN = "out_of_domain"
    CLARIFY_WITH_USER = "clarify_with_user"
    LOAD_LTM_AND_RESPOND = "load_ltm_and_respond"
    UPDATE_LTM_AND_RESPOND = "update_ltm_and_respond"

SUB_AGENT_REGISTRY = {
    IrisRoute.FUNDAMENTALS: {"app": fundamentals_app_instance, "input_key": "question", "name": "Fundamentals Agent"},
    IrisRoute.SENTIMENT: {"app": sentiment_app_instance, "input_key": "query", "name": "Sentiment Agent"},
    IrisRoute.TECHNICALS: {"app": technical_app_instance, "input_key": "question", "name": "Technicals Agent"},
}

# --- Pydantic & State Definitions ---
# <<--- SIMPLIFIED: Removed parameters_for_subagent from Pydantic model ---
class SupervisorDecision(BaseModel):
    route: IrisRoute
    reasoning: str
    ltm_operation_needed: Optional[str] = Field(None)
    new_facts_for_ltm: Optional[TypingList[str]] = Field(None, description="A list of simple string facts to remember.")
    profile_updates_for_ltm: Optional[Dict[str, Any]] = Field(None, description="A dictionary for key-value profile updates.")
    direct_iris_response_content: Optional[str] = Field(None)

class IrisState(TypedDict):
    user_identifier: str
    langgraph_thread_id: str
    db_user_id: Optional[int]
    db_session_id: Optional[int]
    user_input: str
    chat_history: Annotated[TypingList[BaseMessage], lambda x, y: x + y]
    ltm_preferences: Optional[Dict[str, Any]]
    ltm_loaded_this_turn: bool
    supervisor_decision_output: Optional[SupervisorDecision]
    intermediate_response: Optional[str]
    sub_agent_outputs: Optional[Dict[str, str]]
    clarification_question_to_user: Optional[str]
    final_response: str

# --- Prompts ---
# <<--- SIMPLIFIED: Removed instructions about populating parameters ---
SUPERVISOR_PROMPT_TEMPLATE = """You are IRIS, a master AI orchestrator for stock analysis. Your primary job is to analyze the user's query and conversation history to choose the correct `route`.

User's Query: "{user_input}"
Conversation History:
{chat_history_formatted}
User's Long-Term Memory (if loaded):
{ltm_preferences_formatted}

--- ROUTING RULES (IN ORDER OF PRIORITY) ---

1.  **LTM Update (`update_ltm_and_respond`):** If the user EXPLICITLY asks to remember/store/update information. For this route, you MUST populate `direct_iris_response_content` with a confirmation message.
    - Example Query: "Please remember I prefer value investing."

2.  **LTM Recall (`load_ltm_and_respond`):** If the user asks to recall stored information. If LTM data is loaded, you MUST populate `direct_iris_response_content` with the answer.
    - Example Query: "What did I say my investment style was?"

3.  **Clarification (`clarify_with_user`):** If the query is extremely vague or a single entity name without context.
    - Example Query: "Adani"

4.  **Standard Sub-Agent Routing:** For any specific financial query.
    - `fundamentals`: "What is the market cap of GOOGL?", "List all distinct industries", "scripcode for reliance".
    - `sentiment`: "sentiment for AAPL?".
    - `technicals`: "RSI for TSLA?".
    - **Contextual Awareness:** A query like "What's its RSI?" after discussing "MSFT" should be routed to `technicals`. The system will handle passing the context.

5.  **Cross-Agent Reasoning:** For BROAD, open-ended investment questions.
    - Example Query: "Should I invest in Google?"

6.  **In-Domain General:** ONLY for simple greetings or definitions IRIS can answer directly.
    - Example Query: "Hi", "What is a stock?"

Your output MUST be a single, valid JSON object matching the `SupervisorDecision` schema.
"""
SYNTHESIS_PROMPT_TEMPLATE = """You are IRIS, a master stock analysis AI. Synthesize the following specialist reports into a single, coherent, and well-structured final answer for the user.
User's Query: "{user_input}"
--- Reports ---
Fundamentals: {fundamentals}
Sentiment: {sentiment}
Technicals: {technicals}
---
Based on all the information, provide a comprehensive, balanced, and easy-to-understand answer. Conclude with a clear disclaimer that this is not financial advice."""
FINAL_SUMMARY_PROMPT_TEMPLATE = """You are a 'bottom-line' financial analyst. Based on the detailed analysis below, create a 1-2 sentence executive summary of the overall conclusion. Do not give direct advice.
--- Detailed Analysis ---
{detailed_synthesis}
---
Provide only the 1-2 sentence summary:"""

# --- Asynchronous Agent Nodes ---
async def initialize_session_node(state: IrisState) -> Dict:
    print("---NODE: Initialize Session & LTM IDs---")
    user_identifier, langgraph_thread_id = state["user_identifier"], state["langgraph_thread_id"]
    db_user_id = await asyncio.to_thread(db_ltm.get_or_create_user, user_identifier)
    db_session_id = await asyncio.to_thread(db_ltm.get_or_create_session, db_user_id, langgraph_thread_id)
    print(f"LTM User ID: {db_user_id}, Session ID: {db_session_id}")
    return {"db_user_id": db_user_id, "db_session_id": db_session_id, "chat_history": state.get("chat_history", []), "ltm_preferences": None, "ltm_loaded_this_turn": False, "sub_agent_outputs": {}, "clarification_question_to_user": None}

async def supervisor_decide_node(state: IrisState) -> Dict:
    print("---NODE: Supervisor Decide---")
    history = state["chat_history"][-MAX_HISTORY_MESSAGES_FOR_PROMPT:]
    formatted_history = "\n".join([f"{m.type}: {m.content}" for m in history]) or "No history."
    ltm_prefs = json.dumps(state["ltm_preferences"]) if state.get("ltm_preferences") else "Not loaded."
    prompt = SUPERVISOR_PROMPT_TEMPLATE.format(user_input=state["user_input"], chat_history_formatted=formatted_history, ltm_preferences_formatted=ltm_prefs)
    structured_llm = groq_llm.with_structured_output(SupervisorDecision)
    try:
        decision = await structured_llm.ainvoke(prompt)
        print(f"Supervisor Decision: Route='{decision.route.value}', Reason='{decision.reasoning}'")
        return {"supervisor_decision_output": decision}
    except Exception as e:
        print(f"Error in supervisor: {e}. Defaulting to clarification.")
        return {"supervisor_decision_output": SupervisorDecision(route=IrisRoute.CLARIFY_WITH_USER, reasoning=f"Internal error: {e}", direct_iris_response_content="I'm having trouble understanding. Could you rephrase?")}

async def load_ltm_preferences_node(state: IrisState) -> Dict:
    print("---NODE: Load LTM Preferences---")
    preferences = await asyncio.to_thread(db_ltm.get_user_preferences, state["db_user_id"])
    return {"ltm_preferences": preferences, "ltm_loaded_this_turn": True}

async def update_ltm_in_db_node(state: IrisState) -> None:
    print("---NODE: Update LTM in DB---")
    decision, user_id = state["supervisor_decision_output"], state["db_user_id"]
    if not user_id or not decision or decision.ltm_operation_needed != "update": return
    async def _update_ltm_in_thread():
        if decision.profile_updates_for_ltm:
            for k, v in decision.profile_updates_for_ltm.items(): db_ltm.set_user_preference(user_id, k, v)
        if decision.new_facts_for_ltm:
            db_ltm.add_explicit_facts(user_id, decision.new_facts_for_ltm)
    await asyncio.to_thread(_update_ltm_in_thread)

async def prepare_direct_iris_action_node(state: IrisState) -> Dict:
    print("---NODE: Prepare Direct IRIS Action---")
    decision = state["supervisor_decision_output"]
    response = decision.direct_iris_response_content if decision and decision.direct_iris_response_content else "I'm not sure how to respond."
    is_clarification = decision and decision.route == IrisRoute.CLARIFY_WITH_USER
    return {"intermediate_response": response, "clarification_question_to_user": response if is_clarification else None}

# <<--- MODIFIED: This node now constructs the payload itself ---
async def call_single_agent_node(state: IrisState) -> Dict:
    route = state["supervisor_decision_output"].route
    print(f"---NODE: Calling Single Agent: {route.value}---")
    agent_info = SUB_AGENT_REGISTRY.get(route)
    if not agent_info: return {"intermediate_response": f"Error: No agent for route '{route.value}'."}
    
    # Construct the payload using the original user input from the state
    input_key = agent_info['input_key']
    input_payload = {input_key: state["user_input"]}
    
    print(f"Invoking {route.value} with payload: {input_payload}")
    try:
        result_state = await agent_info["app"].ainvoke(input_payload)
        return {"intermediate_response": result_state.get("final_answer", "Agent provided no answer.")}
    except Exception as e:
        print(f"Error in sub-agent {route.value}: {traceback.format_exc()}")
        return {"intermediate_response": f"Sorry, the {route.value} specialist encountered a critical error."}

# <<--- MODIFIED: This node now constructs the payload itself ---
async def parallel_agent_call_node(state: IrisState) -> Dict:
    print("---NODE: Staggered Parallel Agent Calls (Cross-Reasoning)---")
    stagger_delay_seconds = 2.0
    outputs = {}
    for route, agent_info in SUB_AGENT_REGISTRY.items():
        try:
            input_key, payload_name = agent_info["input_key"], agent_info["name"]
            # Construct payload from state's user_input
            payload = {input_key: state["user_input"]}
            print(f"Dispatching to {payload_name} with payload: {payload}")
            result_state = await agent_info["app"].ainvoke(payload)
            outputs[route.value] = result_state.get("final_answer", "Agent provided no answer.")
        except Exception as e:
            print(f"Error invoking {payload_name}: {e}")
            outputs[route.value] = f"The {payload_name} encountered an error."
        await asyncio.sleep(stagger_delay_seconds)
    print(f"Staggered Parallel Collection Results: {json.dumps(outputs, indent=2)}")
    return {"sub_agent_outputs": outputs}

async def synthesize_and_summarize_node(state: IrisState) -> Dict:
    print("---NODE: Synthesize & Summarize---")
    outputs = state["sub_agent_outputs"]
    synth_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        user_input=state["user_input"],
        fundamentals=outputs.get(IrisRoute.FUNDAMENTALS.value, "N/A"),
        sentiment=outputs.get(IrisRoute.SENTIMENT.value, "N/A"),
        technicals=outputs.get(IrisRoute.TECHNICALS.value, "N/A"),
    )
    synthesis_response = await groq_llm.ainvoke(synth_prompt)
    summary_prompt = FINAL_SUMMARY_PROMPT_TEMPLATE.format(detailed_synthesis=synthesis_response.content)
    summary_response = await groq_llm.ainvoke(summary_prompt)
    print(f"Executive Summary: {summary_response.content}")
    return {"intermediate_response": summary_response.content}

async def finalize_response_and_log_chat_node(state: IrisState) -> Dict:
    print("---NODE: Finalize & Log---")
    user_input = state["user_input"]
    final_answer = state.get("clarification_question_to_user") or state.get("intermediate_response") or "I'm sorry, I couldn't process that."
    full_history = state["chat_history"] + [HumanMessage(content=user_input), AIMessage(content=final_answer)]
    await asyncio.to_thread(db_ltm.log_chat_message, state["db_session_id"], "user", user_input)
    await asyncio.to_thread(db_ltm.log_chat_message, state["db_session_id"], "assistant", final_answer)
    pruned_history = full_history[-MAX_HISTORY_MESSAGES_FOR_PROMPT:]
    print(f"Final Response: {final_answer}")
    return {"final_response": final_answer, "chat_history": pruned_history}

# --- Graph Definition ---
def route_after_supervisor(state: IrisState) -> str:
    route = state["supervisor_decision_output"].route
    if route == IrisRoute.LOAD_LTM_AND_RESPOND and not state.get("ltm_loaded_this_turn"):
        return "load_ltm"
    if route in [IrisRoute.FUNDAMENTALS, IrisRoute.SENTIMENT, IrisRoute.TECHNICALS]:
        return "call_single_agent"
    if route == IrisRoute.CROSS_AGENT_REASONING:
        return "parallel_agent_call"
    return route.value

graph_builder = StateGraph(IrisState)
graph_builder.add_node("initialize_session", initialize_session_node)
graph_builder.add_node("supervisor_decide", supervisor_decide_node)
graph_builder.add_node("load_ltm", load_ltm_preferences_node)
graph_builder.add_node("update_ltm", update_ltm_in_db_node)
graph_builder.add_node("direct_iris_action", prepare_direct_iris_action_node)
graph_builder.add_node("call_single_agent", call_single_agent_node)
graph_builder.add_node("parallel_agent_call", parallel_agent_call_node)
graph_builder.add_node("synthesize_and_summarize", synthesize_and_summarize_node)
graph_builder.add_node("finalize_response", finalize_response_and_log_chat_node)

graph_builder.set_entry_point("initialize_session")
graph_builder.add_edge("initialize_session", "supervisor_decide")
graph_builder.add_edge("load_ltm", "supervisor_decide")
graph_builder.add_edge("update_ltm", "direct_iris_action")
graph_builder.add_conditional_edges("supervisor_decide", route_after_supervisor, {
    "load_ltm": "load_ltm", "load_ltm_and_respond": "direct_iris_action",
    "update_ltm_and_respond": "update_ltm", "clarify_with_user": "direct_iris_action",
    "in_domain_general": "direct_iris_action", "out_of_domain": "direct_iris_action",
    "call_single_agent": "call_single_agent", "parallel_agent_call": "parallel_agent_call"
})
graph_builder.add_edge("parallel_agent_call", "synthesize_and_summarize")
graph_builder.add_edge("synthesize_and_summarize", "finalize_response")
graph_builder.add_edge("call_single_agent", "finalize_response")
graph_builder.add_edge("direct_iris_action", "finalize_response")
graph_builder.add_edge("finalize_response", END)

iris_graph_builder = graph_builder

# --- Main Execution for Testing ---
if __name__ == "__main__":
    async def run_test_session():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as memory_checkpointer:
            print("Compiling IRIS graph with async checkpointer...")
            app = graph_builder.compile(checkpointer=memory_checkpointer)
            print("Graph compiled.")
            
            test_user = f"user_iris_final_{uuid.uuid4().hex[:6]}"
            thread_id = f"thread_iris_final_{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}
            
            print(f"\n--- Starting IRIS Test Session ---")
            print(f"User: {test_user}, Thread: {thread_id}")

            test_interactions = [
                 "Hello IRIS!",
                "Please remember my favorite stock is TSLA.",
                "What is the RSI for Reliance Industries?",
                "What did I say my favorite stock was?",
                "Should I invest in Reliance now?",
                "What is the market cap of Reliance Industries?",
                "I prefer value investing",
                "Adani",
                "I mean Adani Enterprises",
                "whats my investment style?",
                "List all distinct industries",
                "What's its current sentiment?",
                "Is Aegis Logistics overbought?",
                "What are the top 5 companies by market cap?",
                "Compare the market cap of Reliance Industries and Ambalal Sarabhai",
                "How volatile is Aegis Logistics now?",
                "Should I buy reliance based on bollinger bands?",
                "Who invented the radio?",
                "What is the scripcode and fincode of Amabalal Sarabhai stock?",
                "Thank you!"

            ]

            base_inputs = {"user_identifier": test_user, "langgraph_thread_id": thread_id}
            for turn, user_input in enumerate(test_interactions):
                print(f"\n--- Turn {turn+1}: User Input ---")
                print(f">>> {user_input}\n")
                inputs = {**base_inputs, "user_input": user_input}
                try:
                    final_state = await app.ainvoke(inputs, config)
                    print(f"\n<<< IRIS Final Response:\n{final_state.get('final_response')}")
                except Exception as e:
                    print(f"\n<<< ERROR during invocation: {e}")
                    traceback.print_exc()
                print("-----------------------------")

    asyncio.run(run_test_session())