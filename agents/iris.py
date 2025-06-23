import os
import json
import uuid
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field 
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver


# Import LLM
from model_config import groq_llm

# Import sub-agent apps
from .fundamentals_agent import app as fundamentals_app_instance
from .sentiment_agent import app as sentiment_app_instance
from .technicals_agent import app as technical_app_instance

# Import LTM database module
import db_ltm # Our new module


load_dotenv()

# --- Pydantic Models for Structured LLM Outputs (SupervisorDecision remains largely the same) ---
class SupervisorDecision(BaseModel):
    route: Annotated[str, Field(description="Next step: 'fundamentals', 'sentiment', 'technicals', 'direct_iris_answer', or 'clarify_with_user'.")]
    reasoning: Annotated[str, Field(description="Brief reasoning for the chosen route.")]
    # LTM updates are now more structured:
    # `new_facts_for_ltm` will store explicit "remember this fact" type statements.
    # `profile_updates_for_ltm` will store key-value preference updates.
    new_facts_for_ltm: Annotated[Optional[TypingList[str]], Field(description="List of new EXPLICIT facts user asked to remember (e.g., 'Remember I like apples').")] = None
    profile_updates_for_ltm: Annotated[Optional[Dict[str, Any]], Field(description="Dictionary of user profile/preference updates (e.g., {'favorite_color': 'blue', 'risk_tolerance': 'high'}).")] = None
    parameters_for_subagent: Annotated[Optional[Dict[str, Any]], Field(description="Parameters to pass to the sub-agent, typically {'question': user_input}.")] = None
    direct_iris_response_content: Annotated[Optional[str], Field(description="Content for IRIS's direct response or clarification question if route is 'direct_iris_answer' or 'clarify_with_user'.")] = None


# --- IRIS Agent State ---
class IrisState(TypedDict):
    # Identifiers
    user_identifier: str # e.g., email or unique username provided at input
    langgraph_thread_id: str # LangGraph's session/thread ID

    # LTM Database IDs (populated after initialization)
    db_user_id: Optional[int]
    db_session_id: Optional[int] # MySQL PK for the 'sessions' table

    # User's current input for this turn
    user_input: str
    
    # Short-Term Memory (managed by LangGraph checkpointer)
    chat_history: Annotated[TypingList[BaseMessage], lambda x, y: x + y]

    # Long-Term Memory (loaded from DB)
    # This will hold preferences and facts fetched from db_ltm.user_preferences
    ltm_preferences: Dict[str, Any]

    # Supervisor's decision and intermediate results
    supervisor_decision_output: Optional[SupervisorDecision]
    intermediate_response: Optional[str]
    
    # Final output
    final_response: str

# --- IRIS Agent Nodes ---

def initialize_session_and_ltm_ids_node(state: IrisState):
    """
    Initializes session, gets/creates user in LTM, gets/creates LTM session.
    This node now handles the LTM user and session ID setup.
    """
    print("---NODE: Initialize Session & LTM IDs---")
    user_identifier = state["user_identifier"] # e.g. "test_user_001@example.com"
    langgraph_thread_id = state["langgraph_thread_id"] # LangGraph's own session ID

    # Get or create user in LTM and get their DB ID
    db_user_id = db_ltm.get_or_create_user(user_identifier)
    
    # Get or create LTM session (linked to user and LangGraph thread_id) and get its DB ID
    db_session_id = db_ltm.get_or_create_session(db_user_id, langgraph_thread_id)

    print(f"LTM User ID: {db_user_id} for identifier: {user_identifier}")
    print(f"LTM Session ID: {db_session_id} for LangGraph Thread: {langgraph_thread_id}")
    
    # Ensure chat_history is initialized if this is the first interaction in the LangGraph thread
    current_chat_history = state.get("chat_history", [])
    if not current_chat_history:
         # Optional: Load chat history from LTM for this session if desired for the *very first* prompt to supervisor
         # For now, LangGraph's checkpointer will handle STM continuity.
         # LTM chat_logs are primarily for UI display and long-term archival.
        pass

    return {
        "db_user_id": db_user_id,
        "db_session_id": db_session_id,
        "chat_history": current_chat_history,
    }

def load_ltm_preferences_node(state: IrisState):
    """Loads Long-Term Memory (preferences) for the current user from MySQL."""
    print("---NODE: Load LTM Preferences---")
    db_user_id = state["db_user_id"]
    if not db_user_id:
        print("Error: db_user_id not found in state. Cannot load LTM preferences.")
        return {"ltm_preferences": {}}

    preferences = db_ltm.get_user_preferences(db_user_id)
    print(f"LTM Preferences loaded for user {db_user_id}: {json.dumps(preferences, indent=2)[:300]}...")
    return {"ltm_preferences": preferences}



SUPERVISOR_PROMPT_TEMPLATE = """
You are IRIS, a sophisticated AI assistant for stock analysis.
Your goal is to understand the user's query, leverage conversation history and long-term memory (preferences & facts),
and decide the best course of action. You also need to identify information to be stored in Long-Term Memory.

User Identifier (LTM): {user_identifier} (DB User ID: {db_user_id})

Conversation History (Short-Term Memory):
{chat_history_formatted}

User's Long-Term Memory (loaded from database):
Preferences & Facts: {ltm_preferences_formatted}

User's Current Query: "{user_input}"

--- Definitions for Routing ---
-   **Fundamental Data Query**: Relates to a company's financial health, performance, and intrinsic value. This includes:
    -   Financial statements (e.g., balance sheets, income statements, cash flow statements).
    -   Financial ratios (e.g., P/E ratio, ROE, Debt-to-Equity, EPS, Book Value).
    -   Valuation metrics (e.g., Market Capitalization / Market Cap).
    -   Company structure and details (e.g., shareholding patterns, promoter holding, dividends).
    -   Company events impacting financials (e.g., bonuses, splits).
    If the user asks for any of these for a specific company, route to 'fundamentals'.

-   **Sentiment Query**: Relates to market mood, news analysis, public opinion, or price targets based on analyst sentiment for a stock. Route to 'sentiment'.

-   **Technical Indicator Query**: Relates to chart patterns, price/volume trends, and calculations like RSI, MACD, Moving Averages, Bollinger Bands, VWAP, ATR, Supertrend for a stock. Route to 'technicals'.

-   **Direct IRIS Answer**: For general greetings, simple clarifications IRIS can handle directly, or when the user explicitly asks IRIS to remember something not covered by other agents.

-   **Clarify with User**: If the query is ambiguous, lacks necessary detail for other agents, or if IRIS is unsure.
--- End of Definitions ---

Based on all the above, make a decision.
You MUST respond with a single, valid JSON object that strictly adheres to the following Pydantic schema for 'SupervisorDecision'.
Do NOT include any other text, explanations, or markdown formatting around the JSON object.

Pydantic Schema for SupervisorDecision:
{{
  "route": "string (Next step: 'fundamentals', 'sentiment', 'technicals', 'direct_iris_answer', or 'clarify_with_user')",
  "reasoning": "string (Brief reasoning for the chosen route)",
  "new_facts_for_ltm": "Optional[List[string]] (List of new EXPLICIT facts user asked to remember. Null if none.)",
  "profile_updates_for_ltm": "Optional[Dict[string, Any]] (Dictionary of user profile/preference updates. Null if none.)",
  "parameters_for_subagent": "Optional[Dict[string, Any]] (Parameters for the sub-agent. IMPORTANT: If routing to 'sentiment', this MUST be `{{\\"query\\": \\"original user question\\"}}`. If routing to 'technicals' or 'fundamentals', this MUST be `{{\\"question\\": \\"original user question\\"}}`. Null if not routing.)",
  "direct_iris_response_content": "Optional[string] (Content for IRIS's direct response or clarification. Null if routing to sub-agent.)"
}}


Example for routing to sentiment agent (ensure your output is ONLY the JSON part):
{{
  "route": "sentiment",
  "reasoning": "User is asking about market sentiment for a stock.",
  "new_facts_for_ltm": null,
  "profile_updates_for_ltm": null,
  "parameters_for_subagent": {{ "query": "{user_input}" }}, # Corrected key to "query"
  "direct_iris_response_content": null
}}

Example for routing to technicals agent:
{{
  "route": "technicals",
  "reasoning": "User is asking for a technical indicator.",
  "new_facts_for_ltm": null,
  "profile_updates_for_ltm": null,
  "parameters_for_subagent": {{ "question": "{user_input}" }}, # Use "question" key
  "direct_iris_response_content": null
}}


Think step-by-step to arrive at the values for the JSON fields:
1.  Analyze query: {user_input}
2.  Consider history and LTM.
3.  Determine intent for 'route'.
4.  If routing, set 'parameters_for_subagent'.
5.  If direct/clarify, set 'direct_iris_response_content'.
6.  Identify LTM updates for 'new_facts_for_ltm' and 'profile_updates_for_ltm'.

Now, provide ONLY the JSON object for the query: "{user_input}"
"""


def supervisor_decide_node(state: IrisState):
    """Supervisor LLM decides the route and extracts LTM updates."""
    print("---NODE: Supervisor Decide---")

    

    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])
    ltm_preferences = state.get("ltm_preferences", {})
    user_identifier = state["user_identifier"]
    db_user_id = state["db_user_id"]

    MAX_HISTORY_TURNS_FOR_PROMPT = 5 # Example: last 5 user/AI pairs (10 messages)

    recent_chat_history = chat_history[-(MAX_HISTORY_TURNS_FOR_PROMPT * 2):] # Get last N pairs

    formatted_history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in chat_history])
    if not formatted_history:
        formatted_history = "No conversation history yet."
    elif len(chat_history) > MAX_HISTORY_TURNS_FOR_PROMPT * 2:
        formatted_history = "[...previous messages truncated...]\n" + formatted_history

    
    MAX_LTM_STR_LEN = 1000 # Example

    ltm_preferences_str = json.dumps(ltm_preferences) if ltm_preferences else "No preferences or facts loaded from LTM."
    if len(ltm_preferences_str) > MAX_LTM_STR_LEN:
        ltm_preferences_str = ltm_preferences_str[:MAX_LTM_STR_LEN] + "... [LTM truncated]"


    # --- Truncate chat history for the prompt to avoid token limits ---
    MAX_HISTORY_MESSAGES_FOR_PROMPT = 10 # Keep last 10 messages (5 turns)

    if len(chat_history) > MAX_HISTORY_MESSAGES_FOR_PROMPT:
        history_for_prompt = chat_history[-MAX_HISTORY_MESSAGES_FOR_PROMPT:]
        history_prefix = f"[INFO: Showing last {len(history_for_prompt)//2} turns of conversation. Full history is longer.]\n"
    else:
        history_for_prompt = chat_history
        history_prefix = ""

    formatted_history = history_prefix + "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in history_for_prompt])
    if not history_for_prompt: # Check original chat_history if needed or history_for_prompt
        formatted_history = "No conversation history yet."
    # --- End of truncation ---

    ltm_preferences_str = json.dumps(ltm_preferences) if ltm_preferences else "No preferences or facts loaded from LTM."
    # Consider truncating ltm_preferences_str too if it can be very long.
    MAX_LTM_PREF_LEN_FOR_PROMPT = 1500
    if len(ltm_preferences_str) > MAX_LTM_PREF_LEN_FOR_PROMPT:
        ltm_preferences_str = ltm_preferences_str[:MAX_LTM_PREF_LEN_FOR_PROMPT] + \
                              "... [LTM preferences truncated due to length]"

    prompt_str = SUPERVISOR_PROMPT_TEMPLATE.format(
        user_identifier=user_identifier,
        db_user_id=db_user_id,
        chat_history_formatted=formatted_history, # Use truncated version
        ltm_preferences_formatted=ltm_preferences_str, # Use potentially truncated version
        user_input=user_input
    )
    
    try:
        # First attempt with "json_mode"
        structured_llm_json_mode = groq_llm.with_structured_output(
            SupervisorDecision,
            method="json_mode", 
            # include_raw=False # Default is False, usually fine
        )
        print("Attempting structured output with method='json_mode'")
        decision = structured_llm_json_mode.invoke(prompt_str)
        print(f"Supervisor Decision: {decision.model_dump()}")
        return {"supervisor_decision_output": decision}
    except Exception as e_json_mode:
        print(f"Error with method='json_mode': {e_json_mode}")
        print("Falling back to default method for structured_output (likely 'function_calling')")

        # Fallback to the default method (which was likely 'function_calling')
        # This also helps confirm if 'json_mode' itself is the issue or if the original error persists

        try:
            structured_llm_default = groq_llm.with_structured_output(SupervisorDecision)
            decision = structured_llm_default.invoke(prompt_str)
            print(f"Supervisor Decision (default method): {decision.model_dump()}")
            return {"supervisor_decision_output": decision}

        except Exception as e_default:
            print(f"Error in supervisor_decide_node (default method after json_mode failed): {e_default}")
            # Construct the fallback decision as before
            fallback_decision = SupervisorDecision(
                route="clarify_with_user",
                reasoning="An internal error occurred while processing the request with both json_mode and default structured output.",
                direct_iris_response_content="I'm sorry, I encountered an issue. Could you please rephrase or try again?"
            )
            return {"supervisor_decision_output": fallback_decision}


# call_sub_agent_node, call_fundamentals_agent_node, call_sentiment_agent_node,
# call_technicals_agent_node, prepare_direct_iris_response_node, route_after_supervisor_decision
# remain THE SAME as in the previous version. I'll omit them here for brevity but they are needed.

def call_sub_agent_node(state: IrisState, agent_app: Any, agent_name: str):
    """Generic node to call a sub-agent."""
    print(f"---NODE: Call {agent_name} Agent---")
    decision = state["supervisor_decision_output"]
    if not decision or not decision.parameters_for_subagent:
        error_msg = f"Error: Missing parameters for {agent_name} agent."
        print(error_msg)
        return {"intermediate_response": error_msg}

    input_payload = decision.parameters_for_subagent
    print(f"Input to {agent_name} agent: {input_payload}")
    
    try:
        result = agent_app.invoke(input_payload) # Sub-agents are invoked simply
        response = result.get("final_answer", f"No 'final_answer' key from {agent_name} agent.")
        print(f"Response from {agent_name} agent: {response[:300]}...")
        return {"intermediate_response": response}
    except Exception as e:
        error_msg = f"Error calling {agent_name} agent: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"intermediate_response": error_msg}

def call_fundamentals_agent_node(state: IrisState):
    return call_sub_agent_node(state, fundamentals_app_instance, "fundamentals")

def call_sentiment_agent_node(state: IrisState):
    return call_sub_agent_node(state, sentiment_app_instance, "sentiment")

def call_technicals_agent_node(state: IrisState):
    return call_sub_agent_node(state, technical_app_instance, "technicals")

def prepare_direct_iris_response_node(state: IrisState):
    print("---NODE: Prepare Direct IRIS Response---")
    decision = state["supervisor_decision_output"]
    if not decision or not decision.direct_iris_response_content:
        return {"intermediate_response": "I'm not sure how to respond to that directly. Can you clarify?"}
    response = decision.direct_iris_response_content
    print(f"Direct IRIS Response/Clarification: {response}")
    return {"intermediate_response": response}

def route_after_supervisor_decision(state: IrisState):
    decision = state["supervisor_decision_output"]
    if not decision: return "direct_iris_action" # Fallback
    print(f"Routing based on: {decision.route}")
    route_map = {
        "fundamentals": "call_fundamentals",
        "sentiment": "call_sentiment",
        "technicals": "call_technicals",
        "direct_iris_answer": "direct_iris_action",
        "clarify_with_user": "direct_iris_action"
    }
    chosen_route = route_map.get(decision.route)
    if not chosen_route:
        print(f"Warning: Unknown route '{decision.route}'. Defaulting to clarification.")
        state["supervisor_decision_output"].direct_iris_response_content = (
            f"I'm not sure how to handle the category '{decision.route}'. Could you please clarify?"
        )
        state["supervisor_decision_output"].route = "clarify_with_user"
        return "direct_iris_action"
    return chosen_route


def update_ltm_in_db_node(state: IrisState):
    """Updates LTM (preferences & facts) in MySQL based on supervisor's decision."""
    print("---NODE: Update LTM in DB---")
    decision = state["supervisor_decision_output"]
    db_user_id = state["db_user_id"]
    
    if not db_user_id:
        print("Error: db_user_id not found in state. Cannot update LTM.")
        return {} # No changes to LTM state if error

    if decision:
        if decision.profile_updates_for_ltm:
            for key, value in decision.profile_updates_for_ltm.items():
                db_ltm.set_user_preference(db_user_id, key, value)
            print(f"LTM DB: Updated profile/preferences: {decision.profile_updates_for_ltm}")

        if decision.new_facts_for_ltm:
            db_ltm.add_explicit_facts(db_user_id, decision.new_facts_for_ltm)
            print(f"LTM DB: Added new explicit facts: {decision.new_facts_for_ltm}")
    else:
        print("LTM DB: No supervisor decision found, no LTM updates.")
        
    # No need to return ltm_preferences here as it's not modified by this node directly.
    # The next load_ltm_preferences_node in a new turn would fetch the updated data.
    return {}


def finalize_response_and_log_chat_node(state: IrisState):
    """
    Finalizes the response, updates LangGraph's chat_history (STM),
    and logs the user/assistant messages to the LTM chat_logs table in MySQL.
    """
    print("---NODE: Finalize Response & Log Chat to STM/LTM---")
    user_input = state["user_input"]
    intermediate_response = state.get("intermediate_response", "I'm sorry, I couldn't generate a response.")
    db_session_id = state["db_session_id"] # MySQL LTM session ID

    final_answer = intermediate_response # For now, no final LLM polish

    # 1. Update LangGraph's chat_history (STM for checkpointer)
    current_stm_chat_history = state.get("chat_history", [])
    updated_stm_chat_history = current_stm_chat_history + [
        HumanMessage(content=user_input),
        AIMessage(content=final_answer)
    ]
    
    # 2. Log messages to MySQL LTM chat_logs table
    if db_session_id:
        db_ltm.log_chat_message(db_session_id, "user", user_input)
        db_ltm.log_chat_message(db_session_id, "assistant", final_answer)
        print(f"Messages logged to LTM chat_logs for LTM session ID: {db_session_id}")
    else:
        print("Warning: db_session_id not found. Cannot log messages to LTM chat_logs.")

    print(f"Final Response to User: {final_answer}")
    return {"final_response": final_answer, "chat_history": updated_stm_chat_history}




# --- Build IRIS Graph ---

iris_graph_builder = StateGraph(IrisState)

iris_graph_builder.add_node("initialize_session_and_ltm_ids", initialize_session_and_ltm_ids_node)
iris_graph_builder.add_node("load_ltm_preferences", load_ltm_preferences_node)
iris_graph_builder.add_node("supervisor_decide", supervisor_decide_node)

iris_graph_builder.add_node("call_fundamentals_agent", call_fundamentals_agent_node)
iris_graph_builder.add_node("call_sentiment_agent", call_sentiment_agent_node)
iris_graph_builder.add_node("call_technicals_agent", call_technicals_agent_node)
iris_graph_builder.add_node("direct_iris_action_node", prepare_direct_iris_response_node)

iris_graph_builder.add_node("update_ltm_in_db", update_ltm_in_db_node)
iris_graph_builder.add_node("finalize_response_and_log_chat", finalize_response_and_log_chat_node)


# --- Define Edges ---

iris_graph_builder.add_edge(START, "initialize_session_and_ltm_ids")
iris_graph_builder.add_edge("initialize_session_and_ltm_ids", "load_ltm_preferences")
iris_graph_builder.add_edge("load_ltm_preferences", "supervisor_decide")

iris_graph_builder.add_conditional_edges(
    "supervisor_decide",
    route_after_supervisor_decision,
    {
        "call_fundamentals": "call_fundamentals_agent",
        "call_sentiment": "call_sentiment_agent",
        "call_technicals": "call_technicals_agent",
        "direct_iris_action": "direct_iris_action_node",
    }
)

iris_graph_builder.add_edge("call_fundamentals_agent", "update_ltm_in_db")
iris_graph_builder.add_edge("call_sentiment_agent", "update_ltm_in_db")
iris_graph_builder.add_edge("call_technicals_agent", "update_ltm_in_db")
iris_graph_builder.add_edge("direct_iris_action_node", "update_ltm_in_db")

iris_graph_builder.add_edge("update_ltm_in_db", "finalize_response_and_log_chat")
iris_graph_builder.add_edge("finalize_response_and_log_chat", END)

# Compile with Checkpointer for Short-Term Memory (LangGraph's state inc. chat_history)
# DO NOT COMPILE iris_app here at the module level with the checkpointer like this:
# memory_checkpointer = SqliteSaver.from_conn_string("iris_stm_checkpoint.sqlite")
# iris_app = iris_graph_builder.compile(checkpointer=memory_checkpointer)


# --- Main Execution for Testing IRIS ---

if __name__ == "__main__":
    print("IRIS Supervisor Agent (with MySQL LTM) Compiled. Ready for testing.")


    # In local test mode, we compile the app with a checkpointer context
    with SqliteSaver.from_conn_string("iris_stm_checkpoint.sqlite") as actual_test_checkpointer:
        print("Compiling IRIS graph for local testing...")
        iris_app_for_test = iris_graph_builder.compile(checkpointer=actual_test_checkpointer)
        print("IRIS graph compiled for local testing.")

        # Unique user identifier (e.g., email or username from your auth system)
        test_user_identifier = f"user_{str(uuid.uuid4())[:8]}@example.com" # Ensures a new user for each full script run for easier testing

        # LangGraph thread_id for the session
        # A new thread_id simulates a completely new chat session for LangGraph's STM.
        # If you reuse a thread_id, LangGraph will resume that STM session.
        current_lg_thread_id = f"lg_thread_{str(uuid.uuid4())[:12]}"

        print(f"\n--- Starting Test Session ---")
        print(f"User Identifier (for LTM): {test_user_identifier}")
        print(f"LangGraph Thread ID (for STM): {current_lg_thread_id}")

        config_for_stream = {"configurable": {"thread_id": current_lg_thread_id}}

        test_queries = [
            "Hello IRIS, nice to meet you!",
            
            "What's the latest Net Cash Flow of Reliance",
           
        ]

        # Initial state for the stream must include fields expected by the first node
        # and any fields that are part of the 'configurable' context.
        # `user_input` changes per turn. `user_identifier` and `langgraph_thread_id` are session-level.
        base_inputs = {
            "user_identifier": test_user_identifier,
            "langgraph_thread_id": current_lg_thread_id,
            # db_user_id, db_session_id, chat_history, ltm_preferences will be populated by the graph.
        }

        for q_text in test_queries:
            print(f"\n>>> USER INPUT: {q_text}")

            current_turn_inputs = {**base_inputs, "user_input": q_text}

            try:
                # USE THE CORRECTLY COMPILED APP: iris_app_for_test
                for event_value_map in iris_app_for_test.stream(current_turn_inputs, config=config_for_stream, stream_mode="values"):
                    last_node_event_key = list(event_value_map.keys())[-1]
                    print(f"\nState after node '{last_node_event_key}':")

                    if "final_response" in event_value_map and event_value_map["final_response"]:
                        print(f"  IRIS FINAL RESPONSE: {event_value_map['final_response']}")
                    elif "intermediate_response" in event_value_map and event_value_map["intermediate_response"]:
                        print(f"  IRIS Intermediate Response: {event_value_map['intermediate_response'][:200]}...")
                    elif "supervisor_decision_output" in event_value_map and event_value_map["supervisor_decision_output"]:
                        print(f"  Supervisor Decision: {event_value_map['supervisor_decision_output'].model_dump()}")
                    
                    # You can add more detailed logging for db_user_id, db_session_id, ltm_preferences etc.
                    # if last_node_event_key == "initialize_session_and_ltm_ids":
                        # print(f"  DB User ID: {event_value_map.get('db_user_id')}, DB Session ID: {event_value_map.get('db_session_id')}")


                    # if last_node_event_key == "load_ltm_preferences":
                        # print(f"  LTM Preferences Loaded: {json.dumps(event_value_map.get('ltm_preferences'), indent=2)[:200]}...")

            except Exception as e:
                print(f"Error invoking IRIS graph for question '{q_text}': {e}")
                import traceback
                traceback.print_exc()
            

            print("---------------------------------------\n")


        print("\n--- Test Session Ended ---")
        print(f"Check your '{db_ltm.DB_NAME}' MySQL database for user, preferences, session, and chat_log entries.")
        print(f"User Identifier used: {test_user_identifier}")
        print(f"LangGraph Thread ID used: {current_lg_thread_id}")


        # Example: Retrieve LTM chat history for the session just completed
        # You'd need the db_user_id and then the db_session_id associated with current_lg_thread_id
        # This is a bit manual for a test script; in an app, you'd manage these IDs.


        try:
            final_db_user_id = db_ltm.get_or_create_user(test_user_identifier) # Get the ID again
            # Find the LTM session ID. In a real app, you'd store/retrieve this mapping.
            # For testing, we might need to query sessions table by langgraph_thread_id.
            session_row = db_ltm._execute_query(
                "SELECT id FROM sessions WHERE langgraph_thread_id = :lg_id AND user_id = :db_uid",
                {"lg_id": current_lg_thread_id, "db_uid": final_db_user_id},
                fetch_one=True
            )

            if session_row:
                final_db_session_id = session_row[0]
                print(f"\nRetrieving LTM chat history for LTM Session ID: {final_db_session_id}")
                ltm_chat_history = db_ltm.get_chat_history_for_ltm_session(final_db_session_id, limit=100)

                for role, content, ts in ltm_chat_history:
                    print(f"  {ts} [{role.upper()}]: {content}")

            else:
                print(f"Could not find LTM session for thread {current_lg_thread_id} to display chat history.")

        
        except Exception as e:
            print(f"Error retrieving LTM chat history for testing: {e}")


        
        # Visualize the graph
        try:
            print("\nAttempting to generate IRIS graph visualization...")
            img_data = iris_app_for_test.get_graph().draw_mermaid_png() 
            with open("iris_supervisor_mysql_ltm_graph.png", "wb") as f:
                f.write(img_data)
            print("IRIS graph saved to iris_supervisor_mysql_ltm_graph.png")
        except Exception as e:
            print(f"Could not generate IRIS graph visualization: {e}. Ensure graphviz/pygraphviz/playwright are installed.")
