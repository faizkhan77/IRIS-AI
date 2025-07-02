# iris.py
import os
import json
import uuid
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # SystemMessage removed (not directly used by nodes)
from pydantic import BaseModel, Field 
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver # For persistent STM

from model_config import groq_llm
# Import sub-agent apps - ensure these are the corrected versions
from .fundamentals_agent import app as fundamentals_app_instance
from .sentiment_agent import app as sentiment_app_instance
from .technicals_agent import app as technical_app_instance
import db_ltm

load_dotenv()

# --- Pydantic Models for Structured LLM Outputs ---
class SupervisorDecision(BaseModel):
    route: Annotated[str, Field(description="Next step. Choose ONE from: 'fundamentals', 'sentiment', 'technicals', 'in_domain_general', 'out_of_domain', 'clarify_with_user', 'load_ltm_and_respond', 'update_ltm_and_respond'.")]
    reasoning: Annotated[str, Field(description="Brief reasoning for the chosen route.")]
    
    ltm_operation_needed: Optional[str] = Field(None, description="'load' if LTM data needs to be recalled for the current query, 'update' if user wants to store/modify LTM data. Null otherwise.")
    new_facts_for_ltm: Optional[TypingList[str]] = Field(None, description="List of new EXPLICIT facts user asked to remember (e.g., 'Remember I like apples'). Only if ltm_operation_needed is 'update'.")
    profile_updates_for_ltm: Optional[Dict[str, Any]] = Field(None, description="Dictionary of user profile/preference updates (e.g., {'favorite_stock': 'AAPL'}). Only if ltm_operation_needed is 'update'.")
    
    parameters_for_subagent: Optional[Dict[str, Any]] = Field(None, description="Parameters for sub-agent. For fundamentals/technicals: {'question': user_input}. For sentiment: {'query': user_input}. Null if not routing to a sub-agent.")
    direct_iris_response_content: Optional[str] = Field(None, description="Content for IRIS's direct response (general, OOD, LTM confirmation) or clarification question. Null if routing to a sub-agent that will generate the response.")

# --- IRIS Agent State ---
class IrisState(TypedDict):
    user_identifier: str
    langgraph_thread_id: str # For LangGraph's checkpointer
    
    db_user_id: Optional[int] # LTM User ID
    db_session_id: Optional[int] # LTM Session ID

    user_input: str # Current user message
    chat_history: Annotated[TypingList[BaseMessage], lambda x, y: x + y] # STM (managed by checkpointer)
    
    # LTM data, loaded conditionally
    ltm_preferences: Optional[Dict[str, Any]] 
    # To track if LTM was loaded in the current interaction path for supervisor re-evaluation
    ltm_loaded_this_turn: bool 

    supervisor_decision_output: Optional[SupervisorDecision]
    intermediate_response: Optional[str] # From sub-agents or direct Iris actions before finalization
    
    # For HITL: stores the clarification question if IRIS needs more info
    clarification_question_to_user: Optional[str]
    final_response: str # The actual response sent to the user for the current turn


# --- IRIS Agent Nodes ---

def initialize_session_node(state: IrisState):
    """Initializes LTM user/session IDs and resets turn-specific state,
       BUT PRESERVES the chat_history loaded by the checkpointer."""
    print("---NODE: Initialize Session & LTM IDs (IRIS)---")
    user_identifier = state["user_identifier"]
    langgraph_thread_id = state["langgraph_thread_id"]

    db_user_id = db_ltm.get_or_create_user(user_identifier)
    db_session_id = db_ltm.get_or_create_session(db_user_id, langgraph_thread_id)

    print(f"LTM User ID: {db_user_id} for identifier: {user_identifier}")
    print(f"LTM Session ID: {db_session_id} for LangGraph Thread: {langgraph_thread_id}")
    
    # *** THE KEY FIX IS HERE ***
    # Get the chat history that was passed into the state by the checkpointer.
    # If it's the very first run, this will be an empty list provided by main.py's initial state.
    # On subsequent runs for the same thread, this will be the populated list.
    current_chat_history = state.get("chat_history", [])

    # Return a dictionary that includes the preserved chat_history along with other setup values.
    # This ensures we don't wipe the STM at the start of each turn.
    return {
        "db_user_id": db_user_id,
        "db_session_id": db_session_id,
        "chat_history": current_chat_history, # ** PRESERVE THE HISTORY **
        "ltm_preferences": None,
        "ltm_loaded_this_turn": False,
        "clarification_question_to_user": None,
        "intermediate_response": None,
        "supervisor_decision_output": None
    }

SUPERVISOR_PROMPT_TEMPLATE = """
    You are IRIS, a sophisticated AI assistant for stock analysis. Your primary goal is to determine the best course of action based on the user's query and the conversation history.

    --- Your Thought Process ---
    1.  **Synthesize Context**: First, analyze the user's latest message (`user_input`) in the context of the `chat_history`. If the `user_input` is a direct answer to your last question, combine them to form a complete, coherent query.
    2.  **Decide Action**: Based on the synthesized query, decide the best course of action by determining `ltm_operation_needed` and `route`.
    3.  **Format Output**: If routing to a sub-agent, set `parameters_for_subagent` using the full, synthesized query. If providing a direct answer or clarification, set `direct_iris_response_content`.

    --- Conversation History (Short-Term Memory) ---
    {chat_history_formatted}

    --- User's Long-Term Memory (if loaded for this turn) ---
    {ltm_preferences_formatted}

    --- Routing Definitions (`route`) ---
    - 'update_ltm_and_respond': If user EXPLICITLY asks to remember/store info.
    - 'load_ltm_and_respond': If user asks to recall stored info.
    - 'clarify_with_user': If the synthesized query is still too vague (e.g., "tell me about Adani").
    - 'out_of_domain': For questions clearly unrelated to finance, stocks, or your capabilities.
    - 'in_domain_general': For greetings or simple financial questions you can answer directly.
    - 'fundamentals': For queries about company financials, ratios, valuation.
    - 'sentiment': For market mood, news sentiment, analyst targets.
    - 'technicals': For chart patterns, technical indicators (RSI, MACD), price/volume trends.

    --- LTM Operations (`ltm_operation_needed`) ---
    - 'update': If the route is 'update_ltm_and_respond'. Populate `new_facts_for_ltm` or `profile_updates_for_ltm`.
    - 'load': If the route is 'load_ltm_and_respond'.
    - null: For all other routes.

    --- EXAMPLES ---

    Example 1: Clarification Loop (THIS IS THE KEY NEW EXAMPLE)
    (Previous History in prompt: HUMAN: "what is the market cap?", AI: "For which company?")
    Current Query: "Reliance Industries"
    Output:
    {{
    "route": "fundamentals",
    "reasoning": "The user provided the company name 'Reliance Industries' in response to my clarification question. The combined query is now 'What is the market cap of Reliance Industries?', which is a fundamentals question.",
    "ltm_operation_needed": null,
    "new_facts_for_ltm": null,
    "profile_updates_for_ltm": null,
    "parameters_for_subagent": {{"question": "What is the market cap of Reliance Industries?"}},
    "direct_iris_response_content": null
    }}

    Example 2: LTM Update
    Current Query: "Remember my favorite stock is MSFT and I like blue chip stocks."
    Output:
    {{
    "route": "update_ltm_and_respond", "reasoning": "User wants to store favorite stock and preference.",
    "ltm_operation_needed": "update",
    "new_facts_for_ltm": ["Likes blue chip stocks"], "profile_updates_for_ltm": {{"favorite_stock": "MSFT"}},
    "parameters_for_subagent": null,
    "direct_iris_response_content": "Okay, I've noted that your favorite stock is MSFT and you like blue chip stocks."
    }}

    Example 3: LTM Load
    Current Query: "What was my favorite stock?"
    Output:
    {{
    "route": "load_ltm_and_respond", "reasoning": "User is asking to recall LTM data. The system needs to load it first.",
    "ltm_operation_needed": "load",
    "new_facts_for_ltm": null, "profile_updates_for_ltm": null,
    "parameters_for_subagent": null,
    "direct_iris_response_content": null
    }}

    --- CURRENT TASK ---
    User Identifier (LTM): {user_identifier} (DB User ID: {db_user_id})
    Current User Input: "{user_input}"
    Your output MUST be a single, valid JSON object matching the 'SupervisorDecision' schema. NO OTHER TEXT.

    Provide ONLY the JSON decision:
"""
def supervisor_decide_node(state: IrisState):
    print("---NODE: Supervisor Decide (IRIS)---")
    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])
    ltm_preferences = state.get("ltm_preferences")
    user_identifier = state["user_identifier"]
    db_user_id = state["db_user_id"]

    # --- OPTIMIZATION START ---
    # Define a maximum number of messages to include in the prompt.
    # 8 messages = the last 4 turns (4 user, 4 AI). This is usually plenty for context.
    MAX_HISTORY_MESSAGES_FOR_PROMPT = 8

    # Slice the history to get only the most recent messages.
    # The full history remains in the state, this is ONLY for the prompt.
    history_for_prompt = chat_history[-MAX_HISTORY_MESSAGES_FOR_PROMPT:]
    # --- OPTIMIZATION END ---

    # Format the TRUNCATED chat history for the prompt
    formatted_history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in history_for_prompt])
    if not history_for_prompt: formatted_history = "No conversation history yet."

    ltm_preferences_str = "LTM not loaded for this decision."
    if ltm_preferences:
        ltm_preferences_str = json.dumps(ltm_preferences)
        if len(ltm_preferences_str) > 1000: ltm_preferences_str = ltm_preferences_str[:1000] + "... [LTM truncated]"

    prompt_str = SUPERVISOR_PROMPT_TEMPLATE.format(
        user_identifier=user_identifier, db_user_id=db_user_id,
        chat_history_formatted=formatted_history, # This is now the shorter, truncated history
        ltm_preferences_formatted=ltm_preferences_str,
        user_input=user_input
    )

    try:
        structured_llm_json_mode = groq_llm.with_structured_output(SupervisorDecision, method="json_mode")
        decision = structured_llm_json_mode.invoke(prompt_str)
        print(f"Supervisor Decision: {decision.model_dump_json(indent=2)}")

        if decision.route == "load_ltm_and_respond" and state.get("ltm_loaded_this_turn") and not decision.direct_iris_response_content:
            print("Supervisor re-evaluating after LTM load to formulate response...")
            current_ltm_data_for_reprompt = json.dumps(state["ltm_preferences"]) if state["ltm_preferences"] else "No LTM data found after loading."
            prompt_str_with_loaded_ltm = SUPERVISOR_PROMPT_TEMPLATE.format(
                user_identifier=user_identifier, db_user_id=db_user_id,
                chat_history_formatted=formatted_history, # Use the same truncated history for consistency
                ltm_preferences_formatted=current_ltm_data_for_reprompt,
                user_input=user_input
            )
            decision = structured_llm_json_mode.invoke(prompt_str_with_loaded_ltm)
            print(f"Supervisor Decision (after LTM load & re-evaluation): {decision.model_dump_json(indent=2)}")

        return {"supervisor_decision_output": decision}
    except Exception as e:
        print(f"Error in supervisor_decide_node: {e}. Defaulting to clarification.")
        fallback_decision = SupervisorDecision(
            route="clarify_with_user",
            reasoning=f"Internal error during decision making: {str(e)[:100]}",
            direct_iris_response_content="I'm having a bit of trouble. Could you please rephrase your request or try again?"
        )
        return {"supervisor_decision_output": fallback_decision}

def load_ltm_preferences_node(state: IrisState):
    print("---NODE: Load LTM Preferences (IRIS)---")
    db_user_id = state["db_user_id"]
    if not db_user_id: # Should not happen if init_session ran
        return {"ltm_preferences": {}, "ltm_loaded_this_turn": True, "intermediate_response": "Error: User ID missing, cannot load LTM."}

    preferences = db_ltm.get_user_preferences(db_user_id)
    print(f"LTM Preferences loaded for user {db_user_id}: {json.dumps(preferences, indent=2)[:200]}...")
    # Update state with loaded LTM and flag that it was loaded this turn.
    # The flow will go back to supervisor_decide.
    return {"ltm_preferences": preferences, "ltm_loaded_this_turn": True}

def update_ltm_in_db_node(state: IrisState):
    print("---NODE: Update LTM in DB (IRIS)---")
    decision = state["supervisor_decision_output"]
    db_user_id = state["db_user_id"]
    
    if not db_user_id or not decision or decision.ltm_operation_needed != "update":
        print("LTM DB: Update skipped (no user ID, no decision, or not an 'update' operation).")
        return {} 

    if decision.profile_updates_for_ltm:
        for key, value in decision.profile_updates_for_ltm.items():
            db_ltm.set_user_preference(db_user_id, key, value)
        print(f"LTM DB: Updated profile/preferences: {decision.profile_updates_for_ltm}")

    if decision.new_facts_for_ltm:
        db_ltm.add_explicit_facts(db_user_id, decision.new_facts_for_ltm)
        print(f"LTM DB: Added new explicit facts: {decision.new_facts_for_ltm}")
    
    # After LTM update, the supervisor should have already prepared a confirmation message
    # in direct_iris_response_content. Flow goes to prepare_direct_iris_action_node.
    # No need to change ltm_preferences in state here, it will be reloaded if needed.
    return {}

def call_sub_agent_node(state: IrisState, agent_app: Any, agent_name: str):
    """Generic node to call a sub-agent."""
    print(f"---NODE: Call {agent_name} Agent---")
    decision = state["supervisor_decision_output"]
    if not decision or not decision.parameters_for_subagent:
        return {"intermediate_response": f"Error: Missing parameters for {agent_name} agent call."}

    input_payload = decision.parameters_for_subagent
    print(f"Input to {agent_name} agent: {input_payload}")
    try:
        result_state = agent_app.invoke(input_payload) 
        response = result_state.get("final_answer", f"No 'final_answer' from {agent_name}.")
        return {"intermediate_response": response}
    except Exception as e:
        return {"intermediate_response": f"Sorry, error calling {agent_name} specialist: {str(e)[:100]}."}

def call_fundamentals_agent_node(state: IrisState):
    return call_sub_agent_node(state, fundamentals_app_instance, "fundamentals")

def call_sentiment_agent_node(state: IrisState):
    return call_sub_agent_node(state, sentiment_app_instance, "sentiment")

def call_technicals_agent_node(state: IrisState):
    return call_sub_agent_node(state, technical_app_instance, "technicals")

def prepare_direct_iris_action_node(state: IrisState):
    """Handles direct answers, OOD, LTM confirmations, and clarification questions."""
    print("---NODE: Prepare Direct IRIS Action/Response---")
    decision = state["supervisor_decision_output"]
    response = "I'm not sure how to respond to that. Can you please clarify?" # Default
    
    if decision and decision.direct_iris_response_content:
        response = decision.direct_iris_response_content
    elif decision: # If no direct content but a route that should have it
        response = f"I'm processing that as a '{decision.route}' action." # Fallback if LLM missed content

    print(f"Direct IRIS Action/Response content: {response}")
    
    # If this is a clarification request, store it in clarification_question_to_user
    # This field in state will be used by finalize_response to make it the output for this turn
    if decision and decision.route == "clarify_with_user":
        return {"intermediate_response": response, "clarification_question_to_user": response}
    
    return {"intermediate_response": response, "clarification_question_to_user": None}


def finalize_response_and_log_chat_node(state: IrisState):
    print("---NODE: Finalize Response & Log Chat (IRIS)---")
    user_input = state["user_input"]
    # Prioritize clarification question if one was set for HITL
    final_answer_for_this_turn = state.get("clarification_question_to_user") or \
                                 state.get("intermediate_response") or \
                                 "I'm sorry, I couldn't process that completely."
    
    db_session_id = state["db_session_id"]

    # Update LangGraph's STM (chat_history)
    current_stm_chat_history = state.get("chat_history", [])
    updated_stm_chat_history = current_stm_chat_history + [
        HumanMessage(content=user_input),
        AIMessage(content=final_answer_for_this_turn)
    ]
    
    # Log to LTM database
    if db_session_id:
        db_ltm.log_chat_message(db_session_id, "user", user_input)
        db_ltm.log_chat_message(db_session_id, "assistant", final_answer_for_this_turn)
        print(f"Messages logged to LTM chat_logs for LTM session ID: {db_session_id}")

    print(f"Final Response to User (this turn): {final_answer_for_this_turn}")
    # Reset ltm_loaded_this_turn for the next independent user input
    return {
        "final_response": final_answer_for_this_turn, 
        "chat_history": updated_stm_chat_history,
        "ltm_loaded_this_turn": False # Reset for the next user input cycle
        # clarification_question_to_user is implicitly cleared if not set by prepare_direct_iris_action_node
    }

# --- Routing Logic ---
def route_after_supervisor_decision(state: IrisState):
    decision = state["supervisor_decision_output"]
    if not decision:
        # This is a fallback, should ideally not be hit if supervisor_decide_node is robust
        state["supervisor_decision_output"] = SupervisorDecision(
            route="clarify_with_user", 
            reasoning="Critical error: Supervisor decision missing.",
            direct_iris_response_content="I had an internal hiccup. Could you rephrase?"
        )
        return "direct_iris_action" 

    route = decision.route
    print(f"IRIS Supervisor Routing to: {route} (Reason: {decision.reasoning})")

    # If supervisor wants to load LTM, and it hasn't been loaded this turn yet
    if route == "load_ltm_and_respond" and not state.get("ltm_loaded_this_turn"):
        return "load_ltm" 
    # If it was 'load_ltm_and_respond' but LTM is now loaded (or was already), supervisor should provide direct_iris_response_content
    elif route == "load_ltm_and_respond" and state.get("ltm_loaded_this_turn"):
         # Supervisor should have ideally populated direct_iris_response_content in the re-evaluation step
        return "direct_iris_action"

    if route == "update_ltm_and_respond":
        return "update_ltm"
    
    route_map = {
        "fundamentals": "call_fundamentals",
        "sentiment": "call_sentiment",
        "technicals": "call_technicals",
        "in_domain_general": "direct_iris_action",
        "out_of_domain": "direct_iris_action",
        "clarify_with_user": "direct_iris_action" # This will set clarification_question_to_user
    }
    chosen_node = route_map.get(route)
    if not chosen_node: # Should not happen if supervisor adheres to schema
        print(f"Warning: Unknown route '{route}' from supervisor. Defaulting to clarification.")
        decision.route = "clarify_with_user"
        decision.direct_iris_response_content = f"I'm unsure how to handle '{route}'. Can you clarify?"
        return "direct_iris_action"
    return chosen_node

def route_after_ltm_load(state: IrisState):
    """After LTM is loaded into state, go back to supervisor to re-decide with LTM context."""
    print("LTM loaded. Routing back to supervisor_decide for re-evaluation.")
    return "supervisor_decide"

def route_after_ltm_update(state: IrisState):
    """After LTM is updated in DB, supervisor already set a confirmation message."""
    print("LTM updated in DB. Routing to direct_iris_action for confirmation message.")
    return "direct_iris_action" # Supervisor should have set direct_iris_response_content

# --- Build IRIS Graph ---
iris_graph_builder = StateGraph(IrisState)

iris_graph_builder.add_node("initialize_session", initialize_session_node)
iris_graph_builder.add_node("supervisor_decide", supervisor_decide_node)
iris_graph_builder.add_node("load_ltm", load_ltm_preferences_node)
iris_graph_builder.add_node("update_ltm", update_ltm_in_db_node)
iris_graph_builder.add_node("call_fundamentals_agent", call_fundamentals_agent_node)
iris_graph_builder.add_node("call_sentiment_agent", call_sentiment_agent_node)
iris_graph_builder.add_node("call_technicals_agent", call_technicals_agent_node)
iris_graph_builder.add_node("direct_iris_action_node", prepare_direct_iris_action_node)
iris_graph_builder.add_node("finalize_response_and_log_chat", finalize_response_and_log_chat_node)

iris_graph_builder.set_entry_point("initialize_session")
iris_graph_builder.add_edge("initialize_session", "supervisor_decide")

iris_graph_builder.add_conditional_edges("supervisor_decide", route_after_supervisor_decision, {
    "load_ltm": "load_ltm",
    "update_ltm": "update_ltm",
    "call_fundamentals": "call_fundamentals_agent",
    "call_sentiment": "call_sentiment_agent",
    "call_technicals": "call_technicals_agent",
    "direct_iris_action": "direct_iris_action_node",
})

iris_graph_builder.add_edge("load_ltm", "supervisor_decide") # Re-evaluate after loading
iris_graph_builder.add_edge("update_ltm", "direct_iris_action_node") # Go to confirmation message

iris_graph_builder.add_edge("call_fundamentals_agent", "finalize_response_and_log_chat")
iris_graph_builder.add_edge("call_sentiment_agent", "finalize_response_and_log_chat")
iris_graph_builder.add_edge("call_technicals_agent", "finalize_response_and_log_chat")
iris_graph_builder.add_edge("direct_iris_action_node", "finalize_response_and_log_chat")

iris_graph_builder.add_edge("finalize_response_and_log_chat", END) # END node if no clarification

# --- Compile with Checkpointer ---
# It's good practice to define the checkpointer separately and pass it during compilation,
# especially if you might have different checkpointer configs for dev/test/prod.
# For the main `app` instance that might be imported, compile without checkpointer.
# The checkpointer is applied in the `if __name__ == "__main__":` block for testing.
iris_app_compiled_no_checkpoint = iris_graph_builder.compile()


# --- Main Execution for Testing IRIS ---
if __name__ == "__main__":
    print("IRIS Supervisor Agent (Enhanced for LTM, HITL, New Routes) Compiled.")

    # In local test mode, we compile the app with a checkpointer context
    # THIS IS THE CORRECT PATTERN YOU ORIGINALLY HAD (and should work)
    with SqliteSaver.from_conn_string(":memory:") as memory_checkpointer_instance: # Use :memory: for fresh tests
        print("Compiling IRIS graph for local testing with checkpointer...")
        iris_app_for_test = iris_graph_builder.compile(checkpointer=memory_checkpointer_instance)
        print("IRIS graph compiled for local testing.")

        test_user_identifier = f"user_iris_hitl_{str(uuid.uuid4())[:6]}" 
        current_lg_thread_id = f"lg_thread_hitl_{str(uuid.uuid4())[:8]}"

        print(f"\n--- Starting IRIS Test Session ---")
        print(f"User Identifier (LTM): {test_user_identifier}")
        print(f"LangGraph Thread ID (STM): {current_lg_thread_id}")

        config_for_stream = {"configurable": {"thread_id": current_lg_thread_id}}

        test_interactions = [
            {"user_input": "Hello IRIS!"},
            {"user_input": "What is the scripcode and fincode of Ambalal Sarabhai?"},
            # {"user_input": "What's the capital of Germany?"},
            # {"user_input": "Please remember my favorite stock is GOOGL and I prefer value investing."},
            # {"user_input": "What is the market cap of Reliance Industries?"}, 
            # {"user_input": "What did I say my favorite stock was?"},
            # {"user_input": "And my investment style?"},
            # {"user_input": "Tell me about Adani"},
            # {"user_input": "I mean Adani Enterprises. What's its current sentiment?"},
            # {"user_input": "Is Tesla overbought?"},
            # {"user_input": "Thanks IRIS!"},
            # {"user_input":"List all distinct industries from company_master."},
            # {"user_input":"What is the market cap of Reliance Industries and Aegis Logistics"},
            # {"user_input":"What are the top 5 companies by market cap?"},
            # {"user_input":"Compare the market cap of Reliance Industries and Ambalal Sarabhai"},
            # {"user_input":"Is Reliance overbought?"},
            # {"user_input":"How volatile is Aegis Logistics now?"},
            # {"user_input":"Should I buy reliance based on bollinger bands?"},
            # {"user_input":"Who invented the radio?"}


            
        ]

        base_inputs = {
            "user_identifier": test_user_identifier,
            "langgraph_thread_id": current_lg_thread_id,
        }

        for interaction in test_interactions:
            q_text = interaction["user_input"]
            print(f"\n\n>>> USER INPUT: '{q_text}'")
            
            current_turn_inputs = {**base_inputs, "user_input": q_text}
            
            final_turn_response = None
            try:
                for event_value_map in iris_app_for_test.stream(current_turn_inputs, config=config_for_stream, stream_mode="values"):
                    last_node_event_key = list(event_value_map.keys())[-1] 
                    print(f"\n  State after node ~'{last_node_event_key}':")

                    if "supervisor_decision_output" in event_value_map and event_value_map["supervisor_decision_output"]:
                        print(f"    Supervisor Decision: {event_value_map['supervisor_decision_output'].model_dump_json(indent=2)}")
                    if "ltm_preferences" in event_value_map and event_value_map["ltm_preferences"] is not None:
                         print(f"    LTM Preferences in State: {json.dumps(event_value_map['ltm_preferences'], indent=2)[:200]}...")
                    if "intermediate_response" in event_value_map and event_value_map["intermediate_response"]:
                        print(f"    IRIS Intermediate: {event_value_map['intermediate_response']}")
                    if "clarification_question_to_user" in event_value_map and event_value_map["clarification_question_to_user"]:
                        print(f"    IRIS Clarification Question to User: {event_value_map['clarification_question_to_user']}")
                    if "final_response" in event_value_map and event_value_map["final_response"]:
                        final_turn_response = event_value_map['final_response']
                        print(f"    IRIS FINAL RESPONSE TO USER: {final_turn_response}")
                
                if final_turn_response and "clarify" in final_turn_response.lower(): # Rudimentary check
                    print(f"    >>> IRIS IS WAITING FOR CLARIFICATION (HITL). Next input should be the answer.")

            except Exception as e:
                print(f"Error invoking IRIS graph for question '{q_text}': {e}")
                import traceback
                traceback.print_exc()
            
            print("---------------------------------------\n")

        print("\n--- IRIS Test Session Ended ---")
        print(f"User Identifier used: {test_user_identifier}")
        print(f"LangGraph Thread ID used: {current_lg_thread_id}")
        print(f"Check your '{db_ltm.DB_NAME}' MySQL database. STM was in-memory for this test run.")
            
        try:
            print("\nAttempting to generate IRIS graph visualization (iris_supervisor_v3_graph.png)...")
            # Use the compiled app instance that has the checkpointer for get_graph if needed
            img_data = iris_app_for_test.get_graph().draw_mermaid_png() 
            with open("iris_supervisor_v3_graph.png", "wb") as f:
                f.write(img_data)
            print("IRIS graph saved to iris_supervisor_v3_graph.png")
        except Exception as e:
            print(f"Could not generate IRIS graph visualization: {e}.")
