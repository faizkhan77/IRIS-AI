
import os
import json
import uuid
import asyncio
import traceback
from enum import Enum
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, RootModel 

# --- LangChain/LangGraph Imports ---
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --- Local Imports ---
from model_config import groq_llm, groq_llm_fast
from .fundamentals_agent import app as fundamentals_app_instance
from .sentiment_agent import app as sentiment_app_instance
from .technicals_agent import app as technical_app_instance
import db_ltm

load_dotenv()

# --- Configuration & Enums ---
class IrisRoute(str, Enum):
    SAVE_LTM = "save_ltm"
    LOAD_LTM = "load_ltm"
    FUNDAMENTALS = "fundamentals"
    SENTIMENT = "sentiment"
    TECHNICALS = "technicals"
    CROSS_AGENT_REASONING = "cross_agent_reasoning"
    CLARIFICATION = "clarification"
    OUT_OF_DOMAIN = "out_of_domain" # For non-financial questions
    GENERAL = "general" # For greetings

# --- Pydantic & State Definitions ---
class SupervisorDecision(BaseModel):
    route: IrisRoute
    reasoning: str
class LtmSaveRequest(RootModel[Dict[str, Any]]):
    pass

class IrisState(TypedDict):
    user_identifier: str
    thread_id: str
    db_user_id: Optional[int]
    db_session_id: Optional[int]
    user_input: str
    final_response: Optional[str]
    full_chat_history: Annotated[TypingList[BaseMessage], lambda x, y: x + y]
    supervisor_decision: Optional[SupervisorDecision]
    sub_agent_outputs: Annotated[dict, lambda x, y: {**x, **y}]

# --- Prompts ---

# --- Prompts ---
# (keep SUPERVISOR_PROMPT_TEMPLATE and others)

LTM_UPDATE_PROMPT_TEMPLATE = """You are an AI assistant that specializes in intelligently updating a user's profile preferences based on new information.

**Your Task:**
Analyze the user's new request and update the JSON of their current preferences. Your output must be a single JSON object containing the *complete, final, and updated set of all preferences*.

**Current Preferences (as JSON):**
{existing_preferences}

**User's New Request:**
"{user_input}"

**--- Rules for Updating ---**
1.  **Overwrite:** If the user provides a new, single value for an existing key (e.g., changes their favorite stock), overwrite the old value.
2.  **Merge & Append:** If the user provides additional items for a key that could be a list (like 'fav_indicator'), merge the new items with any existing ones. Create a list if the key previously held a single value. Ensure there are no duplicates in the final list.
3.  **Add New:** If the user provides a completely new preference, add it as a new key-value pair.
4.  **Infer Keys:** Use clear, consistent keys like 'fav_stock', 'investment_style', 'fav_indicator'.
5.  **Output:** Your entire output must be ONLY the final, complete JSON object of facts to save.

**Example 1: Overwrite**
- Current Preferences: {{"fav_stock": "Reliance"}}
- User Request: "My fav stock is ABB India"
- Your Output JSON: {{"fav_stock": "ABB India"}}

**Example 2: Append to List**
- Current Preferences: {{"fav_indicator": ["RSI"]}}
- User Request: "I also love EMA and MACD"
- Your Output JSON: {{"fav_indicator": ["RSI", "EMA", "MACD"]}}

**Example 3: Create List from String**
- Current Preferences: {{"fav_indicator": "RSI"}}
- User Request: "I also like MACD"
- Your Output JSON: {{"fav_indicator": ["RSI", "MACD"]}}

Now, based on the provided preferences and the new request, generate the final JSON object.
"""


# In iris.py

SUPERVISOR_PROMPT_TEMPLATE = """
You are IRIS, a master AI supervisor. Your job is to analyze the user's query to determine their INTENT and route them to the correct tool. You must be extremely precise and follow a logical thought process.

**--- Conversation History (for context) ---**
{chat_history}
- **User's Current Query:** "{user_input}"

**--- YOUR THOUGHT PROCESS (You MUST follow this logic): ---**

1.  **Is the user stating a preference for me to remember?**
    - Trigger Phrases: "My favorite...", "I prefer...", "My style is..."
    - If YES, route to **`save_ltm`**.

2.  **Is the user asking about a preference I have already saved?**
    - Trigger Phrases: "What is my favorite...", "Which indicator do I like?"
    - If YES, route to **`load_ltm`**.

3.  **Is the query just a company name and nothing else?**
    - Examples: "Reliance", "Infosys", "Adani stock"
    - If YES, route to **`clarification`**.

4.  **Is the user asking for my OPINION or RECOMMENDATION on a stock?**
    - This includes questions like "Should I buy/sell/hold...?", "Is it a good buy?", "Is it strong/weak?".
    - If YES, now check for a sub-condition:
        - **Does the question mention a SPECIFIC analysis method?**
            - e.g., "...based on xyz?", "...based on its xyz?", "...based on a,b and c", etc
            - If YES, route to the specific agent (`technicals` or `fundamentals`).
        - **If NO specific method is mentioned**, it's a broad question requiring a full analysis. Route to **`cross_agent_reasoning`**. THIS IS THE DEFAULT FOR OPINION QUESTIONS.

5.  **Is the user asking for a FACTUAL data point?**
    - If the fact is a **technical indicator** (RSI, MACD, volatility, momentum), route to **`technicals`**.
    - If the fact is a **fundamental data point** (market cap, sales, P/E), route to **`fundamentals`**.
    - If the fact is about **news or sentiment**, route to **`sentiment`**.

6.  **Is the question about something other than finance?**
    - If YES, route to **`out_of_domain`**.

7.  **Is it a simple greeting or closing?**
    - If YES, route to **`general`**.

---
Now, apply this thought process to the user's query and provide your output as a single, valid JSON object matching the `SupervisorDecision` schema.
"""

# --- FIX: New prompt for rewriting questions to be self-contained. This logic now lives in IRIS. ---
REWRITE_QUESTION_PROMPT = """You are an expert at rephrasing questions to be self-contained.
Given a chat history and a follow-up question, your task is to rewrite the follow-up question to be a standalone question that can be understood without the context of the chat history.
Resolve any pronouns (like "it", "they", "its", "that") by replacing them with the specific entities they refer to from the history.
If the question is already self-contained, simply return it as is.

**Chat History:**
{chat_history}

**Follow-up Question:**
{question}

**Standalone Question:**"""

SYNTHESIS_PROMPT_TEMPLATE = """You are IRIS, a sharp and confident financial analyst AI. Your task is to synthesize reports into a single, direct, and easy-to-understand response.
**User's Query:** "{user_input}"
**--- Analyst Reports ---**
- **Fundamental Analysis:** {fundamentals}
- **Technical Analysis:** {technicals}
- **Sentiment Analysis:** {sentiment}
**--- Instructions ---**
1. Summarize each finding. If an analyst reported an error, state that plainly.
2. Conclude with a direct opinion (buy/sell/hold).
3. **Formatting:** Use full company names, 'crores'/'lakhs'. No intros or disclaimers.
Begin your synthesized response now.
"""
LTM_RESPONSE_PROMPT = """You are IRIS. You have retrieved the user's saved preferences. Answer their question based on this information and the conversation history.
**Conversation History:**
{chat_history}
**User's Saved Preferences:**
{ltm_data}
**User's Question:**
{user_input}
Answer the question naturally and directly.
"""
CLARIFICATION_PROMPT_TEMPLATE = """You are IRIS, a helpful financial assistant. A user has provided a vague query. Your goal is to ask a smart, clarifying question to better understand their intent. Offer them concrete options.
**User's Vague Query:** "{user_input}"
Generate a suitable clarifying question, starting directly.
"""
GENERAL_PROMPT_TEMPLATE = """You are IRIS, a friendly and professional financial assistant. The user has said something general (like a greeting or a simple question about you). Respond naturally and conversationally. Keep it brief.
**Chat History (for context):**
{chat_history}
**User's Message:** "{user_input}"
Your response:
"""

# --- Helper Function for Rewriting Question ---
async def _get_standalone_question(state: IrisState) -> str:
    """Uses chat history to rewrite a question to be self-contained."""
    history_str = "\n".join([f"{m.type}: {m.content}" for m in state.get('full_chat_history', [])[-6:]])
    if not history_str:
        return state["user_input"]
        
    prompt = REWRITE_QUESTION_PROMPT.format(chat_history=history_str, question=state["user_input"])
    response = await groq_llm.ainvoke(prompt)
    rewritten_question = response.content.strip()
    print(f"--- Rewrote question for agent: '{rewritten_question}' ---")
    return rewritten_question

# --- Main Orchestrator Class ---
class IrisOrchestrator:
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    def _build_graph(self) -> StateGraph:
        graph_builder = StateGraph(IrisState)
        graph_builder.add_node("initialize_session", self.initialize_session_node)
        graph_builder.add_node("supervisor_decide", self.supervisor_decide_node)
        graph_builder.add_node(IrisRoute.SAVE_LTM.value, self.save_ltm_node)
        graph_builder.add_node(IrisRoute.LOAD_LTM.value, self.load_ltm_and_respond_node)
        graph_builder.add_node(IrisRoute.FUNDAMENTALS.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.TECHNICALS.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.SENTIMENT.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.CROSS_AGENT_REASONING.value, self.staggered_parallel_agent_call_node)
        graph_builder.add_node(IrisRoute.CLARIFICATION.value, self.clarification_node)
        graph_builder.add_node(IrisRoute.GENERAL.value, self.general_node)
        graph_builder.add_node(IrisRoute.OUT_OF_DOMAIN.value, self.out_of_domain_node)
        graph_builder.add_node("synthesize_results", self.synthesize_results_node)
        graph_builder.add_node("log_to_db_and_finalize", self.log_to_db_and_finalize_node)
        
        graph_builder.set_entry_point("initialize_session")
        graph_builder.add_edge("initialize_session", "supervisor_decide")
        graph_builder.add_conditional_edges("supervisor_decide", lambda s: s["supervisor_decision"].route.value, {r.value: r.value for r in IrisRoute})
        
        graph_builder.add_edge(IrisRoute.CROSS_AGENT_REASONING.value, "synthesize_results")
        graph_builder.add_edge("synthesize_results", "log_to_db_and_finalize")
        for route in [r for r in IrisRoute if r != IrisRoute.CROSS_AGENT_REASONING]:
            graph_builder.add_edge(route.value, "log_to_db_and_finalize")
        graph_builder.add_edge("log_to_db_and_finalize", END)
        return graph_builder

    # --- Node Implementations ---
    async def initialize_session_node(self, state: IrisState) -> Dict:
        if state.get("db_user_id"): return {}
        user_identifier, thread_id = state["user_identifier"], state["thread_id"]
        db_user_id = await asyncio.to_thread(db_ltm.get_or_create_user, user_identifier)
        db_session_id = await asyncio.to_thread(db_ltm.get_or_create_session, db_user_id, thread_id)
        return {"db_user_id": db_user_id, "db_session_id": db_session_id, "full_chat_history": []}

    async def supervisor_decide_node(self, state: IrisState) -> Dict:
        print("---NODE: Supervisor Decide---")
        history_str = "\n".join([f"{m.type}: {m.content}" for m in state.get('full_chat_history', [])[-6:]])
        prompt = SUPERVISOR_PROMPT_TEMPLATE.format(chat_history=history_str, user_input=state["user_input"])
        structured_llm = groq_llm.with_structured_output(SupervisorDecision)
        try:
            decision = await structured_llm.ainvoke(prompt)
            print(f"Supervisor Decision: Route='{decision.route.value}', Reason='{decision.reasoning}'")
            return {"supervisor_decision": decision}
        except Exception as e:
            return {"supervisor_decision": SupervisorDecision(route=IrisRoute.CLARIFICATION, reasoning=f"Supervisor Error: {e}")}

    async def save_ltm_node(self, state: IrisState) -> Dict:
        print("---NODE: Save LTM (Read-Update-Write)---")
        try:
            # 1. READ existing preferences
            user_id = state["db_user_id"]
            existing_prefs = await asyncio.to_thread(db_ltm.get_user_preferences, user_id)
            print(f"LTM Read: Found existing preferences: {existing_prefs}")

            # 2. UPDATE using an intelligent LLM call
            update_prompt = LTM_UPDATE_PROMPT_TEMPLATE.format(
                existing_preferences=json.dumps(existing_prefs),
                user_input=state["user_input"]
            )
            
            extractor = groq_llm_fast.with_structured_output(LtmSaveRequest)
            
            extracted_data = await extractor.ainvoke(update_prompt)
            # --- FIX: Access the root dictionary directly ---
            updated_facts = extracted_data.root

            print(f"LTM Update: LLM generated updated facts: {updated_facts}")

            # 3. WRITE the complete updated set back to the DB
            for key, value in updated_facts.items():
                value_to_save = json.dumps(value) if isinstance(value, list) else str(value)
                await asyncio.to_thread(db_ltm.set_user_preference, user_id, str(key), value_to_save)
            
            print("LTM Write: Successfully saved updated preferences to DB.")
            return {"final_response": "Got it, I've saved that for you."}
        except Exception as e:
            traceback.print_exc()
            return {"final_response": f"Sorry, I had trouble saving that. Error: {e}"}

    async def load_ltm_and_respond_node(self, state: IrisState) -> Dict:
        print("---NODE: Load LTM and Respond---")
        try:
            # Step 1: Get raw preferences from DB
            raw_ltm_data = await asyncio.to_thread(db_ltm.get_user_preferences, state["db_user_id"])
            if not raw_ltm_data: 
                return {"final_response": "I don't have any preferences saved for you yet."}
            
            # --- FIX: Parse JSON strings back into Python objects (like lists) ---
            parsed_ltm_data = {}
            for key, value in raw_ltm_data.items():
                try:
                    # Try to load value as JSON (for lists, etc.)
                    parsed_ltm_data[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # If it's not valid JSON (i.e., a simple string), use it as is
                    parsed_ltm_data[key] = value
            
            print(f"LTM Load: Parsed preferences: {parsed_ltm_data}")

            # Step 2: Use the parsed data to answer the user's question
            history_str = "\n".join([f"{m.type}: {m.content}" for m in state['full_chat_history'][-6:]])
            prompt = LTM_RESPONSE_PROMPT.format(
                chat_history=history_str, 
                ltm_data=json.dumps(parsed_ltm_data), 
                user_input=state["user_input"]
            )
            response = await groq_llm.ainvoke(prompt)
            return {"final_response": response.content}
        except Exception as e: 
            traceback.print_exc()
            return {"final_response": f"Sorry, I had trouble retrieving your preferences. Error: {e}"}

    async def call_agent_node(self, state: IrisState) -> Dict:
        # Rewrite the question to be self-contained (this part is correct)
        standalone_question = await _get_standalone_question(state)
        
        route = state["supervisor_decision"].route

        # --- FIX: Create the correct payload based on the agent being called ---
        if route == IrisRoute.SENTIMENT:
            payload = {"query": standalone_question}
        else: # For Fundamentals and Technicals
            payload = {"question": standalone_question}
        
        agent_map = {
            IrisRoute.FUNDAMENTALS: fundamentals_app_instance,
            IrisRoute.SENTIMENT: sentiment_app_instance,
            IrisRoute.TECHNICALS: technical_app_instance
        }
        try:
            # Sub-agents now receive a payload with the correct key
            print(f"--- Calling {route.name} agent with payload: {payload} ---")
            result_state = await agent_map[route].ainvoke(payload)
            response = result_state.get("final_answer")
            return {"final_response": response or "The agent could not find an answer."}
        except Exception as e: 
            return {"final_response": f"The {route.name.title()} agent encountered an error: {e!r}"}

    async def synthesize_results_node(self, state: IrisState) -> Dict:
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(user_input=state["user_input"], **state["sub_agent_outputs"])
        response = await groq_llm.ainvoke(prompt)
        return {"final_response": response.content}

    async def clarification_node(self, state: IrisState) -> Dict:
        prompt = CLARIFICATION_PROMPT_TEMPLATE.format(user_input=state["user_input"])
        response = await groq_llm_fast.ainvoke(prompt)
        return {"final_response": response.content}
    
    async def out_of_domain_node(self, state: IrisState) -> Dict:
        return {"final_response": "I am IRIS, an AI financial analyst. I can only answer questions related to the stock market and financial data."}

    async def general_node(self, state: IrisState) -> Dict:
        history_str = "\n".join([f"{m.type}: {m.content}" for m in state.get('full_chat_history', [])[-6:]])
        prompt = GENERAL_PROMPT_TEMPLATE.format(chat_history=history_str, user_input=state["user_input"])
        response = await groq_llm_fast.ainvoke(prompt)
        return {"final_response": response.content}

    async def log_to_db_and_finalize_node(self, state: IrisState) -> Dict:
        user_input, final_response = state["user_input"], state.get("final_response") or "I'm sorry, I cannot respond."
        await asyncio.to_thread(db_ltm.log_chat_message, state["db_session_id"], "user", user_input)
        await asyncio.to_thread(db_ltm.log_chat_message, state["db_session_id"], "assistant", final_response)
        new_history = state["full_chat_history"] + [HumanMessage(content=user_input), AIMessage(content=final_response)]
        print(f"Final Response:\n{final_response}")
        return {"final_response": final_response, "full_chat_history": new_history}


    async def staggered_parallel_agent_call_node(self, state: IrisState) -> Dict:
        # Rewrite the question ONCE (this part is correct)
        standalone_question = await _get_standalone_question(state)

        # --- NEW STEP: Extract the primary entity from the question ---
        # This gives us a clean entity name to pass to all agents.
        entity_extractor_prompt = f"From the following user question, extract the single primary company name or stock symbol. Question: \"{standalone_question}\""
        entity_response = await groq_llm_fast.ainvoke(entity_extractor_prompt)
        entity_name = entity_response.content.strip()
        print(f"--- Extracted entity for parallel call: '{entity_name}' ---")
        
        sub_agent_outputs = {}

        # Fundamentals and Technicals take 'question'
        try:
            print(f"--- Calling parallel fundamentals agent ---")
            payload = {"question": standalone_question}
            result = await fundamentals_app_instance.ainvoke(payload)
            sub_agent_outputs["fundamentals"] = result.get("final_answer", "No response.")
        except Exception as e: 
            sub_agent_outputs["fundamentals"] = f"Analyst error: {e!r}"

        await asyncio.sleep(1.0) # Stagger

        try:
            print(f"--- Calling parallel technicals agent ---")
            payload = {"question": standalone_question}
            result = await technical_app_instance.ainvoke(payload)
            sub_agent_outputs["technicals"] = result.get("final_answer", "No response.")
        except Exception as e: 
            sub_agent_outputs["technicals"] = f"Analyst error: {e!r}"

        await asyncio.sleep(1.0) # Stagger

        # Sentiment agent takes 'query' and now also gets the explicit company name
        try:
            print(f"--- Calling parallel sentiment agent ---")
            # The sentiment agent is now called with the query AND the extracted entity
            # This requires a small change in the sentiment agent to accept `company_name` directly
            # For now, we assume it will re-extract, but the fix is to pass it.
            # Let's assume the Sentiment agent is updated to handle this:
            payload = {"query": standalone_question, "company_name": entity_name}
            result = await sentiment_app_instance.ainvoke(payload)
            sub_agent_outputs["sentiment"] = result.get("final_answer", "No response.")
        except Exception as e: 
            sub_agent_outputs["sentiment"] = f"Analyst error: {e!r}"

        return {"sub_agent_outputs": sub_agent_outputs}


    async def ainvoke(self, payload: dict, config: dict) -> Dict:
        return await self.app.ainvoke(payload, config)

# --- Main Execution for Testing ---
if __name__ == "__main__":
    async def run_test_session():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as memory_checkpointer:
            orchestrator = IrisOrchestrator(checkpointer=memory_checkpointer)
            test_user, thread_id = f"user_{uuid.uuid4().hex[:6]}", f"thread_{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}
            print(f"\n--- Starting IRIS Test Session ---")
            print(f"User: {test_user}, Thread: {thread_id}")

            test_interactions = [
            #     # technicals questions
            #     "What is RSI?", 
            #     "Is Reliance overbought?", 
            #     "How volatile is Aegis Logistics now?", 
            #     "stochastic for Ambalal Sarabhai", 
            #  "Is Reliance strong?", 
            #     "Should i buy Ambalal Sarabhai?",
            #     "Should I buy reliance based on Bollinger Bands",
            #     "Should I buy reliance based on RSI?", 
            #     "Should I buy reliance based on MACD?",
            #     "Should I buy Reliance Industries based on RSI, MACD, and Supertrend?", # Explicit multi-indicator
            #     "Is Reliance Industries a good buy right now?", 

            # "What is the market cap of Reliance Industries and the latest sales for ABB India?",
            # "What is the business description of Ambalal Sarabhai?",
            # "What are the top 5 companies by market cap?",
            # "Compare the market cap of Reliance Industries and Ambalal Sarabhai",
            # "Scripcode and fincode of ABB India",
            # "What is the latest promoter shareholding percentage for Reliance Industries?",
            # "List all distinct industries",
            # "I prefer long term investments and my fav stock is Reliance Industries",
            # "What is my investment style?",
            # "and my fav stock?"
            # "My fav stock is Reliance Industies",
            # "My fav stock is ABB India now",
            # "But i also love Aegis Logistics",
            # "and i prefer to use EMA"
            # "Whats my fav stocks?",
            # "and which indicator do i use?",
            # "I also love RSI and MACD",
            # "Now tell me which are the indicators that i use?",
            # "Reliance Industries",
            # "tell me the market cap of it",
            # "Now tell me its latest closing price",
            # "Reliance",
            # "adani",
            # "hi",
            # "How are you",
            # "who invented TV",

            # cross agent reasoning
            "Is Reliance Industries strong?",
            "Should i buy Ambalal Sarabhai?",
            "Is Reliance Industries a good buy right now?"
            ]

            
            initial_payload = {"user_identifier": test_user, "thread_id": thread_id}
            for turn, user_input in enumerate(test_interactions):
                print(f"\n========================= Turn {turn + 1} =========================")
                print(f">>> User: {user_input}\n")
                current_payload = {**initial_payload, "user_input": user_input}
                try:
                    await orchestrator.ainvoke(current_payload, config)
                except Exception as e:
                    print(f"\n<<< CRITICAL ERROR during invocation: {e}")
                    traceback.print_exc()
                print("----------------------------------------------------")
    asyncio.run(run_test_session())