
import os
import json
import uuid
import asyncio
import traceback
from enum import Enum
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, RootModel 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate 
# --- LangChain/LangGraph Imports ---
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --- Local Imports ---
from model_config import groq_llm, groq_llm_fast
from .fundamentals_agent import app as fundamentals_app_instance
from .sentiment_agent import app as sentiment_app_instance
from .technicals_agent import app as technical_app_instance
from . import db_ltm
from .db_ltm import update_session_summary

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
    RECOMMENDATION = "recommendation"

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

LTM_UPDATE_PROMPT_TEMPLATE = """
    You are an AI assistant that specializes in intelligently updating a user's profile preferences based on new information.

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

SUPERVISOR_PROMPT_TEMPLATE = """
    You are a precise, efficient routing AI. Your sole purpose is to analyze the user's query and conversation history and output a single, valid JSON object to route the query to the correct tool.

    **--- Conversation History (for context) ---**
    {chat_history}
    - **User's Current Raw Query:** "{user_input}"

    **--- You MUST follow this logic in this exact order: ---**

    **1. **Recommendation & Screener Check:**
        - Is the user asking for a **recommendation**, **suggestion**, or a **list of stocks based on criteria**? (e.g., "recommend me stocks," "top 5 stocks by P/E," "best stocks for beginners").
        - If YES, route to **`recommendation`**. This is a high-priority task.
    
    **2. Answer to Clarification Check (Highest Priority):**
        - Was the ASSISTANT's last message a question? (e.g., "Which company?").
        - Does the User's Current Query look like a direct ANSWER to that question? (e.g., "Tata Motors").
        - If YES to both, find the user's *original question* from before the clarification and determine the route based on THAT intent. For example, if the original question was "What are the fundamentals for it?" and the user now says "Tata Motors", the correct route is `fundamentals`.

    **3. Long-Term Memory (LTM) Check:**
        - Is the user **TELLING ME** a new preference to remember? (e.g., "My favorite stock is...", "I prefer..."). If YES, route to **`save_ltm`**.
        - Is the user's primary goal to **RECALL** a saved preference? (e.g., "What is my favorite stock?", "Remind me which indicators I like?"). If YES, route to **`load_ltm`**.
        - **IMPORTANT:** If the user wants to **APPLY** a known preference to a new task (e.g., "Analyze xyz based on my favorite indicators"), DO NOT route to `load_ltm`.

    **4. Contextual Follow-up Check:**
        - If not an answer to clarification, does the query refer to a previous turn with pronouns or vague references (e.g., "the first option", "tell me more about that", "what about its PE ratio?")?
        - If YES, identify the topic from the history (e.g., "Financial Performance of x") and route to the appropriate agent (`fundamentals`, `technicals`, `sentiment`).

    **5. Specific Fact Check:**
        - Is the user asking for specific **shareholding details** (promoter, public, DII, FII)? Route to **`fundamentals`**.
        - Is the user asking for other specific **fundamental facts** (P/E, market cap, net sales, revenue)? Route to **`fundamentals`**.
        - Is the user asking for specific **technical facts** (RSI, MACD, moving averages)? Route to **`technicals`**.
        - Is the user asking about **news, public mood, or headlines**? Route to **`sentiment`**.

    **6. Broad Opinion Check:**
        - Is the user asking for a broad opinion (e.g., "Should I buy/sell/hold...?", "Is it a good buy?") without specifying a method? If YES, route to **`cross_agent_reasoning`**.

    **7. Vague Query Check:**
        - Is the query just a company name (e.g., "Reliance", "Infosys") with no other context? If YES, route to **`clarification`**.

    **8. Catch-Alls:**
        - Is it a simple greeting or closing? Route to **`general`**.
        - Is it about something other than finance? Route to **`out_of_domain`**.


    **--- EXAMPLES of correct routing ---**

    **Example 1: Answer to Clarification**
    - History: "assistant: Which company are you asking about?"
    - User's Current Raw Query: "HDFC Bank"
    - Your Output JSON: {{"route": "fundamentals", "reasoning": "User answered clarification. The original intent was to get a fundamental fact."}}

    **Example 2: Specific Standalone Question**
    - History: ""
    - User's Current Raw Query: "what are the shareholding patterns for Titan?"
    - Your Output JSON: {{"route": "fundamentals", "reasoning": "User is asking for a specific fundamental fact (shareholding)."}}

    **Example 3: Vague Initial Query**
    - History: ""
    - User's Current Raw Query: "Reliance"
    - Your Output JSON: {{"route": "clarification", "reasoning": "User provided only a company name with no specific question."}}

    **Example 4: Simple Greeting**
    - History: ""
    - User's Current Raw Query: "hello there"
    - Your Output JSON: {{"route": "general", "reasoning": "User provided a simple greeting."}}

    **--- OUTPUT INSTRUCTIONS ---**
    YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT AND NOTHING ELSE.
    DO NOT INCLUDE ANY OTHER TEXT, EXPLANATIONS, OR MARKDOWN FORMATTING LIKE ```json.
    JUST THE JSON OBJECT.

    ---
    Now, apply this thought process to the user's raw query and provide your output as a single, valid JSON object.
"""

REWRITE_QUESTION_PROMPT = """
    You are an expert at creating self-contained, standalone queries for another AI.
    Your task is to analyze the user's latest input and the recent conversation history, then rewrite the user's input into a complete query that can be understood without any of the previous context.

    **Strict Rules:**
    1.  **Resolve Context for Follow-ups:** ONLY if the user's input is a clear follow-up (e.g., "the first option", "what about them?", "tell me more"), use the history to resolve pronouns and references.
    2.  **DO NOT Add Context to New Topics:** If the user's input introduces a new, specific subject (e.g., "What are the top 5 companies by market cap?", "How is Tata Motors doing?"), you MUST assume it is a new, unrelated query. DO NOT inject context from the previous conversation into it.
    3.  **Preserve Intent:** If the original input is a question, the output must be a question. If it's a statement/command, the output must be a statement/command.
    4.  **Be Concise:** Your entire output must be ONLY the final, rewritten query. Do not add any extra text or explanations.
    5.  **No-Op:** If the user's input is already a complete, standalone query, return it EXACTLY AS IS.

    **--- Example of a Follow-up (Rule 1) ---**
    **Chat History:**
    assistant: To provide analysis of Reliance Industries, we need more info...
    - **Financial Performance:** Stock price, revenue, and profit margins.
    **User Input:**
    I wanna know the First option
    **Rewritten, Self-Contained User Input:**
    What is the financial performance of Reliance Industries?

    **--- Example of a New Topic (Rule 2) ---**
    **Chat History:**
    assistant: The analysis for Reliance Industries is complete.
    **User Input:**
    What are the top 5 companies by market capitalization?
    **Rewritten, Self-Contained User Input:**
    What are the top 5 companies by market capitalization?
    **--- End Examples ---**

    Now, perform this task on the following inputs.

    **Chat History:**
    {chat_history}

    **User Input:**
    {question}

    **Rewritten, Self-Contained User Input:**
"""

SYNTHESIS_PROMPT_TEMPLATE = """
    You are IRIS, a sharp and confident financial analyst AI. Your task is to synthesize reports into a single, direct, and easy-to-understand response.
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

LTM_RESPONSE_PROMPT = """
    You are IRIS. You have retrieved the user's saved preferences. Answer their question based on this information and the conversation history.
    **Conversation History:**
    {chat_history}
    **User's Saved Preferences:**
    {ltm_data}
    **User's Question:**
    {user_input}
    Answer the question naturally and directly.
"""

CLARIFICATION_PROMPT_TEMPLATE = """
    You are IRIS, a helpful financial assistant. A user has provided a vague query. Your goal is to ask a smart, clarifying question to better understand their intent. Offer them concrete options.
    **User's Vague Query:** "{user_input}"
    Generate a suitable clarifying question, starting directly.
"""

GENERAL_PROMPT_TEMPLATE = """
    You are IRIS, a friendly and professional financial assistant. The user has said something general (like a greeting or a simple question about you). Respond naturally and conversationally. Keep it brief.
    **Chat History (for context):**
    {chat_history}
    **User's Message:** "{user_input}"
    Your response:
"""

SESSION_SUMMARY_PROMPT_TEMPLATE = """
    You are an expert at summarizing conversation histories into a short, descriptive title for a UI sidebar.

    **Conversation History:**
    {chat_history}

    **Your Task:**
    Based on the full conversation, generate a single, concise title (max 5-7 words) that captures the main topic or key entity discussed. Your entire output should be ONLY the title itself.

    **Example 1:**
    - History: "User: hi, Assistant: hello, User: what is the P/E of Reliance?, Assistant: The P/E is 24.5"
    - Your Output: "Analysis of Reliance P/E Ratio"

    **Example 2:**
    - History: "User: my fav stock is INFY, Assistant: Got it."
    - Your Output: "User Preference: Favorite Stock"

    Now, generate the title for the provided history.
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

    def should_i_summarize(self, state: IrisState) -> str:
        """Checks if the conversation is at a point where it should be summarized."""
        # A "turn" consists of 2 messages (user + assistant)
        num_messages = len(state.get("full_chat_history", []))
        
        # Don't summarize if there's no history yet
        if num_messages < 2:
            return "end"

        num_turns = num_messages // 2
        
        SUMMARY_INTERVAL = 3 

        # Summarize on the 1st turn, then on the 4th, 7th, 10th etc.
        # This condition captures the initial summary and periodic updates.
        if num_turns > 0 and (num_turns == 1 or (num_turns - 1) % SUMMARY_INTERVAL == 0):
            print(f"--- ROUTING: Turn {num_turns}. Condition met. Routing to summarizer. ---")
            return "summarize"
        else:
            print(f"--- ROUTING: Turn {num_turns}. Skipping summarizer. ---")
            return "end"

    def _build_graph(self) -> StateGraph:
        graph_builder = StateGraph(IrisState)
        
        # --- Define all nodes (this part is unchanged) ---
        graph_builder.add_node("initialize_session", self.initialize_session_node)
        graph_builder.add_node("supervisor_decide", self.supervisor_decide_node)
        graph_builder.add_node("rewrite_for_agent", self.rewrite_for_agent_node)
        graph_builder.add_node(IrisRoute.SAVE_LTM.value, self.save_ltm_node)
        graph_builder.add_node(IrisRoute.LOAD_LTM.value, self.load_ltm_and_respond_node)
        graph_builder.add_node(IrisRoute.FUNDAMENTALS.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.RECOMMENDATION.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.TECHNICALS.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.SENTIMENT.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.CROSS_AGENT_REASONING.value, self.staggered_parallel_agent_call_node)
        graph_builder.add_node(IrisRoute.CLARIFICATION.value, self.clarification_node)
        graph_builder.add_node(IrisRoute.GENERAL.value, self.general_node)
        graph_builder.add_node(IrisRoute.OUT_OF_DOMAIN.value, self.out_of_domain_node)
        graph_builder.add_node("synthesize_results", self.synthesize_results_node)
        graph_builder.add_node("format_and_stream", self.format_and_stream_node)
        graph_builder.add_node("log_to_db_and_finalize", self.log_to_db_and_finalize_node)
        graph_builder.add_node("summarize_session", self.summarize_session_node)
        
        
        # Entry point and supervisor routing are the same
        graph_builder.set_entry_point("initialize_session")
        graph_builder.add_edge("initialize_session", "supervisor_decide")

        agent_routes_that_need_rewrite = {
            IrisRoute.FUNDAMENTALS, IrisRoute.TECHNICALS,
            IrisRoute.SENTIMENT, IrisRoute.CROSS_AGENT_REASONING,
            IrisRoute.RECOMMENDATION
        }

        def route_from_supervisor(state: IrisState):
            route = state["supervisor_decision"].route
            if route in agent_routes_that_need_rewrite:
                return "rewrite_for_agent"
            return route.value
            
        graph_builder.add_conditional_edges("supervisor_decide", route_from_supervisor, {
            "rewrite_for_agent": "rewrite_for_agent",
            IrisRoute.SAVE_LTM.value: IrisRoute.SAVE_LTM.value,
            IrisRoute.LOAD_LTM.value: IrisRoute.LOAD_LTM.value,
            IrisRoute.CLARIFICATION.value: IrisRoute.CLARIFICATION.value,
            IrisRoute.GENERAL.value: IrisRoute.GENERAL.value,
            IrisRoute.OUT_OF_DOMAIN.value: IrisRoute.OUT_OF_DOMAIN.value,
        })

        def route_after_rewrite(state: IrisState):
            return state["supervisor_decision"].route.value

        graph_builder.add_conditional_edges("rewrite_for_agent", route_after_rewrite,
            {route.value: route.value for route in agent_routes_that_need_rewrite})
        
        # --- MODIFIED: More intelligent decision logic for formatting ---
        
        # Define which routes produce simple text that should NOT be formatted
        routes_that_bypass_formatter = {
            IrisRoute.GENERAL.value,
            IrisRoute.CLARIFICATION.value,
            IrisRoute.OUT_OF_DOMAIN.value,
            IrisRoute.SAVE_LTM.value,
            IrisRoute.LOAD_LTM.value,
        }

        def decide_on_formatting(state: IrisState) -> str:
            """
            Checks the supervisor's original routing decision and response type
            to determine if heavy markdown formatting is needed.
            """
            route = state["supervisor_decision"].route
            final_response = state.get("final_response")
            
            # Case 1: The response is a structured dict (e.g., for a chart). Skip formatting.
            if isinstance(final_response, dict):
                print(f"--- ROUTING: Structured response found. Bypassing formatter. ---")
                return "log_to_db_and_finalize"

            # Case 2: The original intent was for a simple, conversational response. Skip formatting.
            if route.value in routes_that_bypass_formatter:
                print(f"--- ROUTING: Bypassing formatter for simple route '{route.value}'. ---")
                return "log_to_db_and_finalize"

            # Case 3: It's a complex response from an analysis agent. Send to formatter.
            else:
                print(f"--- ROUTING: Sending to formatter for complex route '{route.value}'. ---")
                return "format_and_stream"

        # Define all nodes that produce a final response that needs a formatting decision
        nodes_before_decision = [
            IrisRoute.FUNDAMENTALS.value, IrisRoute.TECHNICALS.value, IrisRoute.SENTIMENT.value,
            IrisRoute.SAVE_LTM.value, IrisRoute.LOAD_LTM.value, IrisRoute.CLARIFICATION.value,
            IrisRoute.GENERAL.value, IrisRoute.OUT_OF_DOMAIN.value, "synthesize_results"
        ]

        # Route all these nodes to our new, smarter decision point
        for node_name in nodes_before_decision:
            graph_builder.add_conditional_edges(
                node_name,
                decide_on_formatting,
                {
                    "format_and_stream": "format_and_stream",
                    "log_to_db_and_finalize": "log_to_db_and_finalize"
                }
            )

        # Cross-agent reasoning still goes to synthesis first
        graph_builder.add_edge(IrisRoute.CROSS_AGENT_REASONING.value, "synthesize_results")
        
        # The output of the formatter ALWAYS goes to the logger
        graph_builder.add_edge("format_and_stream", "log_to_db_and_finalize")

        # --- MODIFIED FINAL EDGES FOR SUMMARIZATION ---
        # The logging node now routes to our conditional check
        graph_builder.add_conditional_edges(
            "log_to_db_and_finalize",
            self.should_i_summarize,
            {
                "summarize": "summarize_session",
                "end": END,
            }
        )
        
        # The final step is always logging, which then ends the graph
        graph_builder.add_edge("summarize_session", END)

        return graph_builder

    async def initialize_session_node(self, state: IrisState) -> Dict:
        """
        This node is now robust. It ensures user and session IDs are present
        in the state on EVERY turn, fetching them from the DB if necessary.
        This is the definitive fix for the state loss issue.
        """
        print("\n---NODE: Initialize or Verify Session---")
        thread_id = state["thread_id"]
        user_identifier = state["user_identifier"]

        # On every turn, try to fetch existing session details from DB using thread_id
        existing_ids = await asyncio.to_thread(db_ltm.get_session_and_user_ids_by_thread, thread_id)

        if existing_ids:
            # Session already exists in the database.
            db_session_id, db_user_id = existing_ids
            print(f"--- Session verified. UserID: {db_user_id}, SessionID: {db_session_id} ---")
            
            # We return the IDs to ensure they are correctly set in the state for this turn,
            # overwriting any potentially corrupted (None) values from a previous turn's state.
            # We do NOT touch full_chat_history, allowing the accumulator to work.
            return {
                "db_user_id": db_user_id,
                "db_session_id": db_session_id,
            }
        else:
            # This is the very first turn for this thread_id.
            # Create the user and session records.
            print(f"--- First turn for thread {thread_id}. Creating new session in DB. ---")
            db_user_id = await asyncio.to_thread(db_ltm.get_or_create_user, user_identifier)
            db_session_id = await asyncio.to_thread(db_ltm.get_or_create_session, db_user_id, thread_id)
            
            print(f"--- New session created. UserID: {db_user_id}, SessionID: {db_session_id} ---")

            # On the first turn, we must initialize the history list
            # and populate the state with the newly created IDs.
            return {
                "db_user_id": db_user_id,
                "db_session_id": db_session_id,
                "full_chat_history": []
            }

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
            traceback.print_exc()
            return {"supervisor_decision": SupervisorDecision(route=IrisRoute.CLARIFICATION, reasoning=f"Supervisor Error: {e}")}

    async def rewrite_for_agent_node(self, state: IrisState) -> Dict:
        print("---NODE: Rewrite and Enrich Question for Agent---")
        user_input = state["user_input"]
        history = state.get('full_chat_history', [])

        if not history:
            print("No history, not rewriting.")
            return {} 

        history_str = "\n".join([f"{m.type}: {m.content}" for m in history[-6:]])
        
        # --- Part 1: Standard Rewrite for context and pronouns ---
        rewrite_prompt = REWRITE_QUESTION_PROMPT.format(chat_history=history_str, question=user_input)
        rewrite_response = await groq_llm_fast.ainvoke(rewrite_prompt)
        rewritten_str = rewrite_response.content.strip().strip('"')
        print(f"Initial rewrite: '{rewritten_str}'")

        # --- Part 2: Intelligent Enrichment with LTM data ---
        # Check if the query implies using saved preferences
        enrichment_check_prompt = f"""
            Analyze the following user query and conversation history.
            Does the query ask to APPLY or USE a previously mentioned preference (like 'favorite indicators', 'my style', 'based on them')?
            Answer with a single word: YES or NO.

            History: {history_str}
            Query: {user_input}
        """
        check_response = await groq_llm_fast.ainvoke(enrichment_check_prompt)
        needs_enrichment = "yes" in check_response.content.lower()

        if needs_enrichment:
            print("Query implies use of LTM. Fetching preferences for enrichment...")
            try:
                # Fetch the preferences from the database
                raw_ltm_data = await asyncio.to_thread(db_ltm.get_user_preferences, state["db_user_id"])
                if raw_ltm_data:
                    # Parse and format the preferences for injection
                    parsed_prefs = {}
                    for key, value in raw_ltm_data.items():
                        try:
                            parsed_prefs[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            parsed_prefs[key] = value
                    
                    # Create a clear, readable string of preferences
                    pref_str_parts = []
                    if 'fav_indicator' in parsed_prefs:
                        indicators = parsed_prefs['fav_indicator']
                        if isinstance(indicators, list):
                            pref_str_parts.append(f"indicators: {', '.join(indicators)}")
                        else:
                            pref_str_parts.append(f"indicator: {indicators}")
                    
                    if 'investment_style' in parsed_prefs:
                        pref_str_parts.append(f"investment style: {parsed_prefs['investment_style']}")

                    if pref_str_parts:
                        # Construct the final, enriched query
                        final_pref_str = " and ".join(pref_str_parts)
                        enriched_question = f"{rewritten_str} (using my preferences: {final_pref_str})"
                        print(f"Original question: '{user_input}'")
                        print(f"Enriched question for agent: '{enriched_question}'")
                        return {"user_input": enriched_question}

            except Exception as e:
                print(f"--- WARNING: Failed to enrich query with LTM data: {e} ---")
                # Fallback to the simply rewritten question if enrichment fails
        
        # If no enrichment was needed or it failed, use the standard rewritten string
        if rewritten_str.lower() != user_input.lower():
            print(f"Original question: '{user_input}'")
            print(f"Final (rewritten only) question for agent: '{rewritten_str}'")
            return {"user_input": rewritten_str}
        else:
            print("Question is already self-contained, no changes made.")
            return {}

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
        standalone_question = state["user_input"]
        
        route = state["supervisor_decision"].route

        if route == IrisRoute.SENTIMENT:
            payload = {"query": standalone_question}
        else:
            payload = {"question": standalone_question}
        
        agent_map = {
            IrisRoute.FUNDAMENTALS: fundamentals_app_instance,
            IrisRoute.RECOMMENDATION: fundamentals_app_instance,
            IrisRoute.SENTIMENT: sentiment_app_instance,
            IrisRoute.TECHNICALS: technical_app_instance
        }
        try:
            # Sub-agents now receive a payload with the correct key
            print(f"--- Calling {route.name} agent with payload: {payload} ---")
            result_state = await agent_map[route].ainvoke(payload)
            
            # --- MODIFIED: Pass the entire final_answer object (str OR dict) ---
            response = result_state.get("final_answer")
            
            if response:
                return {"final_response": response}
            else:
                return {"final_response": f"The {route.name.title()} agent could not find an answer."}

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

    async def format_and_stream_node(self, state: IrisState):
        """
        This is the final gatekeeper node. It takes the generated response,
        formats it for the end-user, and streams it back, while also ensuring
        the final, complete response is passed on in the state.
        """
        print("---NODE: Format and Stream Final Answer---")
        
        response_from_agent = state.get("final_response", "I'm sorry, I encountered an issue.")

        simple_responses = [
            "Got it, I've saved that for you.",
            "I am IRIS, an AI financial analyst. I can only answer questions related to the stock market and financial data."
        ]
        if response_from_agent in simple_responses or "Sorry, I had trouble" in response_from_agent:
             return {"final_response": response_from_agent}
        
        formatting_prompt = PromptTemplate.from_template(
        """
            You are IRIS, a financial assistant AI. Your final and most important job is to take an internal data summary and format it into a final, user-facing response. The response must be perfectly formatted as markdown AND be a direct, conversational answer to the user's original question.

            **User's Original Question:**
            ---
            {user_question}
            ---

            **Internal Analysis Summary (raw data):**
            ---
            {raw_answer}
            ---

            **--- Part 1: Your Thought Process (for content) ---**
            1.  **Analyze Intent:** First, I will read the User's Original Question to understand their specific goal (e.g., "what is..?", "should I buy..?", "is it strong?").
            2.  **Extract Key Data:** I will read the Internal Analysis Summary to find the core data points and verdicts.
            3.  **Synthesize Answer:** I will craft a direct, conversational answer that uses the data to address the user's specific intent. If they asked "Should I buy?", my answer will address the "buy" decision. I will not say "Hold" to a user who doesn't own the stock. I will be concise and use simple language.
            4.  **Layman Language:** Keep in mind that user might not be very technical or stock market expert, so answer in simple language.

            **--- Part 2: Your Formatting Rules (for appearance) ---**
            -   **No Fluff:** My response MUST NOT begin with any conversational filler like "Here is the response:".
            -   **Main Title:** I will begin with a Level 2 Markdown Heading (`##`) that includes the company name if applicable.
            -   **Sections:** I will use Level 3 Markdown Headings (`###`) for different analysis types (e.g., Fundamental, Technical).
            -   **Lists:** I will use a hyphen (`- `) for all bullet points.
            -   **Emphasis:** I will use bold markdown (`**text**`) for key terms, data points, and final verdicts like **Buy**, **Sell**, or **Hold**.
            -   **Structure:** The entire output must be valid GitHub-flavored Markdown.
            -   **Sentiment: Instead of using technical phrases such as bullish or bearish, explain it in simpler words

            **--- Example of a PERFECT Final Output ---**
            *If the user asked "Is xyz strong?" and the internal summary was "Fundamental: Hold. Technical: Buy. Sentiment: Neutral. Overall: Buy."*

            ```markdown
            ## xyz Analysis
            ### Summary
            Overall, the analysis suggests that **Reliance Industries** is looking quite **strong**.

            ### Detailed Breakdown
            - **Fundamental Analysis:** The data suggests a **Hold**, indicating solid but potentially fully-priced core metrics.
            - **Technical Analysis:** The chart indicators are giving a clear **Buy** signal, showing positive momentum.
            - **Sentiment Analysis:** Market sentiment is currently **Neutral**.

            Now, applying BOTH your thought process and formatting rules, transform the internal summary into the final, perfect, user-facing markdown response.
            """
        )

        user_question = state.get("user_input", "the user's request.")

        formatter_chain = formatting_prompt | groq_llm

        formatted_response = await formatter_chain.ainvoke(
            {"raw_answer": response_from_agent, "user_question": user_question},
            config={"run_name": "final_user_output_formatter"}
        )

       

        # 3. After the stream is complete, return the final, accumulated
        #    response. This correctly updates the state for the next node.
        return {"final_response": formatted_response.content}

    async def log_to_db_and_finalize_node(self, state: IrisState) -> Dict:
            """
            Logs the conversation to the database and prepares the final state.
            This node is now correctly implemented to avoid wiping state keys.
            """
            print("---NODE: Log to DB and Finalize---")
            user_input = state.get("original_user_input", state["user_input"])
            final_response_obj = state.get("final_response") or "I'm sorry, I cannot respond."

            # Handle both string and dict responses for logging
            if isinstance(final_response_obj, dict):
                assistant_response_for_log = final_response_obj.get("text_response", "Structured response generated.")
            else:
                assistant_response_for_log = str(final_response_obj)

            # The DB logging calls are correct. The issue is what this node returns.
            session_id = state.get("db_session_id")
            if not session_id:
                print("--- CRITICAL ERROR: db_session_id is missing in log_to_db_and_finalize_node. ---")
                # You might want to handle this more gracefully, but for now, we'll log it.
                # This check helps confirm the fix.
                return {"final_response": "A critical session error occurred. Please start a new chat."}

            await asyncio.to_thread(db_ltm.log_chat_message, session_id, "user", user_input)
            await asyncio.to_thread(db_ltm.log_chat_message, session_id, "assistant", assistant_response_for_log)

            # --- FIX ---
            # 1. Use the `Annotated` accumulator correctly by only returning the NEW messages.
            #    LangGraph will automatically append them to the existing `full_chat_history`.
            new_messages = [HumanMessage(content=user_input), AIMessage(content=assistant_response_for_log)]

            print(f"Final Response Object being sent to API:\n{final_response_obj}")

            # 2. Return a dictionary of ONLY the keys you want to update.
            #    LangGraph preserves all other keys (like db_session_id) in the state.
            return {
                "final_response": final_response_obj,
                "full_chat_history": new_messages, # Let the accumulator do the append
                "original_user_input": None        # Clear the original input for the next turn
            }

    async def staggered_parallel_agent_call_node(self, state: IrisState) -> Dict:
        # Rewrite the question ONCE (this part is correct)
        standalone_question = state["user_input"]

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

    async def summarize_session_node(self, state: IrisState) -> Dict:
        """
        As a final, non-blocking step, summarizes the conversation and saves
        it to the database for display in the UI.
        """
        print("---NODE: Summarize Session---")
        try:
            # Prepare the prompt using the *complete* chat history
            history_str = "\n".join([f"{m.type}: {m.content}" for m in state['full_chat_history']])
            prompt = SESSION_SUMMARY_PROMPT_TEMPLATE.format(chat_history=history_str)

            # Get summary from a fast LLM
            response = await groq_llm_fast.ainvoke(prompt)
            summary_text = response.content.strip().strip('"')

            # Update the database in the background
            await asyncio.to_thread(
                update_session_summary,
                state["db_session_id"],
                summary_text
            )
            print(f"Updated session summary to: '{summary_text}'")
        except Exception as e:
            # We don't want to crash the whole flow if summarization fails
            print(f"--- ERROR in summarize_session_node: {e} ---")
            traceback.print_exc()
        
        # This node doesn't modify the state, it's a "fire-and-forget" side effect
        return {}

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
                "hi",
                "hello",
                "my fav stock is Titan Company and i mostly use RSI and EMA indicators",
                "What is my fav stock and which indicators do i prefer to use?",
                "Reliance Industries",
                "I wanna know the First option",
                "What are the top 5 companies by market capitalization?",
                "what is the market cap of ABB India",
                "What is the latest market capitalization and net sales of Reliance Industries?"
         
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

            # --- Graph Visualization Logic ---
            try:
                print("\nAttempting to generate IRIS graph visualization (graphs/iris_v1.png)...")
                
                # Ensure the 'graphs' directory exists
                if not os.path.exists("graphs"):
                    os.makedirs("graphs")
                    
                # Get the graph from the compiled app instance
                img_data = orchestrator.app.get_graph().draw_mermaid_png()
                
                with open("graphs/iris_v1.png", "wb") as f:
                    f.write(img_data)
                print("IRIS graph saved to graphs/iris_v1.png")
            except Exception as e:
                print(f"Could not generate IRIS graph visualization: {e}")
            

    asyncio.run(run_test_session())


