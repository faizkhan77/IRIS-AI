
import os
import json
import uuid
import asyncio
import traceback
from enum import Enum
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional, List

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

from .db_ltm import find_matching_companies
import re

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
    PERFORMANCE_ANALYSIS = "performance_analysis" 

# --- Pydantic & State Definitions ---
class SupervisorDecision(BaseModel):
    route: IrisRoute
    reasoning: str
    time_period_years: Optional[int] = Field(None, description="The time period in years, if mentioned (e.g., '5 years').")
    sector: Optional[str] = Field(None, description="The industry sector, if mentioned (e.g., 'Minerals sector').")

class LtmSaveRequest(RootModel[Dict[str, Any]]):
    pass

class IrisState(TypedDict):
    user_identifier: str
    thread_id: str
    db_user_id: Optional[int]
    db_session_id: Optional[int]
    user_input: str
    final_response: Optional[Any]
    full_chat_history: Annotated[TypingList[BaseMessage], lambda x, y: x + y]
    supervisor_decision: Optional[SupervisorDecision]
    sub_agent_outputs: Annotated[dict, lambda x, y: {**x, **y}]

class ClarificationOption(BaseModel):
    label: str = Field(description="The user-facing text for the button/option. Should be descriptive and may include emojis (e.g., '📊 Get Fundamental Analysis').")
    query: str = Field(description="The self-contained query that will be sent back to the agent if the user clicks this option (e.g., 'Fundamental analysis of Reliance Industries').")

# THIS IS THE NEW, CORRECT MODEL
class LLMSuggestions(BaseModel):
    """A model to represent the list of clarification options the LLM should generate."""
    items: List[ClarificationOption] = Field(description="A list of suggested clarification options.")

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


# In agents/iris.py

SUPERVISOR_PROMPT_TEMPLATE = """
    You are an AI routing supervisor. Your sole purpose is to analyze the user's query and conversation history and output a single, valid JSON object to route the query to the correct tool.

    **Conversation History (for context):**
    {chat_history}
    - **User's Current Raw Query:** "{user_input}"

    **--- Tool Routing Logic ---**

    Based on the user's query, determine the most appropriate tool from the following options. Your response MUST be a single JSON object.

    1.  **`performance_analysis`**:
        - Use for queries asking for a **LIST or RANKING** of stocks based on **HISTORICAL PERFORMANCE** over a **TIME PERIOD**.
        - Keywords: "top performing", "most consistent", "best stocks over [time]".
        - Must extract `time_period_years` and `sector`.

    2.  **`recommendation`**:
        - Use for queries asking for a **general recommendation** based on an investment style or theme.
        - Keywords: "recommend", "suggest", "good stocks for".

    3.  **`fundamentals`**:
        - Use for queries asking for **specific, current fundamental data points** for one or more companies.
        - Also use for **simple factual rankings** based on a single metric.
        - Keywords: "market cap", "P/E ratio", "shareholding", "net sales", "top 5 by [metric]".

    4.  **`technicals`**:
        - Use for queries asking for **specific technical indicators** or the **historical technical performance of a SINGLE company**.
        - Keywords: "RSI", "MACD", "technical analysis of [company]", "performance of [company]".

    5.  **`sentiment`**:
        - Use for queries about **news, headlines, or public mood**.
        - Keywords: "sentiment", "news about", "market mood".

    6.  **`cross_agent_reasoning`**:
        - Use for broad, open-ended questions like **"should I buy/sell/hold [stock]?"** that require a synthesized opinion from multiple analysis types.

    7.  Long-Term Memory (LTM) Check:**
        - Is the user **TELLING ME** a new preference to remember? (e.g., "My favorite stock is...", "I prefer...", 'I am a xyz', 'I like xyz', 'I used to xyz..', etc). If YES, route to **`save_ltm`**.
        - Is the user's primary goal to **RECALL** a saved preference? (e.g., "What is my favorite stock?", "Remind me which indicators I like?"). If YES, route to **`load_ltm`**.
        - **IMPORTANT:** If the user wants to **APPLY** a known preference to a new task (e.g., "Analyze xyz based on my favorite indicators"), DO NOT route to `load_ltm`.

    8.  **`clarification`**:
        - Use ONLY when the query is extremely vague and contains just a company name with no other context (e.g., "Reliance").

    9.  **`general`**:
        - Use for simple greetings, closings, or conversational filler (e.g., "hello", "thanks", "how are you?").

    10. **`out_of_domain`**:
        - Use for any query not related to finance or the stock market.

    **--- Contextual Follow-up Rule ---**
    - If the user's query is a follow-up (e.g., "what about its P/E ratio?", "tell me more"), use the conversation history to understand the original topic (e.g., "Reliance Industries") and route to the appropriate tool (`fundamentals`, `technicals`, etc.).


    **--- OUTPUT INSTRUCTIONS ---**
    - Your entire response MUST be ONLY the single, valid JSON object.
    - Do NOT include any other text, explanations, or markdown formatting.
    - JUST THE JSON.

    ---
    Now, apply this logic to the user's query and provide your output.
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


SUPERVISOR_FALLBACK_PROMPT_TEMPLATE = """
    You are an expert routing assistant. Your task is to analyze the user's query and the conversation history, then output the single most appropriate tool name from the provided list.

    **--- Tool Routing Logic ---**

    1.  **`clarification`**: Use this tool if and ONLY IF the user's query is just a company name without any other context (e.g., "reliance", "adani", "apple"). This is your HIGHEST PRIORITY. If a query is just a name, you MUST choose 'clarification'.

    2.  **`fundamentals`**: Use for specific data points like "market cap of reliance".
    3.  **`technicals`**: Use for technical indicators like "rsi of reliance".
    4.  **`sentiment`**: Use for news or mood, like "sentiment for reliance".
    5.  **`cross_agent_reasoning`**: Use for "should I buy reliance?".
    
    (Other tools: {other_route_names})

    **--- Conversation History (for context) ---**
    {chat_history}

    **--- User's Current Query ---**
    "{user_input}"

    **--- INSTRUCTIONS ---**
    -   Your entire output MUST BE only the single, most appropriate tool name from the list.
    -   Do not add any explanation or other text.
    -   Just the single word.
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
        self.llm = groq_llm
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
        graph_builder.add_node("pre_routing_classifier", self.pre_routing_classifier_node)
        graph_builder.add_node("supervisor_decide", self.supervisor_decide_node)
        graph_builder.add_node("rewrite_for_agent", self.rewrite_for_agent_node)
        graph_builder.add_node(IrisRoute.SAVE_LTM.value, self.save_ltm_node)
        graph_builder.add_node(IrisRoute.LOAD_LTM.value, self.load_ltm_and_respond_node)
        graph_builder.add_node(IrisRoute.FUNDAMENTALS.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.RECOMMENDATION.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.PERFORMANCE_ANALYSIS.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.TECHNICALS.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.SENTIMENT.value, self.call_agent_node)
        graph_builder.add_node(IrisRoute.CROSS_AGENT_REASONING.value, self.staggered_parallel_agent_call_node)
        graph_builder.add_node(IrisRoute.CLARIFICATION.value, self.clarification_node)
        graph_builder.add_node(IrisRoute.GENERAL.value, self.general_node)
        graph_builder.add_node(IrisRoute.OUT_OF_DOMAIN.value, self.out_of_domain_node)
        graph_builder.add_node("synthesize_results", self.synthesize_results_node)
        graph_builder.add_node("final_response_validator", self.final_response_validator_node)
        graph_builder.add_node("format_and_stream", self.format_and_stream_node)
        graph_builder.add_node("log_to_db_and_finalize", self.log_to_db_and_finalize_node)
        graph_builder.add_node("summarize_session", self.summarize_session_node)
        
        
        graph_builder.set_entry_point("initialize_session")
        graph_builder.add_edge("initialize_session", "pre_routing_classifier")

        graph_builder.add_conditional_edges(
            "pre_routing_classifier",
            lambda state: (
                state["supervisor_decision"].route.value
                if state.get("supervisor_decision") and hasattr(state["supervisor_decision"], "route")
                else "supervisor_decide"
            ),
            {
                IrisRoute.FUNDAMENTALS.value: IrisRoute.FUNDAMENTALS.value,
                IrisRoute.SENTIMENT.value: IrisRoute.SENTIMENT.value,
                IrisRoute.TECHNICALS.value: IrisRoute.TECHNICALS.value,
                IrisRoute.CROSS_AGENT_REASONING.value: IrisRoute.CROSS_AGENT_REASONING.value,
                IrisRoute.PERFORMANCE_ANALYSIS.value: IrisRoute.PERFORMANCE_ANALYSIS.value,
                IrisRoute.RECOMMENDATION.value: IrisRoute.RECOMMENDATION.value,
                IrisRoute.GENERAL.value: IrisRoute.GENERAL.value,
                IrisRoute.OUT_OF_DOMAIN.value: IrisRoute.OUT_OF_DOMAIN.value,
                IrisRoute.SAVE_LTM.value: IrisRoute.SAVE_LTM.value,         # Optional but good to include
                IrisRoute.LOAD_LTM.value: IrisRoute.LOAD_LTM.value,         # Optional but good to include
                IrisRoute.CLARIFICATION.value: IrisRoute.CLARIFICATION.value,
                "supervisor_decide": "supervisor_decide",  # fallback
            }
        )


        agent_routes_that_need_rewrite = {
            IrisRoute.FUNDAMENTALS, IrisRoute.TECHNICALS,
            IrisRoute.SENTIMENT, IrisRoute.CROSS_AGENT_REASONING,
            IrisRoute.RECOMMENDATION, IrisRoute.PERFORMANCE_ANALYSIS,
        }

        def route_from_supervisor(state: IrisState):
            route = state["supervisor_decision"].route
            if route in agent_routes_that_need_rewrite:
                return "rewrite_for_agent"
            return route.value
            
        graph_builder.add_conditional_edges("supervisor_decide", route_from_supervisor, {
            "rewrite_for_agent": "rewrite_for_agent",
            IrisRoute.SAVE_LTM.value: IrisRoute.SAVE_LTM.value,
            IrisRoute.CLARIFICATION.value: IrisRoute.CLARIFICATION.value,  # <--- ADD THIS LINE
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
            IrisRoute.SAVE_LTM.value, IrisRoute.LOAD_LTM.value,
            IrisRoute.CLARIFICATION.value, IrisRoute.GENERAL.value,
            IrisRoute.OUT_OF_DOMAIN.value
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

        # These nodes now go to validator first (if applicable)
        nodes_before_validation = [
            IrisRoute.FUNDAMENTALS.value, IrisRoute.TECHNICALS.value, IrisRoute.SENTIMENT.value,
            IrisRoute.RECOMMENDATION.value, IrisRoute.PERFORMANCE_ANALYSIS.value
        ]

        for node_name in nodes_before_validation:
            graph_builder.add_edge(node_name, "final_response_validator")

        # After validation, apply formatting logic
        graph_builder.add_conditional_edges(
            "final_response_validator",
            decide_on_formatting,
            {
                "format_and_stream": "format_and_stream",
                "log_to_db_and_finalize": "log_to_db_and_finalize",
            }
        )

        # Cross-agent reasoning still goes to synthesis first
        graph_builder.add_edge(IrisRoute.CROSS_AGENT_REASONING.value, "synthesize_results")
        graph_builder.add_edge("synthesize_results", "final_response_validator")

        
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
            in the state on EVERY turn, fetching them from the DB if necessary,
            AND it clears the supervisor decision from the previous turn.
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
                
                # --- THIS IS THE FIX ---
                # On every turn, reset the previous decision and ensure IDs are set.
                return {
                    "db_user_id": db_user_id,
                    "db_session_id": db_session_id,
                    "supervisor_decision": None # Explicitly clear the decision
                }
            else:
                # This is the very first turn for this thread_id.
                print(f"--- First turn for thread {thread_id}. Creating new session in DB. ---")
                db_user_id = await asyncio.to_thread(db_ltm.get_or_create_user, user_identifier)
                db_session_id = await asyncio.to_thread(db_ltm.get_or_create_session, db_user_id, thread_id)
                
                print(f"--- New session created. UserID: {db_user_id}, SessionID: {db_session_id} ---")

                # --- ALSO APPLY THE FIX HERE ---
                # Initialize state for the very first turn.
                return {
                    "db_user_id": db_user_id,
                    "db_session_id": db_session_id,
                    "full_chat_history": [],
                    "supervisor_decision": None # Explicitly initialize as None
                }

    async def pre_routing_classifier_node(self, state: IrisState) -> Dict:
        print("---NODE: Pre-Routing Classifier (Keyword-Based)---")
        user_input = state["user_input"].lower()

        # Rule-based keyword mapping in order of PRIORITY
        # Higher priority items should come first in this dictionary.
        ROUTE_KEYWORDS = {
            # --- HIGHEST PRIORITY: User is talking ABOUT their preferences ---
            IrisRoute.SAVE_LTM: [
                "my favorite", "i prefer", "i like", "remember this", "save my", 
                "i am a", "save my preferences", "i mostly use", "i use rsi", 
                "i mostly invest in"
            ],
            IrisRoute.LOAD_LTM: [
                "what is my", "remind me", "what did i say", "recall", 
                "do you remember", "what did i ask"
            ],

            # --- SECOND PRIORITY: High-confidence analytical commands ---
            IrisRoute.TECHNICALS: [
                "technical analysis", "rsi", "macd", "bollinger", "ema", "sma", "supertrend"
            ],
            IrisRoute.SENTIMENT: ["sentiment", "mood", "news", "headlines"],
            IrisRoute.RECOMMENDATION: [
                "what should i invest in", "stock recommendation", "suggest good stocks",
                "any stocks to buy", "top stocks to invest", "recommend", "suggest"
            ],
            IrisRoute.PERFORMANCE_ANALYSIS: [
                "top performing", "most consistent", "over the years", "best stocks in"
            ],
            IrisRoute.CROSS_AGENT_REASONING: [
                "should i buy", "should i sell", "hold or sell", "is it a good time",
                "good buy", "worth buying", "is it overpriced"
            ],
            
            # --- THIRD PRIORITY: Fundamental is often a fallback for specific data queries ---
            IrisRoute.FUNDAMENTALS: [
                "market cap", "net sales", "p/e", "pe ratio", "shareholding", 
                "profit", "valuation", "fundamental analysis"
            ],
            
            # --- LOWEST PRIORITY: Conversational / Out of Domain ---
            IrisRoute.GENERAL: ["hi", "hello", "thanks", "who are you", "how are you"],
            IrisRoute.OUT_OF_DOMAIN: ["weather", "movie", "football", "travel", "non finance"],
        }

        for route, keywords in ROUTE_KEYWORDS.items():
            # A simple check for any keyword in the input
            if any(kw in user_input for kw in keywords):
                # --- CRITICAL FIX: Ensure LTM intent isn't overridden by content keywords ---
                # If we match a technical/fundamental keyword, we must double-check if the user
                # was actually trying to SAVE a preference.
                if route in [IrisRoute.TECHNICALS, IrisRoute.FUNDAMENTALS]:
                    save_intent_present = any(kw in user_input for kw in ROUTE_KEYWORDS[IrisRoute.SAVE_LTM])
                    if save_intent_present:
                        print(f"⚠️ Keyword overlap detected. Prioritizing SAVE_LTM over {route.value}.")
                        # Skip this rule and let the loop find the SAVE_LTM rule later.
                        # Since SAVE_LTM is first, this will now work correctly.
                        # This logic is a safety net in case the dict order isn't guaranteed.
                        continue
                
                print(f"✅ High-confidence match: Routing to {route.value}")
                return {"supervisor_decision": SupervisorDecision(route=route, reasoning=f"Keyword match for: {route.value}")}

        print("⚠️ No high-confidence match. Falling back to LLM-based routing.")
        return {}

    async def supervisor_decide_node(self, state: IrisState) -> Dict:
        print("---NODE: Supervisor Decide (with Rule-Based Override)---")
        user_input = state["user_input"]
        history_str = "\n".join([f"{m.type}: {m.content}" for m in state.get('full_chat_history', [])[-6:]])
        
        # --- STAGE 1: Rule-Based Keyword Override (Guardrail) ---
        # This provides deterministic routing for high-confidence queries.
        
        # Technical keywords are very specific and reliable.
        technical_keywords = ["technical analysis", "rsi", "macd", "bollinger", "ema", "supertrend", "indicator"]
        if any(kw in user_input.lower() for kw in technical_keywords):
            print("--- Supervisor Override: Technical keyword detected. Forcing TECHNICALS route. ---")
            return {"supervisor_decision": SupervisorDecision(
                route=IrisRoute.TECHNICALS,
                reasoning="Forced route due to presence of a specific technical keyword."
            )}
            
        # Sentiment keywords
        sentiment_keywords = ["sentiment", "news", "mood", "headlines"]
        if any(kw in user_input.lower() for kw in sentiment_keywords):
            print("--- Supervisor Override: Sentiment keyword detected. Forcing SENTIMENT route. ---")
            return {"supervisor_decision": SupervisorDecision(
                route=IrisRoute.SENTIMENT,
                reasoning="Forced route due to presence of a sentiment-related keyword."
            )}

        # --- STAGE 2: LLM-Based Routing (for everything else) ---
        # This part remains the same as our previous best version.
        
        prompt = SUPERVISOR_PROMPT_TEMPLATE.format(chat_history=history_str, user_input=user_input)
        try:
            # Try structured call first
            structured_llm = groq_llm.with_structured_output(SupervisorDecision)
            decision = await structured_llm.ainvoke(prompt)
            print(f"Supervisor Decision (Structured Success): Route='{decision.route.value}', Reason='{decision.reasoning}'")
            return {"supervisor_decision": decision}
        except Exception as e:
            print(f"--- SUPERVISOR structured call failed: {e}. Attempting smart fallback. ---")
            try:
                # Fallback to a smarter, non-structured call
                other_routes = [r.value for r in IrisRoute if r.value not in ["clarification", "fundamentals", "technicals", "sentiment", "cross_agent_reasoning"]]
                fallback_prompt = SUPERVISOR_FALLBACK_PROMPT_TEMPLATE.format(
                    other_route_names=", ".join(other_routes),
                    chat_history=history_str,
                    user_input=user_input
                )
                fallback_response = await groq_llm_fast.ainvoke(fallback_prompt)
                chosen_route_str = fallback_response.content.strip().lower().replace('"', '').replace('`', '')
                chosen_route = IrisRoute(chosen_route_str)
                print(f"--- Supervisor Decision (Fallback Success): Chose route: '{chosen_route.value}' ---")
                return {"supervisor_decision": SupervisorDecision(
                    route=chosen_route,
                    reasoning=f"Smart fallback from structured output failure. LLM selected '{chosen_route_str}'."
                )}
            except (ValueError, Exception) as fallback_e:
                # Ultimate safety net
                print(f"--- SUPERVISOR fallback also failed: {fallback_e}. Defaulting to CLARIFICATION. ---")
                traceback.print_exc()
                return {"supervisor_decision": SupervisorDecision(
                    route=IrisRoute.CLARIFICATION,
                    reasoning="Ultimate fallback: Both structured and simple LLM routing failed."
                )}

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
        standalone_question = state["user_input"]
        route = state["supervisor_decision"].route
        
        agent_map = {
            IrisRoute.FUNDAMENTALS: fundamentals_app_instance,
            IrisRoute.RECOMMENDATION: fundamentals_app_instance,
            IrisRoute.PERFORMANCE_ANALYSIS: fundamentals_app_instance,
            IrisRoute.SENTIMENT: sentiment_app_instance,
            IrisRoute.TECHNICALS: technical_app_instance
        }
        
        # --- FIX: Re-introduce the supervisor_decision into the payload ---
        # The payload MUST contain all the keys expected by the sub-agent's entry point.
        payload = {
            "question": standalone_question,
            "supervisor_decision": state["supervisor_decision"].model_dump() 
        }

        # Override for sentiment agent's specific input key
        if route == IrisRoute.SENTIMENT:
            payload = {"query": standalone_question}

        try:
            print(f"--- Calling {route.name} agent with payload: {payload} ---")
            agent_to_call = agent_map[route]
            result_state = await agent_to_call.ainvoke(payload)
            response = result_state.get("final_answer")
            
            return {"final_response": response or f"The {route.name.title()} agent could not find an answer."}
        except Exception as e:
            traceback.print_exc()
            return {"final_response": f"The {route.name.title()} agent encountered an error: {e!r}"}

    async def synthesize_results_node(self, state: IrisState) -> Dict:
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(user_input=state["user_input"], **state["sub_agent_outputs"])
        response = await groq_llm.ainvoke(prompt)
        return {"final_response": response.content}


    async def clarification_node(self, state: IrisState) -> Dict:
            print("---NODE: Smart Clarification (State-Aware v6, Structured Output)---")
            user_input = state["user_input"].strip()
            history = state.get("full_chat_history", [])
            
            last_assistant_message = ""
            if len(history) > 1 and history[-1].type == 'ai':
                last_assistant_message = history[-1].content
            elif len(history) > 2 and history[-2].type == 'ai':
                last_assistant_message = history[-2].content

            # --- STATE 1 & 2: These are deterministic and don't need the LLM. They remain unchanged. ---
            if "Please select one" in last_assistant_message:
                # (Your logic for handling a selected company is correct and stays here)
                print(f"Context: User selected a company ('{user_input}'). Offering actions.")
                proper_company_name_list = await asyncio.to_thread(find_matching_companies, user_input, limit=1)
                proper_company_name = proper_company_name_list[0] if proper_company_name_list else user_input
                action_options = [
                    {"label": "📊 Fundamental Analysis", "query": f"Fundamental analysis of {proper_company_name}"},
                    {"label": "📈 Technical Analysis", "query": f"Technical analysis of {proper_company_name}"},
                    {"label": "📰 Latest News & Sentiment", "query": f"Latest news sentiment for {proper_company_name}"},
                    {"label": "📋 Shareholding Pattern", "query": f"Shareholding pattern of {proper_company_name}"}
                ]
                return {"final_response": {
                    "text_response": f"What information would you like for **{proper_company_name}**?",
                    "ui_components": [{"type": "clarification_options", "title": "Suggested Analyses", "options": action_options}]
                }}

            matching_companies = await asyncio.to_thread(find_matching_companies, user_input)
            if matching_companies:
                # (Your logic for offering company choices is correct and stays here)
                if len(matching_companies) == 1 and matching_companies[0].lower() == user_input.lower():
                    # ... same as before ...
                    action_options = [
                        {"label": "📊 Fundamental Analysis", "query": f"Fundamental analysis of {user_input}"},
                        {"label": "📈 Technical Analysis", "query": f"Technical analysis of {user_input}"},
                        {"label": "📰 Latest News & Sentiment", "query": f"Latest news sentiment for {user_input}"},
                        {"label": "📋 Shareholding Pattern", "query": f"Shareholding pattern of {user_input}"}
                    ]
                    return {"final_response": {
                        "text_response": f"What information would you like for **{user_input}**?",
                        "ui_components": [{"type": "clarification_options", "title": "Suggested Analyses", "options": action_options}]
                    }}
                options = [{"label": name, "query": name} for name in matching_companies]
                return {"final_response": {
                    "text_response": f"By `{user_input}`, what do you mean? Please select one:",
                    "ui_components": [{"type": "clarification_options", "title": "Matching Companies", "options": options}]
                }}

            # --- STATE 3: Fallback using RELIABLE structured output ---
            print(f"Query '{user_input}' is vague and no companies were found. Using LLM for structured clarification.")
            
            LLM_SUGGESTION_PROMPT = """
            You are IRIS, a helpful financial assistant. A user has provided a vague query. Your goal is to generate a list of 3-4 concrete, actionable suggestions that showcase your main capabilities.

            **User's Vague Query:** "{user_input}"
            
            **CRITICAL INSTRUCTIONS:**
            1.  Your entire output must be a valid JSON object matching the `LLMSuggestions` schema, with a root key "items" containing a list of objects.
            2.  Each object must have a "label" and a "query".
            3.  The "label" should be a user-friendly action (e.g., "📊 Get a fundamental analysis").
            4.  **MOST IMPORTANTLY:** The "query" for each suggestion MUST be a simple, unique, keyword-based intent string. It should start with `intent_clarify_`. This tells the system what the user wants to do next. Do NOT make the query a full question.
            
            **Example Output:**
            ```json
            {{
              "items": [
                {{
                  "label": "📊 Get a fundamental analysis",
                  "query": "intent_clarify_fundamentals"
                }},
                {{
                  "label": "📈 See a technical chart",
                  "query": "intent_clarify_technicals"
                }},
                {{
                  "label": "🏆 Find top performing stocks",
                  "query": "intent_clarify_performance"
                }},
                {{
                  "label": "💡 Get a stock recommendation",
                  "query": "intent_clarify_recommendation"
                }}
              ]
            }}
            ```
            
            Now, generate the JSON object for the user's vague query, following all instructions.
        """
           

            prompt = LLM_SUGGESTION_PROMPT.format(user_input=user_input)

            try:
                # This is the key change: we bind the Pydantic model to the LLM call.
                structured_llm = groq_llm_fast.with_structured_output(LLMSuggestions)
                
                # The result is now a Pydantic model instance, not a raw string.
                response_model = await structured_llm.ainvoke(prompt)
                
                # We access the list of options via .root and convert to dict for JSON serialization
                llm_options = [opt.model_dump() for opt in response_model.items]

                # Now, we wrap this guaranteed-to-be-correct list in our response structure.
                return {"final_response": {
                    "text_response": "I'm not sure how to help with that. Here are some things I can do:",
                    "ui_components": [{"type": "vertical_suggestions", "title": "Perhaps you meant...", "options": llm_options}]
                }}
            except Exception as e:
                # This block now only runs if the structured call itself fails, which is much rarer.
                print(f"--- Structured LLM suggestion generation failed: {e}. Falling back to simple text. ---")
                traceback.print_exc()
                fallback_prompt = CLARIFICATION_PROMPT_TEMPLATE.format(user_input=user_input)
                fallback_response = await groq_llm_fast.ainvoke(fallback_prompt)
                return {"final_response": fallback_response.content}

    async def out_of_domain_node(self, state: IrisState) -> Dict:
        return {"final_response": "I am IRIS, an AI financial analyst. I can only answer questions related to the stock market and financial data."}

    async def general_node(self, state: IrisState) -> Dict:
        history_str = "\n".join([f"{m.type}: {m.content}" for m in state.get('full_chat_history', [])[-6:]])
        prompt = GENERAL_PROMPT_TEMPLATE.format(chat_history=history_str, user_input=state["user_input"])
        response = await groq_llm_fast.ainvoke(prompt)
        return {"final_response": response.content}

    async def final_response_validator_node(self, state: IrisState) -> IrisState:
            response = state.get("final_response")
            user_query = state.get("user_input")

            if not response:
                return state

            # Check if the response is a string and contains verbose patterns
            if isinstance(response, str) and (
                "step 1" in response.lower() or
                "step 2" in response.lower() or
                "the final answer is:" in response.lower()
            ):
                print("⚠️ Final response looks verbose. Rewriting...")
                
                # --- THIS IS THE CORRECTED CODE ---
                rewriter_prompt = f"""The user asked: "{user_query}"

    Below is a very verbose explanation. Return a clean response that is a direct answer to the user's question. Eliminate step-by-step formatting.

    {response}
    """
                # Use the module-level groq_llm and the async ainvoke method
                rewriter_response = await groq_llm.ainvoke(rewriter_prompt)
                
                # The output of .ainvoke() is a message object, so we access its content
                state["final_response"] = rewriter_response.content.strip()
                # --- END OF CORRECTED CODE ---
                
            return state


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
        """
        This node now intelligently constructs the perfect payload for each sub-agent
        it calls in parallel during a cross-agent reasoning task.
        """
        standalone_question = state["user_input"]
        
        # 1. Extract the entity name. This is reliable and necessary for all agents.
        entity_extractor_prompt = f"From the following user question, extract the single primary company name or stock symbol. Question: \"{standalone_question}\""
        entity_response = await groq_llm_fast.ainvoke(entity_extractor_prompt)
        entity_name = entity_response.content.strip().strip('"')
        print(f"--- Extracted entity for parallel call: '{entity_name}' ---")
        
        sub_agent_outputs = {}

        # 2. Define the concurrent tasks, each with a perfectly crafted payload.
        async def call_funda():
            try:
                print(f"--- Calling parallel fundamentals agent (for health check) ---")
                # Create a synthetic but valid payload. This forces the agent down its
                # 'health_checklist_query' path without needing a second LLM extraction.
                payload = {
                    "question": f"Fundamental analysis of {entity_name}",
                    "supervisor_decision": {"route": "fundamentals"}, # Satisfies the entry point
                }
                result = await fundamentals_app_instance.ainvoke(payload)
                final_answer = result.get("final_answer", {})
                sub_agent_outputs["fundamentals"] = final_answer.get("text_response", "Unable to retrieve fundamental data.")
            except Exception as e: 
                sub_agent_outputs["fundamentals"] = f"Fundamental analyst error: {e!r}"

        async def call_tech():
            try:
                print(f"--- Calling parallel technicals agent ---")
                # The technicals agent is smart enough to infer its task from this question.
                payload = {"question": f"Technical analysis of {entity_name}"}
                result = await technical_app_instance.ainvoke(payload)
                sub_agent_outputs["technicals"] = result.get("final_answer", "No technical analysis available.")
            except Exception as e: 
                sub_agent_outputs["technicals"] = f"Technical analyst error: {e!r}"

        async def call_sentiment():
            try:
                print(f"--- Calling parallel sentiment agent ---")
                # The sentiment agent is most reliable when given the entity name directly.
                payload = {"query": standalone_question, "company_name": entity_name}
                result = await sentiment_app_instance.ainvoke(payload)
                sub_agent_outputs["sentiment"] = result.get("final_answer", "No sentiment data available.")
            except Exception as e: 
                sub_agent_outputs["sentiment"] = f"Sentiment analyst error: {e!r}"

        # 3. Run all tasks concurrently for maximum speed.
        await asyncio.gather(call_funda(), call_tech(), call_sentiment())
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


