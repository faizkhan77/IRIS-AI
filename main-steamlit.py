# main.py
import streamlit as st
import os
import time # For execution time
from dotenv import load_dotenv

# Agent imports
from agents.sentiment_agent import app as sentiment_analysis_app # type: ignore
from agents.sentiment_agent import get_sentiment_model_and_tokenizer # type: ignore
from agents.fundamentals_agent import app as fundamentals_agent_app # type: ignore
from agents.technicals_agent import app_technical as technical_analysis_app # Import your technicals agent # type: ignore
from db_config import DATABASE_URL # type: ignore
import plotly.express as px

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="📈 IQUN-AI Financial Suite 📉",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global State & Helper Functions ---

# Initialize session state
if 'active_agent' not in st.session_state:
    st.session_state.active_agent = "Sentiment Analysis"
if 'sentiment_model_loaded' not in st.session_state:
    st.session_state.sentiment_model_loaded = False
# Add any pre-loading flags for other agents if necessary
# For technicals, assume no special pre-loading for now.

# --- API Key Checks (Do this once) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# DATABASE_URL is imported

# --- Sidebar Navigation ---
st.sidebar.title("IQUN-AI Agents")
st.sidebar.markdown("---")

agent_options = ["Sentiment Analysis", "Fundamental Analysis", "Technical Analysis"] # Updated
st.session_state.active_agent = st.sidebar.radio(
    "Choose an Agent:",
    agent_options,
    index=agent_options.index(st.session_state.active_agent)
)
st.sidebar.markdown("---")
st.sidebar.info("Select an agent to begin your financial data exploration.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        Built with ❤️ by Brainfog
    </div>
    """, unsafe_allow_html=True
)


# --- Agent Specific UI and Logic ---

# ==============================================================================
# 1. SENTIMENT ANALYSIS AGENT
# ==============================================================================
if st.session_state.active_agent == "Sentiment Analysis":
    st.title("📈 Financial News Sentiment Analyzer")
    st.markdown("""
    Enter a query (e.g., "latest news of Reliance stock", "outlook for NVIDIA") to fetch news,
    analyze sentiment for each article, and get an AI-generated summary.
    """)
    st.markdown("---")

    if not TAVILY_API_KEY or not GROQ_API_KEY:
        st.error("🚨 Missing API Keys for Sentiment Agent! Please set TAVILY_API_KEY and GROQ_API_KEY.")
        st.stop()

    if not st.session_state.sentiment_model_loaded:
        with st.spinner("Initializing sentiment analysis model... This may take a moment."):
            try:
                get_sentiment_model_and_tokenizer()
                st.session_state.sentiment_model_loaded = True
            except Exception as e:
                st.sidebar.error(f"Failed to load sentiment model: {e}")
                st.error(f"Critical error: Could not load the sentiment model. Details: {e}")
                st.stop()
    
    query_sentiment = st.text_input(
        "Enter your financial news query:",
        placeholder="e.g., latest news of Tesla stock",
        key="sentiment_query_input"
    )

    if st.button("Analyze Sentiment", key="sentiment_analyze_button"):
        if not query_sentiment:
            st.warning("Please enter a news query.")
        else:
            start_time = time.time()
            with st.spinner("🔍 Fetching news, analyzing sentiments, and generating summary... Please wait."):
                try:    
                    inputs = {"query": query_sentiment}
                    results = sentiment_analysis_app.invoke(inputs) # Standard invoke for sentiment
                    st.success("✅ Analysis Complete!")
                    st.subheader("📊 Processed Sentiment Results")

                    if "sentiment_results" in results and results["sentiment_results"]:
                        st.write("#### Individual News Sentiments:")
                        for i, item in enumerate(results["sentiment_results"]):
                            expander_title = f"📰 News {i+1}: {item.get('title', 'N/A')}"
                            with st.expander(expander_title):
                                st.markdown(f"**🔗 URL:** [{item.get('url', '#')}]({item.get('url', '#')})")
                                st.markdown(f"**📝 Content Preview:**")
                                st.caption(item.get('content_preview', 'No preview available.'))
                                st.markdown(f"**💬 Sentiment:** **{item.get('sentiment_label', 'N/A').upper()}**")
                                if item.get("sentiment_scores"):
                                    st.markdown("**Percentage Scores:**")
                                    score_order = ['bullish', 'bearish', 'neutral']
                                    processed_scores_md = ""
                                    for key_score in score_order:
                                        if key_score in item["sentiment_scores"]:
                                            processed_scores_md += f"- {key_score.capitalize()}: {item['sentiment_scores'][key_score]*100:.1f}%\n"
                                    st.markdown(processed_scores_md)
                                st.markdown("---")
                    else:
                        st.warning("No individual news sentiment results to display.")

                    if "aggregated_sentiment" in results and results["aggregated_sentiment"]:
                        st.write("#### Overall Aggregated Sentiment:")
                        agg = results["aggregated_sentiment"]
                        st.metric(label="Majority Sentiment", value=agg.get("majority_vote", "N/A").upper())
                        if agg.get("percentages"):
                            percentages_data = agg["percentages"]
                            if any(p > 0 for p in percentages_data.values()):
                                labels = [k.capitalize() for k in percentages_data.keys()]
                                values = list(percentages_data.values())
                                color_map = {'Bullish': 'lightgreen', 'Bearish': 'lightcoral', 'Neutral': 'lightgrey'}
                                fig = px.pie(values=values, names=labels, title="Sentiment Distribution",
                                             color=labels, color_discrete_map=color_map)
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write("No percentage data for pie chart.")
                    else:
                        st.warning("No aggregated sentiment data to display.")

                    if "final_summary" in results and results["final_summary"]:
                        st.write("#### 🤖 LLM Financial Summary:")
                        with st.chat_message("ai", avatar="🤖"):
                            st.info(results["final_summary"])
                    else:
                        st.warning("No LLM summary generated.")

                except Exception as e:
                    st.error(f"An error occurred during sentiment analysis: {e}")
                    st.exception(e) # Provides full traceback in the app
                finally:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.info(f"⏱️ Sentiment analysis took: {elapsed_time:.2f} seconds")

# ==============================================================================
# 2. FUNDAMENTAL ANALYSIS AGENT
# ==============================================================================
elif st.session_state.active_agent == "Fundamental Analysis":
    st.title("🏦 Fundamental Data Query Engine")
    st.markdown("""
    Ask questions about company fundamentals, financials, shareholding, and more.
    The AI will attempt to generate and execute SQL queries to find the answer from our database.
    """)
    st.markdown("---")

    if not GROQ_API_KEY or not DATABASE_URL:
        st.error("🚨 Missing API Key or Database URL for Fundamental Agent! Please set GROQ_API_KEY and DATABASE_URL.")
        st.stop()

    query_fundamental = st.text_input(
        "Ask a question about financial data:",
        placeholder="e.g., What is the market cap of Reliance Industries?",
        key="fundamental_query_input"
    )

    if 'generated_sql' not in st.session_state:
        st.session_state.generated_sql = ""
    if 'final_answer_fundamental' not in st.session_state: # Use unique key for this agent's answer
        st.session_state.final_answer_fundamental = ""


    if st.button("Get Fundamental Data", key="fundamental_analyze_button"):
        if not query_fundamental:
            st.warning("Please enter a question about fundamentals.")
        else:
            start_time = time.time()
            st.markdown("---")
            st.subheader("🧠 Processing Your Query...")
            
            st.session_state.generated_sql = ""
            st.session_state.final_answer_fundamental = ""

            sql_query_display_area = st.empty()
            final_answer_display_area = st.empty()

            with st.status("IQUN-AI is processing your request...", expanded=True) as status_box:
                try:
                    completed_outputs = set() 
                    for event_chunk in fundamentals_agent_app.stream(
                        {"question": query_fundamental},
                        stream_mode="values" 
                    ):
                        if "extract" in event_chunk and event_chunk["extract"] and "extract" not in completed_outputs:
                            extract_info = event_chunk["extract"]
                            status_box.write(f"Understanding query... DB Required: {extract_info.get('is_required_database', 'N/A')}, Tables: {extract_info.get('tables_required', 'N/A')}")
                            completed_outputs.add("extract")
                        if "columns_context" in event_chunk and "columns_context" not in completed_outputs:
                            status_box.write("Fetching table schemas...")
                            completed_outputs.add("columns_context")
                        if "query" in event_chunk and event_chunk["query"] and event_chunk["query"] != st.session_state.generated_sql:
                            st.session_state.generated_sql = event_chunk["query"]
                            with sql_query_display_area.container():
                                st.markdown("### 🧾 Generated SQL Query")
                                st.code(st.session_state.generated_sql, language="sql")
                            status_box.write(f"SQL Query Generated. Executing...")
                            completed_outputs.add("query")
                        if "result" in event_chunk and "result" not in completed_outputs:
                            status_box.write(f"Query Executed. Result preview: {str(event_chunk['result'])[:100]}...")
                            completed_outputs.add("result")
                        if "answer" in event_chunk and "answer" not in completed_outputs: # Draft answer
                            status_box.write(f"Formulating draft answer...")
                            completed_outputs.add("answer")
                        if "final_answer" in event_chunk and event_chunk["final_answer"]:
                            st.session_state.final_answer_fundamental = event_chunk["final_answer"]
                            # Don't break here, let the stream finish naturally with __end__

                        if "__end__" in event_chunk: # LangGraph's end signal
                            status_box.update(label="✅ Processing Complete!", state="complete", expanded=False)
                            break
                    
                    # Display final answer after stream completion
                    if st.session_state.final_answer_fundamental:
                        with final_answer_display_area.container():
                            st.markdown("### 💬 IQUN-AI's Response:")
                            with st.chat_message("ai", avatar="🤖"):
                                st.markdown(st.session_state.final_answer_fundamental)
                    elif not st.session_state.generated_sql : # Likely a general question or error before SQL
                         final_answer_display_area.warning("Could not determine a conclusive database query or response for your question.")
                    else: # SQL was generated, but no final answer (should be rare if graph is robust)
                        final_answer_display_area.warning("A SQL query was processed, but a final natural language answer could not be formulated. Check agent logs.")

                except Exception as e:
                    st.error(f"An error occurred during fundamental analysis: {e}")
                    st.exception(e)
                    status_box.update(label="⚠️ Error Occurred!", state="error", expanded=True)
                finally:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.info(f"⏱️ Fundamental analysis took: {elapsed_time:.2f} seconds")

# ==============================================================================
# 3. TECHNICAL ANALYSIS AGENT
# ==============================================================================
elif st.session_state.active_agent == "Technical Analysis":
    st.title("⚙️ Technical Indicator Analyzer")
    st.markdown("""
    Ask for technical indicator calculations (e.g., "RSI for Reliance Industries", "What is MACD?")
    or other stock-related questions. The AI will attempt to provide the information or calculations.
    """)
    st.markdown("---")

    if not GROQ_API_KEY or not DATABASE_URL: # Technical agent also needs DB for price data
        st.error("🚨 Missing API Key or Database URL for Technical Agent! Please set GROQ_API_KEY and DATABASE_URL.")
        st.stop()

    query_technical = st.text_input(
        "Ask a question about technical indicators or stocks:",
        placeholder="e.g., What is the RSI of INFY? or Explain Bollinger Bands",
        key="technical_query_input" # Unique key
    )

    if 'final_answer_technical' not in st.session_state: # Use unique key for this agent's answer
        st.session_state.final_answer_technical = ""
    # Store intermediate results if you want to display them (optional)
    if 'technical_extraction_output' not in st.session_state:
        st.session_state.technical_extraction_output = None
    if 'technical_target_fincode' not in st.session_state:
        st.session_state.technical_target_fincode = None
    if 'technical_indicator_result' not in st.session_state:
        st.session_state.technical_indicator_result = None


    if st.button("Analyze Technicals", key="technical_analyze_button"): # Unique key
        if not query_technical:
            st.warning("Please enter a question.")
        else:
            start_time = time.time()
            st.markdown("---")
            st.subheader("⚙️ Processing Your Technical Query...")

            # Reset previous results for a new query
            st.session_state.final_answer_technical = ""
            st.session_state.technical_extraction_output = None
            st.session_state.technical_target_fincode = None
            st.session_state.technical_indicator_result = None
            
            # UI Placeholders
            extraction_display_area = st.empty()
            fincode_display_area = st.empty()
            indicator_result_display_area = st.empty()
            final_answer_display_area = st.empty()


            with st.status("IQUN-AI is analyzing your technical query...", expanded=True) as status_box:
                try:
                    # Stream values from the technical analysis agent
                    for event_chunk in technical_analysis_app.stream(
                        {"question": query_technical},
                        stream_mode="values" # Ensure your agent supports this or adjust
                    ):
                        # Displaying intermediate states from the technical agent
                        if "extraction_output" in event_chunk and event_chunk["extraction_output"]:
                            st.session_state.technical_extraction_output = event_chunk["extraction_output"]
                            with extraction_display_area.container():
                                st.markdown("##### Query Understanding:")
                                st.json(st.session_state.technical_extraction_output) # Display as JSON for clarity
                            if st.session_state.technical_extraction_output: # Add a check to ensure it's not None
                                status_box.write(f"Query understood. Stock Market Related: {st.session_state.technical_extraction_output.is_stock_market_related}")
                        
                        if "target_fincode" in event_chunk and event_chunk["target_fincode"] is not None:
                             st.session_state.technical_target_fincode = event_chunk["target_fincode"]
                             with fincode_display_area.container():
                                 st.markdown(f"##### Resolved Fincode: `{st.session_state.technical_target_fincode}`")
                             status_box.write(f"Fincode resolved: {st.session_state.technical_target_fincode}. Fetching price data...")
                        
                        if "price_data" in event_chunk and event_chunk["price_data"]: # Acknowledge price data fetched
                            status_box.write(f"Price data fetched ({len(event_chunk['price_data'])} records). Calculating indicator...")

                        if "indicator_result" in event_chunk and event_chunk["indicator_result"]:
                            st.session_state.technical_indicator_result = event_chunk["indicator_result"]
                            with indicator_result_display_area.container():
                                st.markdown("##### Indicator Calculation/Error:")
                                st.json(st.session_state.technical_indicator_result)
                            status_box.write("Indicator calculation complete. Generating response...")
                        
                        # The 'draft_answer' key can be used to show progress if your agent yields it
                        if "draft_answer" in event_chunk and event_chunk["draft_answer"]:
                            status_box.write(f"Drafting answer: {str(event_chunk['draft_answer'])[:100]}...")

                        if "final_answer" in event_chunk and event_chunk["final_answer"]:
                            st.session_state.final_answer_technical = event_chunk["final_answer"]
                            # Let the __end__ signal update the status box finally

                        if "__end__" in event_chunk: # LangGraph's end signal
                            status_box.update(label="✅ Technical Analysis Complete!", state="complete", expanded=False)
                            break
                    
                    # After the stream is complete, display the final answer
                    if st.session_state.final_answer_technical:
                        with final_answer_display_area.container():
                            st.markdown("### 💬 IQUN-AI's Response:")
                            with st.chat_message("ai", avatar="🤖"):
                                st.markdown(st.session_state.final_answer_technical)
                    else: # Should be rare if the graph handles all paths to a final_answer
                        final_answer_display_area.warning("Could not formulate a final response for your technical query. Please check agent logs.")


                except Exception as e:
                    st.error(f"An error occurred during technical analysis: {e}")
                    st.exception(e)
                    status_box.update(label="⚠️ Error Occurred!", state="error", expanded=True)
                finally:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.info(f"⏱️ Technical analysis took: {elapsed_time:.2f} seconds")