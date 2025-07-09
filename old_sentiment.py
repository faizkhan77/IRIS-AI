# sentiment_agent.py
import os
import json
import re # For removing <think> tags
from typing import List, TypedDict, Dict, Any
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from model_config import groq_llm # Use the configured LLM from model_config
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import re
from html import unescape

load_dotenv()

# --- Configuration ---
MODEL_PATH_FULL = "artifacts/finbert_finetuned_full.pt"
MODEL_WEIGHTS_PATH = "artifacts/finbert_finetuned.pt"
MODEL_CONFIG_PATH = "finbert_finetuned_config/config.json" # Ensure this path is correct
TOKENIZER_NAME = "ProsusAI/finbert"

_sentiment_model = None
_sentiment_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_sentiment_model_and_tokenizer():
    global _sentiment_model, _sentiment_tokenizer
    if _sentiment_model is None or _sentiment_tokenizer is None:
        print(f"Loading sentiment model onto {_device}...")
        model_loaded_successfully = False
        try:
            _sentiment_model = torch.load(MODEL_PATH_FULL, map_location=_device, weights_only=False)
            if isinstance(_sentiment_model, torch.nn.Module):
                _sentiment_model.eval()
                print(f"Successfully loaded full model object from {MODEL_PATH_FULL}")
                model_loaded_successfully = True
            else:
                print(f"Warning: Loaded object from {MODEL_PATH_FULL} is not a nn.Module. Type: {type(_sentiment_model)}")
                _sentiment_model = None
        except Exception as e1:
            print(f"Failed to load full model from {MODEL_PATH_FULL}: {e1}")

        if not model_loaded_successfully:
            print("Attempting to load model from weights and config...")
            try:
                if not os.path.exists(MODEL_CONFIG_PATH) or not os.path.exists(MODEL_WEIGHTS_PATH):
                    # Try relative path from sentiment_agent.py if absolute fails
                    # This assumes config is in the same dir or a subdir relative to execution
                    base_dir = os.path.dirname(__file__) # Gets directory of sentiment_agent.py
                    config_path_rel = os.path.join(base_dir, MODEL_CONFIG_PATH)
                    weights_path_rel = os.path.join(base_dir, MODEL_WEIGHTS_PATH)

                    if not os.path.exists(config_path_rel) or not os.path.exists(weights_path_rel):
                         raise FileNotFoundError(f"Model config or weights file not found. Checked: {MODEL_CONFIG_PATH}, {MODEL_WEIGHTS_PATH}, {config_path_rel}, {weights_path_rel}")

                    actual_config_path = config_path_rel
                    actual_weights_path = weights_path_rel
                else:
                    actual_config_path = MODEL_CONFIG_PATH
                    actual_weights_path = MODEL_WEIGHTS_PATH

                config = AutoConfig.from_pretrained(actual_config_path)
                if not hasattr(config, 'id2label') or config.id2label is None or len(config.id2label) != 3:
                     config.id2label = {0: 'bearish', 1: 'bullish', 2: 'neutral'}
                     config.label2id = {v: k for k, v in config.id2label.items()}

                _sentiment_model = AutoModelForSequenceClassification.from_config(config)
                _sentiment_model.load_state_dict(torch.load(actual_weights_path, map_location=_device))
                _sentiment_model.to(_device)
                _sentiment_model.eval()
                model_loaded_successfully = True
                print(f"Successfully loaded model from weights ({actual_weights_path}) and config ({actual_config_path}).")
            except Exception as e2:
                print(f"Failed to load model from weights and config: {e2}")

        if not model_loaded_successfully:
            raise RuntimeError("Could not load sentiment model. Please check paths and file integrity.")

        default_id2label = {0: 'bearish', 1: 'bullish', 2: 'neutral'}
        default_label2id = {v: k for k, v in default_id2label.items()}

        if not hasattr(_sentiment_model, 'config') or _sentiment_model.config is None:
            _sentiment_model.config = AutoConfig.from_dict({})

        if not hasattr(_sentiment_model.config, 'id2label') or \
           not isinstance(_sentiment_model.config.id2label, dict) or \
           len(_sentiment_model.config.id2label) != 3:
            _sentiment_model.config.id2label = default_id2label
            _sentiment_model.config.label2id = default_label2id
        else:
            try:
                current_id2label = {int(k): str(v) for k, v in _sentiment_model.config.id2label.items()}
                _sentiment_model.config.id2label = current_id2label
                _sentiment_model.config.label2id = {v: k for k, v in current_id2label.items()}
            except (ValueError, TypeError):
                _sentiment_model.config.id2label = default_id2label
                _sentiment_model.config.label2id = default_label2id

        _sentiment_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print("Sentiment model and tokenizer loaded.")

    return _sentiment_model, _sentiment_tokenizer


def predict_sentiment(text: str) -> Dict[str, Any]:
    model, tokenizer = get_sentiment_model_and_tokenizer()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(_device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities_tensor = torch.softmax(logits, dim=1).squeeze()
    predicted_class_id = torch.argmax(logits, dim=1).item()

    config_id2label = model.config.id2label

    scores = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}

    for class_idx_from_config, raw_label_from_config in config_id2label.items():
        score_value = probabilities_tensor[class_idx_from_config].item()
        raw_label_lower = raw_label_from_config.lower()

        if 'positive' in raw_label_lower or 'bullish' in raw_label_lower:
            scores['bullish'] += score_value
        elif 'negative' in raw_label_lower or 'bearish' in raw_label_lower:
            scores['bearish'] += score_value
        elif 'neutral' in raw_label_lower:
            scores['neutral'] += score_value
        else: # Fallback for unexpected labels
            # Try to map based on common variations or assign to neutral
            if "pos" in raw_label_lower: scores['bullish'] += score_value
            elif "neg" in raw_label_lower: scores['bearish'] += score_value
            else: scores['neutral'] += score_value # Default unknown to neutral

    # Normalize scores if they don't sum to 1 (e.g. due to fallback logic)
    total_score = sum(scores.values())
    if total_score > 0 and not (0.99 < total_score < 1.01) : # Check if not close to 1
        scores = {k: v / total_score for k, v in scores.items()}


    predicted_raw_label = config_id2label.get(predicted_class_id, "unknown").lower()
    final_predicted_label = "neutral"
    if 'positive' in predicted_raw_label or 'bullish' in predicted_raw_label:
        final_predicted_label = "bullish"
    elif 'negative' in predicted_raw_label or 'bearish' in predicted_raw_label:
        final_predicted_label = "bearish"
    elif 'neutral' in predicted_raw_label:
        final_predicted_label = "neutral"

    return {"predicted_label": final_predicted_label, "scores": scores}

# --- LangGraph State ---
class AgentState(TypedDict):
    query: str
    stock_name_for_summary: str # Extracted stock name for the final summary
    news_items: List[Dict[str, str]]
    sentiment_results: List[Dict[str, Any]]
    aggregated_sentiment: Dict[str, Any]
    final_answer: str # Changed from final_summary to final_answer for consistency

def clean_tavily_content(raw_content: str) -> str:
    if not raw_content: return ""
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', raw_content)
    cleaned = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cleaned)
    cleaned = re.sub(r'#+\s*', '', cleaned)
    cleaned = re.sub(r'\*+', '', cleaned)
    cleaned = re.sub(r'\|.*?\|', '', cleaned)
    cleaned = re.sub(r'\n+', ' ', cleaned)
    cleaned = re.sub(r'\t+', ' ', cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()

# --- LangGraph Nodes ---

def extract_stock_name_node(state: AgentState):
    """Extracts the primary stock name from the query for concise summary."""
    print("---NODE: Extract Stock Name for Sentiment Summary---")
    query = state["query"]
    # Simple regex to find stock names/symbols. This can be improved.
    # Assumes common patterns like "sentiment for GOOGL", "news about Apple", "Paytm stock"
    # This might need an LLM call for more robust extraction if regex is insufficient.
    # For now, let's try a simple heuristic or pass the original query subject.

    # Attempt to find a clear subject. This is a placeholder for better entity extraction.
    # We will use the original query as a fallback if specific extraction fails.
    # For a more robust solution, an LLM call or a dedicated NER tool would be better.
    # For now, the prompt to the summarizer LLM will use the original query's subject.
    # A simple approach: use the query itself as the "subject" or try to find a capitalized word.

    # Let's use an LLM call to extract the company name, as it's more robust.
    prompt = f"Extract the primary company or stock symbol mentioned in the following user query. If multiple are mentioned, pick the first or most prominent one. If none, return 'the analyzed subject'. Query: '{query}'\nCompany/Symbol:"
    try:
        response = groq_llm.invoke(prompt)
        stock_name = response.content.strip()
        if not stock_name or stock_name.lower() == "the analyzed subject" or len(stock_name) > 30: # Basic validation
            # Fallback: try to find a capitalized word or use a generic term
            matches = re.findall(r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)", query) # Matches multi-word capitalized names
            if matches:
                stock_name = matches[0] # Take the first likely name
            else:
                stock_name = "the subject of your query" # Generic fallback
    except Exception as e:
        print(f"Error extracting stock name with LLM: {e}. Using generic fallback.")
        stock_name = "the subject of your query"

    print(f"Extracted/determined stock name for summary: {stock_name}")
    return {"stock_name_for_summary": stock_name}


def tavily_search_node(state: AgentState):
    print("---NODE: Tavily Search---")
    query = state["query"] # Use the original query for searching
    tavily_tool = TavilySearchResults(max_results=5, # Reduced for faster processing & conciseness
                                     include_raw_content=False, # We only need summaries/snippets
                                     include_answer=False) # No need for Tavily's direct answer

    # Construct a search query focused on news
    search_query = f"latest news headlines and summaries for {state['stock_name_for_summary']}"
    print(f"Tavily search query: {search_query}")

    news_results = tavily_tool.invoke(search_query)
    processed_news = []
    for item in news_results:
        # Tavily results are dicts with 'url', 'content' (summary), 'title'
        if isinstance(item, dict) and "content" in item and "url" in item:
            # The 'content' from Tavily is usually already a summary/snippet
            cleaned_content = clean_tavily_content(item["content"])
            if len(cleaned_content) < 30 : continue # Skip very short/unusable content
            processed_news.append({
                "title": item.get("title", "N/A"),
                "content": cleaned_content, # This is the summary from Tavily
                "url": item["url"]
            })
        else:
            print(f"Warning: Skipping malformed Tavily result: {item}")
    print(f"Found {len(processed_news)} news items via Tavily.")
    return {"news_items": processed_news}

def sentiment_analysis_node(state: AgentState):
    print("---NODE: Sentiment Analysis---")
    news_items = state["news_items"]
    sentiment_results = []
    if not news_items:
        print("No news items to analyze.")
        return {"sentiment_results": []}

    for i, item in enumerate(news_items):
        print(f"Analyzing sentiment for news item {i+1}/{len(news_items)}: {item['title'][:50]}...")
        # Use item["content"] which is the summary from Tavily
        sentiment_details = predict_sentiment(item["content"])
        sentiment_results.append({
            "title": item["title"],
            "content_preview": item["content"][:150] + "...", # Shorter preview
            "url": item["url"],
            "sentiment_label": sentiment_details["predicted_label"],
            "sentiment_scores": sentiment_details["scores"]
        })
    print(f"Sentiment analysis complete for {len(sentiment_results)} items.")
    return {"sentiment_results": sentiment_results}

def aggregation_node(state: AgentState):
    print("---NODE: Aggregation---")
    sentiment_results = state["sentiment_results"]
    if not sentiment_results:
        print("No sentiment results to aggregate.")
        return {"aggregated_sentiment": {"majority_vote": "N/A", "percentages": {}, "has_data": False}}

    sentiments = [res["sentiment_label"] for res in sentiment_results]
    sentiment_counts = Counter(sentiments)

    total_sentiments = len(sentiments)
    percentages = {
        sentiment: (count / total_sentiments) * 100
        for sentiment, count in sentiment_counts.items()
    }
    for std_label in ["bullish", "bearish", "neutral"]:
        if std_label not in percentages:
            percentages[std_label] = 0.0

    majority_vote = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "N/A"

    print(f"Aggregated sentiment: {majority_vote}, Percentages: {percentages}")
    return {"aggregated_sentiment": {"majority_vote": majority_vote, "percentages": percentages, "has_data": True}}

def llm_summary_node(state: AgentState):
    print("---NODE: LLM Summary (Sentiment)---")
    aggregated = state["aggregated_sentiment"]
    stock_name = state["stock_name_for_summary"]

    if not aggregated or not aggregated.get("has_data"):
        print("Skipping LLM summary due to no aggregated sentiment data.")
        return {"final_answer": f"I couldn't find enough recent news for {stock_name} to perform a sentiment analysis."}

    # news_with_sentiments_str = "\n".join([ # Removed for brevity
    #     f"- {res['title'][:60]}... (Sentiment: {res['sentiment_label']})"
    #     for res in state["sentiment_results"][:3] # Max 3 for brevity
    # ])

    bullish_perc = aggregated["percentages"].get('bullish', 0.0)
    bearish_perc = aggregated["percentages"].get('bearish', 0.0)
    neutral_perc = aggregated["percentages"].get('neutral', 0.0)

    # Prompt for concise, direct summary
    prompt = f"""
        Based on the latest news analysis for {stock_name}:
        - Bullish sentiment is {bullish_perc:.1f}%
        - Bearish sentiment is {bearish_perc:.1f}%
        - Neutral sentiment is {neutral_perc:.1f}%

        Provide a very short, one-sentence summary in the format:
        "Based on recent news analysis, {stock_name} shows: Bullish {bullish_perc:.1f}%, Bearish {bearish_perc:.1f}%, Neutral {neutral_perc:.1f}%."
        Do not add any other information or commentary.
    """
    try:
        response = groq_llm.invoke(prompt) # Using the globally configured groq_llm
        summary = response.content.strip()

        # Ensure it adheres to the format strictly
        expected_start = f"Based on recent news analysis, {stock_name} shows:"
        if not summary.startswith(expected_start):
            # Force the format if LLM deviates
            summary = f"{expected_start} Bullish {bullish_perc:.1f}%, Bearish {bearish_perc:.1f}%, Neutral {neutral_perc:.1f}%."

        print(f"LLM sentiment summary: {summary}")
    except Exception as e:
        print(f"Error calling Groq LLM for sentiment summary: {e}")
        # Fallback to a template if LLM fails
        summary = f"Based on recent news analysis, {stock_name} shows: Bullish {bullish_perc:.1f}%, Bearish {bearish_perc:.1f}%, Neutral {neutral_perc:.1f}%."

    return {"final_answer": summary}

# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("extract_stock_name", extract_stock_name_node) # New first step
workflow.add_node("tavily_search", tavily_search_node)
workflow.add_node("sentiment_analysis", sentiment_analysis_node)
workflow.add_node("aggregation", aggregation_node)
workflow.add_node("llm_summary", llm_summary_node)

workflow.set_entry_point("extract_stock_name")
workflow.add_edge("extract_stock_name", "tavily_search")
workflow.add_edge("tavily_search", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", "aggregation")
workflow.add_edge("aggregation", "llm_summary")
workflow.add_edge("llm_summary", END)

app = workflow.compile()

if __name__ == "__main__":
    if not os.getenv("TAVILY_API_KEY") or not os.getenv("GROQ_API_KEY"):
        print("Please set TAVILY_API_KEY and GROQ_API_KEY environment variables.")
    else:
        # Ensure model is loaded once if running directly
        try:
            get_sentiment_model_and_tokenizer()
        except Exception as e:
            print(f"Could not load sentiment model on startup: {e}")
            print("Sentiment analysis might fail.")

        user_inp = input("Enter your query for sentiment analysis (e.g., 'sentiment for GOOGL stock'): ")

        inputs = {"query": user_inp} # Initial input for the graph

        print("\n--- Running Sentiment Agent Workflow ---")
        try:
            for event_part in app.stream(inputs):
                # Print node name and the content of that node's output
                node_name = list(event_part.keys())[0]
                node_output = event_part[node_name]
                print(f"\n--- Output from Node: {node_name} ---")
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        if key == "news_items" or key == "sentiment_results":
                            print(f"  {key}: (Count: {len(value)})")
                            if value: print(f"    First item preview: {str(value[0])[:200]}...")
                        else:
                            print(f"  {key}: {str(value)[:300]}") # Print limited preview
                else:
                    print(node_output)

            # final_state = app.invoke(inputs) # Or get final state after stream
            # print("\n\n--- FINAL RESULTS (CLI Test) ---")
            # print(f"Query: {final_state['query']}")
            # print(f"Stock Name for Summary: {final_state.get('stock_name_for_summary', 'N/A')}")
            # print(f"\nFinal Answer by LLM:\n{final_state.get('final_answer', 'N/A')}")

        except Exception as e:
            print(f"Error during agent execution: {e}")
            import traceback
            traceback.print_exc()