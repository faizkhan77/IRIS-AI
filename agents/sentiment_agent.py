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
from model_config import groq_llm
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import re
from html import unescape

load_dotenv()

# --- Configuration ---
MODEL_PATH_FULL = "artifacts/finbert_finetuned_full.pt"
MODEL_WEIGHTS_PATH = "artifacts/finbert_finetuned.pt"
MODEL_CONFIG_PATH = "finbert_finetuned_config/config.json"
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
            _sentiment_model = torch.load(MODEL_PATH_FULL, map_location=_device, weights_only=False) # weights_only=False for full model
            if isinstance(_sentiment_model, torch.nn.Module):
                _sentiment_model.eval()
                print(f"Successfully loaded full model object from {MODEL_PATH_FULL}")
                model_loaded_successfully = True
            else:
                print(f"Warning: Loaded object from {MODEL_PATH_FULL} is not a nn.Module. Type: {type(_sentiment_model)}")
                _sentiment_model = None # Reset if not a model
        except Exception as e1:
            print(f"Failed to load full model from {MODEL_PATH_FULL}: {e1}")

        if not model_loaded_successfully:
            print("Attempting to load model from weights and config...")
            try:
                if not os.path.exists(MODEL_CONFIG_PATH) or not os.path.exists(MODEL_WEIGHTS_PATH):
                    raise FileNotFoundError("Model config or weights file not found for fallback.")
                
                config = AutoConfig.from_pretrained(MODEL_CONFIG_PATH)
                # Fallback id2label (user's preference)
                if not hasattr(config, 'id2label') or config.id2label is None or len(config.id2label) != 3:
                     config.id2label = {0: 'bearish', 1: 'bullish', 2: 'neutral'}
                     config.label2id = {v: k for k, v in config.id2label.items()}
                
                _sentiment_model = AutoModelForSequenceClassification.from_config(config)
                _sentiment_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=_device))
                _sentiment_model.to(_device)
                _sentiment_model.eval()
                model_loaded_successfully = True
                print("Successfully loaded model from weights and config.")
            except Exception as e2:
                print(f"Failed to load model from weights and config: {e2}")
        
        if not model_loaded_successfully:
            raise RuntimeError("Could not load sentiment model. Please check paths and file integrity.")

        # Ensure model config's id2label is standardized post-loading (especially for full model)
        default_id2label = {0: 'bearish', 1: 'bullish', 2: 'neutral'} # User's preferred default
        default_label2id = {v: k for k, v in default_id2label.items()}

        if not hasattr(_sentiment_model, 'config') or _sentiment_model.config is None:
            print("Warning: Loaded model does not have a 'config' attribute. Initializing one.")
            _sentiment_model.config = AutoConfig.from_dict({}) 

        if not hasattr(_sentiment_model.config, 'id2label') or \
           not isinstance(_sentiment_model.config.id2label, dict) or \
           len(_sentiment_model.config.id2label) != 3:
            print(f"Model's id2label is missing/malformed. Setting default: {default_id2label}")
            _sentiment_model.config.id2label = default_id2label
            _sentiment_model.config.label2id = default_label2id
        else:
            # Ensure keys are integers and values are strings
            try:
                current_id2label = {int(k): str(v) for k, v in _sentiment_model.config.id2label.items()}
                _sentiment_model.config.id2label = current_id2label
                # Rebuild label2id from the (potentially corrected) id2label
                _sentiment_model.config.label2id = {v: k for k, v in current_id2label.items()}
            except (ValueError, TypeError):
                print(f"Error processing model's id2label: {_sentiment_model.config.id2label}. Setting default.")
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

    config_id2label = model.config.id2label # Should be sanitized by get_sentiment_model_and_tokenizer

    scores = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    
    # Map probabilities to standard labels using model's config_id2label
    # and normalize raw labels (e.g., "positive" to "bullish")
    for class_idx_from_config, raw_label_from_config in config_id2label.items():
        # class_idx_from_config is int (e.g. 0, 1, 2)
        # raw_label_from_config is str (e.g. "bearish", "positive")
        score_value = probabilities_tensor[class_idx_from_config].item()
        raw_label_lower = raw_label_from_config.lower()

        if 'positive' in raw_label_lower or 'bullish' in raw_label_lower:
            scores['bullish'] += score_value
        elif 'negative' in raw_label_lower or 'bearish' in raw_label_lower:
            scores['bearish'] += score_value
        elif 'neutral' in raw_label_lower:
            scores['neutral'] += score_value
        else:
            print(f"Warning: Unmapped raw label '{raw_label_from_config}' from model config. Score {score_value} not assigned.")

    # Determine the final predicted label string based on predicted_class_id
    predicted_raw_label = config_id2label.get(predicted_class_id, "unknown").lower()
    final_predicted_label = "neutral" # Default
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
    news_items: List[Dict[str, str]] # Original news items
    sentiment_results: List[Dict[str, Any]] # title, preview, url, sentiment_label, sentiment_scores
    aggregated_sentiment: Dict[str, Any] # majority_vote, percentages
    final_summary: str


def clean_tavily_content(raw_content: str) -> str:
    if not raw_content:
        return ""

    # Remove markdown images: ![alt](url)
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', raw_content)

    # Remove markdown links but keep the text: [text](url) -> text
    cleaned = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cleaned)

    # Remove any leftover markdown headers or formatting
    cleaned = re.sub(r'#+\s*', '', cleaned)  # Remove headers like ##, ### etc.
    cleaned = re.sub(r'\*+', '', cleaned)    # Remove bullets or bold asterisks

    # Remove HTML tables or lines resembling tabular data
    cleaned = re.sub(r'\|.*?\|', '', cleaned)

    # Remove multiple newlines or tabs
    cleaned = re.sub(r'\n+', ' ', cleaned)
    cleaned = re.sub(r'\t+', ' ', cleaned)

    # Unescape HTML characters
    cleaned = unescape(cleaned)

    # Remove extra spaces
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)

    return cleaned.strip()

# --- LangGraph Nodes ---
def tavily_search_node(state: AgentState):
    print("---NODE: Tavily Search---")
    query = state["query"]
    tavily_tool = TavilySearchResults(max_results=5)
    news_results = tavily_tool.invoke(query)
    processed_news = []
    for item in news_results:
        if isinstance(item, dict) and "content" in item and "url" in item:
            cleaned_content = clean_tavily_content(item["content"])
            processed_news.append({
                "title": item.get("title", "N/A"),
                "content": cleaned_content,
                "url": item["url"]
            })
            # print(f"TITLE CHECKING: {item['title']}")
            # print(f"CONTENT CHECKING: {item['content']}")
        else:
            print(f"Warning: Skipping malformed Tavily result: {item}")
    print(f"Found {len(processed_news)} news items.")
    return {"news_items": processed_news}

def sentiment_analysis_node(state: AgentState):
    print("---NODE: Sentiment Analysis---")
    news_items = state["news_items"]
    sentiment_results = []
    if not news_items:
        print("No news items to analyze.")
        return {"sentiment_results": []}

    for i, item in enumerate(news_items):
        print(f"Analyzing sentiment for news item {i+1}/{len(news_items)}...")
        sentiment_details = predict_sentiment(item["content"]) # Returns dict
        sentiment_results.append({
            "title": item["title"],
            "content_preview": item["content"][:2000] + "...",
            "url": item["url"],
            "sentiment_label": sentiment_details["predicted_label"],
            "sentiment_scores": sentiment_details["scores"] 
        })
    print("=====================================================================")
    print(f"SENTIMENT RESULT KEYS: {sentiment_results[0].keys()}")
    print(f"SENTIMENT RESULT: {sentiment_results}")
    print("=====================================================================")
    print(f"Sentiment analysis complete for {len(sentiment_results)} items.")
    return {"sentiment_results": sentiment_results}

def aggregation_node(state: AgentState):
    print("---NODE: Aggregation---")
    sentiment_results = state["sentiment_results"]
    if not sentiment_results:
        print("No sentiment results to aggregate.")
        return {"aggregated_sentiment": {"majority_vote": "N/A", "percentages": {}}}

    sentiments = [res["sentiment_label"] for res in sentiment_results] # Use sentiment_label
    sentiment_counts = Counter(sentiments)
    
    total_sentiments = len(sentiments)
    percentages = {
        sentiment: (count / total_sentiments) * 100
        for sentiment, count in sentiment_counts.items()
    }
    # Ensure all three standard sentiments are in percentages, even if 0%
    for std_label in ["bullish", "bearish", "neutral"]:
        if std_label not in percentages:
            percentages[std_label] = 0.0
            
    if sentiment_counts:
        majority_vote = sentiment_counts.most_common(1)[0][0]
    else:
        majority_vote = "N/A"
        
    print(f"Aggregated sentiment: {majority_vote}, Percentages: {percentages}")
    return {"aggregated_sentiment": {"majority_vote": majority_vote, "percentages": percentages}}

def llm_summary_node(state: AgentState):
    print("---NODE: LLM Summary---")
    if not state.get("sentiment_results") or not state.get("aggregated_sentiment"):
        print("Skipping LLM summary due to missing prior data.")
        return {"final_summary": "Could not generate summary due to errors in previous steps."}

    news_with_sentiments_str = "\n\n".join([
        f"News Title: {res['title']}\nURL: {res['url']}\nSentiment: {res['sentiment_label']}\nPreview: {res['content_preview']}" # Use sentiment_label
        for res in state["sentiment_results"]
    ])
    
    aggregated = state["aggregated_sentiment"]
    percentages_str = ", ".join([f"{s.capitalize()}: {p:.2f}%" for s, p in aggregated["percentages"].items()])
    
    prompt = f"""You are a financial news analyst.
        Based on the following news items, their individual sentiments, and the overall aggregated sentiment, provide a concise natural language summary.
        Focus on the implications for the subject of the query: '{state['query']}'.

        Here are the news items and their sentiments:
        {news_with_sentiments_str}

        Overall Sentiment Analysis:
        Majority Sentiment: {aggregated['majority_vote'].capitalize()}
        Sentiment Percentages: {percentages_str}

        Please provide your summary:
    """
    try:
        # Using the model name from your provided sentiment_agent.py
        # groq_llm = ChatGroq(temperature=0.2, model_name="deepseek-r1-distill-llama-70b", groq_api_key=os.getenv("GROQ_API_KEY"))
        # If 'qwen-qwq-32b' gives issues, try common ones like "mixtral-8x7b-32768" or "llama3-8b-8192"
        response = groq_llm.invoke(prompt)
        summary = response.content
        
        # Remove <think>...</think> blocks
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL | re.IGNORECASE).strip()
        print("LLM summary generated and cleaned.")
    except Exception as e:
        print(f"Error calling Groq LLM: {e}")
        summary = f"Error generating summary with LLM: {e}"
        
    return {"final_summary": summary}

# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("tavily_search", tavily_search_node)
workflow.add_node("sentiment_analysis", sentiment_analysis_node)
workflow.add_node("aggregation", aggregation_node)
workflow.add_node("llm_summary", llm_summary_node)

workflow.set_entry_point("tavily_search")
workflow.add_edge("tavily_search", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", "aggregation")
workflow.add_edge("aggregation", "llm_summary")
workflow.add_edge("llm_summary", END)

app = workflow.compile()

if __name__ == "__main__":
    if not os.getenv("TAVILY_API_KEY") or not os.getenv("GROQ_API_KEY"):
        print("Please set TAVILY_API_KEY and GROQ_API_KEY environment variables.")
    else:
        get_sentiment_model_and_tokenizer() # Pre-load

        user_inp = input("Enter your query: ")
        
        inputs = {"query": user_inp}
        final_state = app.invoke(inputs)
        
        print("\n\n--- FINAL RESULTS (CLI Test) ---")
        print(f"Query: {final_state['query']}")
        
        print(f"\nSentiment Results:")
        for res in final_state.get('sentiment_results', []):
            print(f"  - Title: {res['title']}")
            print(f"    URL: {res['url']}")
            print(f"    Sentiment Label: {res['sentiment_label']}")
            print(f"    Sentiment Scores: {res['sentiment_scores']}")
            print(f"    Preview: {res['content_preview'][:100]}...")
        
        agg_sentiment = final_state.get('aggregated_sentiment', {})
        print(f"\nAggregated Sentiment:")
        print(f"  Majority: {agg_sentiment.get('majority_vote', 'N/A')}")
        print(f"  Percentages: {agg_sentiment.get('percentages', {})}")
        
        print(f"\nFinal Summary by LLM:\n{final_state.get('final_summary', 'N/A')}")