# rag_context_provider.py
from collections import defaultdict
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import colorama
from colorama import Fore, Style

# Import the new deterministic parser
from .schema_parser import parse_schema_markdown

# Initialize colorama
colorama.init(autoreset=True)

# --- RAG Initialization (Expensive operations, run only once on module import) ---
print("Initializing RAG Context Provider (Metadata-Driven)...")
_RAG_INITIALIZED = False
vectorstore = None

try:
    # 1. Load and Parse the Schema with our new parser
    with open("context/table_context.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
    
    # This gives us a clean list of dicts: [{'table_name': ..., 'column_name': ..., 'description': ...}, ...]
    all_columns_info = parse_schema_markdown(markdown_content)

    # 2. Create Metadata-Rich Documents for the Vector Store
    docs_for_embedding = []
    for col_info in all_columns_info:
        # The content is a descriptive sentence optimized for semantic search
        page_content = f"Column: {col_info['column_name']} in Table: {col_info['table_name']}. Description: {col_info['description']}"
        
        # The metadata holds the clean, ground-truth data we need to retrieve
        metadata = {
            "table_name": col_info['table_name'],
            "column_name": col_info['column_name'],
            "description": col_info['description']
        }
        docs_for_embedding.append(Document(page_content=page_content, metadata=metadata))

    # 3. Initialize Embeddings Model and create the Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = InMemoryVectorStore.from_documents(
        documents=docs_for_embedding, embedding=embeddings
    )

    print("RAG Context Provider Initialized Successfully.")
    _RAG_INITIALIZED = True

except Exception as e:
    print(f"{Fore.RED}FATAL: Failed to initialize RAG Context Provider: {e}{Style.RESET_ALL}")
    import traceback
    traceback.print_exc()

# --- Main Function to be Called by the Agent ---
def get_rag_based_context(task_query: str) -> str:
    """
    Takes a generic task query, uses a vector search to find relevant columns,
    and returns them as a formatted string by extracting table/column names from
    the document METADATA. This function does NOT use an LLM.
    """
    if not _RAG_INITIALIZED or not vectorstore:
        return "Error: RAG context provider is not initialized."

    print(f"---RAG: Getting context for task: '{task_query}'---")
    
    # 1. Retrieve documents. The score allows us to filter low-relevance results.
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(task_query, k=10)
    
    # Filter out results with a high distance score (low relevance)
    # The score is L2 distance, so lower is better. 1.0 is a reasonable cutoff.
    relevant_docs = [doc for doc, score in retrieved_docs_with_scores if score < 1.0]

    if not relevant_docs:
        print(f"{Fore.YELLOW}---RAG: No relevant columns found for task: '{task_query}'---{Style.RESET_ALL}")
        return ""

    # 2. Group results by table_name using the metadata
    grouped_by_table = defaultdict(list)
    for doc in relevant_docs:
        table = doc.metadata['table_name']
        column_info = {
            "name": doc.metadata['column_name'],
            "desc": doc.metadata['description']
        }
        # Avoid adding duplicate columns for the same table
        if column_info['name'] not in [c['name'] for c in grouped_by_table[table]]:
            grouped_by_table[table].append(column_info)

    # 3. Format the grouped results into a clean string for the SQL-writing LLM
    formatted_parts = []
    for table_name, columns in grouped_by_table.items():
        table_part = f"Table: {table_name}\nRelevant Columns:\n"
        columns_part = "\n".join([f"- {col['name']}: {col['desc']}" for col in columns])
        formatted_parts.append(table_part + columns_part)
        
    final_context_str = "\n\n".join(formatted_parts)
    
    print(Style.BRIGHT + Fore.CYAN + f"--- RAG Context Found for '{task_query}' ---" + Style.RESET_ALL)
    print(Fore.CYAN + final_context_str)
    print(Style.BRIGHT + Fore.CYAN + "------------------------------------" + Style.RESET_ALL)
    
    return final_context_str

# --- Standalone Testing Block ---
if __name__ == "__main__":
    if not _RAG_INITIALIZED:
        print(f"{Fore.RED}Could not run test because RAG failed to initialize.{Style.RESET_ALL}")
    else:
        print("\n--- RAG Context Provider Standalone Test (Metadata-Driven) ---")
        print("Enter a generic question about schema (e.g., 'company market cap'), or type 'exit' to quit.")
        
        test_queries = [
            "company market cap",
            "market capitalization",
            "company description",
            "promoter holding percentage",
            "latest stock price",
            "earnings per share",
        ]
        
        for query in test_queries:
            print(f"\n> {query}")
            get_rag_based_context(query)

        while True:
            query = input("\n> ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
            get_rag_based_context(query)