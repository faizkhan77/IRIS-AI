# agents/build_rag_store.py
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from .schema_parser import parse_schema_markdown

def build_store():
    """Builds and saves the RAG vector store."""
    print("--- RAG Vector Store Builder ---")
    print("This script will create a pre-computed vector store to be used by the agents.")

    try:
        VECTOR_STORE_PATH = "context/rag_vector_store.pkl"

        # 1. Load and Parse the Schema
        print("Reading and parsing schema from context/table_context.md...")
        with open("context/table_context.md", "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        all_columns_info = parse_schema_markdown(markdown_content)
        if not all_columns_info:
            print("\nERROR: No column information was parsed. Cannot build vector store. Aborting.")
            return

        # 2. Create Metadata-Rich Documents
        docs_for_embedding = []
        for col_info in all_columns_info:
            page_content = f"Column name: {col_info['column_name']}. Description: {col_info['description']}. This column is in the table named {col_info['table_name']}."
            metadata = {
                "table_name": col_info['table_name'],
                "column_name": col_info['column_name'],
                "description": col_info['description']
            }
            docs_for_embedding.append(Document(page_content=page_content, metadata=metadata))
        
        # 3. Initialize Embeddings Model
        print("Loading sentence-transformers embedding model (this may take a moment)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 4. Create the Vector Store
        print(f"Building vector store from {len(docs_for_embedding)} documents...")
        vectorstore = InMemoryVectorStore.from_documents(
            documents=docs_for_embedding, embedding=embeddings
        )
        
        # 5. Save the vector store
        print(f"Saving vector store to {VECTOR_STORE_PATH}...")
        with open(VECTOR_STORE_PATH, "wb") as f_out:
            pickle.dump(vectorstore, f_out)
            
        print("\n--- SUCCESS ---")
        print(f"Vector store has been successfully built and saved to {VECTOR_STORE_PATH}.")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred while building the vector store: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    build_store()