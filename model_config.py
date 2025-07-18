
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain.chat_models import init_chat_model
import os




# A powerful model for complex reasoning, SQL generation, and synthesis
groq_llm = ChatGroq(
    temperature=0.2, 
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# A fast, smaller model for routing, simple extractions, and rewriting questions
groq_llm_fast = ChatGroq(
    temperature=0.1, # Lower temperature for more deterministic routing
    model_name="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)
# llm = init_chat_model("llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"),model_provider="groq")

# groq_llm = ChatOllama(
#     model="llama3:latest",       
#     temperature=0.2,
# )