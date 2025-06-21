from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
import os




groq_llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

# groq_llm = ChatOllama(
#     model="llama3:latest",       
#     temperature=0.2,
# )