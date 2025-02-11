from langchain_cohere import CohereEmbeddings
from config import COHERE_API_KEY
from langchain.chat_models import ChatCohere

def init_embeddings():
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

def init_llm():
    return ChatCohere(
        model="command",
        temperature=0,
        cohere_api_key=COHERE_API_KEY
    )
