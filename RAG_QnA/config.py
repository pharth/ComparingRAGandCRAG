import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200