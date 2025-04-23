from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import List, Dict, Any
from langchain_core.documents import Document
import cohere
import time
import random
from config import COHERE_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP

class ContextualPDFProcessor:
    def __init__(self, embeddings, llm, persist_directory=None):
        """Initialize contextual PDF processor with text splitter and vector store"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        # Store the LangChain LLM for compatibility but we won't use it directly
        self.llm = llm
        
        # Create a direct Cohere client
        self.co = cohere.Client(COHERE_API_KEY)
        
        # Rate limiting properties
        self.last_api_call = 0
        self.min_time_between_calls = 6.0  # seconds (allow max 10 calls per minute)
    
    def _wait_for_rate_limit(self):
        """Wait to ensure we respect rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_time_between_calls:
            # Calculate sleep time with a small random factor to avoid thundering herd
            sleep_time = self.min_time_between_calls - time_since_last_call + random.uniform(0.1, 0.5)
            print(f"Rate limiting: Waiting {sleep_time:.2f} seconds before next API call")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def generate_chunk_context(self, chunk_content: str, max_retries=3) -> str:
        """Generate contextual summary for a chunk using direct Cohere API with retries"""
        retries = 0
        backoff_factor = 1
        
        while retries <= max_retries:
            try:
                # Wait for rate limit
                self._wait_for_rate_limit()
                
                # Call the Cohere API directly
                response = self.co.chat(
                    message=f"""
                    Provide a brief context for the following text chunk.
                    
                    Provide 1-2 sentences that explain:
                    1. What is the main topic of this chunk?
                    2. What key information does it contain?
                    
                    Text chunk:
                    {chunk_content}
                    
                    Provide ONLY the contextual summary in 1-2 sentences. Be concise but informative.
                    """,
                    model="command",
                    temperature=0.0
                )
                
                # Return the text response
                return response.text
                    
            except Exception as e:
                retries += 1
                
                if "429" in str(e) and retries <= max_retries:
                    # Rate limit hit, apply exponential backoff
                    wait_time = backoff_factor * 15  # 15, 30, 60 seconds
                    backoff_factor *= 2
                    print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Error generating context with direct Cohere API: {str(e)}")
                    if retries <= max_retries:
                        wait_time = backoff_factor * 5
                        print(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                        backoff_factor *= 2
                    else:
                        return "Context unavailable due to API error after multiple retries."
        
        return "Failed to generate context after multiple attempts."
    
    def create_contextual_document(self, document: Document) -> Document:
        """Enrich a document with contextual information"""
        try:
            # Generate context using just the chunk content
            context = self.generate_chunk_context(document.page_content)
            
            # Create new document with context + original content
            contextual_content = f"Context: {context}\n\nContent: {document.page_content}"
            
            # Create new document with same metadata but enriched content
            return Document(
                page_content=contextual_content,
                metadata={
                    **document.metadata,
                    "original_content": document.page_content,
                    "context_summary": context
                }
            )
        except Exception as e:
            print(f"Error creating contextual document: {str(e)}")
            return document
    
    def load_and_process(self, pdf_path: str) -> List[Document]:
        """Load and process a PDF document with contextual enrichment"""
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = pdf_path
            
            # Split text
            splits = self.text_splitter.split_documents(documents)
            print(f"Document split into {len(splits)} chunks.")
            
            # Option to process only a subset of chunks while testing
            max_chunks = 95  # Set to None to process all chunks
            if max_chunks and len(splits) > max_chunks:
                print(f"Limiting processing to first {max_chunks} chunks for testing.")
                splits = splits[:max_chunks]
            
            # Enrich each chunk with context
            contextual_documents = []
            print(f"Generating contextual embeddings for {len(splits)} chunks...")
            
            for i, chunk in enumerate(splits):
                try:
                    print(f"Processing chunk {i+1}/{len(splits)}")
                    contextual_doc = self.create_contextual_document(chunk)
                    contextual_documents.append(contextual_doc)
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {str(e)}")
                    contextual_documents.append(chunk)
            
            # Add to vector store
            if contextual_documents:
                self.vector_store.add_documents(documents=contextual_documents)
            
            return contextual_documents
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return []