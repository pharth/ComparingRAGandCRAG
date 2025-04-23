from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class QAChain:
    """Question-answering chain for Simple RAG"""
    
    def __init__(self, vector_store: Chroma, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Setup RAG prompt
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that provides accurate information based on the context provided.
        
        Answer the question based only on the following context:
        
        {context}
        
        Question: {question}
        """)
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate_answer(self, query: str) -> str:
        """Generate an answer for the query using RAG"""
        try:
            # Get relevant documents directly
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format the prompt manually
            prompt_content = f"""
            You are a helpful assistant that provides accurate information based on the context provided.
            
            Answer the question based only on the following context:
            
            {context}
            
            Question: {query}
            """
            
            # Call LLM directly
            response = self.llm.invoke(prompt_content)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"