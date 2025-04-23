# 🤖 RAG System with Cohere

This repository contains implementations of two Retrieval-Augmented Generation (RAG) systems using the Cohere API. The project provides both a simple RAG implementation and an enhanced contextual RAG implementation to demonstrate different approaches and performance characteristics.

## 📚 Overview

The project implements two RAG architectures:

1. **Simple RAG** 🔍: An implementation following the basic RAG pattern - document loading, chunking, embedding, retrieval, and generation.

2. **Contextual RAG** 🧠: A more sophisticated implementation with additional features:
   - Context generation for document chunks
   - Query reformulation based on conversation history
   - Enhanced document retrieval with contextual awareness

Both systems leverage Cohere's API for embeddings and language models, and process PDF documents as knowledge sources.

## 🛠️ Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-system-cohere.git
   cd rag-system-cohere
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your Cohere API key:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   ```
   
API keys can be obtained by signing up at [cohere.ai](https://cohere.ai).


## 🖥️ Usage

The system can be used in two modes: single query mode or interactive mode.

### Command Line Interface

#### Simple RAG 🔍
```bash
# Single query mode
python app.py simple --pdf_path path/to/your/document.pdf --query "Your question here"

# Interactive mode
python app.py simple --pdf_path path/to/your/document.pdf --interactive

# Process a directory of PDFs
python app.py simple --pdf_path path/to/pdf/directory --interactive
```

#### Contextual RAG 🧠
```bash
# Single query mode
python app.py contextual --pdf_path path/to/your/document.pdf --query "Your question here"

# Interactive mode
python app.py contextual --pdf_path path/to/your/document.pdf --interactive

# Process a directory of PDFs
python app.py contextual --pdf_path path/to/pdf/directory --interactive
```

Both implementations can be tested to determine which best fits specific use cases.

### 📊 Comparison Notebook

The included Jupyter notebook `comparison.ipynb` provides a detailed comparison between the Simple and Contextual RAG implementations. The notebook:

- Loads identical document sets into both systems
- Executes benchmark queries through both systems
- Compares response quality and relevance
- Analyzes performance differences between approaches
- Visualizes results for comparative analysis

To run the notebook:
```bash
jupyter notebook comparison.ipynb
```

## ✨ Key Features

### Simple RAG 🔍

- Document processing pipeline with PyPDF for loading PDFs
- Text chunking with RecursiveCharacterTextSplitter
- Vector storage with Chroma
- Basic retrieval using similarity search
- Answer generation using retrieved context

### Contextual RAG 🧠

- Enhanced document processing with context generation
- Context summaries created for each document chunk
- Conversation-aware query reformulation
- Improved retrieval with contextual information
- Rate limiting to manage API usage

## 🔧 Technical Details

### Document Processing 📄

Both systems process PDF documents by:

1. Loading the PDF using PyPDFLoader
2. Splitting the text into manageable chunks
3. Storing chunks in a vector database (Chroma)

The Contextual RAG system adds an additional step of generating contextual summaries for each chunk using Cohere's API.

### Query Processing 🔎

- **Simple RAG**: Directly uses the user's query for retrieval
- **Contextual RAG**: Reformulates queries based on conversation history for improved context awareness

### Configuration ⚙️

The `config.py` file contains configurable parameters:

- API keys
- Vector database settings
- Text chunking parameters
- Default retrieval settings

### Performance Considerations ⏱️

- The Contextual RAG system makes more API calls and has higher latency due to the additional context generation step.
- The Simple RAG system is faster but may lack contextual awareness in multi-turn conversations.
- Rate limiting has been implemented in the Contextual RAG system to avoid API throttling.

## ⚖️ System Comparison

The two RAG systems offer different trade-offs:

- **Simple RAG** 🔍: Provides a lightweight, faster implementation suitable for straightforward question-answering tasks.
- **Contextual RAG** 🧠: Delivers more contextually relevant answers, especially in multi-turn conversations, at the cost of increased latency and API usage.

Selection between implementations depends on specific use case requirements regarding response quality, conversation handling, and performance constraints.

## 🚀 Future Development

Potential enhancements:

- 📚 Integration with additional document types (beyond PDFs)
- 🔄 Implementation of hybrid retrieval techniques
- ⚡ Caching mechanisms to reduce API calls
- 🔧 Fine-tuning of chunking and retrieval parameters
- 🖥️ UI implementation for improved user interaction

Contributions welcome for any of these enhancements or other improvements.