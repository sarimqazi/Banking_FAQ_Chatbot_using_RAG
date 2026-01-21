# Banking FAQ Chatbot using RAG

A Retrieval-Augmented Generation (RAG) chatbot designed to answer frequently asked questions about banking services. This project uses LLMs, vector embeddings, and semantic search to provide accurate, context-aware responses to user queries.

## Overview

This chatbot leverages a RAG architecture to:
- Load FAQ data from a CSV file
- Split documents into manageable chunks
- Generate embeddings using HuggingFace's sentence transformers
- Store embeddings in a vector database (Chroma)
- Retrieve relevant context for queries
- Generate responses using an LLM (Ollama Llama3.2)

## Features

- **FAQ Database**: Banking FAQs stored in structured CSV format
- **Vector Database**: Chroma for efficient semantic search
- **Embeddings**: HuggingFace sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM**: Ollama Llama3.2 for response generation
- **RAG Chain**: LangChain implementation using LCEL for robust question-answering
- **Context-Aware**: Uses retrieved documents to ground responses in actual FAQ data

## Project Structure

```
.
├── Chatbot.ipynb              # Main Jupyter notebook with chatbot implementation
├── banking_app_faqs.csv       # FAQ dataset
├── requirements.txt           # Python dependencies
├── chroma_db/                 # Vector database storage
│   ├── chroma.sqlite3         # Database file
│   └── [collection data]/     # Embedded vector data
└── README.md                  # This file
```

## Prerequisites

- Python 3.8+
- Ollama with Llama3.2 model installed and running
- pip package manager

## Installation

1. **Clone/Download the project** to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Ollama is running**:
   ```bash
   ollama run llama3.2:latest
   ```
   Keep this running in a separate terminal while using the chatbot.

## Usage

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open `Chatbot.ipynb`**

3. **Run the cells in order**:
   - Load FAQ data
   - Split documents into chunks
   - Create embeddings
   - Store in vector database
   - Create retriever and RAG chain
   - Query the chatbot

4. **Ask questions** using the RAG chain:
   ```python
   response = rag_chain.invoke("How do I transfer money to another bank account?")
   print(response)
   ```

## How It Works

### 1. Data Loading
- FAQ data is loaded from `banking_app_faqs.csv` using LangChain's `CSVLoader`

### 2. Document Chunking
- Documents are split using `RecursiveCharacterTextSplitter`
- Chunk size: 1000 tokens with 200-token overlap

### 3. Embeddings
- Text is converted to vectors using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings capture semantic meaning of text

### 4. Vector Storage
- Embeddings stored in Chroma vector database
- Persisted locally in `chroma_db/` directory

### 5. Retrieval
- User queries are embedded and compared against stored embeddings
- Top 3 most similar documents are retrieved

### 6. Response Generation
- Retrieved context is passed to Ollama Llama3.2 LLM
- LLM generates response based on context using a system prompt

## Dependencies

Key dependencies (see requirements.txt for full list):
- **langchain-core**: RAG pipeline framework
- **langchain-huggingface**: HuggingFace integrations
- **sentence-transformers**: Embedding models
- **chromadb**: Vector database
- **langchain-ollama**: Ollama LLM integration
- **python-dotenv**: Environment variable management

## Configuration

### Embedding Model
To change the embedding model, modify:
```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Change here
)
```

### LLM Model
To change the LLM, modify:
```python
llm = ChatOllama(model="llama3.2:latest")  # Change model name
```

### Retriever Parameters
Adjust the number of retrieved documents:
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Change number of results
)
```

## Notes

- The Ollama Llama3.2 model must be installed and running before executing the chatbot
- First run may take time to generate and store embeddings
- The vector database is cached in `chroma_db/` for faster subsequent runs
- The chatbot is designed to stay within the context of banking FAQs

## Troubleshooting

**Issue**: "Connection refused" error
- **Solution**: Ensure Ollama is running with `ollama run llama3.2:latest`

**Issue**: Slow response times
- **Solution**: This is normal for the first run. Subsequent queries are faster due to caching.

**Issue**: Missing embeddings
- **Solution**: Delete `chroma_db/` folder and re-run the notebook to regenerate embeddings.

## Future Improvements

- Add support for multiple LLMs (GPT-4, Claude, etc.)
- Implement response caching
- Add user feedback mechanism to improve retrieval
- Deploy as web application (FastAPI/Streamlit)
- Add conversation history tracking
- Implement query optimization

## License

[Add your license here]

## Author

[Your Name/Organization]

---

**Last Updated**: January 2026
