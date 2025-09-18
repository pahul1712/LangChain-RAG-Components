# LangChain-RAG-Components

This repository demonstrates the core building blocks of a Retrieval-Augmented Generation (RAG) pipeline using LangChain. The project is organized step-by-step into modular folders for clarity and reusability.


## Folder Structure

1-Langchain/
│── 1.1-openai/                 # OpenAI-based LLM demos
│── 1.2-ollama/                 # Ollama-based LLM demos
│── 3.2-DataIngestion/          # Loaders for documents, web pages, etc.
│── 3.3-Data Transformer/       # Text splitters (Recursive, Character, JSON, HTML, etc.)
│── 4-Embeddings/               # Embedding generation (OpenAI, HuggingFace, Ollama, etc.)
│── 5-VectorStore/              # Vector DB implementations (FAISS, Chroma, etc.)



## Components

### 1. Data Ingestion

Load and preprocess raw data from multiple sources into LangChain Document objects.

Implemented in:

3.2-DataIngestion.ipynb

🔹 Examples:

PDF and CSV loaders

Web scraping with WebBaseLoader

JSON ingestion

### 2. Data Transformer

Transform large documents into smaller, semantically meaningful chunks for embeddings.

Implemented in:

3.3-RecursiveCharacterTextSplitter.ipynb

3.4-CharacterTextSplitter.ipynb

3.5-HTMLTextSplitter.ipynb

3.6-RecursiveJsonSplitter.ipynb

🔹 Techniques used:

Recursive splitting (token-aware)

Character-based splitting

HTML parsing

JSON recursive splitting

### 3. Embeddings

Convert chunks into vector representations for semantic search and retrieval.

Implemented in:

4.3-huggingface.ipynb

🔹 Models used:

OpenAI Embeddings

HuggingFace Transformers

Ollama embeddings

### 4. Vector Stores

Store embeddings in a vector database for similarity search and retrieval.

Implemented in:

5.1-Faiss.ipynb

5.2-Chroma.ipynb

🔹 Vector DBs covered:

FAISS → lightweight, local vector search

Chroma → simple and production-ready

## 🔄 Workflow

Data Ingestion → Load PDF/HTML/JSON into LangChain.

Data Transformation → Split documents into smaller chunks.

Embedding → Convert chunks into numerical vectors.

Vector Store → Store vectors in FAISS/Chroma for semantic search.

Retrieval & Generation (future step) → Retrieve top-k relevant chunks and feed into an LLM for contextual answers.

## 🚀 Next Steps

Add Retrievers to query vector databases.

Connect retrievers to an LLM chain for answering questions.

Implement a full RAG pipeline demo (Ask questions about your documents!).

✨ This structure makes it easy to understand each stage of the RAG pipeline, experiment with different components, and extend to production-ready applications.
