# RAG Technical Documentation Assistant

A chatbot that answers questions about technical documentation using Retrieval-Augmented Generation (RAG).

## About

This project allows users to upload PDF documents and ask questions about them in natural language. The system retrieves relevant information from the documents and generates accurate answers using OpenAI's GPT-3.5.

I built this to learn how RAG systems work and to practice implementing LangChain pipelines.

## How it works

1. **Load documents**: PDF files are loaded from the `data/` folder
2. **Split into chunks**: Documents are split into smaller pieces (1000 characters each with 200 character overlap)
3. **Create embeddings**: Each chunk is converted to a vector using OpenAI embeddings
4. **Store in FAISS**: Vectors are stored in a local FAISS database for fast similarity search
5. **Answer questions**: When you ask a question, the system finds the 3 most relevant chunks and uses them to generate an answer with GPT-3.5

## Tech Stack

- **LangChain**: Framework for building the RAG pipeline
- **OpenAI API**: For embeddings (text-embedding-ada-002) and text generation (GPT-3.5-turbo)
- **FAISS**: Vector database for similarity search
- **Streamlit**: Web interface
- **Python 3.11**

## Setup

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (get one at https://platform.openai.com/api-keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/rag-technical-docs.git
cd rag-technical-docs

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Add your documents

Create a `data/` folder and add your PDF files:

```bash
mkdir data
# Copy your PDF files to the data/ folder
```

### Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`



## Contact

Oumayma Lamjar  
lamjar.oumayma@gmail.com  
[LinkedIn](https://linkedin.com/in/oumayma-lamjar)
