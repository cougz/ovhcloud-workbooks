# RAG Tutorial Setup Guide

This guide walks you through setting up a Retrieval-Augmented Generation (RAG) system using OVHcloud AI Endpoints.

## Prerequisites

Before you begin, make sure you have:

- An OVHcloud account with access to AI Endpoints
- Python 3.8+ installed on your system
- Basic knowledge of Python programming
- A terminal or command prompt

## Step 1: Set Up Your Environment

First, create a new directory for your project and set up a virtual environment:

```bash
mkdir ovh-rag-project
cd ovh-rag-project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install langchain openai python-dotenv faiss-cpu tiktoken
```

## Step 2: Configure OVHcloud Credentials

Create a `.env` file in your project directory with your OVHcloud credentials:

```bash
touch .env
```

Add the following content to the `.env` file:

```
OVH_ENDPOINT="https://api.ovh.com/1.0"
OVH_APPLICATION_KEY="your_application_key"
OVH_APPLICATION_SECRET="your_application_secret"
OVH_CONSUMER_KEY="your_consumer_key"
```

Replace the placeholder values with your actual OVHcloud credentials.

## Step 3: Test OVHcloud Connection

Create a file named `test_ovh_connection.py`:

```python
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test connection to OVHcloud AI Endpoints"""
    endpoint = os.getenv("OVH_ENDPOINT")
    app_key = os.getenv("OVH_APPLICATION_KEY")
    app_secret = os.getenv("OVH_APPLICATION_SECRET")
    consumer_key = os.getenv("OVH_CONSUMER_KEY")
    
    headers = {
        "X-Ovh-Application": app_key,
        "X-Ovh-Consumer": consumer_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(f"{endpoint}/auth/time", headers=headers)
        if response.status_code == 200:
            print("✅ Successfully connected to OVHcloud API!")
            return True
        else:
            print(f"❌ Connection failed with status code: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error connecting to OVHcloud API: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()
```

Run the script to test your connection:

```bash
python test_ovh_connection.py
```

## Step 4: Implement Basic RAG System

Create a file named `test_rag_ovh.py`:

```python
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Load environment variables
load_dotenv()

# Configure OVH endpoints
os.environ["OPENAI_API_BASE"] = "https://ai-endpoints.ovh.com/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("OVH_CONSUMER_KEY")

def create_knowledge_base(file_path):
    """Create a vector store from a text file"""
    # Load document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

def query_knowledge_base(vectorstore, query):
    """Query the knowledge base with a question"""
    # Create retrieval chain
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Execute query
    result = qa_chain.run(query)
    return result

if __name__ == "__main__":
    # Create a sample knowledge base file
    with open("knowledge.txt", "w") as f:
        f.write("""
        OVHcloud AI Endpoints provide access to powerful AI models.
        The service offers both embedding models and large language models.
        You can use these endpoints for various AI applications including
        text generation, summarization, and retrieval-augmented generation.
        The API is compatible with OpenAI's API format, making it easy to
        integrate with existing applications and frameworks like LangChain.
        """)
    
    # Create knowledge base
    vectorstore = create_knowledge_base("knowledge.txt")
    
    # Test with a query
    query = "What can I use OVHcloud AI Endpoints for?"
    result = query_knowledge_base(vectorstore, query)
    
    print("\nQuery:", query)
    print("\nResponse:", result)
```

Run the script to test your RAG system:

```bash
python test_rag_ovh.py
```

## Step 5: Optimize Your RAG System

For a production-ready RAG system, you'll want to optimize various parameters:

1. **Chunk Size**: Experiment with different chunk sizes (500-2000) to find the optimal balance
2. **Embedding Model**: Test different embedding models available on OVHcloud
3. **Retrieval Parameters**: Adjust top_k and similarity thresholds
4. **Prompt Engineering**: Refine your prompts for better context integration

You can use the provided scripts in the [scripts directory](../../../../public-cloud/ai-endpoints/rag-tutorial/scripts/) to experiment with these parameters.

## Next Steps

Now that you have a basic RAG system working with OVHcloud AI Endpoints, you can:

1. Expand your knowledge base with more documents
2. Implement a web interface using Flask or Streamlit
3. Add caching mechanisms for better performance
4. Implement feedback loops to improve responses over time

Check out the complete scripts in the [scripts directory](../../../../public-cloud/ai-endpoints/rag-tutorial/scripts/) for more advanced implementations and examples.

[Back to Tutorial Home](index.md){ .md-button }
