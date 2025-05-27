import os
import requests
from langchain.schema import Document
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OVHEmbeddings(Embeddings):
    """OVHcloud embeddings wrapper for LangChain."""
    
    def __init__(self, model_name="bge-multilingual-gemma2"):
        self.token = os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
        
        # Model configurations
        self.models = {
            "bge-multilingual-gemma2": {
                "url": "https://bge-multilingual-gemma2.endpoints.kepler.ai.cloud.ovh.net/api/text2vec",
                "dimensions": 3584
            },
            "bge-base-en-v1.5": {
                "url": "https://bge-base-en-v1-5.endpoints.kepler.ai.cloud.ovh.net/api/text2vec",
                "dimensions": 768
            },
            "bge-m3": {
                "url": "https://bge-m3.endpoints.kepler.ai.cloud.ovh.net/api/text2vec",
                "dimensions": 1024
            }
        }
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported. Available: {list(self.models.keys())}")
        
        self.model_name = model_name
        self.url = self.models[model_name]["url"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = requests.post(
                self.url,
                data=text,
                headers={
                    "Content-Type": "text/plain",
                    "Authorization": f"Bearer {self.token}"
                }
            )
            if response.status_code == 200:
                embeddings.append(response.json())
            else:
                raise Exception(f"Embedding failed for model {self.model_name}: {response.status_code} - {response.text}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class OVHLLM(LLM):
    """OVHcloud LLM wrapper for LangChain."""
    
    def __init__(self):
        super().__init__()
        token = os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
        if not token:
            raise ValueError("OVH_AI_ENDPOINTS_ACCESS_TOKEN environment variable is required")
        
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
                api_key=token
            )
        except Exception as e:
            raise Exception(f"Failed to setup OpenAI client: {e}")
    
    def _call(self, prompt: str, stop=None) -> str:
        try:
            response = self._client.chat.completions.create(
                model="Meta-Llama-3_3-70B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM call failed: {e}")
    
    @property
    def _llm_type(self) -> str:
        return "ovh_llm"

def setup_rag_system():
    """Set up the complete RAG system with LangChain."""
    
    print("🔧 Setting up RAG system...")
    
    # Create knowledge base
    chunks = [
        Document(
            page_content="Export crashes occur with files over 4GB on Mac due to memory limits. This affects ProRes format specifically.",
            metadata={"source": "troubleshooting_guide", "chunk_id": 1}
        ),
        Document(
            page_content="Solution for 4GB+ export crashes: Enable compression in export settings, then switch from ProRes to H.264 codec. This reduces memory usage by 60%.",
            metadata={"source": "troubleshooting_guide", "chunk_id": 2}
        ),
        Document(
            page_content="Windows users experiencing export issues should update graphics drivers to latest version. Download from manufacturer website.",
            metadata={"source": "troubleshooting_guide", "chunk_id": 3}
        )
    ]
    
    # Build vector store with similarity threshold
    print("📊 Creating embeddings...")
    # Use bge-multilingual-gemma2 since it worked in your connection test
    embeddings = OVHEmbeddings(model_name="bge-multilingual-gemma2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Use regular similarity search instead of score threshold (due to embedding scoring issues)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Get top 3 results
    )
    
    # Create RAG chain
    print("🤖 Setting up LLM...")
    llm = OVHLLM()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    print("✅ RAG system ready!")
    return rag_chain

def main():
    try:
        rag_system = setup_rag_system()
        
        # Test queries
        test_queries = [
            "My video export keeps crashing",
            "How do I fix export crashes on Mac?",
            "Why does my coffee taste bitter?"  # No-match scenario
        ]
        
        for question in test_queries:
            print(f"\n{'='*60}")
            print(f"❓ Question: {question}")
            
            # Use the new invoke method instead of deprecated __call__
            result = rag_system.invoke({"query": question})
            print(f"💬 Answer: {result['result']}")
            
            if result.get('source_documents'):
                sources = [f"Chunk {doc.metadata['chunk_id']}" 
                          for doc in result['source_documents']]
                print(f"📚 Sources: {', '.join(sources)}")
            else:
                print("📚 Sources: No matching sources found")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your OVH_AI_ENDPOINTS_ACCESS_TOKEN is correct in .env file")

if __name__ == "__main__":
    main()
