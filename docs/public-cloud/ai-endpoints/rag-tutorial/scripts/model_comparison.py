from test_rag_ovh import OVHEmbeddings
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def compare_embeddings():
    """Compare how different models embed the same text"""
    test_text = "OVHcloud provides data sovereignty with predictable pricing and innovative water cooling technology."
    
    models = ["bge-base-en-v1.5", "bge-m3", "bge-multilingual-gemma2"]
    
    for model in models:
        try:
            embeddings = OVHEmbeddings(model_name=model)
            result = embeddings.embed_query(test_text)
            print(f"{model}: {len(result)} dimensions, first 3 values: {result[:3]}")
        except Exception as e:
            print(f"{model}: Failed - {e}")

if __name__ == "__main__":
    compare_embeddings()
