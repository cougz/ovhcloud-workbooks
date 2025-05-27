from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from test_rag_ovh import OVHEmbeddings

def test_similarity_thresholds():
    """Experiment with different similarity approaches"""
    
    # Create test documents about cloud providers
    docs = [
        Document(page_content="OVHcloud ensures data sovereignty by keeping your data in specific geographic regions with transparent policies"),
        Document(page_content="AWS offers global cloud services with pay-as-you-use pricing across multiple availability zones"), 
        Document(page_content="Cats are domestic animals that make popular pets and enjoy playing with yarn"),
        Document(page_content="Dogs are loyal companion animals known for their intelligence and trainability")
    ]
    
    embeddings = OVHEmbeddings(model_name="bge-base-en-v1.5")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    query = "Tell me about cloud data sovereignty"
    
    # Test different k values
    for k in [1, 2, 3, 4]:
        results = vectorstore.similarity_search(query, k=k)
        print(f"\nTop {k} results for '{query}':")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content}")

if __name__ == "__main__":
    test_similarity_thresholds()
