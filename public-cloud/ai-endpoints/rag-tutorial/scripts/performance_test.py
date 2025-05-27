import time
from test_rag_ovh import OVHEmbeddings, OVHLLM, Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def test_model_performance(model_name):
    """Test performance for a specific embedding model"""
    print(f"\n🧪 Testing {model_name}...")
    
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
    
    # Setup timing
    start_time = time.time()
    
    # Create embeddings and vector store
    embeddings = OVHEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Create RAG chain
    llm = OVHLLM()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    setup_time = time.time() - start_time
    print(f"⏱️  Setup time: {setup_time:.2f} seconds")
    
    # Test queries
    test_query = "My video export keeps crashing"
    
    start_time = time.time()
    result = rag_chain.invoke({"query": test_query})
    query_time = time.time() - start_time
    
    print(f"⏱️  Query time: {query_time:.2f} seconds")
    print(f"💬 Answer: {result['result']}")
    
    return setup_time, query_time

def test_performance():
    """Test performance across all embedding models"""
    print("🚀 Starting comprehensive performance test...")
    
    models = [
        "bge-base-en-v1.5",      # 768 dimensions
        "bge-m3",                # 1024 dimensions  
        "bge-multilingual-gemma2" # 3584 dimensions
    ]
    
    results = {}
    
    for model in models:
        try:
            setup_time, query_time = test_model_performance(model)
            results[model] = {"setup": setup_time, "query": query_time}
        except Exception as e:
            print(f"❌ {model} failed: {e}")
            results[model] = {"setup": None, "query": None}
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Dimensions':<12} {'Setup':<10} {'Query':<10}")
    print("-" * 60)
    
    model_dims = {"bge-base-en-v1.5": 768, "bge-m3": 1024, "bge-multilingual-gemma2": 3584}
    
    for model, times in results.items():
        dims = model_dims[model]
        setup = f"{times['setup']:.2f}s" if times['setup'] else "Failed"
        query = f"{times['query']:.2f}s" if times['query'] else "Failed"
        print(f"{model:<25} {dims:<12} {setup:<10} {query:<10}")
    
    print(f"\n💡 Performance Insights:")
    print("   • Higher dimensions = better accuracy, slower performance")
    print("   • Lower dimensions = faster performance, potentially less accuracy")
    print("   • Choose based on your speed vs accuracy requirements")

if __name__ == "__main__":
    test_performance()
