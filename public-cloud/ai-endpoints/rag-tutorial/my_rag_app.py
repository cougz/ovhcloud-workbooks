from test_rag_ovh import OVHEmbeddings, OVHLLM, Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def create_ovhcloud_rag():
    """Create a RAG system for OVHcloud information"""
    
    # OVHcloud knowledge base with fun facts
    ovhcloud_knowledge = [
        Document(
            page_content="OVHcloud pioneered using water cooling in standard server designs, making data centers 30% more energy efficient than traditional air cooling. They're literally keeping it cool while saving the planet - one server at a time!",
            metadata={"source": "ovh_tech", "chunk_id": 1}
        ),
        Document(
            page_content="OVHcloud's pay-as-you-go model means you only pay for what you actually use, with per-second billing on many services. No more paying for that VM you forgot to turn off after testing - your wallet will thank you!",
            metadata={"source": "ovh_pricing", "chunk_id": 2}
        ),
        Document(
            page_content="OVHcloud guarantees data sovereignty by ensuring your data stays within your chosen geographic region. Unlike providers who treat your data like a world traveler, OVH keeps it exactly where you put it.",
            metadata={"source": "ovh_sovereignty", "chunk_id": 3}
        ),
        Document(
            page_content="OVHcloud offers the best price-performance ratio in the industry by controlling their entire supply chain, from server manufacturing to data center operations. They cut out the middleman and pass savings to customers - refreshingly honest business!",
            metadata={"source": "ovh_value", "chunk_id": 4}
        ),
        Document(
            page_content="OVHcloud's Go Global initiative provides local presence in 4 continents while maintaining consistent service quality. They're globally distributed but locally committed - like having a neighborhood shop that happens to be worldwide!",
            metadata={"source": "ovh_global", "chunk_id": 5}
        )
    ]
    
    # Set up RAG system
    embeddings = OVHEmbeddings(model_name="bge-multilingual-gemma2")
    vectorstore = FAISS.from_documents(ovhcloud_knowledge, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    llm = OVHLLM()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True
    )
    
    return rag_chain

def interactive_ovhcloud_chat():
    """Simple interactive OVHcloud information assistant"""
    rag_system = create_ovhcloud_rag()
    
    print("🌍 Your OVHcloud Expert Assistant is ready!")
    print("Ask me about data sovereignty, pricing, cooling technology, or global presence!")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nYou: ")
        if question.lower() in ['quit', 'exit', 'bye']:
            break
            
        try:
            result = rag_system.invoke({"query": question})
            print(f"OVH Expert: {result['result']}")
            
            if result.get('source_documents'):
                sources = [f"Source {doc.metadata['chunk_id']}" 
                          for doc in result['source_documents']]
                print(f"📚 Sources: {', '.join(sources)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_ovhcloud_chat()
