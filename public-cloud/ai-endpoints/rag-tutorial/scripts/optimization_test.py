import time
from test_rag_ovh import OVHEmbeddings, OVHLLM, Document
from langchain_community.vectorstores import FAISS

def test_chunk_sizes():
    """Experiment with different text chunk sizes"""
    
    ovhcloud_story = """OVHcloud started as a small French hosting company and grew into Europe's largest cloud provider 
    through innovation and customer focus. They revolutionized the industry by manufacturing their own servers with 
    water cooling technology, making data centers 30% more energy efficient than traditional air-cooled systems. 
    Their commitment to data sovereignty means customer data stays exactly where it's supposed to be, not scattered 
    across unknown international servers. The company's predictable pricing model eliminates surprise bills by 
    charging only for actual usage with transparent per-second billing. Their Go Global initiative expanded operations 
    to 4 continents while maintaining local presence and support in each region. OVHcloud controls the entire supply 
    chain from server manufacturing to data center operations, delivering the best price-performance ratio in the 
    industry. Their pay-as-you-go platform scales from startups to enterprises, making advanced cloud technology 
    accessible to businesses of all sizes. The water cooling innovation alone saves millions in energy costs while 
    reducing environmental impact across their global infrastructure."""
    
    chunk_sizes = [20, 40, 60, 100]  # Different word counts
    
    for chunk_size in chunk_sizes:
        words = ovhcloud_story.split()
        chunks = []
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            if len(chunk_words) >= 10:  # Only include substantial chunks
                chunks.append(" ".join(chunk_words))
        
        docs = [Document(page_content=chunk, metadata={"chunk_id": i}) 
                for i, chunk in enumerate(chunks)]
        
        start_time = time.time()
        embeddings = OVHEmbeddings(model_name="bge-base-en-v1.5")
        vectorstore = FAISS.from_documents(docs, embeddings)
        setup_time = time.time() - start_time
        
        print(f"OVH story chunk size {chunk_size} words: {setup_time:.2f}s setup, {len(docs)} chunks created")

if __name__ == "__main__":
    test_chunk_sizes()
