import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_ovh_embedding_api():
    """Test OVHcloud embedding API connectivity"""
    token = os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
    
    if not token:
        print("❌ No token found. Check your .env file.")
        return False, None
    
    # Test the three OVHcloud embedding models with correct URLs from documentation
    models_to_test = [
        {
            "name": "bge-multilingual-gemma2",
            "url": "https://bge-multilingual-gemma2.endpoints.kepler.ai.cloud.ovh.net/api/text2vec",
            "dimensions": 3584
        },
        {
            "name": "bge-base-en-v1.5", 
            "url": "https://bge-base-en-v1-5.endpoints.kepler.ai.cloud.ovh.net/api/text2vec",
            "dimensions": 768
        },
        {
            "name": "bge-m3",
            "url": "https://bge-m3.endpoints.kepler.ai.cloud.ovh.net/api/text2vec", 
            "dimensions": 1024
        }
    ]
    
    test_text = "This is a test sentence for embedding."
    
    for model in models_to_test:
        try:
            print(f"Testing {model['name']}...")
            response = requests.post(
                model["url"],
                data=test_text,
                headers={
                    "Content-Type": "text/plain",
                    "Authorization": f"Bearer {token}"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json()
                print(f"✅ {model['name']} works! Dimensions: {len(embedding)} (expected: {model['dimensions']})")
                print(f"First 5 values: {embedding[:5]}")
                return True, model  # Return the working model
            else:
                print(f"❌ {model['name']} failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ {model['name']} connection error: {e}")
    
    return False, None

def test_ovh_llm_api():
    """Test OVHcloud LLM API connectivity"""
    from openai import OpenAI
    
    token = os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
    
    try:
        client = OpenAI(
            base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
            api_key=token
        )
        
        response = client.chat.completions.create(
            model="Meta-Llama-3_3-70B-Instruct",
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            temperature=0,
            max_tokens=50
        )
        
        print(f"✅ LLM API works! Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ LLM API failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing OVHcloud AI Endpoints connectivity...\n")
    
    embedding_ok, working_model = test_ovh_embedding_api()
    print()
    llm_ok = test_ovh_llm_api()
    
    if embedding_ok and llm_ok:
        print(f"\n🎉 All OVHcloud APIs are working!")
        print(f"✅ Working embedding model: {working_model['name']}")
        print("You can proceed with RAG testing.")
    else:
        print("\n⚠️  Some APIs failed. Check your token and try again.")
