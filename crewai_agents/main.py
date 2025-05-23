from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from crewai_agents import RetrieverAgent, SummarizerAgent, ComposerAgent

# Initialize dependencies
model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host="localhost", port=6333, timeout=60.0)
collection_name = "hotel_reviews"

# Create agent instances
retriever = RetrieverAgent(model=model, client=client, collection_name=collection_name)
summarizer = SummarizerAgent()
composer = ComposerAgent()

def agentic_pipeline(user_query: str):
    retrieved_texts = retriever.run(user_query)
    summary = summarizer.run(retrieved_texts)
    final_response = composer.run(summary)
    return final_response

# Test the pipeline
if __name__ == "__main__":
    query = "best hotel near downtown with free wifi"
    response = agentic_pipeline(query)
    print(response)