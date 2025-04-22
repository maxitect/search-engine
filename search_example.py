from src.inference import SearchEngine
from src.data_preparation import load_msmarco_data
import os

def main():
    # Initialize the search engine with the trained model
    model_path = os.path.join('models', 'best_model.pth')
    search_engine = SearchEngine(model_path)
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_msmarco_data()
    
    # Extract documents and their IDs
    documents = dataset['passage'].tolist()
    doc_ids = dataset['id'].tolist()
    
    # Cache document encodings
    print("Caching document encodings...")
    search_engine.cache_document_encodings(documents, doc_ids)
    
    # Save the cache for future use
    cache_path = os.path.join('models', 'document_cache.pth')
    search_engine.save_cache(cache_path)
    
    # Example queries
    queries = [
        "What is machine learning?",
        "How does a neural network work?",
        "What are the applications of artificial intelligence?"
    ]
    
    # Perform searches
    print("\nPerforming searches...")
    for query in queries:
        print(f"\nQuery: {query}")
        results = search_engine.search(query, top_k=5)
        print("Top 5 results:")
        for doc_id, score in results:
            print(f"Document ID: {doc_id}, Similarity Score: {score:.4f}")

if __name__ == "__main__":
    main() 