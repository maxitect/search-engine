from src.inference import SearchEngine
from src.data_preparation import load_msmarco_data
import os
import argparse

def main(max_documents=100000000):  # Default to 1000 documents for quick testing
    # Initialize the search engine with the trained model
    model_path = os.path.join('models', 'best_model.pth')
    search_engine = SearchEngine(model_path)
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_msmarco_data()
    
    # Extract documents and their IDs from the training set
    documents = []
    doc_ids = []
    
    # Process the training set to get all passages
    for example in dataset['train']:
        passages = example['passages']['passage_text']
        is_selected = example['passages']['is_selected']
        
        # We'll use all passages, not just the selected ones
        for passage in passages:
            if len(documents) >= max_documents:
                break
            documents.append(passage)
            # Generate a unique ID for each passage
            doc_ids.append(f"doc_{len(doc_ids)}")
        
        if len(documents) >= max_documents:
            break
    
    print(f"Loaded {len(documents)} documents (limited for testing)")
    
    # Cache document encodings
    print("Caching document encodings...")
    search_engine.cache_document_encodings(documents, doc_ids)
    
    # Save the cache for future use
    cache_path = os.path.join('models', 'document_cache.pth')
    search_engine.save_cache(cache_path)
    
    # Example queries
    queries = [
        "What is a color?",
        "How does a neural network work?",
        "What are the applications of artificial intelligence?"
    ]
    
    # Create a mapping from doc_id to document text
    doc_id_to_text = {doc_id: text for doc_id, text in zip(doc_ids, documents)}
    
    # Perform searches
    print("\nPerforming searches...")
    for query in queries:
        print(f"\nQuery: {query}")
        results = search_engine.search(query, top_k=5)
        print("Top 5 results:")
        for doc_id, score in results:
            print(f"\nDocument ID: {doc_id}")
            print(f"Similarity Score: {score:.4f}")
            print("Content:")
            print("-" * 80)
            print(doc_id_to_text[doc_id])
            print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-docs', type=int, default=1000, 
                       help='Maximum number of documents to process (default: 1000)')
    args = parser.parse_args()
    main(args.max_docs) 