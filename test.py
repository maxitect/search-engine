import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Load vocabulary and embeddings
vocab_path = 'data/word2vec/vocabulary.csv'
embedding_path = 'data/word2vec/embeddings.npy'

print("Loading vocabulary and embeddings...")
vocab_df = pd.read_csv(vocab_path)
word_to_idx = {row['word']: int(row['idx']) for _, row in vocab_df.iterrows()}
idx_to_word = {int(row['idx']): row['word'] for _, row in vocab_df.iterrows()}

embeddings = np.load(embedding_path)
embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)

def get_most_similar_words(word, top_k=5):
    if word not in word_to_idx:
        print(f"Word '{word}' not found in vocabulary.")
        return

    word_idx = word_to_idx[word]
    word_vec = embeddings_tensor[word_idx]

    # Compute cosine similarities
    similarities = F.cosine_similarity(word_vec.unsqueeze(0), embeddings_tensor)
    top_k_idx = torch.topk(similarities, top_k + 1).indices.tolist()  # +1 because first is the word itself

    print(f"\nTop {top_k} similar words to '{word}':")
    for idx in top_k_idx[1:]:  # skip the word itself
        print(f"{idx_to_word[idx]} (score: {similarities[idx].item():.4f})")

if __name__ == "__main__":
    while True:
        word = input("\nEnter a word (or type 'exit' to quit): ").strip().lower()
        if word == 'exit':
            break
        get_most_similar_words(word)
