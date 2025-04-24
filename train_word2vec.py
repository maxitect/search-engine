import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm import tqdm
import wandb
import os
import re
import collections
import time

def preprocess(text: str, min_count=15) -> list[str]:
    """Enhanced text preprocessing for Word2Vec with better filtering"""
    # Standardize and protect punctuation
    text = text.lower()
    replacements = {
        '.': ' <PERIOD> ',
        ',': ' <COMMA> ',
        '"': ' <QUOTATION_MARK> ',
        ';': ' <SEMICOLON> ',
        '!': ' <EXCLAMATION_MARK> ',
        '?': ' <QUESTION_MARK> ',
        '(': ' <LEFT_PAREN> ',
        ')': ' <RIGHT_PAREN> ',
        '--': ' <HYPHENS> ',
        ':': ' <COLON> ',
        "'": " <APOSTROPHE> ",
        "\n": " <NEWLINE> "
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Remove remaining special characters (except protected ones)
    text = re.sub(r'[^a-z0-9<>\s]', ' ', text)
    
    # Handle contractions and possessives
    text = re.sub(r"(\w+)'s", r"\1 's", text)
    text = re.sub(r"(\w+)n't", r"\1 n't", text)
    text = re.sub(r"(\w+)'re", r"\1 're", text)
    
    # Split and filter
    words = text.split()
    
    # Additional filtering - but less aggressive on removing word forms
    words = [w for w in words if len(w) > 1]  # Remove single characters
    words = [w for w in words if not w.isdigit()]  # Remove numbers
    
    # Keep some useful word forms that were previously filtered out
    # but still filter out words with numbers and URLs
    words = [w for w in words if not any(c.isdigit() for c in w)]  # Remove words with numbers
    words = [w for w in words if not w.startswith('http')]  # Remove URLs
    words = [w for w in words if not w.startswith('www')]  # Remove URLs
    
    # Keep word forms - we want to preserve these semantic distinctions
    # Don't filter out words ending with 'ing', 'ed', 'ly' as they carry important semantic info
    
    word_counts = collections.Counter(words)
    
    # Keep only frequent words and protected tokens
    keep_words = {word for word, count in word_counts.items() 
                 if (count >= min_count) or word.startswith('<')}
    
    return [word for word in words if word in keep_words]

def get_combined_words():
    """Loads text8 + MS-MARCO passages into a single word list with optimized memory usage"""
    # 1. Load text8
    print("Loading text8 dataset...")
    with open("data/text8", "r") as f:
        text8_text = f.read()
        text8_words = preprocess(text8_text)
        print(f"Loaded {len(text8_words)} words from text8")
    
    # 2. Load MS-MARCO passages with memory optimization
    print("Loading MS-MARCO passages...")
    # Load the full dataset
    dataset = load_dataset("ms_marco", "v1.1", split="train")
    msmarco_words = []
    
    # Process in chunks to manage memory
    chunk_size = 10000
    total_examples = len(dataset)
    print(f"Total MS-MARCO examples: {total_examples}")
    
    for chunk_start in tqdm(range(0, total_examples, chunk_size), desc="Processing MS-MARCO chunks"):
        chunk_end = min(chunk_start + chunk_size, total_examples)
        chunk = dataset.select(range(chunk_start, chunk_end))
        
        chunk_words = []
        for example in chunk:
            passages = example['passages']['passage_text']
            for passage in passages:
                processed_words = preprocess(passage)
                chunk_words.extend(processed_words)
        
        msmarco_words.extend(chunk_words)
        # Print progress
        print(f"Processed {chunk_end}/{total_examples} examples, current word count: {len(msmarco_words)}")
    
    print(f"Loaded {len(msmarco_words)} words from MS-MARCO")
    
    # Combine and filter words
    all_words = text8_words + msmarco_words
    word_counts = Counter(all_words)
    filtered_words = [word for word in all_words if word_counts[word] >= 15]  # Increased threshold
    
    print(f"Total combined words after filtering: {len(filtered_words)}")
    return filtered_words

def get_similar_words(model, word, word2idx, idx2word, top_k=10):
    """Get similar words using cosine similarity with more results"""
    if word not in word2idx:
        return f"Word '{word}' not in vocabulary"
    
    word_idx = word2idx[word]
    word_embedding = model.embeddings.weight[word_idx]
    
    # Calculate cosine similarity with all words
    similarities = torch.nn.functional.cosine_similarity(
        word_embedding.unsqueeze(0),
        model.embeddings.weight,
        dim=1
    )
    
    # Get top k similar words (excluding the word itself)
    top_k_indices = torch.topk(similarities, k=top_k+1)[1][1:]  # +1 to exclude the word itself
    similar_words = [(idx2word[idx.item()], similarities[idx].item()) for idx in top_k_indices]
    
    return similar_words

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights using a better distribution for word embeddings
        self.embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
    
    def forward(self, context):
        embeds = self.embeddings(context).mean(dim=1)
        return self.linear(embeds)

def evaluate_analogy(model, word2idx, idx2word, test_cases):
    """Evaluate analogy tasks like 'king - man + woman = queen'"""
    correct = 0
    skipped = 0
    
    for a, b, c, expected in test_cases:
        if a not in word2idx or b not in word2idx or c not in word2idx:
            skipped += 1
            continue
            
        a_vec = model.embeddings.weight[word2idx[a]]
        b_vec = model.embeddings.weight[word2idx[b]]
        c_vec = model.embeddings.weight[word2idx[c]]
        
        # d = c + (b - a)
        result_vec = c_vec + (b_vec - a_vec)
        
        # Find closest word (excluding a, b, c)
        exclude = {word2idx[a], word2idx[b], word2idx[c]}
        
        similarities = torch.nn.functional.cosine_similarity(
            result_vec.unsqueeze(0),
            model.embeddings.weight,
            dim=1
        )
        
        # Set similarities of excluded words to very negative value
        for idx in exclude:
            similarities[idx] = -100
            
        # Get most similar word
        most_similar_idx = torch.argmax(similarities).item()
        predicted = idx2word[most_similar_idx]
        
        if predicted == expected:
            correct += 1
    
    if len(test_cases) - skipped > 0:
        accuracy = correct / (len(test_cases) - skipped)
        return accuracy, skipped
    else:
        return 0, skipped

def train():
    start_time = time.time()
    
    # Initialize wandb with more detailed config
    wandb.init(project="word2vec-cbow", config={
        "architecture": "CBOW",
        "dataset": "text8+MS-MARCO-full",
        "embedding_dim": 384,  # Increased from 300
        "window_size": 10,     # Increased from 8
        "batch_size": 24576,   # Increased for RTX 4090
        "test_size": 0.1,
        "min_count": 15,       # Increased from previous value
        "initial_lr": 0.01,
        "min_lr": 0.0001,
        "epochs": 10,          # Increased from 5
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "mixed_precision": True
    })
    
    # Create data directory if it doesn't exist
    os.makedirs("data/word2vec", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable cuDNN benchmark mode for faster training
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # 1. Get all words
    print("Loading and combining datasets...")
    words = get_combined_words()
    
    # 2. Build vocabulary with stricter filtering
    print("Building vocabulary...")
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= 15]  # Increased threshold
    print(f"Vocabulary size: {len(vocab)}")
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Define test words for similarity evaluation
    test_words = ["computer", "technology", "data", "learning", "system"]
    # Filter to only include words in vocabulary
    test_words = [w for w in test_words if w in word2idx]
    
    # 3. Prepare CBOW training data with larger window
    print("Preparing training data...")
    window_size = 10  # Increased window size for better context
    train_data = []
    
    for i in tqdm(range(window_size, len(words)-window_size), desc="Creating training pairs"):
        context = words[i-window_size:i] + words[i+1:i+window_size+1]
        target = words[i]
        if target in word2idx and all(c in word2idx for c in context):
            train_data.append((
                [word2idx[c] for c in context],
                word2idx[target]
            ))
    
    print(f"Created {len(train_data)} training pairs")
    
    # Split data into train and test sets
    np.random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)  # 90% train, 10% test
    train_set = train_data[:split_idx]
    test_set = train_data[split_idx:]
    
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # 4. Initialize model and optimizer
    model = CBOW(len(vocab), embedding_dim=384).to(device)  # Increased embedding dimension
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Add learning rate scheduler with more aggressive reduction
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True,
        min_lr=0.0001
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # 5. Training loop with mixed precision and improved metrics
    batch_size = 24576  # Increased batch size for RTX 4090
    num_epochs = 10     # Increased epochs
    
    # Define some simple analogy test cases (if words are in vocabulary)
    analogy_tests = [
        ("man", "king", "woman", "queen"),
        ("good", "best", "bad", "worst"),
        ("city", "cities", "child", "children"),
        ("go", "went", "see", "saw"),
        ("small", "smaller", "big", "bigger")
    ]
    
    for epoch in range(num_epochs):
        model.train()
        np.random.shuffle(train_set)
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        with tqdm(range(0, len(train_set), batch_size)) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Training")
            
            for i in pbar:
                batch = train_set[i:i+batch_size]
                if not batch:
                    continue
                    
                contexts, targets = zip(*batch)
                context_tensor = torch.LongTensor(contexts).to(device)
                target_tensor = torch.LongTensor(targets).to(device)
                
                optimizer.zero_grad()
                
                # Use mixed precision training
                with autocast():
                    outputs = model(context_tensor)
                    loss = criterion(outputs, target_tensor)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                _, predicted = torch.max(outputs.data, 1)
                total += target_tensor.size(0)
                correct += (predicted == target_tensor).sum().item()
                total_loss += loss.item()
                
                # Update progress bar with more metrics
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{total_loss/(i/batch_size + 1):.2f}",
                    "acc": f"{100*correct/total:.1f}%",
                    "lr": f"{current_lr:.6f}"
                })
                
                # Log batch metrics at intervals
                if i % (batch_size * 10) == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch_accuracy": 100*correct/total,
                        "learning_rate": current_lr,
                        "epoch": epoch + (i/len(train_set))
                    })
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / (len(train_set) // batch_size)
        epoch_time = time.time() - epoch_start_time
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for i in range(0, len(test_set), batch_size):
                batch = test_set[i:i+batch_size]
                if not batch:
                    continue
                    
                contexts, targets = zip(*batch)
                context_tensor = torch.LongTensor(contexts).to(device)
                target_tensor = torch.LongTensor(targets).to(device)
                
                outputs = model(context_tensor)
                loss = criterion(outputs, target_tensor)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += target_tensor.size(0)
                test_correct += (predicted == target_tensor).sum().item()
                test_loss += loss.item()
        
        # Evaluate word similarities
        similarity_results = {}
        for test_word in test_words:
            if test_word in word2idx:
                similar = get_similar_words(model, test_word, word2idx, idx2word)
                # Format results for logging
                if isinstance(similar, list):
                    similarity_results[test_word] = ", ".join([f"{word} ({score:.3f})" for word, score in similar[:5]])
                else:
                    similarity_results[test_word] = similar
        
        # Evaluate analogies
        analogy_accuracy, skipped = evaluate_analogy(model, word2idx, idx2word, analogy_tests)
        
        # Log comprehensive metrics to wandb
        wandb.log({
            "train_loss": total_loss/len(train_set),
            "train_accuracy": 100*correct/total,
            "test_loss": test_loss/len(test_set),
            "test_accuracy": 100*test_correct/test_total,
            "epoch_time_seconds": epoch_time,
            "analogy_accuracy": analogy_accuracy,
            "analogies_skipped": skipped,
            "epoch": epoch
        })
        
        # Log word similarities
        for word, similar in similarity_results.items():
            wandb.log({f"similar_to_{word}": similar})
        
        # Print results
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds")
        print(f"Train accuracy: {100*correct/total:.2f}%, Test accuracy: {100*test_correct/test_total:.2f}%")
        print(f"Analogy accuracy: {analogy_accuracy*100:.2f}% (skipped {skipped}/{len(analogy_tests)})")
        
        # Print similarity results
        for word, similar in similarity_results.items():
            print(f"Similar words to '{word}': {similar}")
        
        # Save checkpoint with embedding norms info
        embedding_norms = torch.norm(model.embeddings.weight, dim=1)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': total_loss/len(train_set),
            'test_loss': test_loss/len(test_set),
            'embedding_norm_mean': embedding_norms.mean().item(),
            'embedding_norm_std': embedding_norms.std().item(),
            'embedding_norm_max': embedding_norms.max().item(),
            'embedding_norm_min': embedding_norms.min().item()
        }, f"data/word2vec/checkpoint_epoch_{epoch}.pt")
    
    # 6. Save final embeddings
    final_embeddings = model.embeddings.weight.data.cpu().numpy()
    torch.save({
        'embeddings': final_embeddings,
        'vocab': vocab,
        'word2idx': word2idx,
        'idx2word': idx2word
    }, "data/word2vec/word2vec_cbow_final.pt")
    
    # Save in a more interoperable format (numpy)
    np.save("data/word2vec/embeddings.npy", final_embeddings)
    with open("data/word2vec/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")
    
    total_time = time.time() - start_time
    print(f"Training complete in {total_time/60:.2f} minutes!")
    wandb.log({"total_training_time_minutes": total_time/60})
    wandb.finish()

if __name__ == "__main__":
    train()