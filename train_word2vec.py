import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm import tqdm
import wandb
import os
import re
import time
from contextlib import nullcontext

def preprocess(text: str) -> list[str]:
    """Enhanced text preprocessing with punctuation protection"""
    text = text.lower()
    
    # Replace punctuation with special tokens
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
    
    # Split into words and filter
    words = text.split()
    word_counts = Counter(words)
    words = [word for word in words if word_counts[word] > 5]
    return words

def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create word to index and index to word mappings"""
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab

def load_text8():
    """Load and preprocess text8 dataset"""
    print("Loading text8...")
    with open("data/text8", "r") as f:
        text = f.read()
    words = preprocess(text)
    print(f"Loaded {len(words)} words from text8")
    return words

def load_msmarco():
    """Load and preprocess MS-MARCO dataset"""
    print("Loading MS-MARCO...")
    dataset = load_dataset("ms_marco", "v1.1", split="train")
    words = []
    
    # Process in chunks to manage memory
    chunk_size = 10000
    total_examples = len(dataset)
    print(f"Total MS-MARCO examples: {total_examples}")
    
    for chunk_start in tqdm(range(0, total_examples, chunk_size), desc="Processing MS-MARCO"):
        chunk_end = min(chunk_start + chunk_size, total_examples)
        chunk = dataset.select(range(chunk_start, chunk_end))
        
        chunk_words = []
        for example in chunk:
            passages = example['passages']['passage_text']
            for passage in passages:
                chunk_words.extend(preprocess(passage))
        
        words.extend(chunk_words)
        print(f"Processed {chunk_end}/{total_examples} examples, current word count: {len(words)}")
    
    print(f"Loaded {len(words)} words from MS-MARCO")
    return words

def combine_datasets():
    """Combine text8 and MS-MARCO datasets into a single corpus"""
    text8_words = load_text8()
    msmarco_words = load_msmarco()
    
    # Combine words
    all_words = text8_words + msmarco_words
    print(f"Total words after combining: {len(all_words)}")
    
    # Create lookup tables
    word2idx, idx2word = create_lookup_tables(all_words)
    print(f"Vocabulary size: {len(word2idx)}")
    
    # Convert words to indices
    tokens = [word2idx[word] for word in all_words]
    
    # Save tokens to file
    with open("data/word2vec/tokens.txt", "w", encoding="utf-8") as f:
        f.write('\n'.join(map(str, tokens)))
    
    return tokens, word2idx, idx2word

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Split vocabulary into chunks for memory efficiency
        self.chunk_size = 10000  # Process 10k words at a time
        self.num_chunks = (vocab_size + self.chunk_size - 1) // self.chunk_size
        
        # Create embedding chunks
        self.embeddings = nn.ModuleList([
            nn.Embedding(min(self.chunk_size, vocab_size - i * self.chunk_size), embedding_dim)
            for i in range(self.num_chunks)
        ])
        
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights with smaller values
        for emb in self.embeddings:
            emb.weight.data.uniform_(-0.1 / embedding_dim, 0.1 / embedding_dim)
        self.linear.weight.data.uniform_(-0.1 / embedding_dim, 0.1 / embedding_dim)
        self.linear.bias.data.zero_()
    
    def get_embedding(self, indices):
        # Get embeddings from appropriate chunk
        embeddings = []
        for i in range(self.num_chunks):
            mask = (indices >= i * self.chunk_size) & (indices < (i + 1) * self.chunk_size)
            if mask.any():
                chunk_indices = indices[mask] - i * self.chunk_size
                chunk_emb = self.embeddings[i](chunk_indices)
                embeddings.append((mask, chunk_emb))
        
        # Combine embeddings
        if not embeddings:
            return torch.zeros(len(indices), self.embedding_dim, device=indices.device)
        
        result = torch.zeros(len(indices), self.embedding_dim, device=indices.device)
        for mask, emb in embeddings:
            result[mask] = emb
        return result
    
    def forward(self, context):
        if self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, context)
        return self._forward(context)
    
    def _forward(self, context):
        # Get embeddings for context
        embeds = self.get_embedding(context)
        embeds = embeds.view(-1, context.size(1), self.embedding_dim)
        embeds = embeds.mean(dim=1)
        return self.linear(embeds)

def get_similar_words(model, word, word2idx, idx2word, top_k=5):
    """Get similar words using cosine similarity"""
    if word not in word2idx:
        return f"Word '{word}' not in vocabulary"
    
    word_idx = word2idx[word]
    word_embedding = model.embeddings.weight[word_idx]
    
    # Calculate cosine similarity with all words
    similarities = F.cosine_similarity(
        word_embedding.unsqueeze(0),
        model.embeddings.weight,
        dim=1
    )
    
    # Get top k similar words (excluding the word itself)
    top_k_indices = torch.topk(similarities, k=top_k+1)[1][1:]  # +1 to exclude the word itself
    similar_words = [(idx2word[idx.item()], similarities[idx].item()) for idx in top_k_indices]
    
    return similar_words

def train():
    # Set environment variables for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Initialize wandb with minimal memory footprint
    wandb.init(project="word2vec-cbow", config={
        "architecture": "CBOW",
        "dataset": "text8+MS-MARCO",
        "embedding_dim": 25,  # Further reduced from 50
        "window_size": 5,
        "batch_size": 128,    # Further reduced from 256
        "learning_rate": 0.001,
        "epochs": 5,
        "gradient_accumulation_steps": 64  # Increased from 32
    })
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable cuDNN benchmarking and disable deterministic mode
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Load and combine datasets
    print("Loading and combining datasets...")
    tokens, word2idx, idx2word = combine_datasets()
    
    # Create training pairs
    print("Creating training pairs...")
    window_size = wandb.config.window_size
    train_data = []
    
    # Process in chunks to manage memory
    chunk_size = 50000  # Further reduced from 100000
    for i in tqdm(range(0, len(tokens), chunk_size), desc="Creating training pairs"):
        chunk_end = min(i + chunk_size, len(tokens))
        chunk = tokens[i:chunk_end]
        
        for j in range(window_size, len(chunk)-window_size):
            context = chunk[j-window_size:j] + chunk[j+1:j+window_size+1]
            target = chunk[j]
            train_data.append((context, target))
        
        # Free memory
        del chunk
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"Created {len(train_data)} training pairs")
    
    # Split into train and test sets
    np.random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)
    train_set = train_data[:split_idx]
    test_set = train_data[split_idx:]
    
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # Free memory
    del train_data
    del tokens
    
    # Initialize model with minimal memory footprint
    vocab_size = len(word2idx)
    embedding_dim = wandb.config.embedding_dim
    
    # Clear CUDA cache before model initialization
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Initialize model with minimal memory
    model = CBOW(vocab_size, embedding_dim=embedding_dim)
    
    # Move model to device with memory optimization
    if device.type == 'cuda':
        # Move embeddings chunk by chunk
        for i, emb in enumerate(model.embeddings):
            model.embeddings[i] = emb.to(device)
            torch.cuda.empty_cache()
        
        # Move linear layer
        model.linear = model.linear.to(device)
        torch.cuda.empty_cache()
        
        # Convert to half precision
        model = model.half()
    
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training loop
    batch_size = wandb.config.batch_size
    num_epochs = wandb.config.epochs
    accumulation_steps = wandb.config.gradient_accumulation_steps
    
    # Define test words
    test_words = ["computer", "technology", "data", "learning", "system"]
    test_words = [w for w in test_words if w in word2idx]
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle training data
        np.random.shuffle(train_set)
        
        with tqdm(range(0, len(train_set), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            optimizer.zero_grad()
            
            for i in pbar:
                batch = train_set[i:i+batch_size]
                if not batch:
                    continue
                
                contexts, targets = zip(*batch)
                context_tensor = torch.LongTensor(contexts).to(device)
                target_tensor = torch.LongTensor(targets).to(device)
                
                # Use mixed precision training
                with torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext():
                    outputs = model(context_tensor)
                    loss = F.cross_entropy(outputs, target_tensor)
                    loss = loss / accumulation_steps
                
                # Scale gradients and optimize
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Step optimizer after accumulating gradients
                if (i + 1) % accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    
                    # Clear CUDA cache after each optimizer step
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                # Update metrics
                _, predicted = torch.max(outputs.data, 1)
                total += target_tensor.size(0)
                correct += (predicted == target_tensor).sum().item()
                total_loss += loss.item() * accumulation_steps
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{total_loss/(i/batch_size + 1):.2f}",
                    "acc": f"{100*correct/total:.1f}%"
                })
                
                # Free memory
                del context_tensor
                del target_tensor
                del outputs
                
                # Clear CUDA cache more frequently
                if device.type == 'cuda' and i % 5 == 0:
                    torch.cuda.empty_cache()
        
        # Evaluation
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
                loss = F.cross_entropy(outputs, target_tensor)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += target_tensor.size(0)
                test_correct += (predicted == target_tensor).sum().item()
                test_loss += loss.item()
                
                # Free memory
                del context_tensor
                del target_tensor
                del outputs
                
                # Clear CUDA cache periodically during evaluation
                if device.type == 'cuda' and i % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate metrics
        train_loss = total_loss / (len(train_set) // batch_size)
        test_loss = test_loss / (len(test_set) // batch_size)
        train_acc = 100 * correct / total
        test_acc = 100 * test_correct / test_total
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "epoch": epoch
        })
        
        # Print results
        print(f"\nEpoch {epoch+1} completed")
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")
        
        # Print similar words
        print("\nSimilar words:")
        for word in test_words:
            similar = get_similar_words(model, word, word2idx, idx2word)
            if isinstance(similar, list):
                print(f"{word}: {', '.join([f'{w} ({s:.3f})' for w, s in similar])}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, f"data/word2vec/checkpoint_epoch_{epoch}.pt")
        
        # Clear CUDA cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save final embeddings
    embeddings = model.embeddings.weight.data.cpu().numpy()
    torch.save({
        'embeddings': embeddings,
        'word2idx': word2idx,
        'idx2word': idx2word
    }, "data/word2vec/word2vec_cbow_final.pt")
    
    # Save in numpy format
    np.save("data/word2vec/embeddings.npy", embeddings)
    with open("data/word2vec/vocab.txt", "w") as f:
        for word in word2idx:
            f.write(word + "\n")
    
    wandb.finish()

if __name__ == "__main__":
    train()
