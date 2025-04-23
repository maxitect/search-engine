import os
import numpy as np
import requests
import pickle
from tqdm import tqdm
import time
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import re
from collections import deque
import random
import pandas as pd
import json
import wandb
from datetime import datetime

# Configuration
OUTPUT_DIR = './models/text8_embeddings'
EMBEDDING_DIM = 200
WINDOW = 8
MIN_COUNT = 1  # Include all words
WORKERS = 8
EPOCHS = 5
LEARNING_RATE = 0.0005
BATCH_WORDS = 5000
HS = 0
NEGATIVE = 5
SAMPLE = 0.001  # Downsample setting
SG = 1  # Skip-gram model (1) or CBOW (0)
TEST_SIZE = 0.1  # 10% of data for testing
USE_WANDB = True  # Set to False to disable wandb

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ProgressCallback(CallbackAny2Vec):
    """Callback to track training progress and loss."""
    
    def __init__(self, log_interval=1000, window_size=100, test_sentences=None):
        self.pbar = None
        self.start_time = time.time()
        self.last_update = self.start_time
        self.last_words = 0
        self.epoch = 0
        self.total_words = 0
        self.log_interval = log_interval
        self.window_size = window_size
        self.loss_history = []
        self.epoch_losses = []
        self.test_losses = []
        self.current_epoch_loss = 0
        self.current_epoch_words = 0
        self.recent_losses = deque(maxlen=window_size)
        self.epoch_start_time = None
        self.test_sentences = test_sentences
        self.run = None
        
        # Initialize wandb if enabled
        if USE_WANDB:
            try:
                self.run = wandb.init(
                    project="word-embeddings",
                    config={
                        "embedding_dim": EMBEDDING_DIM,
                        "window": WINDOW,
                        "min_count": MIN_COUNT,
                        "workers": WORKERS,
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "batch_words": BATCH_WORDS,
                        "hs": HS,
                        "negative": NEGATIVE,
                        "sample": SAMPLE,
                        "sg": SG,
                        "model_type": "Skip-gram" if SG == 1 else "CBOW",
                        "optimization": "Hierarchical Softmax" if HS == 1 else "Negative Sampling",
                        "dataset": "text8 + MS MARCO"
                    }
                )
                print("Weights & Biases initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Weights & Biases: {e}")
                self.run = None
    
    def on_train_begin(self, model):
        """Called when training begins."""
        # model.corpus_count is an integer, not an iterable
        # We need to estimate the total words differently
        if hasattr(model, 'corpus_count'):
            # If corpus_count is available, use it as an estimate
            self.total_words = model.corpus_count
        else:
            # Otherwise, use a reasonable estimate based on the model's vocabulary
            self.total_words = len(model.wv) * 100  # Rough estimate
        
        self.pbar = tqdm(total=self.total_words, desc="Initializing Word2Vec model", unit="words")
        print(f"Building vocabulary from sentences...")
        
        # Initialize loss tracking
        self.loss_history = []
        self.epoch_losses = []
        self.test_losses = []
        self.current_epoch_loss = 0
        self.current_epoch_words = 0
        self.recent_losses = deque(maxlen=self.window_size)
        
        # Log to wandb
        if self.run is not None:
            wandb.log({
                "vocabulary_size": len(model.wv) if hasattr(model, 'wv') else 0,
                "sentences_count": model.corpus_count if hasattr(model, 'corpus_count') else 0,
                "total_words": self.total_words
            })
    
    def on_epoch_begin(self, model):
        """Called at the beginning of each epoch."""
        self.epoch += 1
        self.epoch_start_time = time.time()
        self.current_epoch_loss = 0
        self.current_epoch_words = 0
        print(f"Starting epoch {self.epoch}/{model.epochs}")
        
        # Log to wandb
        if self.run is not None:
            wandb.log({"epoch": self.epoch})
    
    def on_batch_end(self, model):
        """Called after each batch is processed."""
        current_time = time.time()
        
        # Get the current progress from the model
        if hasattr(model, 'corpus_count'):
            words_processed = model.corpus_count
        else:
            # If corpus_count is not available, estimate progress
            words_processed = min(self.last_words + 1000, self.total_words)
        
        words_diff = words_processed - self.last_words
        
        # Update progress bar every 2 seconds
        if current_time - self.last_update >= 2:
            if self.pbar is not None:
                # Calculate average loss over recent batches
                avg_loss = np.mean(list(self.recent_losses)) if self.recent_losses else 0
                
                self.pbar.update(words_diff)
                self.pbar.set_postfix({
                    "vocab_size": len(model.wv) if hasattr(model, 'wv') else "building",
                    "epoch": self.epoch,
                    "loss": f"{avg_loss:.4f}"
                })
            
            self.last_update = current_time
        
        # Track loss if available
        if hasattr(model, 'get_latest_training_loss'):
            try:
                loss = model.get_latest_training_loss()
                self.recent_losses.append(loss)
                self.current_epoch_loss += loss
                self.current_epoch_words += words_diff
                
                # Log loss at intervals
                if words_processed % self.log_interval < words_diff:
                    avg_loss = np.mean(list(self.recent_losses)) if self.recent_losses else 0
                    print(f"Words: {words_processed}, Loss: {avg_loss:.4f}")
                    
                    # Log to wandb
                    if self.run is not None:
                        wandb.log({
                            "batch_loss": avg_loss,
                            "words_processed": words_processed,
                            "epoch": self.epoch
                        })
            except:
                pass
        
        self.last_words = words_processed
    
    def calculate_test_loss(self, model):
        """Calculate loss on test set."""
        if not self.test_sentences or len(self.test_sentences) == 0:
            return None
        
        try:
            # Use a subset of test sentences for efficiency
            test_subset = random.sample(self.test_sentences, min(1000, len(self.test_sentences)))
            
            # Calculate total loss on test set
            total_test_loss = 0
            total_test_words = 0
            
            for sentence in test_subset:
                if len(sentence) < 2:  # Skip very short sentences
                    continue
                
                # For each word in the sentence, predict its context
                for i, target_word in enumerate(sentence):
                    # Get context words (words before and after)
                    context_words = []
                    for j in range(max(0, i - WINDOW), min(len(sentence), i + WINDOW + 1)):
                        if j != i:  # Don't include the target word itself
                            context_words.append(sentence[j])
                    
                    if not context_words:  # Skip if no context words
                        continue
                    
                    # Calculate loss for this word-context pair
                    try:
                        # This is a simplified approach - in a real implementation,
                        # you would use the model's internal methods to calculate loss
                        # For now, we'll use a proxy based on similarity
                        if target_word in model.wv and all(w in model.wv for w in context_words):
                            # Calculate average similarity between target and context words
                            similarities = [model.wv.similarity(target_word, w) for w in context_words]
                            avg_similarity = np.mean(similarities)
                            # Convert similarity to a loss value (higher similarity = lower loss)
                            loss = 1.0 - avg_similarity
                            total_test_loss += loss
                            total_test_words += 1
                    except:
                        pass
            
            # Calculate average test loss
            if total_test_words > 0:
                return total_test_loss / total_test_words
            else:
                return None
        except Exception as e:
            print(f"Error calculating test loss: {e}")
            return None
    
    def on_epoch_end(self, model):
        """Called at the end of each epoch."""
        epoch_time = time.time() - self.epoch_start_time
        
        # Calculate average loss for this epoch
        train_loss = None
        if self.current_epoch_words > 0:
            train_loss = self.current_epoch_loss / self.current_epoch_words
            self.epoch_losses.append(train_loss)
        
        # Calculate test loss
        test_loss = self.calculate_test_loss(model)
        if test_loss is not None:
            self.test_losses.append(test_loss)
        
        # Print epoch summary with clear formatting
        print("\n" + "="*50)
        print(f"EPOCH {self.epoch} SUMMARY")
        print("="*50)
        print(f"Time: {epoch_time:.2f} seconds")
        if train_loss is not None:
            print(f"Training Loss: {train_loss:.4f}")
        if test_loss is not None:
            print(f"Test Loss: {test_loss:.4f}")
        print("="*50 + "\n")
        
        # Log to wandb
        if self.run is not None:
            # Define metrics to ensure proper step ordering
            wandb.define_metric("epoch")
            wandb.define_metric("train_loss", step_metric="epoch")
            wandb.define_metric("test_loss", step_metric="epoch")
            
            log_data = {
                "epoch": self.epoch,
                "epoch_time": epoch_time
            }
            if train_loss is not None:
                log_data["train_loss"] = train_loss
            if test_loss is not None:
                log_data["test_loss"] = test_loss
            
            # Log all metrics at once with the current epoch as the step
            wandb.log(log_data, step=self.epoch)
            
            # Create a custom plot comparing train and test loss
            if train_loss is not None and test_loss is not None:
                wandb.log({
                    "loss_comparison": wandb.plot.line_series(
                        xs=[[self.epoch], [self.epoch]],
                        ys=[[train_loss], [test_loss]],
                        keys=["Train Loss", "Test Loss"],
                        title="Training vs Test Loss",
                        xname="Epoch"
                    )
                }, step=self.epoch)
    
    def on_train_end(self, model):
        """Called when training ends."""
        if self.pbar is not None:
            self.pbar.close()
        
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Vocabulary size: {len(model.wv)} words")
        
        # Print loss history if available
        if self.epoch_losses:
            print("\nLoss history by epoch:")
            print("Epoch\tTrain Loss\tTest Loss")
            for epoch, (train_loss, test_loss) in enumerate(zip(self.epoch_losses, self.test_losses + [None] * (len(self.epoch_losses) - len(self.test_losses))), 1):
                test_str = f"{test_loss:.4f}" if test_loss is not None else "N/A"
                print(f"{epoch}\t{train_loss:.4f}\t{test_str}")
            
            # Save loss history to file
            loss_file = os.path.join(OUTPUT_DIR, 'loss_history.txt')
            with open(loss_file, 'w') as f:
                f.write("Epoch,Train Loss,Test Loss\n")
                for epoch, (train_loss, test_loss) in enumerate(zip(self.epoch_losses, self.test_losses + [None] * (len(self.epoch_losses) - len(self.test_losses))), 1):
                    test_str = f"{test_loss:.6f}" if test_loss is not None else "N/A"
                    f.write(f"{epoch},{train_loss:.6f},{test_str}\n")
            print(f"Loss history saved to {loss_file}")
        
        # Log final metrics to wandb
        if self.run is not None:
            # Log final metrics
            final_metrics = {
                "total_training_time": total_time,
                "vocabulary_size": len(model.wv),
            }
            
            # Add final loss values if available
            if self.epoch_losses:
                final_metrics["final_train_loss"] = self.epoch_losses[-1]
            if self.test_losses:
                final_metrics["final_test_loss"] = self.test_losses[-1]
            
            wandb.log(final_metrics)
            
            # Create a table with loss history
            if self.epoch_losses:
                data = []
                for epoch, (train_loss, test_loss) in enumerate(zip(self.epoch_losses, self.test_losses + [None] * (len(self.epoch_losses) - len(self.test_losses))), 1):
                    data.append([epoch, train_loss, test_loss if test_loss is not None else "N/A"])
                
                wandb.log({
                    "loss_history": wandb.Table(
                        data=data,
                        columns=["Epoch", "Train Loss", "Test Loss"]
                    )
                })
                
                # Create a line plot of the loss history
                epochs = list(range(1, len(self.epoch_losses) + 1))
                train_losses = self.epoch_losses
                test_losses = self.test_losses + [None] * (len(self.epoch_losses) - len(self.test_losses))
                
                # Filter out None values for plotting
                valid_test_epochs = [e for e, t in zip(epochs, test_losses) if t is not None]
                valid_test_losses = [t for t in test_losses if t is not None]
                
                wandb.log({
                    "loss_plot": wandb.plot.line_series(
                        xs=[epochs, valid_test_epochs],
                        ys=[train_losses, valid_test_losses],
                        keys=["Train Loss", "Test Loss"],
                        title="Training and Test Loss Over Time",
                        xname="Epoch"
                    )
                })
            
            # Close wandb run
            wandb.finish()

def download_text8():
    """Download the text8 dataset."""
    text8_path = os.path.join(OUTPUT_DIR, 'text8')
    
    if os.path.exists(text8_path):
        print(f"Text8 dataset already exists at {text8_path}")
        with open(text8_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    print("Downloading text8 dataset...")
    url = 'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8'
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(text8_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading"):
            f.write(chunk)
    
    with open(text8_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Downloaded text8 dataset with {len(text)} characters")
    return text

def download_ms_marco():
    """Download the MS MARCO dataset."""
    ms_marco_path = os.path.join(OUTPUT_DIR, 'ms_marco_train.json')
    
    if os.path.exists(ms_marco_path):
        print(f"MS MARCO dataset already exists at {ms_marco_path}")
        with open(ms_marco_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    print("Downloading MS MARCO dataset...")
    url = 'https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/train-00000-of-00001.parquet'
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save the parquet file
    parquet_path = os.path.join(OUTPUT_DIR, 'ms_marco_train.parquet')
    with open(parquet_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading"):
            f.write(chunk)
    
    # Read the parquet file
    print("Reading MS MARCO parquet file...")
    df = pd.read_parquet(parquet_path)
    
    # Extract passage texts
    print("Extracting passage texts from MS MARCO...")
    passage_texts = []
    for passages in tqdm(df['passages']):
        for passage in passages['passage_text']:
            passage_texts.append(passage)
    
    # Save the extracted passage texts
    with open(ms_marco_path, 'w', encoding='utf-8') as f:
        json.dump(passage_texts, f)
    
    print(f"Extracted {len(passage_texts)} passage texts from MS MARCO")
    return passage_texts

def preprocess_text(text):
    """Preprocess the text by splitting into words and removing punctuation."""
    print("Preprocessing text...")
    # Remove punctuation and split text into words
    words = re.sub(r'[^\w\s]', ' ', text).split()
    print(f"Extracted {len(words)} words")
    return words

def preprocess_passage(passage):
    """Preprocess a passage by splitting into words and removing punctuation."""
    # Remove punctuation, convert to lowercase, and split into words
    words = re.sub(r'[^\w\s]', ' ', passage).lower().split()
    return words

def prepare_sentences(words, window_size=5):
    """Prepare sentences for Word2Vec training."""
    print("Preparing sentences for Word2Vec training...")
    
    # Simple approach: group words into sentences based on window size
    sentences = []
    for i in tqdm(range(0, len(words), window_size)):
        sentence = words[i:i+window_size]
        if len(sentence) > 0:
            sentences.append(sentence)
    
    print(f"Created {len(sentences)} sentences for training")
    return sentences

def prepare_passage_sentences(passages, window_size=5):
    """Prepare sentences from passages for Word2Vec training."""
    print("Preparing sentences from passages...")
    
    sentences = []
    for passage in tqdm(passages):
        # Preprocess the passage
        words = preprocess_passage(passage)
        
        # Group words into sentences based on window size
        for i in range(0, len(words), window_size):
            sentence = words[i:i+window_size]
            if len(sentence) > 0:
                sentences.append(sentence)
    
    print(f"Created {len(sentences)} sentences from passages")
    return sentences

def split_train_test(sentences, test_size=TEST_SIZE):
    """Split sentences into training and test sets."""
    print(f"Splitting data into train ({1-test_size:.1%}) and test ({test_size:.1%}) sets...")
    
    # Shuffle sentences
    random.shuffle(sentences)
    
    # Calculate split index
    split_idx = int(len(sentences) * (1 - test_size))
    
    # Split into train and test
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    
    print(f"Train set: {len(train_sentences)} sentences")
    print(f"Test set: {len(test_sentences)} sentences")
    
    return train_sentences, test_sentences

def train_word2vec_model(sentences, test_sentences=None, embedding_dim=EMBEDDING_DIM, window=WINDOW, 
                         min_count=MIN_COUNT, workers=WORKERS, epochs=EPOCHS,
                         learning_rate=LEARNING_RATE, batch_words=BATCH_WORDS,
                         hs=HS, negative=NEGATIVE, sample=SAMPLE, sg=SG):
    """Train a Word2Vec model on the sentences."""
    print(f"Training Word2Vec model with embedding_dim={embedding_dim}, window={window}...")
    
    # Create progress callback with loss tracking
    progress_callback = ProgressCallback(log_interval=10000, window_size=100, test_sentences=test_sentences)
    
    # Initialize and train the model
    print("Initializing Word2Vec model...")
    start_time = time.time()
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,  # Skip-gram model (1) or CBOW (0)
        epochs=epochs,
        hs=hs,  # Hierarchical softmax
        negative=negative,  # Negative sampling
        alpha=learning_rate,  # Initial learning rate
        min_alpha=learning_rate/100,  # Final learning rate
        batch_words=batch_words,  # Batch size
        sample=sample,  # Downsample setting
        callbacks=[progress_callback]  # Add progress callback
    )
    
    training_time = time.time() - start_time
    print(f"Word2Vec model training completed in {training_time:.2f} seconds")
    print(f"Vocabulary size: {len(model.wv)} words")
    
    return model

def save_embeddings(model, output_path):
    """Save the embeddings to a file."""
    print(f"Saving embeddings to {output_path}...")
    
    # Create a dictionary of word to embedding
    embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
    
    # Save the embeddings
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Saved embeddings for {len(embeddings)} words to {output_path}")
    return embeddings

def save_model(model, output_path):
    """Save the trained model to a file."""
    print(f"Saving model to {output_path}...")
    model.save(output_path)
    print(f"Model saved to {output_path}")

def interactive_word_search(model):
    """Interactive function to search for similar words."""
    print("\n=== Interactive Word Similarity Search ===")
    print("Enter words to find their most similar words (type 'exit' to quit)")
    
    while True:
        word = input("\nEnter a word: ").strip().lower()
        
        if word == 'exit':
            break
        
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f"\nMost similar words to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"Word '{word}' not found in vocabulary")

def main():
    try:
        # Download and preprocess text8
        print("\n=== Processing text8 dataset ===")
        text = download_text8()
        text8_words = preprocess_text(text)
        text8_sentences = prepare_sentences(text8_words)
        print(f"Text8: {len(text8_sentences)} sentences")
        
        # Download and preprocess MS MARCO
        print("\n=== Processing MS MARCO dataset ===")
        ms_marco_passages = download_ms_marco()
        ms_marco_sentences = prepare_passage_sentences(ms_marco_passages)
        print(f"MS MARCO: {len(ms_marco_sentences)} sentences")
        
        # Combine sentences from both datasets
        print("\n=== Combining datasets ===")
        all_sentences = text8_sentences + ms_marco_sentences
        print(f"Total sentences: {len(all_sentences)}")
        
        # Split into train and test sets
        train_sentences, test_sentences = split_train_test(all_sentences)
        
        # Print hyperparameters
        print("\nWord2Vec Training Parameters:")
        print(f"Embedding dimension: {EMBEDDING_DIM}")
        print(f"Context window size: {WINDOW}")
        print(f"Minimum word count: {MIN_COUNT}")
        print(f"Number of workers: {WORKERS}")
        print(f"Number of epochs: {EPOCHS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Batch size: {BATCH_WORDS} words")
        print(f"Model type: {'Skip-gram' if SG == 1 else 'CBOW'}")
        print(f"Hierarchical softmax: {'Enabled' if HS == 1 else 'Disabled'}")
        print(f"Negative sampling: {'Enabled' if NEGATIVE > 0 else 'Disabled'}")
        if NEGATIVE > 0:
            print(f"Negative samples: {NEGATIVE}")
        print(f"Downsample setting: {SAMPLE}")
        
        # Train Word2Vec model
        model = train_word2vec_model(train_sentences, test_sentences)
        
        # Save the model
        model_path = os.path.join(OUTPUT_DIR, 'word2vec_model')
        save_model(model, model_path)
        
        # Save the embeddings
        embeddings_path = os.path.join(OUTPUT_DIR, 'word_embeddings.pkl')
        embeddings = save_embeddings(model, embeddings_path)
        
        # Print some example embeddings
        print("\nExample embeddings:")
        example_words = ['search', 'query', 'document', 'information', 'results']
        for word in example_words:
            if word in embeddings:
                print(f"{word}: {embeddings[word][:5]}...")
            else:
                print(f"{word}: Not in vocabulary")
        
        # Print vocabulary statistics
        print(f"\nVocabulary size: {len(model.wv)} words")
        print(f"Total words in corpus: {len(text8_words) + sum(len(preprocess_passage(p)) for p in ms_marco_passages)}")
        print(f"Unique words: {len(set(text8_words) | set(word for p in ms_marco_passages for word in preprocess_passage(p)))}")
        
        # Print some similar words
        print("\nSimilar words examples:")
        for word in ['search', 'query', 'document', 'information', 'results']:
            if word in model.wv:
                similar = model.wv.most_similar(word, topn=5)
                print(f"{word}: {', '.join([w for w, _ in similar])}")
            else:
                print(f"{word}: Not in vocabulary")
        
        # Interactive word search
        interactive_word_search(model)
                
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 