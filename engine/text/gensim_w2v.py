import gensim.downloader as api
import re
import numpy as np
from typing import List, Optional

class GensimWord2Vec:
    def __init__(self, model_name: str = 'word2vec-google-news-300'):
        """
        Initialize the Word2Vec embedder with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to load
        """
        self.model = api.load(model_name)
        self.embedding_dim = self.model.vector_size
        # Create a zero vector for unknown words
        self.unknown_vector = np.zeros(self.embedding_dim)
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the input text by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Splitting into words
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed words
        """
        # Convert to lowercase
        text = text.lower()
        # Replace punctuation with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split into words and remove empty strings
        words = [word for word in text.split() if word]
        return words
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding for a single word.
        Returns zero vector for unknown words.
        
        Args:
            word: Input word
            
        Returns:
            Word embedding as numpy array
        """
        try:
            return self.model[word]
        except KeyError:
            return self.unknown_vector
    
    def get_sentence_embeddings(self, text: str) -> List[np.ndarray]:
        """
        Get embeddings for each word in the input text.
        Handles unknown words by returning zero vectors.
        
        Args:
            text: Input text string
            
        Returns:
            List of word embeddings as numpy arrays
        """
        words = self.preprocess_text(text)
        embeddings = [self.get_word_embedding(word) for word in words]
        return np.array(embeddings)
    
    def get_mean_embedding(self, text: str) -> np.ndarray:
        """
        Get the mean embedding of all words in the text.
        Returns zero vector if no valid words are found.
        
        Args:
            text: Input text string
            
        Returns:
            Mean embedding as numpy array
        """
        embeddings = self.get_sentence_embeddings(text)
        if not embeddings:
            return self.unknown_vector
        return np.mean(embeddings, axis=0)