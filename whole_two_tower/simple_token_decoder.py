import torch
import json
import pickle
from transformers import BertTokenizer
from gensim.models import Word2Vec
from typing import Dict, List, Union
import numpy as np

class SimpleTokenDecoder:
    def __init__(self, use_bert: bool = False):
        self.use_bert = use_bert
        if use_bert:
            # Load BERT tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            # For Word2Vec, load the vocabulary mapping
            self.token_to_word = {}
            # Load vocabulary mapping from Word2Vec model
            model = Word2Vec.load('/root/search-engine/models/text8_embeddings/word2vec_model')
            for word, idx in model.wv.key_to_index.items():
                self.token_to_word[str(idx)] = word

    def decode_tokens(self, tokens: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        """Convert token IDs back to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        tokens = np.array(tokens)
        
        if self.use_bert:
            # Use BERT's tokenizer to decode
            return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        else:
            # For Word2Vec, use the vocabulary mapping
            words = [self.token_to_word.get(str(t), '[UNK]') for t in tokens if t != 0]  # Skip padding tokens
            return ' '.join(words)

    def print_token_info(self, tokens: Union[torch.Tensor, List[int], np.ndarray]):
        """Print detailed information about tokens and their corresponding words."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        tokens = np.array(tokens)
        
        print("\nToken Information:")
        print("=" * 50)
        print(f"Token IDs: {tokens.tolist()}")
        
        if self.use_bert:
            print("\nSpecial Tokens:")
            print(f"Padding (0): {sum(tokens == 0)} occurrences")
            print(f"CLS (101): {sum(tokens == 101)} occurrences")
            print(f"SEP (102): {sum(tokens == 102)} occurrences")
            
            print("\nToken to Word Mapping:")
            for token in tokens:
                if token in [0, 101, 102]:
                    continue
                word = self.tokenizer.decode([token], skip_special_tokens=True)
                print(f"Token {token} -> {word}")
        else:
            print("\nToken to Word Mapping:")
            for token in tokens:
                if token == 0:
                    continue
                word = self.token_to_word.get(str(token), '[UNK]')
                print(f"Token {token} -> {word}")
        
        decoded_text = self.decode_tokens(tokens)
        print("\nDecoded Text:")
        print(decoded_text)
        print("=" * 50)

def test_decoder():
    """Test the token decoder with example tokens."""
    # Test with BERT
    print("Testing BERT Decoder:")
    bert_decoder = SimpleTokenDecoder(use_bert=True)
    test_tokens = [101, 2054, 2003, 1996, 1037, 102, 0, 0]  # Example BERT tokens
    bert_decoder.print_token_info(test_tokens)
    
    # Test with Word2Vec
    print("\nTesting Word2Vec Decoder:")
    word2vec_decoder = SimpleTokenDecoder(use_bert=False)
    test_tokens = [1, 2, 3, 4, 5]  # Example Word2Vec tokens
    word2vec_decoder.print_token_info(test_tokens)

if __name__ == "__main__":
    test_decoder() 