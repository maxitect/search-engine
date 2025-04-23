import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

def download_nltk_resources():
    """Download required NLTK resources."""
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text and optionally remove stopwords."""
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def preprocess_file(input_path: str, output_path: str, remove_stopwords: bool = True) -> None:
    """Preprocess a text file and save the results."""
    # Download NLTK resources if needed
    download_nltk_resources()
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean and tokenize
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text, remove_stopwords)
    
    # Save processed text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(tokens))
    
    print(f"Preprocessed text saved to {output_path}")

def main():
    input_path = "data/processed/combined.txt"
    output_path = "data/processed/preprocessed.txt"
    preprocess_file(input_path, output_path)

if __name__ == "__main__":
    main() 