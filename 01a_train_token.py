import collections
import pickle
import requests
import pandas as pd

import src.config as config


def preprocess(text):
    if not text:
        return []
    text = str(text).lower()
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace(':',  ' <COLON> ')
    words = text.split()
    return words


r = requests.get(
    'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('text8', 'wb') as f:
    f.write(r.content)
with open('text8') as f:
    text8: str = f.read()

# Filter out low frequency words
corpus: list[str] = preprocess(text8)
word_counts = collections.Counter(corpus)
corpus = [word for word in corpus if word_counts[word] > 5]
print(f"Corpus object type: {type(corpus)}")
print(f"Original corpus size: {len(corpus)}")
print(f"First 7 words: {corpus[:7]}")

with open(config.CORPUS_PATH, 'wb') as f:
    pickle.dump(corpus, f)

# Download MS MARCO dataset
print("Downloading MS MARCO dataset...")
r = requests.get(
    "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/train-00000-of-00001.parquet"
)
with open("ms_marco_train.parquet", "wb") as f:
    f.write(r.content)

# Process parquet file
df = pd.read_parquet("ms_marco_train.parquet")
print(f"DataFrame columns: {df.columns.tolist()}")

# Extract text from the passage_text column directly from the screenshot
passage_texts = []
for idx, row in df.iterrows():
    passages = row.passages['passage_text']
    for passage in passages:
        passage_texts.append(passage)

print(f"Extracted {len(passage_texts)} passage texts")

# Process passages and add to corpus
ms_marco_words = []
for text in passage_texts:
    ms_marco_words.extend(preprocess(text))

# Filter out low frequency words
word_counts = collections.Counter(ms_marco_words)
ms_marco_words = [word for word in ms_marco_words if word_counts[word] > 5]

print(
    f"Extracted {len(ms_marco_words)} "
    "words from MS MARCO passages after filtering")

# Add to existing corpus
corpus.extend(ms_marco_words)
print(f"New corpus size: {len(corpus)}")

# Save updated corpus
with open(config.CORPUS_PATH, 'wb') as f:
    pickle.dump(corpus, f)


# Recreate vocabulary lookup tables
def create_lookup_tables(words):
    word_counts = collections.Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


words_to_ids, ids_to_words = create_lookup_tables(corpus)
print(f"Vocabulary size: {len(words_to_ids)}")
# Create tokens from updated corpus
tokens = [words_to_ids[word] for word in corpus]
print(f"Total tokens: {len(tokens)}")

print(type(tokens))
print(tokens[:7])

print(ids_to_words[5234]
      if 5234 in ids_to_words else "Index 5234 not in vocabulary")
print(words_to_ids.get('anarchism', 'Word not in vocabulary'))
print(words_to_ids.get('have', 'Word not in vocabulary'))

# Save updated vocabulary
with open(config.VOCAB_TO_ID_PATH, 'wb') as f:
    pickle.dump(words_to_ids, f)
with open(config.ID_TO_VOCAB_PATH, 'wb') as f:
    pickle.dump(ids_to_words, f)
