import collections
import pickle
import pandas as pd

import src.config as config
from src.utils.tokenise import create_lookup_tables, preprocess


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

# Read all three parquet files and combine into one DataFrame
df = pd.read_parquet("ms_marco_docs.parquet")

ms_marco_words = []
for passage in df.values:
    ms_marco_words.extend(preprocess(passage))

print(f"Extracted {len(df)} passage texts")

# Filter out low frequency words
word_counts = collections.Counter(ms_marco_words)
ms_marco_words = [word for word in ms_marco_words if word_counts[word] > 5]

print(
    f"Extracted {len(ms_marco_words)} "
    "words from MS MARCO passages after filtering"
)

# Add to existing corpus
corpus.extend(ms_marco_words)
print(f"New corpus size: {len(corpus)}")

# Save updated corpus
with open(config.CORPUS_PATH, 'wb') as f:
    pickle.dump(corpus, f)


words_to_ids, ids_to_words = create_lookup_tables(corpus)
print(f"Vocabulary size: {len(words_to_ids)}")
tokens = [words_to_ids[word] for word in corpus]
print(f"Total tokens: {len(tokens)}")

print(f"Token object type: {type(tokens)}")
print(f"First 7 tokens: {tokens[:7]}")

print(f"This word should be 'beaches': {ids_to_words[5234]}")
print(f"Index of word 'anarchism': {words_to_ids.get('anarchism')}")
print(f"Index of word 'have': {words_to_ids.get('have')}")

# Save updated vocabulary
with open(config.VOCAB_TO_ID_PATH, 'wb') as f:
    pickle.dump(words_to_ids, f)
with open(config.ID_TO_VOCAB_PATH, 'wb') as f:
    pickle.dump(ids_to_words, f)
