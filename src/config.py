import os

# Model parameters
EMBEDDING_DIM = 128
VOCAB_SIZE = 63642
MAX_TITLE_LENGTH = 100

# Training parameters
SKIPGRAM_EPOCHS = 5
SKIPGRAM_BATCH_SIZE = 512
SKIPGRAM_LR = 0.003

REGRESSOR_EPOCHS = 10
REGRESSOR_BATCH_SIZE = 128
REGRESSOR_LR = 0.001

# Model paths
SKIPGRAM_CHECKPOINT_DIR = './models/skipgram'
DOMAIN_CHECKPOINT_DIR = './models/domain'
REGRESSOR_CHECKPOINT_DIR = './models/regressor'

# File paths
CORPUS_PATH = os.path.join(
    SKIPGRAM_CHECKPOINT_DIR, 'corpus.pkl'
)
VOCAB_TO_ID_PATH = os.path.join(
    SKIPGRAM_CHECKPOINT_DIR, 'tkn_words_to_ids.pkl'
)
ID_TO_VOCAB_PATH = os.path.join(
    SKIPGRAM_CHECKPOINT_DIR, 'tkn_ids_to_words.pkl'
)

SKIPGRAM_BEST_MODEL_PATH = os.path.join(
    SKIPGRAM_CHECKPOINT_DIR, 'best_model.pth')
DOMAIN_MAPPING_PATH = os.path.join(DOMAIN_CHECKPOINT_DIR, 'domain_mapping.pth')
REGRESSOR_BEST_MODEL_PATH = os.path.join(
    REGRESSOR_CHECKPOINT_DIR, 'best_model.pth'
)

# API settings
MODEL_VERSION = "0.1.0"
LOG_DIR_PATH = "/var/log/app"
LOG_PATH = f"{LOG_DIR_PATH}/V-{MODEL_VERSION}.log"

# Ensure directories exist
os.makedirs('./models/skipgram', exist_ok=True)
os.makedirs('./models/domain', exist_ok=True)
os.makedirs('./models/regressor', exist_ok=True)
