import os

# Model parameters
EMBEDDING_DIM = 256
VOCAB_SIZE = 140976
MAX_TITLE_LENGTH = 64

# Training parameters
SKIPGRAM_EPOCHS = 15
SKIPGRAM_BATCH_SIZE = 1024
SKIPGRAM_LR = 0.002
SKIPGRAM_LR_SCHEDULE = "cosine"
SKIPGRAM_WARMUP_STEPS = 1000

TWOTOWERS_EPOCHS = 20
TWOTOWERS_BATCH_SIZE = 64
TWOTOWERS_LR = 0.0005
TWOTOWERS_LR_SCHEDULE = "linear_with_warmup"
TWOTOWERS_WARMUP_RATIO = 0.1
MARGIN = 0.3
DROPOUT_RATE = 0.2

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
