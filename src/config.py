import os

# Model parameters
EMBEDDING_DIM = 256
VOCAB_SIZE = 140976
MAX_TITLE_LENGTH = 64

# Skipgram training parameters
SKIPGRAM_EPOCHS = 15
SKIPGRAM_BATCH_SIZE = 512
SKIPGRAM_LR = 0.002
SKIPGRAM_LR_SCHEDULE = "cosine"
SKIPGRAM_WARMUP_STEPS = 1000
NEGATIVE_SAMPLES = 15
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Two towers training parameters
TWOTOWERS_EPOCHS = 20
TWOTOWERS_BATCH_SIZE = 64
TWOTOWERS_LR = 0.0003
TWOTOWERS_LR_SCHEDULE = "linear_with_warmup"
TWOTOWERS_WARMUP_RATIO = 0.1
TWOTOWERS_PATIENCE = 3
MARGIN = 0.3
DROPOUT_RATE = 0.3
MAX_QUERY_LEN = 20
MAX_DOC_LEN = 200
HIDDEN_DIM = 256
OUTPUT_DIM = 128
WEIGHT_DECAY = 1e5

# Model paths
SKIPGRAM_CHECKPOINT_DIR = 'models/skipgram'
TWOTOWERS_CHECKPOINT_DIR = 'models/two-towers'

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
TWOTOWERS_BEST_MODEL_PATH = os.path.join(
    TWOTOWERS_CHECKPOINT_DIR, 'best_model.pth'
)

# API settings
MODEL_VERSION = "0.1.0"
LOG_DIR_PATH = "/var/log/app"
LOG_PATH = f"{LOG_DIR_PATH}/V-{MODEL_VERSION}.log"

# Ensure directories exist
os.makedirs(SKIPGRAM_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TWOTOWERS_CHECKPOINT_DIR, exist_ok=True)
