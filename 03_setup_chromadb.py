import argparse

import src.config as config
from src.database import build_chroma
from src.utils.model_loader import download_model


if __name__ == "__main__":
    # default reads 'ms_marco_docs.parquet' and indexes under 'ms_marco_docs'
    parser = argparse.ArgumentParser(description='Download models')
    parser.add_argument(
        '--download_sg',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--download_tt',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    args = parser.parse_args()

    # Download model if required
    download_model(
        args.download_sg, "skipgram-best", config.SKIPGRAM_CHECKPOINT_DIR
    )
    download_model(
        args.download_tt, "two-towers-best", config.TWOTOWERS_CHECKPOINT_DIR
    )
    build_chroma()
