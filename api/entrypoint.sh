#!/bin/bash
set -e

# Log in to Wandb if credentials are provided
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to Weights & Biases"
    wandb login $WANDB_API_KEY
else
    echo "WANDB_API_KEY not provided, skipping login"
fi

# Run the ChromaDB setup script if requested
if [ "$DOWNLOAD_MODELS" = "true" ]; then
    echo "Downloading models and setting up ChromaDB"
    python 00_train_data.py
    python 01a_train_token.py
    python 03_setup_chromadb.py --download_sg --download_tt
else
    echo "Skipping model download"
    python 03_setup_chromadb.py
fi

# Start the FastAPI application
echo "Starting Search API"
exec uvicorn app:app --host 0.0.0.0 --port 8000