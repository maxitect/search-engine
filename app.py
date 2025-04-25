import logging
import os
import sys

import chromadb
import streamlit as st
import wandb
from pydantic import BaseModel, Field

from create_db import CONFIG_OPTIONS
from inference import setup_semantics_embedder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

embedding_methods = CONFIG_OPTIONS

if len(sys.argv) > 1:
    if sys.argv[1] == 'docker':
        logger.info('Running in Docker, login using Docker secrets')
        from huggingface_hub._login import _login

    # Sets up API Keys from Docker Compose
    with open("/run/secrets/HF_TOKEN") as f:
        HF_TOKEN = f.read().strip()

    with open("/run/secrets/WANDB_API_KEY") as f:
        WANDB_API_KEY = f.read().strip()

    _login(token=HF_TOKEN, add_to_git_credential=False)
    wandb.login(key=WANDB_API_KEY)
else:
    logger.info('Running in local, login using API keys in terminal.')

@st.cache_resource
def setup():
    chroma_client = chromadb.PersistentClient()
    logger.info('Setting up the embedding models...')
    semantics_embedders = {}
    for method in embedding_methods[1:]:
        semantics_embedders[method] = setup_semantics_embedder(method)
    return chroma_client, semantics_embedders


chroma_client, semantics_embedders = setup()

st.title('Search Engine Demo 🕵')

# Make dropdown in the sidebar for selecting the embedding method
with st.sidebar:
    config = st.selectbox(
        'Select the embedding method:', CONFIG_OPTIONS,
    )

if config == CONFIG_OPTIONS[0]:
    logger.info('Using Chroma')
    collection = chroma_client.get_or_create_collection(name=f'train_docs_{config}')
else:
    logger.info(f'Using {config}')
    collection = chroma_client.get_or_create_collection(
        name=f'train_docs_{config}',
    )

# Add a text input for the search query
query = st.text_input('Enter your search query:', '')

# Add a slider for selecting number of results
k = st.slider(
    'Number of results to return:',
    min_value=1, max_value=10, value=2,
)

if query:
    logger.info(f"Received query: '{query}', k={k}")
    if config == CONFIG_OPTIONS[0]:
        # Chroma will embed this for you
        results = collection.query(
            query_texts=[query],
            n_results=k,
        )
    else:
        # Embed the query using the model
        query_embedding = semantics_embedders[config].embed_query(
            query,
        ).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,  # how many results to return
        )
    # Display results
    st.subheader('Search Results:')
    for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0])):
        st.write(f'### Result {i+1} (Score: {1-score:.4f})')
        st.write(doc)
        st.write('---')
