import os

import chromadb
import streamlit as st
from inference import setup_semantics_embedder
# import torch

# from engine.data.ms_marco import load_ms_marco
# from engine.text import setup_language_models

# Initialize ChromaDB client and collection


@st.cache_resource
def setup():
    chroma_client = chromadb.PersistentClient()
    semantics_embedder = setup_semantics_embedder()
    return chroma_client, semantics_embedder

# Load the MS MARCO dataset
# train_ds = load_ms_marco()['train']


chroma_client, semantics_embedder = setup()

st.title('Search Engine Demo 🕵')

# Make dropdown in the sidebar for selecting the embedding method
with st.sidebar:
    embedding_method = st.selectbox(
        'Select the embedding method:', ['Chroma', 'Word2Vec'],
    )

if embedding_method == 'Chroma':
    collection = chroma_client.get_or_create_collection(name='train_docs')
else:
    collection = chroma_client.get_or_create_collection(
        name='train_docs_my_embeddings',
    )

# Add a text input for the search query
query = st.text_input('Enter your search query:', '')

# Add a slider for selecting number of results
k = st.slider(
    'Number of results to return:',
    min_value=1, max_value=10, value=2,
)

if query:
    # # Embed the query
    # if use_chroma:
    #     # query_embedding = embed_string(query, tokeniser, w2v_model, device)
    #     query_embedding = None
    # else:
    #     query_embedding = embed_string(query, tokeniser, w2v_model, device)
    # Search the collection
    # results = collection.query(
    #     query_embeddings=[query_embedding.tolist()],
    #     n_results=k
    # )
    if embedding_method == 'Chroma':
        results = collection.query(
            query_texts=[query],  # Chroma will embed this for you
            n_results=k,  # how many results to return
        )
    else:
        query_embedding = semantics_embedder.embed_query(query).tolist()
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
