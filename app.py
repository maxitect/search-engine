import os

import chromadb
import streamlit as st
from inference import setup_semantics_embedder

@st.cache_resource
def setup():
    chroma_client = chromadb.PersistentClient()
    semantics_embedder = setup_semantics_embedder('gensim_embeddings')
    return chroma_client, semantics_embedder

chroma_client, semantics_embedder = setup()

st.title('Search Engine Demo 🕵')

# Make dropdown in the sidebar for selecting the embedding method
with st.sidebar:
    embedding_method = st.selectbox(
        'Select the embedding method:', ['Chroma', 'Gensim'],
    )

if embedding_method == 'Chroma':
    collection = chroma_client.get_or_create_collection(name='train_docs')
else:
    collection = chroma_client.get_or_create_collection(
        name='train_docs_gensim_embeddings',
    )

# Add a text input for the search query
query = st.text_input('Enter your search query:', '')

# Add a slider for selecting number of results
k = st.slider(
    'Number of results to return:',
    min_value=1, max_value=10, value=2,
)

if query:
    if embedding_method == 'Chroma':
        # Chroma will embed this for you
        results = collection.query(
            query_texts=[query],  
            n_results=k,  
        )
    else:
        # Embed the query using the model
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