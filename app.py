import os

import chromadb
import streamlit as st

# import torch

# from engine.data.ms_marco import load_ms_marco
# from engine.text import setup_language_models

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection(name='train_docs')

# Load the MS MARCO dataset
# train_ds = load_ms_marco()['train']

# # Initialize language models
# tokeniser, w2v_model = setup_language_models()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# w2v_model = w2v_model.to(device)

# def embed_string(q: str, tokeniser, w2v_model, device):
#     """Embed a string into a mean-pooled word embedding."""
#     with torch.no_grad():
#         token = tokeniser.tokenise_string(q)
#         embedding = w2v_model.forward(torch.tensor(token).to(device)).mean(dim=0)
#     return embedding.cpu().numpy()


def main():
    st.title('Search Engine Demo 🕵')

    # Make dropdown in the sidebar for selecting the embedding method
    with st.sidebar:
        embedding_method = st.selectbox(
            'Select the embedding method:', ['Chroma', 'Word2Vec'],
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

        results = collection.query(
            query_texts=[query],  # Chroma will embed this for you
            n_results=k,  # how many results to return
        )
        # Display results
        st.subheader('Search Results:')
        for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0])):
            st.write(f'### Result {i+1} (Score: {1-score:.4f})')
            st.write(doc)
            st.write('---')


if __name__ == '__main__':
    main()
