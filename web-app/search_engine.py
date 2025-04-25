import streamlit as st
import requests
import os
from typing import Dict, Any

# Configuration
SEARCH_API_URL = os.environ.get("SEARCH_API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("Neural Search Engine")
st.markdown("### Search through the MS MARCO dataset using neural embeddings")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        """
        This search engine uses a Two-Tower neural network architecture
        to encode both queries and documents. The model encodes your
        search query and finds the most semantically similar documents.

        The search is powered by:
        - Neural embeddings (Two-Tower architecture)
        - ChromaDB as the vector database
        - FastAPI backend
        """
    )

    st.header("Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

# Search form
with st.form(key="search_form"):
    query = st.text_input("Enter your search query",
                          placeholder="What is machine learning?")
    submit_button = st.form_submit_button(label="Search")

# Function to call the search API


def search(query: str, top_k: int = 5) -> Dict[str, Any]:
    try:
        response = requests.get(
            f"{SEARCH_API_URL}/search",
            params={"q": query, "top_k": top_k},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to search API: {e}")
        return {"query": query, "results": []}


# Perform search when the form is submitted
if submit_button and query:
    with st.spinner("Searching..."):
        search_results = search(query, top_k)

    # Display the results
    if search_results and "results" in search_results:
        st.subheader(f"Search Results for: '{search_results['query']}'")

        if not search_results["results"]:
            st.info("No results found.")

        for i, result in enumerate(search_results["results"]):
            with st.container():
                col1, col2 = st.columns([1, 4])

                with col1:
                    # Display rank and similarity score
                    st.metric(
                        label=f"Result #{i+1}",
                        value=f"{result['similarity']:.4f}",
                        delta="similarity"
                    )
                    st.caption(f"ID: {result['id']}")

                with col2:
                    # Display document content in an expandable element
                    with st.expander("View Document", expanded=True):
                        st.markdown(result['document'])

                # Add a separator between results
                st.divider()
    else:
        st.error("Failed to get search results.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using PyTorch, ChromaDB, FastAPI, and Streamlit")
