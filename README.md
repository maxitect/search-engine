# Search Engine

A neural search engine implementing a two-tower architecture trained on the MS MARCO dataset. This engine efficiently encodes both queries and documents into the same vector space, allowing for fast and relevant document retrieval.

## Setup

1. Clone the repo

   ```bash
   conda env create -f environment.yml
   ```

1. Create environment from root:

   ```bash
   conda env create -f environment.yml
   ```

1. Activate environment:

   ```bash
   conda activate search-engine
   ```

1. Make sure you have access to the data repo at:
   ```
   https://huggingface.co/datasets/microsoft/ms_marco
   ```

## Architecture

The system uses a two-tower neural network approach:

- **Query tower**: Encodes search queries into dense vectors
- **Document tower**: Encodes documents into the same vector space
- Vector similarity between query and document embeddings determines relevance

## Training Process

The model is trained on MS MARCO using triplet loss with negative sampling:

1. For each query, we use:

   - Positive examples: ~10 relevant documents per query from MS MARCO
   - Negative examples: Randomly sampled irrelevant documents (using negative sampling)

2. Negative sampling approach:
   - From 600,000+ potential negative documents, we sample only ~10 per query
   - This creates balanced training triplets of (query, relevant document, irrelevant document)
   - Helps manage computational complexity while maintaining training effectiveness

## Benefits

- **Efficiency**: Embeddings can be pre-computed for the document corpus
- **Scalability**: Retrieval becomes a vector similarity search problem
- **Performance**: Captures semantic relationships beyond keyword matching

## Usage

[Instructions on how to use the search engine]

## Installation

[Installation instructions]

## Contributors

[@maxitect](https://github.com/maxitect)
[@kwokkenton](https://github.com/kwokkenton)
[@FilippoRomeo](https://github.com/FilippoRomeo)
