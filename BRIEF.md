# Search Engine Project Brief

## Project Goal

Build a search engine that takes queries and returns the 5 most relevant documents using the Microsoft Machine Reading Comprehension (MS MARCO) dataset.

## Key Deliverables

1. Data preparation pipeline
2. Two Tower neural network architecture
3. Training implementation
4. Inference function

## Architecture

We'll implement a Two Tower model with:

- A word embedding layer (pretrained word2vec)
- Two RNN (MLP for the MVP) encoders running in parallel:
  - Query encoder
  - Document encoder

## Implementation Steps

1. **Dataset Preparation**

   - Source MS MARCO v1.1 from HuggingFace
   - Split into training, validation and test sets
   - Create triples of (query, relevant document, irrelevant document) using negative sampling
   - Tokenise all data

2. **Model Development**

   - Implement word embedding layer (pretrained, with weights frozen or fine-tuned)
   - Create dual RNN or MLP encoders (may use improved variants like GRU, LSTM or BiRNN)
   - Implement distance function for comparing encodings (cosine similarity)
   - Use triplet loss function during training

3. **Inference**
   - Pre-cache all document encodings
   - Encode incoming queries
   - Find and rank top 5 documents by similarity
   - Use approximate nearest neighbour algorithms for faster retrieval at scale
   - Implement batch processing for multiple simultaneous queries

## Key Concepts

- **Negative Sampling**: Randomly selecting irrelevant documents (same number as relevant ones) to make training manageable
- **Two Tower Architecture**: Separate encoders for queries and documents to handle their different structures
- **Triplet Loss**: Trains model to minimise distance between query and relevant documents while maximising distance to irrelevant ones
- **RNN (Recurrent Neural Network)**: Processes text sequentially to preserve word order and relationships
- **MLP (Multi-Layer Perceptron)**: A simpler feedforward neural network that can be used as an initial MVP before implementing the more complex RNN encoders
- **GRU (Gated Recurrent Unit)**: An RNN variant that better manages information flow using gates to improve learning of long-range dependencies
- **LSTM (Long Short-Term Memory)**: An RNN variant with additional cell state to preserve long-term information throughout the sequence
- **BiRNN (Bidirectional RNN)**: Processes sequences in both directions to capture context from both past and future tokens
- **Layer Normalisation**: Technique that scales neural network outputs to prevent vanishing/exploding gradients
- **Learning Rate Scheduling**: Automatically adjusts learning rate during training to improve convergence
- **Dropout**: Regularisation technique that randomly disables neurons during training to prevent overfitting

## Technical Constraints

- Use MS MARCO v1.1 from HuggingFace
- Train with triplet loss and appropriate margin hyperparameter
- Implement learning rate scheduling to improve convergence
- Use layer normalisation to prevent vanishing/exploding gradients
- Apply dropout (10-50%) for regularisation
- Return exactly 5 documents per query during inference

## Success Criteria

A working search engine that accurately retrieves the 5 most relevant documents for a given query, with good generalisation to unseen queries.
