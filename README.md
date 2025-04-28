# Neural Search Engine

A neural search engine implementing a two-tower architecture trained on the MS MARCO dataset. This engine efficiently encodes both queries and documents into the same vector space, allowing for fast and relevant document retrieval.

## Architecture

The system uses a two-tower neural network approach:

- **Query tower**: Encodes search queries into dense vectors
- **Document tower**: Encodes documents into the same vector space
- **Vector similarity**: Determines relevance between queries and documents
- **ChromaDB**: Provides efficient vector storage and similarity search

The model is trained with:

- Pretrained word embeddings (using SkipGram on text8 corpus + MS MARCO)
- Dual RNN encoders with bidirectional GRU layers
- Triplet loss with negative sampling

## Project Structure

```
search-engine/
├── api/                      # Backend API and model code
│   ├── src/                  # Core search engine code
│   │   ├── config.py         # Configuration settings
│   │   ├── database.py       # ChromaDB interaction
│   │   ├── dataset.py        # MS MARCO dataset handling
│   │   ├── evaluate.py       # Model evaluation
│   │   ├── models/           # Neural network models
│   │   │   ├── skipgram.py   # Word embedding model
│   │   │   └── twotowers.py  # Two-tower architecture
│   │   └── utils/            # Utility functions
│   ├── app.py                # FastAPI application
│   ├── search_cli.py         # Command-line search tool
│   ├── 00_prep_data.py       # Download and transform all data
│   ├── 01a_prep_token.py     # Tokenise vocabulary datasets
│   ├── 01b_train_skipgram.py # Train skipgram model
│   ├── 02_train_twotowers.py # Train two towers model
│   ├── 03_eval_accuracy.py   # Evaluate two towers model accuracy
│   └── 04_setup_chromadb.py  # Setup script for ChromaDB
├── web-app/                  # Streamlit web interface
│   ├── .streamlit/           # Streamlit configuration
│   │   └── config.toml       # Theme and settings
│   ├── search_engine.py      # Streamlit app code
│   └── requirements.txt      # Web app dependencies
└── docker-compose.yml        # Docker configuration
```

## Setup Options

### 1. Using Docker (Recommended)

The easiest way to get started is using Docker:

1. **Clone the repository**:

   - **with https:**

   ```bash
   git clone https://github.com/maxitect/search-engine.git
   cd search-engine
   ```

   - **with ssh:**

   ```bash
   git clone git@github.com:maxitect/search-engine.git
   cd search-engine
   ```

2. **Create a `.env` file with your Weights & Biases API key**:

   - If you have trained the models using the other scripts, they will be saved and downloaded for you when running the container:

   ```
   WANDB_API_KEY=your_api_key_here
   DOWNLOAD_MODELS=true
   ```

3. **Build and start the containers**:

   ```bash
   docker-compose up --build
   ```

4. **Access the web interface**:
   Open http://localhost:8501 in your browser.

### 2. Development Setup

For local development without Docker:

1. **Create conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate search-engine
   ```

2. **Login to Weights and Biases**:

   ```bash
   wandb login
   ```

3. **Start training**:

   - **for skipgram:**

   ```bash
   cd api
   python 00_prep_data.py
   python 01a_prep_token.py
   python 01b_train_skipgram.py
   ```

   - **for two towers:**

   ```bash
   cd api
   python 00_prep_data.py
   python 02_train_twotowers.py --download_sg # download if skipgram best model is not local already
   python 03_eval_accuracy.py # run to evaluate two towers model accuracy on validation data
   ```

4. **Database setup**:

   ```bash
   cd api
   python 03_setup_chromadb.py --download_sg --download_tt
   ```

5. **Start the API**:

   ```bash
   uvicorn app:app --reload
   ```

6. **In a separate terminal, start the web app**:
   ```bash
   cd web-app
   streamlit run search_engine.py
   ```

## Usage

### Web Interface

The easiest way to use the search engine is through the web interface at http://localhost:8501.

### Command Line Interface

For quick searches without the web interface:

```bash
cd api
python search_cli.py "your search query" --top-k 5
```

### API Endpoints

The API is available at http://localhost:8000 with the following endpoints:

- `GET /`: Health check endpoint
- `GET /search?q=your+query&top_k=5`: Search endpoint

API documentation is available at http://localhost:8000/docs

## Training Process

The model is trained on MS MARCO using triplet loss with negative sampling:

1. **Query-Document pairs**: Uses relevant document pairs from MS MARCO
2. **Negative sampling**: Randomly samples irrelevant documents to create triplets
3. **Training stages**:
   - SkipGram embeddings: Trained on text8 corpus + MS MARCO documents
   - Two-Tower model: Fine-tuned on MS MARCO using pretrained embeddings

## Contributors

[@maxitect](https://github.com/maxitect)
[@kwokkenton](https://github.com/kwokkenton)
[@FilippoRomeo](https://github.com/FilippoRomeo)
[@ocmoney](https://github.com/ocmoney)
[@ocmoney](https://github.com/ocmoney)
