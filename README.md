# Mini RAG System (Movie Plots)

This repository contains a simple, end-to-end **Retrieval-Augmented Generation (RAG)** system built in a Jupyter Notebook. It uses **OpenAI Embeddings** for transforming text into vectors, **FAISS** for efficient vector search, and a **Large Language Model (LLM)** via the OpenAI Chat API for generating final, grounded answers.

The system is designed to answer questions about a small dataset of movie plots, retrieving relevant plot snippets before generating an informed answer.

## Key Components

* **Data Source**: A subset of the `wiki_movie_plots_deduped.csv` dataset, using the 'Title' and 'Plot' columns.
* **Chunking**: Movie plots are split into smaller chunks (up to 300 words) to improve retrieval granularity.
* **Embedding**: **`text-embedding-3-small`** is used to create vector representations of the plot chunks.
* **Vector Store**: **FAISS (`IndexFlatL2`)** is used to store and search the embeddings quickly.
* **Generation**: The **`gpt-4o-mini`** model is used to synthesize a final answer based on the user's query and the retrieved context chunks.

## Prerequisites

To run this notebook, you will need:

* **Python 3.x**
* **An OpenAI API Key**: This is required for generating embeddings and LLM responses.
* **Dataset**: The `wiki_movie_plots_deduped.csv` file, which is assumed to be available in the environment where the notebook is run (e.g., in the `/content/` directory in a Colab environment).

## Setup and Installation

### 1. Install Dependencies

The first cell in the notebook handles the required package installation:

```bash
!pip install openai faiss-cpu pandas tiktoken
````

### 2\. Set API Key

You must set your OpenAI API key as an environment variable for the script to function:

```python
import os
import openai

# Set your API key in your environment or replace the line below:
openai.api_key = os.getenv("OPENAI_API_KEY") 
```

## How to Use

The notebook follows a clear sequence of steps:

### 1\. Data Loading and Chunking

The code loads the movie plot data, selecting the first 200 entries and splitting their plots into manageable chunks.

### 2\. Embedding Creation

Each plot chunk is converted into a vector embedding using the OpenAI API.

```python
embeddings = [get_embedding(c['chunk']) for c in tqdm(chunks)]
embeddings = np.array(embeddings).astype("float32")
```

### 3\. FAISS Indexing

The embeddings are indexed using **FAISS** for fast similarity search. The total number of chunks (vectors) is printed after indexing.

### 4\. Search and RAG Functionality

The core logic is contained in the `search` and `generate_answer` functions:

  * The **`search`** function takes a query, converts it to an embedding, and uses the FAISS index to find the top **`k`** most similar plot chunks.
  * The **`generate_answer`** function takes the query and the retrieved context chunks, constructs a prompt, and sends it to `gpt-4o-mini` with a specific **JSON schema** to get a structured, grounded answer.

### Example Query

You can test the system with a question like the following:

```python
query = "Which movie involves a jewel heist planned during a theater performance with explosions as cover?"
result = generate_answer(query)
print(json.dumps(result, indent=2))
```

This query should successfully retrieve context from the movie **"Manhattan Madness"** and use it to form the final answer.

```
