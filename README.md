# MiniRAG: Hybrid Retrieval and Reranking QA System

## Purpose

This is a learning-oriented Python project built to understand how retrieval-augmented question answering works beyond simple framework usage.

The project focuses on retrieval quality, evidence selection, debugging, and extractive answer reliability.

## Core Components

- Dense retrieval with sentence-transformer embeddings
- BM25 lexical retrieval
- Optional Qdrant-backed dense retrieval backend for vector database integration
- Candidate union from dense and lexical retrieval results
- Cross-encoder reranking
- Sentence-level evidence scoring
- Extractive answer generation
- Confidence scoring based on selected evidence
- Debugging outputs for retrieved chunks, ranked sentences, selected evidence, and failure cases
- Small experimental character-level Transformer generator for learning purposes
  
## Project Versions

This repository contains two versions of the MiniRAG pipeline:

- `mini_rag.py`: Original version using local FAISS-based dense retrieval.
- `mini_rag_qdrant.py`: Updated version using Qdrant as the vector database backend for dense retrieval.

Both versions keep the same high-level RAG structure:

Dense retrieval + BM25 lexical retrieval → candidate union → cross-encoder reranking → sentence-level evidence scoring → evidence selection → extractive question answering.

## What Changed in the Qdrant Version

The original `mini_rag.py` version uses FAISS as a local dense retrieval index.

The updated `mini_rag_qdrant.py` version replaces the FAISS dense retriever with a Qdrant-backed vector database retriever. SentenceTransformer embeddings are still generated locally, but the vectors and chunk payloads are stored in a Qdrant collection.

The rest of the RAG pipeline remains intentionally similar:

- BM25 lexical retrieval is still used as the sparse retrieval component.
- Dense and lexical results are still merged through candidate union.
- A cross-encoder still reranks the unioned candidates.
- Sentence-level scoring, evidence selection, and extractive answering remain unchanged.

This makes it easier to compare how the dense retrieval backend changes while keeping the rest of the pipeline stable.

## Data

This project uses small sample text passages for retrieval experiments. No private, customer, or company data is included.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project from the repository root:

Run the original FAISS-based version:
```bash
python mini_rag.py
```
Run the Qdrant-backed version:
```bash
python mini_rag_qdrant.py
```

The sample corpus should be located at:

```text
data/sample_corpus.txt
```

## Example Evaluation Output

The script runs a small evaluation set and reports metrics such as:

- Answer accuracy
- Evidence accuracy
- Overall accuracy
- Sentence recall
- Sentence MRR

## Example Question

Example query:

```text
What was John Shakespeare's profession?
```
Expected behavior:

```text
The system retrieves candidate chunks using dense retrieval and BM25, merges the candidates, reranks them with a cross-encoder, selects evidence sentences, and returns an extractive answer grounded in the selected evidence.
```
Example answer:

```text
alderman and a successful glover
```
Example evidence:

```text
William Shakespeare was the son of John Shakespeare, an alderman and a successful glover...
```
The exact output may vary slightly depending on model versions, local environment, and retrieval scores.

## Why I Built This

I built this project to understand the internal mechanisms of RAG systems instead of only using high-level frameworks.

The main learning goal was to inspect how retrieval results are selected, reranked, scored, and used as evidence for question answering.

## Key Learnings

- Dense retrieval and BM25 capture different relevance signals.
- Candidate quality strongly affects reranking quality.
- Reranking can improve precision but cannot fully recover from weak initial retrieval.
- Extractive answering can be more reliable than weak generation when grounding is important.
- Debugging RAG requires inspecting chunks, sentences, scores, selected evidence, and failure cases.

## Current Limitations

- This is not a production-ready RAG system.
- The small experimental Transformer generator is weak and mainly used for learning.
- The current focus is retrieval quality, evidence selection, and explainability rather than deployment.
- The code is still being improved as part of an ongoing learning process.

## Next Steps

- Compare FAISS-based local retrieval with Qdrant-backed vector retrieval.
- Add a small `compare_retrievers.py` script to inspect retrieval overlap, ranking differences, and score behavior.
- Improve project structure and modularity.
- Add clearer evaluation examples.
- Explore hybrid search and metadata filtering.
- Build a small demo interface or API.

## Run the Offline Demo

The repository includes a lightweight offline demo that uses fake retrievers instead of external models or APIs.

Run it from the repository root with:

```bash
PYTHONPATH=. python examples/offline_minirag_demo.py
```

## Installation for Development

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

This installs the local minirag package together with test tooling such as pytest.

Then run all checks with:

make check


## Module Notes

Special-route logic is separated from the main answer pipeline:

- `minirag/answering.py` contains the main answer pipeline and `MiniRAG` facade.
- `minirag/special_routes.py` contains comparison and structured-spec route handlers.
- `minirag/routing.py` contains lightweight routing and parsing helpers.
