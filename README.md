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
- Qdrant-backed vector database retrieval in the updated version
  
## Project Versions

This repository contains two versions of the MiniRAG pipeline:

- `mini_rag.py`: Original version using local FAISS-based dense retrieval.
- `mini_rag_qdrant.py`: Updated version using Qdrant as the vector database backend for dense retrieval.

Both versions keep the same high-level RAG structure:

Dense retrieval + BM25 lexical retrieval → candidate union → cross-encoder reranking → sentence-level evidence scoring → evidence selection → extractive question answering.

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
- Improve project structure and modularity.
- Add clearer evaluation examples.
- Explore hybrid search and metadata filtering.
- Build a small demo interface or API.
