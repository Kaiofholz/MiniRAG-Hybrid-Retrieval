# MiniRAG: Hybrid Retrieval and Reranking QA System

## Purpose

This is a learning-oriented Python project built to understand how retrieval-augmented question answering works beyond simple framework usage.

The project focuses on retrieval quality, evidence selection, debugging, and extractive answer reliability.

## Core Components

- Dense retrieval with sentence-transformer embeddings
- BM25 lexical retrieval
- Candidate union from dense and lexical retrievers
- Cross-encoder reranking
- Sentence-level evidence scoring
- Extractive answer generation
- Confidence scoring based on selected evidence
- Debugging outputs for retrieved chunks, ranked sentences, selected evidence, and failure cases
- Small experimental character-level Transformer generator for learning purposes

## Data

This project uses small sample text passages for retrieval experiments. No private, customer, or company data is included.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
Run the project from the repository root:
python mini_rag.py
The sample corpus should be located at:
data/sample_corpus.txt

## Example Evaluation Output

The script runs a small evaluation set and reports metrics such as:

Answer accuracy
Evidence accuracy
Overall accuracy
Sentence recall
Sentence MRR
## Why I Built This

I built this project to understand the internal mechanisms of RAG systems instead of only using high-level frameworks.

The main learning goal was to inspect how retrieval results are selected, reranked, scored, and used as evidence for question answering.

## Key Learnings
Dense retrieval and BM25 capture different relevance signals.
Candidate quality strongly affects reranking quality.
Reranking can improve precision but cannot fully recover from weak initial retrieval.
Extractive answering can be more reliable than weak generation when grounding is important.
Debugging RAG requires inspecting chunks, sentences, scores, selected evidence, and failure cases.

## Current Limitations
This is not a production-ready RAG system.
The small experimental Transformer generator is weak and mainly used for learning.
The current focus is retrieval quality, evidence selection, and explainability rather than deployment.
The code is still being improved as part of an ongoing learning process.

## Next Steps
Add Qdrant as a vector database backend.
Improve project structure and modularity.
Add clearer evaluation examples.
Explore hybrid search and metadata filtering.
Build a small demo interface or API.
