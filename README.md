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
