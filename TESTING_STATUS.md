# Testing Status

This project now has CI-backed tests for the MiniRAG package.

## Covered areas

- Package imports
- Answer cache and LLM cache
- Query routing
- Basic extraction and date extraction
- Evidence selection
- Sentence candidate building
- Retrieval wrapper
- Hybrid retrieval / RRF fusion
- Chunk reranking
- Prompt building
- Grounding validation
- Mock generator
- Real LLM wrapper with fake LLM function
- Answer result factory
- Extractive answer validation
- Evidence pipeline
- Retrieval answer pipeline
- Full MiniRAG facade:
  - answer()
  - answer(debug=True)
  - clear_cache()
  - clear_llm_cache()
  - no retrieved evidence path

## CI

GitHub Actions runs pytest and package compilation on push and pull request.

## Current testing philosophy

The CI tests avoid external services and heavy model downloads. They use fake retrievers, fake LLM functions, and lightweight package components so the tests remain fast and deterministic.
