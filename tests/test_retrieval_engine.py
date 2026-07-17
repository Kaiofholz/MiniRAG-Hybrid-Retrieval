from minirag.retrieval import RetrievalEngine, RetrieverWrapper


def test_retrieval_engine_rrf_fuses_dense_and_bm25_results():
    def fake_dense_search(query, **kwargs):
        return [
            ("shared chunk", 0.90, 1),
            ("dense only chunk", 0.80, 2),
        ]

    def fake_bm25_search(query, **kwargs):
        return [
            ("shared chunk", 12.0, 1),
            ("bm25 only chunk", 8.0, 3),
        ]

    engine = RetrievalEngine(
        dense_retriever=RetrieverWrapper(fake_dense_search),
        bm25_retriever=RetrieverWrapper(fake_bm25_search),
        top_n_retrieval=3,
        retrieval_fusion="rrf",
    )

    result = engine.retrieve_candidates_with_details("Who was Shakespeare?")

    assert len(result.dense_results) == 2
    assert len(result.bm25_results) == 2
    assert len(result.retrieved) == 3

    top_chunk = result.retrieved[0]

    assert top_chunk.chunk_id == 1
    assert top_chunk.text == "shared chunk"
    assert top_chunk.dense_score == 0.90
    assert top_chunk.bm25_score == 12.0
    assert top_chunk.dense_rank == 1
    assert top_chunk.bm25_rank == 1
