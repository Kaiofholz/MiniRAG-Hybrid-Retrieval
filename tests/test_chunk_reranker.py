from minirag.retrieval import ChunkReranker
from minirag.schemas import RetrievedChunk


def test_chunk_reranker_uses_cross_encoder_scores():
    def fake_predict(pairs):
        assert pairs == [
            ("Who was Shakespeare?", "Chunk A"),
            ("Who was Shakespeare?", "Chunk B"),
        ]
        return [0.2, 0.9]

    reranker = ChunkReranker(
        cross_encoder_predict_fn=fake_predict,
        top_n_rerank=1,
    )

    chunks = [
        RetrievedChunk(chunk_id=1, text="Chunk A"),
        RetrievedChunk(chunk_id=2, text="Chunk B"),
    ]

    result = reranker.rerank_chunks("Who was Shakespeare?", chunks)

    assert len(result) == 1
    assert result[0].chunk_id == 2
    assert result[0].rerank_score == 0.9


def test_chunk_reranker_fallback_uses_hybrid_score_without_cross_encoder():
    reranker = ChunkReranker(
        cross_encoder_predict_fn=None,
        top_n_rerank=2,
    )

    chunks = [
        RetrievedChunk(chunk_id=1, text="Chunk A", dense_score=0.1, bm25_score=0.1),
        RetrievedChunk(chunk_id=2, text="Chunk B", dense_score=0.8, bm25_score=0.6),
    ]

    result = reranker.rerank_chunks("Who was Shakespeare?", chunks)

    assert result[0].chunk_id == 2
    assert result[0].rerank_score == 0.7
