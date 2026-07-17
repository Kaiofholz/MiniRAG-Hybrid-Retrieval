from minirag.evidence import SentenceCandidateBuilder
from minirag.schemas import RetrievedChunk
import re

def test_sentence_candidate_builder_splits_chunks_into_candidates():
    builder = SentenceCandidateBuilder()

    chunks = [
        RetrievedChunk(
            chunk_id=1,
            text=(
                "William Shakespeare was born in Stratford-upon-Avon. "
                "He later became a playwright and poet."
            ),
            rerank_score=2.5,
        )
    ]

    candidates = builder.build(chunks)

    assert len(candidates) == 2
    assert candidates[0].chunk_id == 1
    assert candidates[0].sentence_id == 0
    assert "born in Stratford-upon-Avon" in candidates[0].text
    assert candidates[0].chunk_rerank_score == 2.5

    assert candidates[1].sentence_id == 1
    assert "playwright and poet" in candidates[1].text
