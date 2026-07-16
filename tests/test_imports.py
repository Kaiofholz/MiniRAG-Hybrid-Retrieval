def test_schema_imports():
    from minirag.schemas import AnswerResult, RetrievedChunk, SentenceCandidate

    chunk = RetrievedChunk(
        chunk_id=1,
        text="William Shakespeare was born in Stratford-upon-Avon."
    )

    result = AnswerResult(
        question="Where was Shakespeare born?",
        answer="Stratford-upon-Avon",
        supported=True,
        confidence=1.0,
        mode="extractive",
        evidence_sentences=[chunk.text],
        debug={},
    )

    assert result.supported is True
    assert "Stratford" in result.answer
