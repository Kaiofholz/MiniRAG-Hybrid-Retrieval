from minirag.extraction import DateExtractor
from minirag.schemas import SentenceCandidate


def test_date_extractor_extracts_date_from_text():
    extractor = DateExtractor()

    result = extractor.extract_date_answer(
        question="When was Shakespeare born?",
        text="William Shakespeare was born on 23 April 1564.",
    )

    assert result is not None
    assert "23 April 1564" in result


def test_date_extractor_extracts_date_with_source():
    extractor = DateExtractor()

    evidence = [
        SentenceCandidate(
            chunk_id=1,
            sentence_id=0,
            text="William Shakespeare was born on 23 April 1564.",
            final_score=0.9,
        )
    ]

    answer, source = extractor.extract_date_with_source(
        question="When was Shakespeare born?",
        evidence_sentences=evidence,
    )

    assert answer is not None
    assert "23 April 1564" in answer
    assert source is evidence[0]
