from minirag.evidence import EvidenceSelector
from minirag.schemas import SentenceCandidate, EvidenceSelectionResult



def test_evidence_selector_selects_strong_candidates():
    selector = EvidenceSelector(
        evidence_threshold=0.5,
        top_k_evidence=2,
    )

    candidates = [
        SentenceCandidate(
            chunk_id="1",
            sentence_id=0,
            text="Strong evidence sentence.",
            final_score=0.8,
        ),
        SentenceCandidate(
            chunk_id="2",
            sentence_id=0,
            text="Weak evidence sentence.",
            final_score=0.3,
        ),
    ]

    result = selector.select(candidates)

    assert result.is_sufficient is True
    assert result.confidence == 0.8
    assert len(result.evidence) == 1
    assert result.evidence[0].text == "Strong evidence sentence."


def test_evidence_selector_handles_empty_candidates():
    selector = EvidenceSelector(
        evidence_threshold=0.5,
        top_k_evidence=2,
    )

    result = selector.select([])

    assert result.is_sufficient is False
    assert result.confidence == 0.0
    assert result.evidence == []
