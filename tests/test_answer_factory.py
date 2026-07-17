from minirag.cache import AnswerCache
from minirag.answering import (
    AnswerFinalizer,
    AnswerResultFactory,
    SafetyAbstentionPolicy,
)
from minirag.formatting import EvidenceFormatter
from minirag.schemas import SentenceCandidate


def build_factory():
    cache = AnswerCache()
    finalizer = AnswerFinalizer(answer_cache=cache)

    factory = AnswerResultFactory(
        evidence_formatter=EvidenceFormatter(),
        safety_abstention_policy=SafetyAbstentionPolicy(),
        answer_finalizer=finalizer,
    )

    return factory, cache


def test_answer_finalizer_saves_result_to_cache():
    cache = AnswerCache()
    finalizer = AnswerFinalizer(answer_cache=cache)

    result = finalizer.finalize_answer_result(
        question="Who were Shakespeare's twins?",
        answer="Hamnet and Judith",
        supported=True,
        confidence=1.0,
        mode="extractive",
        evidence_sentences=["The twins were Hamnet and Judith."],
        debug_info={"route": "retrieval"},
    )

    cached = cache.get("who were shakespeare's twins?")

    assert cached is result
    assert result.answer == "Hamnet and Judith"
    assert result.supported is True


def test_answer_result_factory_builds_insufficient_evidence_answer():
    factory, _ = build_factory()

    evidence = [
        SentenceCandidate(
            chunk_id=1,
            sentence_id=0,
            text="Weak evidence.",
            final_score=0.2,
        )
    ]

    result = factory.answer_insufficient_evidence(
        question="What was Shakespeare's favorite color?",
        evidence=evidence,
        confidence=0.2,
        debug_info={"risk_level": "low"},
    )

    assert result.answer == "Not enough evidence."
    assert result.supported is False
    assert result.mode == "abstain"
    assert result.evidence_sentences == ["Weak evidence."]


def test_answer_result_factory_builds_high_risk_safety_answer():
    factory, _ = build_factory()

    evidence = [
        SentenceCandidate(
            chunk_id=1,
            sentence_id=0,
            text="Weak wiring evidence.",
            final_score=0.2,
        )
    ]

    result = factory.answer_insufficient_evidence(
        question="Can I connect neutral and phase together?",
        evidence=evidence,
        confidence=0.2,
        debug_info={"risk_level": "high"},
    )

    assert result.supported is False
    assert result.mode == "safety_abstain"
    assert "qualified electrician" in result.answer
