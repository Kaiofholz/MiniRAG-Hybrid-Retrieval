from minirag.extraction import ExtractiveAnswerValidator
from minirag.schemas import SentenceCandidate


def test_extractive_answer_validator_accepts_short_supported_answer():
    validator = ExtractiveAnswerValidator()

    source = SentenceCandidate(
        chunk_id=1,
        sentence_id=0,
        text="Shakespeare's twins were Hamnet and Judith.",
    )

    assert validator.is_valid(
        question="Who were Shakespeare's twins?",
        answer="Hamnet and Judith",
        source_evidence=source,
    ) is True


def test_extractive_answer_validator_rejects_empty_answer():
    validator = ExtractiveAnswerValidator()

    source = SentenceCandidate(
        chunk_id=1,
        sentence_id=0,
        text="Shakespeare's twins were Hamnet and Judith.",
    )

    assert validator.is_valid(
        question="Who were Shakespeare's twins?",
        answer="",
        source_evidence=source,
    ) is False


def test_extractive_answer_validator_rejects_favorite_without_preference_evidence():
    validator = ExtractiveAnswerValidator()

    source = SentenceCandidate(
        chunk_id=1,
        sentence_id=0,
        text="William Shakespeare wrote Hamlet and Macbeth.",
    )

    assert validator.is_valid(
        question="What was Shakespeare's favorite play?",
        answer="Hamlet",
        source_evidence=source,
    ) is False
