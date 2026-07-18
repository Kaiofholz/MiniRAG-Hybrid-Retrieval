from minirag.answering import AnswerFinalizer, ComparisonAnswerer
from minirag.cache import AnswerCache
from minirag.schemas import AnswerResult
import re



def test_comparison_answerer_compares_two_supported_numeric_answers():
    calls = []

    def fake_answer_fn(question, use_cache=True):
        calls.append((question, use_cache))

        answers = {
            "What is the max input current of GW10K-ET?": AnswerResult(
                question=question,
                answer="14 A",
                supported=True,
                confidence=1.0,
                mode="structured_lookup",
                evidence_sentences=["GW10K-ET max input current is 14 A."],
                debug={},
            ),
            "What is the max input current of GW8K-ET?": AnswerResult(
                question=question,
                answer="12 A",
                supported=True,
                confidence=1.0,
                mode="structured_lookup",
                evidence_sentences=["GW8K-ET max input current is 12 A."],
                debug={},
            ),
        }

        return answers[question]

    answerer = ComparisonAnswerer(
        answer_fn=fake_answer_fn,
        answer_finalizer=AnswerFinalizer(AnswerCache()),
    )

    result = answerer.answer(
        question="Is GW10K-ET's max input current higher than GW8K-ET's?",
        debug_info={},
    )

    assert result.supported is True
    assert result.mode == "comparison"
    assert "Yes." in result.answer
    assert "14A" in result.answer
    assert "12A" in result.answer
    assert result.evidence_sentences == [
        "GW10K-ET max input current is 14 A.",
        "GW8K-ET max input current is 12 A.",
    ]

    assert calls == [
        ("What is the max input current of GW10K-ET?", False),
        ("What is the max input current of GW8K-ET?", False),
    ]


def test_comparison_answerer_handles_ambiguous_current_question():
    def fake_answer_fn(question, use_cache=True):
        raise AssertionError("answer_fn should not be called for ambiguous comparison")

    answerer = ComparisonAnswerer(
        answer_fn=fake_answer_fn,
        answer_finalizer=AnswerFinalizer(AnswerCache()),
    )

    result = answerer.answer(
        question="Is GW10K-ET's current higher than GW8K-ET's?",
        debug_info={},
    )

    assert result.supported is False
    assert result.mode == "clarification_needed"
    assert "max input current" in result.answer
    assert "max short-circuit current" in result.answer
