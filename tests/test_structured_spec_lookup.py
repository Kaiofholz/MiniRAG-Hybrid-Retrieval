from minirag.answering import AnswerFinalizer, StructuredSpecLookupHandler
from minirag.cache import AnswerCache
from minirag.routing import StructuredSpecQueryParser


def build_handler():
    records = [
        {
            "model": "GW10K-ET",
            "parameter": "max_input_current",
            "value": "14",
            "unit": " A",
            "source_text": "GW10K-ET max input current is 14 A.",
        }
    ]

    return StructuredSpecLookupHandler(
        spec_records=records,
        structured_spec_query_parser=StructuredSpecQueryParser(),
        answer_finalizer=AnswerFinalizer(AnswerCache()),
    )


def test_structured_spec_lookup_finds_exact_parameter():
    handler = build_handler()

    lookup = handler.structured_spec_lookup(
        "What is the max input current of GW10K-ET?"
    )

    assert lookup["status"] == "found"
    assert lookup["model"] == "GW10K-ET"
    assert lookup["parameter"] == "max_input_current"
    assert lookup["value"] == "14"
    assert lookup["unit"] == " A"


def test_structured_spec_lookup_answer_returns_supported_result():
    handler = build_handler()
    debug_info = {}

    result = handler.answer(
        question="What is the max input current of GW10K-ET?",
        debug_info=debug_info,
    )

    assert result.answer == "14 A"
    assert result.supported is True
    assert result.confidence == 1.0
    assert result.mode == "structured_lookup"
    assert result.evidence_sentences == [
        "GW10K-ET max input current is 14 A."
    ]
    assert debug_info["structured_lookup"]["status"] == "found"


def test_structured_spec_lookup_handles_ambiguous_current_question():
    handler = build_handler()

    result = handler.answer(
        question="What is the current of GW10K-ET?",
        debug_info={},
    )

    assert result.supported is False
    assert result.mode == "clarification_needed"
    assert "max input current" in result.answer
    assert "max short-circuit current" in result.answer
