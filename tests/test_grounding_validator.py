from minirag.formatting import PromptBuilder
from minirag.generation import GroundingValidator
import re

def test_grounding_validator_accepts_clean_abstention():
    validator = GroundingValidator(
        prompt_builder=PromptBuilder(),
        support_threshold=0.6,
    )

    validation = validator.validate(
        answer="The provided evidence is insufficient.",
        evidence_sentences=[],
    )

    assert validation["status"] == "abstained"
    assert validation["has_answer"] is False


def test_grounding_validator_detects_missing_citation():
    validator = GroundingValidator(
        prompt_builder=PromptBuilder(),
        support_threshold=0.6,
    )

    validation = validator.validate(
        answer="Shakespeare was born in Stratford-upon-Avon.",
        evidence_sentences=[
            "William Shakespeare was born in Stratford-upon-Avon."
        ],
    )

    assert validation["status"] == "missing_citation"
    assert validation["has_citation"] is False


def test_grounding_validator_accepts_supported_cited_answer():
    validator = GroundingValidator(
        prompt_builder=PromptBuilder(),
        support_threshold=0.6,
    )

    validation = validator.validate(
        answer="Shakespeare was born in Stratford-upon-Avon. [1]",
        evidence_sentences=[
            "William Shakespeare was born in Stratford-upon-Avon."
        ],
    )

    assert validation["status"] == "supported"
    assert validation["valid_citation"] is True
    assert validation["semantic_support"] is True
