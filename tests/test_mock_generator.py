from minirag.formatting import PromptBuilder
from minirag.generation import MockEvidenceGenerator


def test_mock_evidence_generator_answers_twins_question():
    generator = MockEvidenceGenerator(prompt_builder=PromptBuilder())

    answer = generator.generate(
        question="Who were Shakespeare's twins?",
        evidence_sentences=[
            "Shakespeare's twins were Hamnet and Judith."
        ],
    )

    assert "Hamnet" in answer
    assert "Judith" in answer
    assert "[1]" in answer


def test_mock_evidence_generator_abstains_when_no_rule_matches():
    generator = MockEvidenceGenerator(prompt_builder=PromptBuilder())

    answer = generator.generate(
        question="What was Shakespeare's favorite color?",
        evidence_sentences=[
            "William Shakespeare was an English playwright."
        ],
    )

    assert answer == "The provided evidence is insufficient."
