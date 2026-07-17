from minirag.formatting import PromptBuilder
from minirag.schemas import SentenceCandidate
import re

def test_prompt_builder_builds_numbered_evidence_block():
    builder = PromptBuilder(max_evidence=2)

    evidence = [
        SentenceCandidate(
            chunk_id=1,
            sentence_id=0,
            text="William Shakespeare was born in Stratford-upon-Avon. [99]",
        ),
        SentenceCandidate(
            chunk_id=2,
            sentence_id=0,
            text="He was baptised at Holy Trinity Church.",
        ),
        SentenceCandidate(
            chunk_id=3,
            sentence_id=0,
            text="This third evidence item should not appear.",
        ),
    ]

    prompt = builder.build(
        question="Where was Shakespeare born?",
        evidence_sentences=evidence,
    )

    assert "Question:\nWhere was Shakespeare born?" in prompt
    assert "[1] William Shakespeare was born in Stratford-upon-Avon." in prompt
    assert "[2] He was baptised at Holy Trinity Church." in prompt
    assert "This third evidence item should not appear." not in prompt
    assert "[99]" not in prompt
