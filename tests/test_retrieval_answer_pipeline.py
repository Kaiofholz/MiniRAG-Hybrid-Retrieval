
from minirag.answering import (
    AnswerFinalizer,
    AnswerResultFactory,
    HighRiskAnswerHandler,
    RetrievalAnswerPipeline,
    SafetyAbstentionPolicy,
)
from minirag.cache import AnswerCache
from minirag.extraction import (
    BasicAnswerExtractor,
    DateExtractor,
    ExtractiveAnswerValidator,
    ExtractiveAnswerer,
    ExtractiveRuleEngine,
    ProfessionExtractor,
    QueryAnalyzer,
    TechnicalSpecExtractor,
    TextPatternExtractor,
)
from minirag.formatting import EvidenceFormatter
from minirag.routing import QueryRouter
from minirag.schemas import EvidenceContext, SentenceCandidate, RetrievalAnswerPipelineResult


class FakeEvidencePipeline:
    def run(self, question, debug_info, debug=False):
        evidence = [
            SentenceCandidate(
                chunk_id=1,
                sentence_id=0,
                text="William Shakespeare was born in Stratford-upon-Avon.",
                final_score=0.9,
            )
        ]

        return EvidenceContext(
            early_result=None,
            evidence=evidence,
            confidence=0.9,
            sentence_candidates=evidence,
            evidence_is_sufficient=True,
        )


def build_test_retrieval_answer_pipeline():
    cache = AnswerCache()

    answer_factory = AnswerResultFactory(
        evidence_formatter=EvidenceFormatter(),
        safety_abstention_policy=SafetyAbstentionPolicy(),
        answer_finalizer=AnswerFinalizer(cache),
    )

    rule_engine = ExtractiveRuleEngine(
        query_analyzer=QueryAnalyzer(),
        technical_spec_extractor=TechnicalSpecExtractor(),
        profession_extractor=ProfessionExtractor(),
        date_extractor=DateExtractor(),
        text_pattern_extractor=TextPatternExtractor(),
        basic_answer_extractor=BasicAnswerExtractor(),
    )

    extractive_answerer = ExtractiveAnswerer(
        extractive_rule_engine=rule_engine,
        evidence_formatter=EvidenceFormatter(),
        answer_result_factory=answer_factory,
        extractive_answer_validator=ExtractiveAnswerValidator(),
    )

    high_risk_handler = HighRiskAnswerHandler(
        query_router=QueryRouter(),
        answer_result_factory=answer_factory,
    )

    pipeline = RetrievalAnswerPipeline(
        evidence_pipeline=FakeEvidencePipeline(),
        answer_result_factory=answer_factory,
        extractive_answerer=extractive_answerer,
        high_risk_answer_handler=high_risk_handler,
    )

    return pipeline, cache


def test_retrieval_answer_pipeline_returns_extractive_answer():
    pipeline, cache = build_test_retrieval_answer_pipeline()

    pipeline_result = pipeline.run(
        question="Where was Shakespeare born?",
        debug_info={},
        debug=False,
    )

    result = pipeline_result.result

    assert result.answer == "Stratford-upon-Avon"
    assert result.supported is True
    assert result.mode == "extractive"
    assert result.confidence == 0.9
    assert result.evidence_sentences == [
        "William Shakespeare was born in Stratford-upon-Avon."
    ]

    assert pipeline_result.sentence_candidates is not None
    assert len(pipeline_result.sentence_candidates) == 1

    cached = cache.get("Where was Shakespeare born?")
    assert cached is result
