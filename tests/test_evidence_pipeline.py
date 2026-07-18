from minirag.answering import AnswerFinalizer, AnswerResultFactory, SafetyAbstentionPolicy
from minirag.cache import AnswerCache
from minirag.evidence import (
    EvidencePipeline,
    EvidenceSelector,
    SentenceCandidateBuilder,
    SentenceScorer,
)
from minirag.extraction import QueryAnalyzer, TextPatternExtractor
from minirag.formatting import DebugFormatter, EvidenceFormatter
from minirag.retrieval import ChunkReranker, RetrievalEngine, RetrieverWrapper


def build_test_evidence_pipeline():
    def fake_dense_search(query, **kwargs):
        return [
            ("William Shakespeare was born in Stratford-upon-Avon.", 0.9, 1),
        ]

    def fake_bm25_search(query, **kwargs):
        return [
            ("William Shakespeare was born in Stratford-upon-Avon.", 10.0, 1),
        ]

    retrieval_engine = RetrievalEngine(
        dense_retriever=RetrieverWrapper(fake_dense_search),
        bm25_retriever=RetrieverWrapper(fake_bm25_search),
        top_n_retrieval=3,
        retrieval_fusion="rrf",
    )

    pattern_extractor = TextPatternExtractor()

    sentence_scorer = SentenceScorer(
        query_analyzer=QueryAnalyzer(),
        is_child_birth_sentence_fn=pattern_extractor.is_child_birth_sentence,
        cross_encoder_predict_fn=None,
        sentence_alpha=0.8,
        chunk_beta=0.2,
    )

    answer_factory = AnswerResultFactory(
        evidence_formatter=EvidenceFormatter(),
        safety_abstention_policy=SafetyAbstentionPolicy(),
        answer_finalizer=AnswerFinalizer(AnswerCache()),
    )

    return EvidencePipeline(
        retrieval_engine=retrieval_engine,
        chunk_reranker=ChunkReranker(
            cross_encoder_predict_fn=None,
            top_n_rerank=2,
        ),
        sentence_candidate_builder=SentenceCandidateBuilder(),
        sentence_scorer=sentence_scorer,
        evidence_selector=EvidenceSelector(
            evidence_threshold=0.5,
            top_k_evidence=2,
        ),
        debug_formatter=DebugFormatter(),
        answer_result_factory=answer_factory,
    )


def test_evidence_pipeline_selects_sufficient_evidence():
    pipeline = build_test_evidence_pipeline()

    result = pipeline.run(
        question="Where was Shakespeare born?",
        debug_info={},
        debug=False,
    )

    assert result.early_result is None
    assert result.evidence_is_sufficient is True
    assert result.confidence > 0
    assert len(result.evidence) == 1
    assert "Stratford-upon-Avon" in result.evidence[0].text
    assert len(result.sentence_candidates) == 1


def test_evidence_pipeline_debug_mode_does_not_crash(capsys):
    pipeline = build_test_evidence_pipeline()

    result = pipeline.run(
        question="Where was Shakespeare born?",
        debug_info={},
        debug=True,
    )

    captured = capsys.readouterr()

    assert result.evidence_is_sufficient is True
    assert "Top Sentence Candidates" in captured.out
