from minirag.answering import MiniRAG
from minirag.retrieval import RetrieverWrapper
from minirag.cache import AnswerCache, LLMCache
from minirag.formatting import PromptBuilder, DebugFormatter, EvidenceFormatter, DebugInfoBuilder
from minirag.routing import QueryRouter, StructuredSpecQueryParser, SpecialRouteHandler
from minirag.extraction import QueryAnalyzer, BasicAnswerExtractor, TextPatternExtractor, TechnicalSpecExtractor, ProfessionExtractor, DateExtractor, ExtractiveAnswerValidator, ExtractiveRuleEngine, ExtractiveAnswerer
from minirag.retrieval import RetrievalEngine, ChunkReranker
from minirag.evidence import SentenceCandidateBuilder,SentenceScorer, EvidenceSelector, EvidencePipeline
from minirag.generation import GroundingValidator, RealLLMGenerator, MockEvidenceGenerator, GenerativeEvidenceAnswerer, SmallLMGenerator
import re
def test_minirag_answer_returns_extractive_answer_with_fake_retrievers():
    def fake_dense_search(query, **kwargs):
        return [
            ("William Shakespeare was born in Stratford-upon-Avon.", 0.9, 1),
        ]

    def fake_bm25_search(query, **kwargs):
        return [
            ("William Shakespeare was born in Stratford-upon-Avon.", 10.0, 1),
        ]

    rag = MiniRAG(
        dense_retriever=RetrieverWrapper(fake_dense_search),
        bm25_retriever=RetrieverWrapper(fake_bm25_search),
        cross_encoder=None,
        top_n_retrieval=3,
        top_n_rerank=2,
        top_k_evidence=2,
        evidence_threshold=0.5,
        retrieval_fusion="rrf",
    )

    result = rag.answer(
        "Where was Shakespeare born?",
        use_cache=True,
        debug=False,
    )

    assert result.answer == "Stratford-upon-Avon"
    assert result.supported is True
    assert result.mode == "extractive"
    assert result.confidence > 0
    assert result.evidence_sentences == [
        "William Shakespeare was born in Stratford-upon-Avon."
    ]
    assert len(rag.top_sentence_candidates) == 1


def test_minirag_answer_uses_answer_cache():
    calls = {"dense": 0, "bm25": 0}

    def fake_dense_search(query, **kwargs):
        calls["dense"] += 1
        return [
            ("William Shakespeare was born in Stratford-upon-Avon.", 0.9, 1),
        ]

    def fake_bm25_search(query, **kwargs):
        calls["bm25"] += 1
        return [
            ("William Shakespeare was born in Stratford-upon-Avon.", 10.0, 1),
        ]

    rag = MiniRAG(
        dense_retriever=RetrieverWrapper(fake_dense_search),
        bm25_retriever=RetrieverWrapper(fake_bm25_search),
        cross_encoder=None,
        top_n_retrieval=3,
        top_n_rerank=2,
        top_k_evidence=2,
        evidence_threshold=0.5,
        retrieval_fusion="rrf",
    )

    first = rag.answer("Where was Shakespeare born?", use_cache=True)
    second = rag.answer("Where was Shakespeare born?", use_cache=True)

    assert first is second
    assert calls == {"dense": 1, "bm25": 1}
