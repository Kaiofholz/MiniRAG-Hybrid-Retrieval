from dataclasses import dataclass, field
from typing import Optional, Dict, Any,List
import pandas as pd

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: Optional[str] = None
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SentenceCandidate:
    chunk_id: str
    sentence_id: int
    text: str
    chunk_rerank_score: float = 0.0
    sentence_score: float = 0.0
    heuristic_score: float = 0.0
    final_score: float = 0.0
    base_score: float = 0.0

@dataclass
class AnswerResult:
    question: str
    answer: str
    supported: bool
    confidence: float
    mode: str  # "extractive", "generative", "extractive_fallback", "refusal"
    evidence_sentences: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationResult:
    question: str
    evidence: List[str]
    prompt: str
    answer: str
    validation: Dict[str, Any]
    status: str

@dataclass
class EvaluationDashboard:
    rows: list
    df: pd.DataFrame
    summary: object   
  
@dataclass
class EvidenceContext:
    early_result: Optional[Any]
    evidence: Optional[list]
    confidence: Optional[float]
    sentence_candidates: Optional[Any]
    evidence_is_sufficient: Optional[bool] = None

@dataclass
class EvidenceSelectionResult:
    evidence: list
    is_sufficient: bool
    confidence: float

@dataclass
class RAGTestCase:
    query: str
    expected_answer: str
    expected_evidence: str
    
    @classmethod
    def from_tuple(cls, test_tuple):
        query, expected_answer, expected_evidence = test_tuple

        return cls(
            query=query,
            expected_answer=expected_answer,
            expected_evidence=expected_evidence,
        )

@dataclass
class RouteResult:
    route: str
    risk_level: str
    reason: str

    def to_dict(self):
        return {
            "route": self.route,
            "risk_level": self.risk_level,
            "reason": self.reason,
        }

@dataclass
class RetrievalAnswerPipelineResult:
    result: AnswerResult
    sentence_candidates: Optional[Any] = None
  
@dataclass
class RetrievalResult:
    dense_results: list
    bm25_results: list
    retrieved: List[RetrievedChunk]

