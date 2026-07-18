from minirag.schemas import AnswerResult, RetrievalAnswerPipelineResult
from minirag.cache import AnswerCache, LLMCache
from minirag.formatting import PromptBuilder, DebugFormatter, EvidenceFormatter, DebugInfoBuilder
from minirag.routing import QueryRouter, StructuredSpecQueryParser, SpecialRouteHandler
from minirag.extraction import QueryAnalyzer, BasicAnswerExtractor, TextPatternExtractor, TechnicalSpecExtractor, ProfessionExtractor, DateExtractor, ExtractiveAnswerValidator, ExtractiveRuleEngine,  ExtractiveAnswerer
from minirag.retrieval import RetrievalEngine, ChunkReranker
from minirag.evidence import SentenceCandidateBuilder, SentenceScorer, EvidenceSelector, EvidencePipeline
from minirag.generation import GroundingValidator, RealLLMGenerator, MockEvidenceGenerator, GenerativeEvidenceAnswerer, SmallLMGenerator
class AnswerFinalizer:
    def __init__(self, answer_cache):
        self.answer_cache = answer_cache

    def finalize_answer_result(
        self,
        question,
        answer,
        supported,
        confidence,
        mode,
        evidence_sentences,
        debug_info,
    ):
        result = AnswerResult(
            question=question,
            answer=answer,
            supported=supported,
            confidence=confidence,
            mode=mode,
            evidence_sentences=evidence_sentences,
            debug=debug_info,
        )

        self.answer_cache.save(question, result)

        return result
class AnswerResultFactory:
    def __init__(self,
                 evidence_formatter,
                 safety_abstention_policy,
                 answer_finalizer,
    ):
        self.evidence_formatter=evidence_formatter
        self.safety_abstention_policy=safety_abstention_policy
        self.answer_finalizer=answer_finalizer

        
        
    def answer_no_retrieved_evidence(self, question, debug_info):
        return AnswerResult(
            question=question,
            answer=answer_text,
            supported=False,
            confidence=0.0,
            mode=mode,
            evidence_sentences=[],
            debug_info=debug_info,
        )

    def answer_no_sentence_candidates(self, question, debug_info):
        return AnswerResult(
            question=question,
            answer=answer_text,
            supported=False,
            confidence=0.0,
            mode=mode,
            evidence_sentences=[],
            debug_info=debug_info,
        )

    def answer_insufficient_evidence(self, question, evidence, confidence, debug_info):
        debug_info["abstained"] = True
        debug_info["abstain_reason"] = "evidence_not_sufficient"
        debug_info["confidence"] = confidence

        if debug_info.get("risk_level") == "high":
            answer_text = self.safety_abstention_policy.build_safety_abstention_message(question)
            mode = "safety_abstain"
            debug_info["abstain_reason"] = "evidence_not_sufficient_high_risk"
        else:
            answer_text = "Not enough evidence."
            mode = "abstain"

        return self.answer_finalizer.finalize_answer_result(
            question=question,
            answer=answer_text,
            supported=False,
            confidence=confidence,
            mode=mode,
            evidence_sentences=self.evidence_formatter.to_texts(evidence),
            debug_info=debug_info,
        )

    def answer_high_risk_abstention(
        self,
        question,
        evidence,
        confidence,
        debug_info,
    ):
        answer = self.safety_abstention_policy.build_safety_abstention_message(question)

        return self.answer_finalizer.finalize_answer_result(
            question=question,
            answer=answer,
            supported=False,
            confidence=confidence,
            mode="safety_abstention",
            evidence_sentences=self.evidence_formatter.to_texts(evidence),
            debug_info=debug_info,
        )
        
    def answer_after_extractive_failure(
        self,
        question,
        evidence,
        confidence,
        debug_info,
    ):
        # 8.2 abstain before generator fallback
        if self.safety_abstention_policy.should_abstain(question, evidence, confidence):
            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "extractive_failed_and_evidence_not_sufficient"
            debug_info["confidence"] = confidence

            if debug_info.get("risk_level") == "high":
                answer_text = self.safety_abstention_policy.build_safety_abstention_message(question)
                mode = "safety_abstain"
            else:
                answer_text = "Not enough evidence."
                mode = "abstain"

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer=answer_text,
                supported=False,
                confidence=confidence,
                mode=mode,
                evidence_sentences=self.evidence_formatter.to_texts(evidence),
                debug_info=debug_info,
            )
        debug_info["abstained"] = True
        debug_info["abstain_reason"] = (
            "extractive_failed_but_evidence_considered_sufficient_no_generator"
        )
        debug_info["confidence"] = confidence

        return self.answer_finalizer.finalize_answer_result(
            question=question,
            answer="Not enough evidence.",
            supported=False,
            confidence=confidence,
            mode="abstain",
            evidence_sentences=self.evidence_formatter.to_texts(evidence),
            debug_info=debug_info,
        )
    def answer_extractive(
        self,
        question,
        answer,
        evidence_sentences,
        confidence,
        debug_info,
        mode="extractive",
    ):
        return self.answer_finalizer.finalize_answer_result(
            question=question,
            answer=answer,
            supported=True,
            confidence=confidence,
            mode=mode,
            evidence_sentences=evidence_sentences,
            debug_info=debug_info,
        )
class SafetyAbstentionPolicy:
    def should_abstain(
        self,
        query: str,
        evidence_sentences,
        confidence: float,
        min_confidence: float = 4.0,
    ):
        """
        Decide whether the system should abstain instead of generating an unsupported answer.

        Simple first version:
        - If there are no evidence sentences, abstain.
        - If confidence is too low, abstain.
        - If the top evidence is only weakly related, abstain.
        """

        if not evidence_sentences:
            return True

        if confidence < min_confidence:
            return True

        q = query.lower()
        top_text = evidence_sentences[0].text.lower()

        # For "favorite" questions, require explicit favorite/preference evidence.
        if "favorite" in q or "favourite" in q:
            if not any(
                word in top_text
                for word in ["favorite", "favourite", "preferred", "liked most"]
            ):
                return True

        return False

    def build_safety_abstention_message(self, question: str) -> str:
        """
        Build a conservative response for high-risk technical/safety questions
        when evidence is insufficient.
        """

        return (
            "I do not have enough evidence to confirm this safely. "
            "This appears to be a high-risk technical or electrical installation question. "
            "Please check the official documentation and consult a qualified electrician "
            "or manufacturer technical support before proceeding."
        )
class HighRiskAnswerHandler:
    def __init__(
        self,
        query_router,
        answer_result_factory,
    ):
        self.query_router = query_router
        self.answer_result_factory = answer_result_factory

    def try_answer_after_extractive_failure(
        self,
        question,
        evidence,
        confidence,
        debug_info,
    ):
        route_result = self.query_router.route_question(question)

        if route_result.risk_level != "high":
            return None

        return self.answer_result_factory.answer_high_risk_abstention(
            question=question,
            evidence=evidence,
            confidence=confidence,
            debug_info=debug_info,
        )

    def try_answer_high_risk_after_extractive_failure(
        self,
        question,
        evidence,
        confidence,
        debug_info,
    ):
        route = self.query_router.route_question(question)
        
        if debug_info.get("risk_level") != "high":
            return None

        return self.answer_result_factory.answer_high_risk_abstention(
            question=question,
            evidence=evidence,
            confidence=confidence,
            debug_info=debug_info,
        )
class RetrievalAnswerPipeline:
    def __init__(
        self,
        evidence_pipeline,
        answer_result_factory,
        extractive_answerer,
        high_risk_answer_handler,
    ):
        self.evidence_pipeline = evidence_pipeline
        self.answer_result_factory = answer_result_factory
        self.extractive_answerer = extractive_answerer
        self.high_risk_answer_handler = high_risk_answer_handler

    def run(self, question, debug_info, debug=False):
        evidence_context = self.evidence_pipeline.run(
            question=question,
            debug_info=debug_info,
            debug=debug,
        )

        sentence_candidates = evidence_context.sentence_candidates

        if evidence_context.early_result is not None:
            return RetrievalAnswerPipelineResult(
                result=evidence_context.early_result,
                sentence_candidates=sentence_candidates,
            )

        evidence = evidence_context.evidence
        confidence = evidence_context.confidence

        if not evidence_context.evidence_is_sufficient:
            result = self.answer_result_factory.answer_insufficient_evidence(
                question=question,
                evidence=evidence,
                confidence=confidence,
                debug_info=debug_info,
            )

            return RetrievalAnswerPipelineResult(
                result=result,
                sentence_candidates=sentence_candidates,
            )

        extractive_result = self.extractive_answerer.try_answer(
            question=question,
            evidence=evidence,
            confidence=confidence,
            debug_info=debug_info,
            debug=debug,
        )

        if extractive_result is not None:
            return RetrievalAnswerPipelineResult(
                result=extractive_result,
                sentence_candidates=sentence_candidates,
            )

        debug_info["extractive_rejected"] = True

        high_risk_result = (
            self.high_risk_answer_handler.try_answer_high_risk_after_extractive_failure(
                question=question,
                evidence=evidence,
                confidence=confidence,
                debug_info=debug_info,
            )
        )

        if high_risk_result is not None:
            return RetrievalAnswerPipelineResult(
                result=high_risk_result,
                sentence_candidates=sentence_candidates,
            )

        result = self.answer_result_factory.answer_after_extractive_failure(
            question=question,
            evidence=evidence,
            confidence=confidence,
            debug_info=debug_info,
        )

        return RetrievalAnswerPipelineResult(
            result=result,
            sentence_candidates=sentence_candidates,
        )    
class ComparisonAnswerer:
    def __init__(self, answer_fn, answer_finalizer):
        self.answer_fn = answer_fn
        self.answer_finalizer = answer_finalizer

    def answer(self, question: str, debug_info: dict) -> AnswerResult:
        """
        Comparison handler v2:
        - parse comparison question
        - create two atomic subqueries
        - execute both subqueries using normal MiniRAG
        - return both subquery answers
        - numeric comparison is not implemented yet
        """

        debug_info["comparison_handler_used"] = True

        comparison_info = self.parse_comparison_question(question)
        debug_info["comparison_info"] = comparison_info

        if comparison_info.get("ambiguous_parameter"):
            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "comparison_ambiguous_parameter"

            candidates = comparison_info.get("parameter_candidates", [])

            answer_text = (
                "The comparison parameter is ambiguous. Please specify which current you mean: "
                + ", ".join(candidates)
                + "."
            )

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer=answer_text,
                supported=False,
                confidence=0.0,
                mode="clarification_needed",
                evidence_sentences=[],
                debug_info=debug_info,
            )

        if not comparison_info["parsed"]:
            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "comparison_parse_failed"

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer=(
                    "This appears to be a comparison question, but I could not parse "
                    "the two compared items and the parameter clearly enough."
                ),
                supported=False,
                confidence=0.0,
                mode="comparison_parse_failed",
                evidence_sentences=[],
                debug_info=debug_info,
            )

        subqueries = comparison_info["subqueries"]

        # Important:
        # These subqueries should be answered by the normal pipeline.
        # They are simple lookup questions, not comparison questions.
        sub_results = []
        for sq in subqueries:
            sub_result = self.answer_fn(sq, use_cache=False)
            sub_results.append(sub_result)

        debug_info["comparison_subresults"] = [
            {
                "subquery": subqueries[i],
                "answer": sub_results[i].answer,
                "mode": sub_results[i].mode,
                "supported": sub_results[i].supported,
                "confidence": sub_results[i].confidence,
                "route": sub_results[i].debug.get("route"),
                "risk_level": sub_results[i].debug.get("risk_level"),
                "abstain_reason": sub_results[i].debug.get("abstain_reason"),
            }
            for i in range(len(sub_results))
        ]

        # If either subquery is unsupported, the comparison cannot be safely answered.
        if not all(r.supported for r in sub_results):
            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "comparison_subquery_insufficient_evidence"

            answer_text = (
                "I cannot complete the comparison because at least one required value "
                "could not be answered from the available evidence.\n\n"
                f"Subquery 1: {subqueries[0]}\n"
                f"Answer 1: {sub_results[0].answer}\n\n"
                f"Subquery 2: {subqueries[1]}\n"
                f"Answer 2: {sub_results[1].answer}"
            )

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer=answer_text,
                supported=False,
                confidence=0.0,
                mode="comparison_insufficient_evidence",
                evidence_sentences=[],
                debug_info=debug_info,
            )

        # Both subqueries are supported. Now try numeric comparison.
        left_numeric = self.parse_numeric_value(sub_results[0].answer)
        right_numeric = self.parse_numeric_value(sub_results[1].answer)

        debug_info["comparison_numeric_values"] = {
            "left": left_numeric,
            "right": right_numeric,
        }

        if left_numeric is None or right_numeric is None:
            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "comparison_numeric_parse_failed"

            answer_text = (
                "Both comparison subqueries were answered, but I could not parse the "
                "numeric values clearly enough to compare them.\n\n"
                f"Subquery 1: {subqueries[0]}\n"
                f"Answer 1: {sub_results[0].answer}\n\n"
                f"Subquery 2: {subqueries[1]}\n"
                f"Answer 2: {sub_results[1].answer}"
            )

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer=answer_text,
                supported=False,
                confidence=min(r.confidence for r in sub_results),
                mode="comparison_numeric_parse_failed",
                evidence_sentences=[],
                debug_info=debug_info,
            )

        if left_numeric["unit"].lower() != right_numeric["unit"].lower():
            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "comparison_unit_mismatch"

            answer_text = (
                "Both values were found, but their units do not match, so I cannot "
                "safely compare them.\n\n"
                f"Value 1: {sub_results[0].answer}\n"
                f"Value 2: {sub_results[1].answer}"
            )

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer=answer_text,
                supported=False,
                confidence=min(r.confidence for r in sub_results),
                mode="comparison_unit_mismatch",
                evidence_sentences=[],
                debug_info=debug_info,
            )

        left_value = left_numeric["value"]
        right_value = right_numeric["value"]
        unit = left_numeric["unit"]

        operator = comparison_info["operator"]

        if operator == "higher_than":
            comparison_result = left_value > right_value
            relation_text = "higher than"
        elif operator == "lower_than":
            comparison_result = left_value < right_value
            relation_text = "lower than"
        else:
            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "comparison_operator_unknown"

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer="I found both values, but I could not determine the comparison operator.",
                supported=False,
                confidence=min(r.confidence for r in sub_results),
                mode="comparison_operator_unknown",
                evidence_sentences=[],
                debug_info=debug_info,
            )

        debug_info["comparison_result"] = comparison_result

        if comparison_result:
            answer_text = (
                f"Yes. {comparison_info['left_item']}'s {comparison_info['parameter']} "
                f"is {left_value:g}{unit}, which is {relation_text} "
                f"{comparison_info['right_item']}'s value of {right_value:g}{unit}."
            )
        else:
            answer_text = (
                f"No. {comparison_info['left_item']}'s {comparison_info['parameter']} "
                f"is {left_value:g}{unit}, which is not {relation_text} "
                f"{comparison_info['right_item']}'s value of {right_value:g}{unit}."
            )

        combined_evidence = []
        for r in sub_results:
            combined_evidence.extend(r.evidence_sentences)

        return self.answer_finalizer.finalize_answer_result(
            question=question,
            answer=answer_text,
            supported=True,
            confidence=min(r.confidence for r in sub_results),
            mode="comparison",
            evidence_sentences=combined_evidence,
            debug_info=debug_info,
        )
    def parse_comparison_question(self, question: str) -> dict:
        """
        First simple parser for comparison questions.

        Example:
        "Is GW10K-ET's max input current higher than GW8K-ET's?"

        Output:
        {
            "left_item": "GW10K-ET",
            "right_item": "GW8K-ET",
            "parameter": "max input current",
            "operator": "higher_than",
            "subqueries": [...]
        }
        """

        q = question
    
        result = {
            "left_item": None,
            "right_item": None,
            "parameter": None,
            "operator": None,
            "subqueries": [],
            "parsed": False,
            "ambiguous_parameter": False,
            "parameter_candidates": [],
        }

        # detect operator
        q_lower = q.lower()
        if "higher than" in q_lower:
            result["operator"] = "higher_than"
        elif "lower than" in q_lower:
            result["operator"] = "lower_than"
        else:
            result["operator"] = "unknown"

        # very simple model-name pattern
        # captures things like GW10K-ET, GW8K-ET
        models = re.findall(r"\b[A-Z]{1,5}\d+[A-Z]?-?[A-Z]{1,5}\b", q)

        if len(models) >= 2:
            result["left_item"] = models[0]
            result["right_item"] = models[1]

        # very simple parameter detection
        if "max input current" in q_lower:
            result["parameter"] = "max input current"
        elif "short-circuit current" in q_lower or "short circuit current" in q_lower:
            result["parameter"] = "max short-circuit current"
        elif "current" in q_lower:
            result["parameter"] = None
            result["ambiguous_parameter"] = True
            result["parameter_candidates"] = [
                "max input current",
                "max short-circuit current",
            ]
        
        if result["left_item"] and result["right_item"] and result["parameter"]:
            result["parsed"] = True
            result["subqueries"] = [
                f"What is the {result['parameter']} of {result['left_item']}?",
                f"What is the {result['parameter']} of {result['right_item']}?",
            ]

        return result 

    def parse_numeric_value(self, answer: str):
        """
        Extract a numeric value and unit from an answer string.

        Example:
        "16A" -> {"value": 16.0, "unit": "A"}
        "14 A" -> {"value": 14.0, "unit": "A"}
        """

        if not answer:
            return None

        match = re.search(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", answer)

        if not match:
            return None

        return {
            "value": float(match.group(1)),
            "unit": match.group(2),
        }
class StructuredSpecLookupHandler:
    def __init__(
        self,
        spec_records,
        structured_spec_query_parser,
        answer_finalizer,
    ):
        self.spec_records = spec_records
        self.structured_spec_query_parser = structured_spec_query_parser
        self.answer_finalizer = answer_finalizer
    def structured_spec_lookup(self, question: str):
        """
        Look up exact technical spec values from structured records.
        Also detects missing/ambiguous parameters.
        """

        parsed = self.structured_spec_query_parser.parse(question)
        if parsed is None:
            return {
                "status": "parse_failed",
                "parsed": None,
            }
            
        if parsed["status"] != "parsed":
            return {
                "status": parsed["status"],
                "parsed": parsed,
            }
            
        target_model = parsed["model"]
        target_parameter = parsed["parameter"]

        for record in self.spec_records:
            if (
                record["model"] == target_model
                and record["parameter"] == target_parameter
            ):
                return {
                    "status": "found",
                    "model": record["model"],
                    "parameter": record["parameter"],
                    "value": record["value"],
                    "unit": record["unit"],
                    "source_text": record["source_text"],
                    "parsed": parsed,
                }

        return {
        "status": "not_found",
        "parsed": parsed,
        }
    def answer(self, question, debug_info):
        lookup = self.structured_spec_lookup(question)
        debug_info["structured_lookup"] = lookup

        if lookup["status"] == "found":
            answer_text = f"{lookup['value']}{lookup['unit']}"

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer=answer_text,
                supported=True,
                confidence=1.0,
                mode="structured_lookup",
                evidence_sentences=[lookup["source_text"]],
                debug_info=debug_info,
            )

        if lookup["status"] == "ambiguous_parameter":
            parsed = lookup["parsed"]
            candidates = parsed.get("candidates", [])

            answer_text = (
                "The parameter is ambiguous. Please specify which current you mean: "
                + ", ".join(candidates)
                + "."
            )

            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "structured_spec_ambiguous_parameter"

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer=answer_text,
                supported=False,
                confidence=0.0,
                mode="clarification_needed",
                evidence_sentences=[],
                debug_info=debug_info,
            )

        if lookup["status"] == "not_found":
            debug_info["abstained"] = True
            debug_info["abstain_reason"] = "structured_spec_not_found"

            return self.answer_finalizer.finalize_answer_result(
                question=question,
                answer="Not enough evidence.",
                supported=False,
                confidence=0.0,
                mode="abstain",
                evidence_sentences=[],
                debug_info=debug_info,
            )

        # Missing/unknown parameter fallback
        debug_info["abstained"] = True
        debug_info["abstain_reason"] = f"structured_spec_{lookup['status']}"

        return self.answer_finalizer.finalize_answer_result(
            question=question,
            answer="Not enough evidence.",
            supported=False,
            confidence=0.0,
            mode="abstain",
            evidence_sentences=[],
            debug_info=debug_info,
        )
class MiniRAG:
    def __init__(
        self,
        dense_retriever=None,
        bm25_retriever=None,
        cross_encoder=None,
        spec_records=None,
        small_lm=None,
        tokenizer=None,
        llm_generate_fn=None,
        top_n_retrieval: int = 8,
        top_n_rerank: int = 4,
        top_k_evidence: int = 3,
        evidence_threshold: float = 0.5,
        sentence_alpha: float = 0.8,
        chunk_beta: float = 0.2,
        debug: bool = True,
        retrieval_fusion: str = "union",
    ):
        # =========================================================
        # 1. External dependencies
        # =========================================================
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder
        self.spec_records = spec_records or []
        self.small_lm = small_lm
        self.tokenizer = tokenizer
        self.llm_generate_fn = llm_generate_fn

        # =========================================================
        # 2. Config
        # =========================================================
        self.top_n_retrieval = top_n_retrieval
        self.top_n_rerank = top_n_rerank
        self.top_k_evidence = top_k_evidence
        self.evidence_threshold = evidence_threshold
        self.sentence_alpha = sentence_alpha
        self.chunk_beta = chunk_beta
        self.debug = debug
        self.retrieval_fusion = retrieval_fusion

        # =========================================================
        # 3. Runtime state
        # =========================================================
        self.top_sentence_candidates = []

        # =========================================================
        # 4. Caches
        # =========================================================
        self.answer_cache = AnswerCache()
        self.llm_cache = LLMCache()

        # =========================================================
        # 5. Shared formatters / builders / routers
        # =========================================================
        self.prompt_builder = PromptBuilder(max_evidence=3)
        self.debug_formatter = DebugFormatter()
        self.evidence_formatter = EvidenceFormatter()

        self.query_router = QueryRouter()
        self.query_analyzer = QueryAnalyzer()

        self.debug_info_builder = DebugInfoBuilder(
            query_router=self.query_router,
        )

        # =========================================================
        # 6. Extractors / parsers
        # =========================================================
        self.basic_answer_extractor = BasicAnswerExtractor()
        self.text_pattern_extractor = TextPatternExtractor()

        self.technical_spec_extractor = TechnicalSpecExtractor()
        self.profession_extractor = ProfessionExtractor()
        self.date_extractor = DateExtractor()

        self.structured_spec_query_parser = StructuredSpecQueryParser()

        self.extractive_answer_validator = ExtractiveAnswerValidator()

        # =========================================================
        # 7. Safety / result construction
        # =========================================================
        self.safety_abstention_policy = SafetyAbstentionPolicy()

        self.answer_finalizer = AnswerFinalizer(
            answer_cache=self.answer_cache,
        )

        self.answer_result_factory = AnswerResultFactory(
            evidence_formatter=self.evidence_formatter,
            safety_abstention_policy=self.safety_abstention_policy,
            answer_finalizer=self.answer_finalizer,
        )

        # =========================================================
        # 8. Retrieval and evidence components
        # =========================================================
        cross_encoder_predict_fn = (
            self.cross_encoder.predict
            if self.cross_encoder is not None
            else None
        )

        self.retrieval_engine = RetrievalEngine(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
            top_n_retrieval=self.top_n_retrieval,
            retrieval_fusion=self.retrieval_fusion,
        )

        self.chunk_reranker = ChunkReranker(
            cross_encoder_predict_fn=cross_encoder_predict_fn,
            top_n_rerank=self.top_n_rerank,
        )

        self.sentence_candidate_builder = SentenceCandidateBuilder()

        self.sentence_scorer = SentenceScorer(
            query_analyzer=self.query_analyzer,
            is_child_birth_sentence_fn=self.text_pattern_extractor.is_child_birth_sentence,
            cross_encoder_predict_fn=cross_encoder_predict_fn,
            sentence_alpha=self.sentence_alpha,
            chunk_beta=self.chunk_beta,
        )

        self.evidence_selector = EvidenceSelector(
            self.evidence_threshold,
            self.top_k_evidence,
        )

        self.evidence_pipeline = EvidencePipeline(
            retrieval_engine=self.retrieval_engine,
            chunk_reranker=self.chunk_reranker,
            sentence_candidate_builder=self.sentence_candidate_builder,
            sentence_scorer=self.sentence_scorer,
            evidence_selector=self.evidence_selector,
            debug_formatter=self.debug_formatter,
            answer_result_factory=self.answer_result_factory,
        )

        # =========================================================
        # 9. Extractive answer path
        # =========================================================
        self.extractive_rule_engine = ExtractiveRuleEngine(
            query_analyzer=self.query_analyzer,
            technical_spec_extractor=self.technical_spec_extractor,
            profession_extractor=self.profession_extractor,
            date_extractor=self.date_extractor,
            text_pattern_extractor=self.text_pattern_extractor,
            basic_answer_extractor=self.basic_answer_extractor,
        )

        self.extractive_answerer = ExtractiveAnswerer(
            extractive_rule_engine=self.extractive_rule_engine,
            evidence_formatter=self.evidence_formatter,
            answer_result_factory=self.answer_result_factory,
            extractive_answer_validator=self.extractive_answer_validator,
        )

        # =========================================================
        # 10. High-risk / retrieval answer pipeline
        # =========================================================
        self.high_risk_answer_handler = HighRiskAnswerHandler(
            query_router=self.query_router,
            answer_result_factory=self.answer_result_factory,
        )

        self.retrieval_answer_pipeline = RetrievalAnswerPipeline(
            evidence_pipeline=self.evidence_pipeline,
            answer_result_factory=self.answer_result_factory,
            extractive_answerer=self.extractive_answerer,
            high_risk_answer_handler=self.high_risk_answer_handler,
        )

        # =========================================================
        # 11. Special routes
        # =========================================================
        self.comparison_answerer = ComparisonAnswerer(
            answer_fn=self.answer,
            answer_finalizer=self.answer_finalizer,
        )

        self.structured_spec_lookup_handler = StructuredSpecLookupHandler(
            spec_records=self.spec_records,
            structured_spec_query_parser=self.structured_spec_query_parser,
            answer_finalizer=self.answer_finalizer,
        )

        self.special_route_handler = SpecialRouteHandler(
            comparison_answerer=self.comparison_answerer,
            answer_structured_spec_lookup_route_fn=(
                self.structured_spec_lookup_handler.answer
            ),
        )

        # =========================================================
        # 12. Generative evaluation path
        # =========================================================
        self.validator = GroundingValidator(
            prompt_builder=self.prompt_builder,
            support_threshold=0.6,
        )

        self.real_llm_generator = RealLLMGenerator(
            llm_generate_fn=self.llm_generate_fn,
            llm_cache=self.llm_cache,
        )

        self.mock_evidence_generator = MockEvidenceGenerator(
            prompt_builder=self.prompt_builder,
        )

        self.generative_evidence_answerer = GenerativeEvidenceAnswerer(
            prompt_builder=self.prompt_builder,
            validator=self.validator,
            mock_evidence_generator=self.mock_evidence_generator,
            real_llm_generator=self.real_llm_generator,
        )
        self.small_lm_generator = SmallLMGenerator(
            small_lm=self.small_lm,
            tokenizer=self.tokenizer,
        )
    def clear_cache(self):
        self.answer_cache.clear()
    def clear_llm_cache(self):
        self.llm_cache.clear()
    def answer(self, question: str, use_cache: bool = True, debug: bool = False) -> AnswerResult:
        if use_cache:
            cached = self.answer_cache.get(question)
            if cached is not None:
                return cached

        self.top_sentence_candidates = []
        debug_info = self.debug_info_builder.build(question)

        special_route_result = self.special_route_handler.try_answer(
            question=question,
            debug_info=debug_info,
        )

        if special_route_result is not None:
            return special_route_result
            
        pipeline_result = self.retrieval_answer_pipeline.run(
            question=question,
            debug_info=debug_info,
            debug=debug,
        )

        self.top_sentence_candidates = pipeline_result.sentence_candidates

        return pipeline_result.result
