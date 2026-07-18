import re

from minirag.schemas import AnswerResult


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
