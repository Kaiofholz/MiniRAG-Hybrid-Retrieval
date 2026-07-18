from minirag.schemas import RouteResult
import re
class QueryRouter:
    def __init__(self):
        pass

    def route_question(self, question):
        q = question.lower()

        route = "retrieval"
        risk_level = "low"
        reason = "default retrieval question"

        # Preference / unsupported-personal-opinion style
        if "favorite" in q or "favourite" in q:
            route = "normal_rag"
            risk_level = "low"
            reason = "preference-style question; may require abstention if no preference evidence"

        # Exact date / time style
        elif "when" in q or "date" in q or "year" in q:
            route = "normal_rag"
            risk_level = "low"
            reason = "date/time question"

        # Explanation style
        elif q.startswith("why") or q.startswith("how"):
            route = "explanation_rag"
            risk_level = "medium"
            reason = "explanation or troubleshooting style question"

        # Comparison style
        elif "compare" in q or "higher than" in q or "lower than" in q or "difference between" in q:
            route = "comparison"
            risk_level = "medium"
            reason = "comparison question"

        # Technical safety style
        elif any(word in q for word in ["connect", "wire", "wiring", "neutral", "phase", "safety"]):
            route = "technical_safety"
            risk_level = "high"
            reason = "possible electrical/safety-related question"
        # route exact specs to structured lookup
        elif (
            re.search(r"\b[A-Z]{1,5}\d+[A-Z]?-?[A-Z]{1,5}\b", question)
            and (
                "max input current" in q
                or "short-circuit current" in q
                or "short circuit current" in q
            )
        ):
            route = "structured_spec_lookup"
            risk_level = "low"
            reason = "exact technical specification lookup"
        #route catches ambiguous current
        elif (
            re.search(r"\b[A-Z]{1,5}\d+[A-Z]?-?[A-Z]{1,5}\b", question)
            and "current" in q
        ):
            route = "structured_spec_lookup"
            risk_level = "low"
            reason = "technical current specification lookup or clarification"
            
        return RouteResult(
            route=route,
            risk_level=risk_level,
            reason=reason,
        )

class SpecialRouteHandler:
    def __init__(
        self,
        comparison_answerer,
        answer_structured_spec_lookup_route_fn,
    ):
        self.comparison_answerer = comparison_answerer
        self.answer_structured_spec_lookup_route_fn = answer_structured_spec_lookup_route_fn

    def try_answer(self, question, debug_info):
        route = debug_info.get("route")

        if route == "comparison":
            result = self.comparison_answerer.answer(question, debug_info)
            return result

        if route == "structured_spec_lookup":
            return self.answer_structured_spec_lookup_route_fn(
                question=question,
                debug_info=debug_info,
            )

        return None
      
class StructuredSpecQueryParser:
    def parse(self, question: str):
        """
        Parse exact or ambiguous technical spec lookup queries.

        Clear example:
        What is the max input current of GW10K-ET?
        -> {"status": "parsed", "model": "GW10K-ET", "parameter": "max_input_current"}

        Ambiguous example:
        What is the current of GW10K-ET?
        -> {"status": "ambiguous_parameter", "model": "GW10K-ET", "parameter": None}
        """

        q = question.lower()

        model_match = re.search(r"\b[A-Z]{1,5}\d+[A-Z]?-?[A-Z]{1,5}\b", question)
        if not model_match:
            return {
                "status": "missing_model",
                "model": None,
                "parameter": None,
            }

        model = model_match.group(0)

        if "max input current" in q:
            return {
                "status": "parsed",
                "model": model,
                "parameter": "max_input_current",
            }

        if "short-circuit current" in q or "short circuit current" in q:
            return {
                "status": "parsed",
                "model": model,
                "parameter": "max_short_circuit_current",
            }

        if "current" in q:
            return {
                "status": "ambiguous_parameter",
                "model": model,
                "parameter": None,
                "candidates": [
                    "max input current",
                    "max short-circuit current",
                ],
            }

        return {
            "status": "unknown_parameter",
            "model": model,
            "parameter": None,
        }
