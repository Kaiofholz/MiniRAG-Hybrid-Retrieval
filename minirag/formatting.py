import re
class PromptBuilder:
    def __init__(self, max_evidence=3):
        self.max_evidence = max_evidence

    def get_evidence_text(self, evidence_item):
        if isinstance(evidence_item, str):
            return evidence_item
        return evidence_item.text

    def clean_evidence_for_prompt(self, evidence_item):
        text = self.get_evidence_text(evidence_item)
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build(self, question, evidence_sentences):
        selected = evidence_sentences[:self.max_evidence]

        evidence_block = "\n".join(
            f"[{i}] {self.clean_evidence_for_prompt(e)}"
            for i, e in enumerate(selected, start=1)
        )

        prompt = (
            "You are a grounded question-answering assistant.\n\n"
            "Rules:\n"
            "- Answer using only the evidence below.\n"
            "- Do not use outside knowledge.\n"
            "- If the evidence is insufficient, answer exactly: \"The provided evidence is insufficient.\"\n"
            "- If you answer \"The provided evidence is insufficient.\", do not add any explanation, citation, or extra facts.\n"
            "- Use only citation IDs that appear at the start of the evidence items: [1], [2], [3].\n"
            "- Do not write words like \"Evidence:\", \"According to evidence\", or \"This is supported by\".\n"
            "- Do not copy bracketed source footnotes from the evidence.\n"
            "- Keep the answer to one sentence when possible.\n"
            "- If the question asks for a job, profession, role, or occupation, include all roles explicitly stated in the evidence.\n"
            "- Do not omit additional occupations or official positions if they are relevant to the question.\n"
            "- Put citations at the end of the sentence, for example: Shakespeare was born in Stratford-upon-Avon. [1]\n"
            "- You may combine multiple evidence items for simple counting or comparison, but cite all evidence used.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Answer:"
        )
        
        return prompt

class DebugFormatter:
    def format_retrieved_candidates(self, retrieved):
        return [
            {
                "rank": i,
                "chunk_id": ch.chunk_id,
                "dense_score": ch.dense_score,
                "bm25_score": ch.bm25_score,
                "rrf_score": getattr(ch, "rrf_score", None),
                "dense_rank": getattr(ch, "dense_rank", None),
                "bm25_rank": getattr(ch, "bm25_rank", None),
                "text_preview": ch.text[:200],
            }
            for i, ch in enumerate(retrieved, start=1)
        ]

    def format_top_chunks(self, top_chunks):
        return [
            {
                "chunk_id": ch.chunk_id,
                "rerank_score": ch.rerank_score,
                "text_preview": ch.text[:200],
            }
            for ch in top_chunks
        ]

    def format_selected_evidence(self, evidence):
        return [
            {
                "chunk_id": e.chunk_id,
                "sentence_id": e.sentence_id,
                "chunk_rerank_score": e.chunk_rerank_score,
                "sentence_score": e.sentence_score,
                "base_score": e.base_score,
                "heuristic_score": e.heuristic_score,
                "final_score": e.final_score,
                "text": e.text,
            }
            for e in evidence
        ]

class EvidenceFormatter:
    def to_texts(self, evidence):
        return [e.text for e in evidence]

    def ordered_texts_with_source_first(self, evidence, source_evidence):
        ordered_evidence = []

        if source_evidence is not None:
            ordered_evidence.append(source_evidence.text)

        for e in evidence:
            if source_evidence is None or e.text != source_evidence.text:
                ordered_evidence.append(e.text)

        return ordered_evidence
      
class DebugInfoBuilder:
    def __init__(self, query_router):
        self.query_router = query_router

    def build(self, question):
        route_result = self.query_router.route_question(question)

        # If route_result is a dataclass / object
        if hasattr(route_result, "to_dict"):
            route_info = route_result.to_dict()
        elif isinstance(route_result, dict):
            route_info = route_result
        else:
            route_info = {
                "route": route_result.route,
                "risk_level": route_result.risk_level,
                "reason": route_result.reason,
            }

        return {
            "route": route_info.get("route"),
            "risk_level": route_info.get("risk_level"),
            "route_reason": route_info.get("reason"),
        }
