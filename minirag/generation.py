import re
class RealLLMGenerator:
    def __init__(self, llm_generate_fn, llm_cache):
        self.llm_generate_fn = llm_generate_fn
        self.llm_cache = llm_cache

    def generate(self, prompt):
        if self.llm_generate_fn is None:
            raise ValueError("No real LLM generator function has been configured.")

        if prompt is None:
            raise ValueError(
                "Prompt is None. Check PromptBuilder.build(): it probably forgot to return prompt."
            )

        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt)}")

        cached_answer = self.llm_cache.get(prompt)

        if cached_answer is not None:
            return cached_answer

        answer = self.llm_generate_fn(prompt).strip()

        self.llm_cache.save(prompt, answer)

        return answer
class MockEvidenceGenerator:
    def __init__(self, prompt_builder):
        self.prompt_builder = prompt_builder

    def generate(self, question, evidence_sentences):
        q = question.lower()

        for i, e in enumerate(evidence_sentences, start=1):
            s = self.prompt_builder.get_evidence_text(e).lower()

            if "twins" in q and "hamnet" in s and "judith" in s:
                return f"Shakespeare's twins were Hamnet and Judith. [{i}]"

            if "where" in q and "born" in q and "stratford-upon-avon" in s:
                return f"Shakespeare was born in Stratford-upon-Avon. [{i}]"

            if "when" in q and "born" in q and "1564" in s and "23 april" in s:
                return f"Shakespeare was born on 23 April 1564. [{i}]"

            if (
                ("who did shakespeare marry" in q or "who did he marry" in q)
                and "anne hathaway" in s
                and "married" in s
            ):
                return f"Shakespeare married Anne Hathaway. [{i}]"

            if (
                ("age" in q or "how old" in q)
                and ("marry" in q or "married" in q)
                and ("age of 18" in s or "at the age of 18" in s or "18" in s)
            ):
                return f"Shakespeare married Anne Hathaway at the age of 18. [{i}]"

            if (
                "how many children" in q
                and ("three children" in s or "had three" in s)
            ):
                return f"Shakespeare had three children. [{i}]"

        return "The provided evidence is insufficient."
class GenerativeEvidenceAnswerer:
    def __init__(
        self,
        prompt_builder,
        validator,
        mock_evidence_generator,
        real_llm_generator,
    ):
        self.prompt_builder = prompt_builder
        self.validator = validator
        self.mock_evidence_generator = mock_evidence_generator
        self.real_llm_generator = real_llm_generator
    
    def answer_mock_from_evidence(self, question, evidence_sentences, max_evidence=3):
        selected = evidence_sentences[:max_evidence]

        prompt = self.prompt_builder.build(
            question=question,
            evidence_sentences=selected
        )

        answer = self.mock_evidence_generator.generate(
            question=question,
            evidence_sentences=selected
        )

        validation = self.validator.validate(
            answer=answer,
            evidence_sentences=selected
        )

        return GenerationResult(
            question=question,
            evidence=[
                self.prompt_builder.clean_evidence_for_prompt(e)
                for e in selected
            ],
            prompt=prompt,
            answer=answer,
            validation=validation,
            status=validation["status"]
        )
            
    def answer_real_from_evidence(self, question, evidence_sentences, max_evidence=3):
        selected = evidence_sentences[:max_evidence]

        prompt = self.prompt_builder.build(
            question=question,
            evidence_sentences=selected
        )

        answer = self.real_llm_generator.generate(prompt)

        validation = self.validator.validate(
            answer=answer,
            evidence_sentences=selected
        )

        return GenerationResult (
            question=question,
            evidence=[
                self.prompt_builder.clean_evidence_for_prompt(e)
                for e in selected
            ],
            prompt=prompt,
            answer=answer,
            validation=validation,
            status=validation["status"]
        )
class GroundingValidator:
    def __init__(self, prompt_builder, support_threshold=0.6):
        self.prompt_builder = prompt_builder
        self.support_threshold = support_threshold

    def is_abstention(self, answer):
        a = answer.strip().lower()

        abstention_patterns = [
            "the provided evidence is insufficient",
            "insufficient evidence",
            "not enough evidence",
            "does not provide enough information",
            "does not provide sufficient information",
            "cannot be determined from the evidence",
            "cannot be answered from the evidence",
            "the evidence does not state",
            "the evidence does not say",
            "the provided evidence does not",
        ]

        return any(pattern in a for pattern in abstention_patterns)

    def is_clean_abstention(self, answer):
        return answer.strip().lower() == "the provided evidence is insufficient."

    def extract_citation_ids(self, answer):
        ids = []

        bracket_groups = re.findall(r"\[([^\]]+)\]", answer)

        for group in bracket_groups:
            nums = re.findall(r"\d+", group)
            ids.extend(int(n) for n in nums)

        phrase_patterns = [
            r"\bevidence\s+(\d+)\b",
            r"\bsource\s+(\d+)\b",
        ]

        for pattern in phrase_patterns:
            matches = re.findall(pattern, answer, flags=re.IGNORECASE)
            ids.extend(int(m) for m in matches)

        seen = set()
        unique_ids = []

        for cid in ids:
            if cid not in seen:
                unique_ids.append(cid)
                seen.add(cid)

        return unique_ids

    def clean_answer_for_validation(self, answer):
        text = answer.lower()

        text = re.sub(r"\[\d+\]", "", text)

        boilerplate_patterns = [
            r"according to evidence",
            r"according to the evidence",
            r"this is supported by evidence",
            r"this is supported by the evidence",
            r"evidence:",
            r"source:",
            r"sources:",
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, " ", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_key_terms(self, text):
        text = self.clean_answer_for_validation(text)
        STOPWORDS = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by",
            "was", "were", "is", "are", "be", "been", "being",
            "he", "she", "it", "they", "his", "her", "their",
            "who", "what", "where", "when", "why", "how",
            "did", "does", "do", "had", "has", "have",
            "shakespeare", "william",
            "answer",

            # real LLM boilerplate
            "according", "evidence", "source", "sources",
            "supported", "supports", "provided", "given",
            "this", "that", "these", "those",
            "claim", "claims", "fact", "facts",
            "states", "stated", "says", "said",

            # common grammatical words
            "not", "no", "yes", "also", "only",
            "from", "as", "with", "for", "into",
            "years", "year", "old",
        }
        # Lowercase first
        text = text.lower()
        text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)
        tokens = re.findall(r"[a-z0-9][a-z0-9\-]*", text)

        key_terms = [
            tok for tok in tokens
            if tok not in STOPWORDS and len(tok) > 1
        ]

        return key_terms

    def relation_support_bonus(self, answer, cited_evidence_text):
        a = answer.lower()
        e = cited_evidence_text.lower()

        bonuses = []

        if "father" in a and "son of john shakespeare" in e:
            bonuses.append("father_supported_by_son_of_relation")

        if "mother" in a and "mary arden" in e:
            bonuses.append("mother_supported_by_parent_listing")

        return bonuses

    def validate(self, answer, evidence_sentences):
        if self.is_abstention(answer):
            clean = self.is_clean_abstention(answer)

            return {
                "has_answer": False,
                "has_citation": False,
                "valid_citation": None,
                "semantic_support": None,
                "support_ratio": None,
                "key_terms": None,
                "matched_terms": None,
                "status": "abstained" if clean else "abstained_with_extra_info",
            }

        cited_ids = self.extract_citation_ids(answer)

        if not cited_ids:
            return {
                "has_answer": True,
                "has_citation": False,
                "valid_citation": False,
                "semantic_support": False,
                "support_ratio": 0.0,
                "key_terms": None,
                "matched_terms": None,
                "status": "missing_citation",
            }

        valid_ids = all(
            1 <= cid <= len(evidence_sentences)
            for cid in cited_ids
        )

        if not valid_ids:
            return {
                "has_answer": True,
                "has_citation": True,
                "valid_citation": False,
                "semantic_support": False,
                "support_ratio": 0.0,
                "key_terms": None,
                "matched_terms": None,
                "status": "invalid_citation_id",
            }

        key_terms = self.extract_key_terms(answer)

        cited_evidence_text = " ".join(
            self.prompt_builder.clean_evidence_for_prompt(
                evidence_sentences[cid - 1]
            ).lower()
            for cid in cited_ids
        )

        matched_terms = [
            term for term in key_terms
            if term in cited_evidence_text
        ]

        support_ratio = (
            0.0 if not key_terms
            else len(matched_terms) / len(key_terms)
        )

        relation_bonuses = self.relation_support_bonus(
            answer,
            cited_evidence_text
        )

        semantic_support = (
            support_ratio >= self.support_threshold
            or len(relation_bonuses) > 0
        )

        return {
            "has_answer": True,
            "has_citation": True,
            "valid_citation": True,
            "semantic_support": semantic_support,
            "support_ratio": round(support_ratio, 3),
            "key_terms": key_terms,
            "matched_terms": matched_terms,
            "relation_bonuses": relation_bonuses,
            "status": "supported" if semantic_support else "unsupported",
        }
