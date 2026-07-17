import re
class ExtractiveRuleEngine:
    def __init__(
        self,
        query_analyzer,
        technical_spec_extractor,
        profession_extractor,
        date_extractor,
        text_pattern_extractor,
        basic_answer_extractor,
    ):
        self.query_analyzer = query_analyzer
        self.technical_spec_extractor = technical_spec_extractor
        self.profession_extractor = profession_extractor
        self.date_extractor = date_extractor
        self.text_pattern_extractor = text_pattern_extractor
        self.basic_answer_extractor= basic_answer_extractor

    def try_pre_query_extraction_rules(
            self,
            question,
            evidence_sentences,
            debug=False,
        ):
            for e in evidence_sentences:
                answer = self.technical_spec_extractor.extract_technical_spec_answer(question, e.text)
                if answer:
                    return answer, e

            for e in evidence_sentences:
                burial_location = self.text_pattern_extractor.extract_burial_location_from_text(question, e.text)
                if burial_location:
                    return burial_location, e

            for e in evidence_sentences:
                daughter_spouse = self.text_pattern_extractor.extract_daughter_spouse_from_text(question, e.text)
                if debug:
                    print("daughter_spouse rule:", daughter_spouse, "| evidence:", e.text[:120])
                if daughter_spouse:
                    return daughter_spouse, e

            for e in evidence_sentences:
                twins_answer = self.text_pattern_extractor.extract_twins_from_text(question, e.text)
                if debug:
                    print("twins rule:", twins_answer, "| evidence:", e.text[:120])
                if twins_answer:
                    return twins_answer, e

            return None, None

    def try_query_type_extraction_rules(
        self,
        question,
        evidence_sentences,
        query_info,
    ):
        best = evidence_sentences[0].text.strip()

        if query_info["relation"] in {"father", "mother"}:
            answer = self.basic_answer_extractor.extract_family_relation_answer(question, best)
            if answer:
                return answer, evidence_sentences[0]

        if query_info["expected_answer_type"] == "profession":
            answer, source = self.profession_extractor.extract_profession_with_source(
                question,
                evidence_sentences,
            )
            if answer:
                return answer, source

        if query_info["expected_answer_type"] == "date":
            answer, source = self.date_extractor.extract_date_with_source(
                question,
                evidence_sentences,
            )
            if answer:
                return answer, source

        if query_info["expected_answer_type"] == "number":
            answer = self.basic_answer_extractor.extract_number_answer(question, best)
            if answer:
                return answer, evidence_sentences[0]

        if query_info["expected_answer_type"] == "location":
            answer = self.basic_answer_extractor.extract_location_answer(question, best)
            if answer:
                return answer, evidence_sentences[0]

        return None, None

    def try_fallback_extraction_rules(
        self,
        question,
        evidence_sentences,
        query_info,
    ):
        best = evidence_sentences[0].text.strip()

        for e in evidence_sentences:
            age_answer = self.text_pattern_extractor.extract_age_from_text(question, e.text)
            if age_answer:
                return age_answer, e

        for e in evidence_sentences:
            spouse_answer = self.text_pattern_extractor.extract_spouse_from_text(question, e.text)
            if spouse_answer:
                return spouse_answer, e

        if query_info["expected_answer_type"] == "person":
            answer = self.text_pattern_extractor.extract_person_name(best)
            if answer:
                return answer, evidence_sentences[0]

        if query_info["expected_answer_type"] in {"definition", "explanation"}:
            return best, evidence_sentences[0]

        return best, evidence_sentences[0]

    def try_extractive_answer_with_source(self, question: str, evidence_sentences, debug: bool = False):
        if not evidence_sentences:
            return None, None
        #reads evidence objects
        query_info = self.query_analyzer.analyze(question)
        best = evidence_sentences[0].text.strip()

        answer, source = self.try_pre_query_extraction_rules(
        question=question,
        evidence_sentences=evidence_sentences,
        debug=debug,
        )

        if answer:
            return answer, source
            
        if debug:
            print("QUERY INFO:", query_info)
            print("BEST EVIDENCE:", best)
        #extracts answer
        answer, source = self.try_query_type_extraction_rules(
            question=question,
            evidence_sentences=evidence_sentences,
            query_info=query_info,
        )

        if answer:
            return answer, source
        
                
        return self.try_fallback_extraction_rules(
            question=question,
            evidence_sentences=evidence_sentences,
            query_info=query_info,
        )
class ExtractiveAnswerValidator:
    def is_valid(
        self,
        question: str,
        answer: str,
        source_evidence,
    ) -> bool:
        """
        Validate whether an extracted answer is actually supported
        by the evidence for the given question.
        """

        if not answer or not answer.strip():
            return False

        q = question.lower()
        a = answer.strip()
        evidence_text = source_evidence.text.lower() if source_evidence is not None else ""

        # 1. Favorite/preference questions require explicit preference evidence.
        if "favorite" in q or "favourite" in q:
            preference_markers = [
                "favorite",
                "favourite",
                "preferred",
                "liked most",
                "most loved",
                "favourite play",
                "favorite play",
            ]

            if not any(marker in evidence_text for marker in preference_markers):
                return False

        # 2. Reject very long "answers" that are probably fallback sentences.
        if len(a.split()) > 25:
            return False

        return True
class TextPatternExtractor:
    def extract_twins_from_text(self, question, text):
        q = question.lower()

        if "twins" in q:
            # Pattern 1: twins Hamnet and Judith
            m = re.search(
                r"\btwins\s+([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                return f"{m.group(1)} and {m.group(2)}"

            # Pattern 2: twins, son Hamnet and daughter Judith
            m = re.search(
                r"\btwins,\s+son\s+([A-Z][a-z]+)\s+and\s+daughter\s+([A-Z][a-z]+)",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                return f"{m.group(1)} and {m.group(2)}"

        return None

    def extract_age_from_text(self, question, text):
        q = question.lower()

        if (
            "what age" in q
            or "at what age" in q
            or "how old" in q
        ):
            # Pattern 1: At the age of 18
            m = re.search(r"\b[Aa]t the age of\s+(\d{1,3})\b", text)
            if m:
                return m.group(1)

            # Pattern 2: aged 18
            m = re.search(r"\baged\s+(\d{1,3})\b", text, flags=re.IGNORECASE)
            if m:
                return m.group(1)

            # Pattern 3: when he was 18
            m = re.search(
                r"\bwhen (?:he|she|they) was\s+(\d{1,3})\b",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                return m.group(1)

        return None

    def extract_spouse_from_text(self, question, text):
        q = question.lower()

        if "who did" in q and "marry" in q:
            # Pattern: Shakespeare married 26-year-old Anne Hathaway
            m = re.search(
                r"\bShakespeare married (?:\d{1,3}-year-old\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                text,
            )
            if m:
                return m.group(1)

            # Pattern: William Shakespeare married Anne Hathaway
            m = re.search(
                r"\bWilliam Shakespeare married (?:\d{1,3}-year-old\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                text,
            )
            if m:
                return m.group(1)

        return None

    def extract_person_name(self, text: str):
        stopwords = {
            "The", "A", "An", "In", "On", "At", "His", "Her",
            "This", "That", "These", "Those",
        }

        matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        cleaned = [m for m in matches if m.split()[0] not in stopwords]

        if cleaned:
            return cleaned[0]

        single_matches = re.findall(r"\b([A-Z][a-z]+)\b", text)
        for m in single_matches:
            if m not in stopwords:
                return m

        return None

    def extract_burial_location_from_text(self, question, text):
        q = question.lower()

        if "buried" not in q:
            return None

        # Pattern 1:
        # "Holy Trinity Church, Stratford-upon-Avon, where Shakespeare was baptised and is buried"
        m = re.search(
            r"\b(Holy Trinity Church(?:,\s*Stratford-upon-Avon)?)\b[^.]{0,150}\b(?:is\s+buried|buried)\b",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1)

        # Pattern 2:
        # "buried in the chancel of the Holy Trinity Church"
        m = re.search(
            r"\bburied\s+in\s+(?:the\s+chancel\s+of\s+)?(?:the\s+)?(Holy Trinity Church)\b",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1)

        return None
    
    def extract_daughter_spouse_from_text(self, question, text):
        q = question.lower()

        if "shakespeare's daughter" in q and "marry" in q:
            m = re.search(
                r"\bSusanna had married (?:a physician,\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                text
            )
            if m:
                return m.group(1)

        return None

    def is_child_birth_sentence(self, text: str) -> bool:
        s = text.lower()

        child_markers = [
            "children",
            "child",
            "son",
            "daughter",
            "twins",
            "susanna",
            "hamnet",
            "judith",
        ]

        return any(marker in s for marker in child_markers)
class QueryAnalyzer:
    def analyze(self, question: str):
        q = question.lower().strip()

        info = {
            "question_type": None,
            "expected_answer_type": "sentence",
            "relation": None,
            "target_entity": None,
        }

        first_word = q.split()[0] if q.split() else ""
        info["question_type"] = first_word

            # target entity detection
        if "john shakespeare" in q:
            info["target_entity"] = "john shakespeare"
        elif "william shakespeare" in q:
            info["target_entity"] = "william shakespeare"
        elif "shakespeare's father" in q or "father" in q:
            info["target_entity"] = "father"
        elif "shakespeare's mother" in q or "mother" in q:
            info["target_entity"] = "mother"
        elif "shakespeare" in q:
            info["target_entity"] = "shakespeare"

        # profession / occupation
        if any(word in q for word in ["profession", "occupation", "job", "living"]):
            info["relation"] = "profession"
            info["expected_answer_type"] = "profession"
            return info
            
    # family relations
        if "father" in q:
            info["relation"] = "father"
            info["expected_answer_type"] = "person"
            return info

        if "mother" in q:
            info["relation"] = "mother"
            info["expected_answer_type"] = "person"
            return info

    # date / year
        if q.startswith("when") or "what year" in q:
            info["expected_answer_type"] = "date"
            return info

    # number
        if q.startswith("how many"):
            info["expected_answer_type"] = "number"
            return info

    # location
        if q.startswith("where"):
            info["expected_answer_type"] = "location"
            if "born" in q:
                info["relation"] = "birthplace"
            return info

    # person
        if q.startswith("who"):
            info["expected_answer_type"] = "person"
            return info

    # explanation / definition
        if q.startswith("why") or q.startswith("how"):
            info["expected_answer_type"] = "explanation"
            return info

        if q.startswith("what"):
            info["expected_answer_type"] = "definition"
            return info

        return info
class BasicAnswerExtractor:
    def extract_family_relation_answer(self, question: str, text: str):
        q = question.lower()

        if "father" in q:
            patterns = [
                r'was the son of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'son of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'his father,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'father,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'father was ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    return m.group(1)

        if "mother" in q:
            patterns = [
                r'was the daughter of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'daughter of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'his mother,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'mother,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'mother was ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    return m.group(1)

        return None
    def extract_number_answer(self, question: str, text: str):
        q = question.lower()
        s = text.strip()

        word_numbers = (
            "one|two|three|four|five|six|seven|eight|nine|ten|"
            "eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
            "seventeen|eighteen|nineteen|twenty"
        )

    # 1. If the query asks about children, prefer number directly before "children"
        if "children" in q or "child" in q:
            patterns = [
                rf"\b({word_numbers}|\d+)\s+children\b",
                rf"\b({word_numbers}|\d+)\s+child\b",
            ]

            for pat in patterns:
                m = re.search(pat, s, flags=re.IGNORECASE)
                if m:
                    return m.group(1)

    # 2. If the query asks about age, prefer age patterns
        if "age" in q or "how old" in q:
            patterns = [
                rf"\bat the age of\s+({word_numbers}|\d+)\b",
                rf"\bat\s+({word_numbers}|\d+)\b",
                rf"\b({word_numbers}|\d+)-year-old\b",
            ]

            for pat in patterns:
                m = re.search(pat, s, flags=re.IGNORECASE)
                if m:
                    return m.group(1)

    # 3. General fallback: first digit
        match = re.search(r"\b\d+(\.\d+)?\b", s)
        if match:
            return match.group(0)

    # 4. General fallback: first word-number
        word_match = re.search(rf"\b({word_numbers})\b", s, flags=re.IGNORECASE)
        if word_match:
            return word_match.group(1)

        return None
        
    def extract_location_answer(self, question: str, text: str):
        q = question.lower()

        if "born" in q:
            patterns = [
            r"born and raised in ([^\.]+)",
            r"born in ([^\.]+)",
        ]

            for pat in patterns:
                match = re.search(pat, text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        return None 
class TechnicalSpecExtractor:
    def extract_technical_spec_answer(self, question: str, text: str):
        """
        Extract simple technical spec values from evidence text.

        Example:
        question: What is the max input current of GW10K-ET?
        text: The max input current of GW10K-ET is 16A.
        answer: 16A
        """

        q = question.lower()
        t = text.strip()

        model_match = re.search(r"\b[A-Z]{1,5}\d+[A-Z]?-?[A-Z]{1,5}\b", question)
        if not model_match:
            return None

        model = model_match.group(0)

        if "max input current" in q:
            parameter_pattern = r"max input current"
        elif "short-circuit current" in q or "short circuit current" in q:
            parameter_pattern = r"max short[- ]circuit current"
        else:
            return None

        # Pattern A:
        # "The max input current of GW10K-ET is 16A"
        pattern_a = (
            rf"{parameter_pattern}\s+of\s+{re.escape(model)}\s+is\s+"
            rf"(\d+(?:\.\d+)?\s*A)"
        )

        match = re.search(pattern_a, t, flags=re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "")

        # Pattern B:
        # "GW10K-ET max input current is 16A"
        pattern_b = (
            rf"{re.escape(model)}.*?{parameter_pattern}.*?is\s+"
            rf"(\d+(?:\.\d+)?\s*A)"
        )

        match = re.search(pattern_b, t, flags=re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "")

        return None
class ProfessionExtractor:
    def clean_profession_phrase(self, phrase: str):
        """
        Clean extracted profession phrase.

        Example:
        "alderman and a successful glover (glove-maker) originally from Snitterfield in Warwickshire"
        ->
        "alderman and a successful glover (glove-maker)"
        """
        if phrase is None:
            return None

        phrase = phrase.strip()

        # Remove trailing sentence fragments that are not part of the profession
        stop_phrases = [
            " originally",
            " from ",
            " who ",
            " and Mary Arden",
            " ,",
        ]

        lower_phrase = phrase.lower()

        cut_positions = []
        for stop in stop_phrases:
            pos = lower_phrase.find(stop.lower())
            if pos != -1:
                cut_positions.append(pos)

        if cut_positions:
            phrase = phrase[:min(cut_positions)].strip()

        # Clean trailing punctuation/spaces
        phrase = phrase.strip(" ,.;:")

        return phrase

    def extract_profession_answer(self, question: str, text: str):
        q = question.lower()
        s = text.strip()

        if not any(word in q for word in ["profession", "occupation", "job", "living"]):
            return None

        patterns = [
        r'John Shakespeare,\s+an?\s+([^,.]+)',
        r'John Shakespeare\s+was\s+an?\s+([^,.]+)',
        r'John Shakespeare\s+was\s+the\s+([^,.]+)',
        r'John Shakespeare.*?,\s+an?\s+([^,.]+)',
        r'son of John Shakespeare,\s+an?\s+([^,.]+(?:\([^)]*\))?)',
        r'John Shakespeare,\s+an?\s+([^,.]+(?:\([^)]*\))?)',
        r'John Shakespeare\s+was\s+an?\s+([^,.]+)',
        r'John Shakespeare\s+was\s+the\s+([^,.]+)',
        r'John Shakespeare.*?,\s+an?\s+([^,.]+(?:\([^)]*\))?)'    
        ]

        for pat in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                profession = m.group(1).strip()
                return self.clean_profession_phrase(profession)

        return None

    def extract_profession_with_source(self, question: str, evidence_sentences):
        for e in evidence_sentences:
            answer = self.extract_profession_answer(question, e.text)
            if answer:
                return answer, e

        return None, None
class DateExtractor:
    def extract_date_answer(self, question: str, text: str):
        q = question.lower()
        t = text.strip()

        # 1. Birth-specific extraction
        if "born" in q or "birth" in q:
        # A. Wikipedia-style lifespan pattern:
        # "23 April 1564 – 23 April 1616"
        # For birth question, take the left-side date.
            match = re.search(
                r"\b((?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b\s*[–-]\s*\b(?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
                t,
            )
            if match:
                return match.group(1).strip()

    # B. Pattern: "April 1564 – 23 April 1616"
    # Less precise than full date, but still clearly the birth-side date.
            match = re.search(
                r"\b([A-Z][a-z]+\s+\d{4})\b\s*[–-]\s*\b(?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
                t,
            )
            if match:
                return match.group(1).strip()

    # C. Explicit "born on 23 April 1564" / "born ... 23 April 1564"
    # Avoid cases where the only date belongs to baptised/died/buried nearby.
            match = re.search(
                r"\bborn\b.{0,80}?\b((?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b",
                t,
                flags=re.IGNORECASE,
            )
            if match:
                window_start = max(0, match.start() - 40)
                window_end = min(len(t), match.end() + 40)
                window = t[window_start:window_end].lower()

                if not any(bad in window for bad in ["baptised", "baptized", "died", "death", "buried"]):
                    return match.group(1).strip()

            # D. Born near year.
            # This is safer than returning a baptism full date.
            born_year = re.search(
                r"\bborn\b.{0,80}\b(1[0-9]{3}|20[0-9]{2})\b",
                t,
                flags=re.IGNORECASE,
            )
            if born_year:
                return born_year.group(1)

            birth_full_date_fallback = re.search(
                r"\b(?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
                t,
                flags=re.IGNORECASE,
            )
            if birth_full_date_fallback:
            # Avoid returning a baptism/death/burial date as birth date.
                date_start = birth_full_date_fallback.start()
                local_window = t[max(0, date_start - 50): date_start].lower()

                if not any(bad in local_window for bad in ["baptised", "baptized", "died", "death", "buried"]):
                    return birth_full_date_fallback.group(0).strip()


            # E. Do NOT return baptised date as exact birth date.
            # At most, baptism can be weak evidence for year, but it should not answer
            # "When was Shakespeare born?" with "26 April 1564".
            return None

        # 2. Death-specific extraction
        if "die" in q or "died" in q or "death" in q:
        # Pattern: "died on 23 April 1616"
            match = re.search(
                r"\bdied\b.{0,80}?\b((?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b",
                t,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()

        # Pattern: "23 April 1564 – 23 April 1616"
        # For death question, take the right-side date.
            match = re.search(
                r"\b(?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b\s*[–-]\s*\b((?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b",
                t,
            )
            if match:
                return match.group(1).strip()

    # 3. Avoid unrelated ranges like "between 1558 and 1642"
        if re.search(r"\bbetween\s+\d{4}\s+and\s+\d{4}\b", t, flags=re.IGNORECASE):
            return None

    # 5. Generic year fallback
        year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", t)
        if year_match:
            return year_match.group(0)

        return None
        
    def extract_date_with_source(self, question: str, evidence_sentences):
        for e in evidence_sentences:
            answer = self.extract_date_answer(question,e.text)
            if answer:
                return answer, e
        return None, None
class ExtractiveAnswerer:
    def __init__(
        self,
        extractive_rule_engine,
        evidence_formatter,
        answer_result_factory,
        extractive_answer_validator,
    ):
        self.extractive_rule_engine = extractive_rule_engine
        self.evidence_formatter = evidence_formatter
        self.answer_result_factory = answer_result_factory
        self.extractive_answer_validator = extractive_answer_validator

    def try_answer(
        self,
        question,
        evidence,
        confidence,
        debug_info,
        debug=False,
    ):
        extracted, source_evidence = self.extractive_rule_engine.try_extractive_answer_with_source(
            question,
            evidence,
            debug=debug,
        )

        debug_info["extractive_answer"] = extracted
        debug_info["extractive_source"] = (
            source_evidence.text if source_evidence else None
        )

        if not (
            extracted
            and len(extracted.strip()) > 0
            and self.extractive_answer_validator.is_valid(
                question,
                extracted,
                source_evidence,
            )
        ):
            return None

        if source_evidence is not None:
            confidence = source_evidence.final_score

        debug_info["confidence"] = confidence

        ordered_evidence = (
            self.evidence_formatter.ordered_texts_with_source_first(
                evidence=evidence,
                source_evidence=source_evidence,
            )
        )

        return self.answer_result_factory.answer_extractive(
            question=question,
            answer=extracted.strip(),
            evidence_sentences=ordered_evidence,
            confidence=confidence,
            debug_info=debug_info,
        )
