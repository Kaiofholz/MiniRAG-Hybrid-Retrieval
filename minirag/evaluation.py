class RAGEvaluator:
    def __init__(
        self,
        test_cases,
        generative_mode="real",
        use_cache=False,
        verbose_failures=True,
    ):
        self.test_cases = test_cases
        self.generative_mode = generative_mode
        self.use_cache = use_cache
        self.verbose_failures = verbose_failures

    def classify_generative_case(self, row):
        status = row["generative_status"]
        answer_pass = row["generative_answer_pass"]

        if answer_pass and status == "supported":
            return "ideal_supported_answer"

        if answer_pass and status == "abstained":
            return "correct_abstention"

        if not answer_pass and status == "abstained":
            return "unnecessary_abstention_or_synthesis_needed"

        if not answer_pass and status == "supported":
            return "grounded_but_answer_mismatch"

        if status == "abstained_with_extra_info":
            return "dirty_abstention"

        return "other"

    def evaluate_rows(self, rag):
        rows = []
        for test_case in self.test_cases:
            case_eval = self.evaluate_one_case(
                rag=rag,
                test_case=test_case,
            )

            row = case_eval["row"]
            rows.append(row)
            
            if self.verbose_failures and not row["overall_pass"]:
                self.print_verbose_failure(
                query=query,
                expected_answer=expected_answer,
                expected_evidence=expected_evidence,
                row=row,
                evidence_list=evidence_list,
            )

        return rows
        
    def evaluate_summary(self, rag):
        rows = self.evaluate_rows(rag)
        return self.summarize_rows(rows)

    def evaluate_df(self, rag):
        rows = self.evaluate_rows(rag)
        return self.build_df_from_rows(rows)
        
    def summarize(self, df):
        if "generative_case_type" in df.columns:
            return df.groupby("generative_case_type").size()

        if "generative_status" in df.columns:
            return df.groupby("generative_status").size()

        return df.describe(include="all")

    def dashboard(self, rag):
        rows = self.evaluate_rows(rag)
        summary = self.summarize_rows(rows)
        df = self.build_df_from_rows(rows)

        return EvaluationDashboard(
            rows=rows,
            df=df,
            summary=summary,
        )

        #adjust abstention phrase for real generative LLM
    def generative_answer_matches(self, rag, generated_answer, expected_answer):
        validator = rag.validator

        if self.expected_answer_is_abstention(expected_answer):
            return validator.is_abstention(generated_answer)

        # First try strict substring/contains check.
        if self.contains_text(generated_answer, expected_answer):
            return True

        # Then use flexible token-overlap check.
        return self.flexible_generative_answer_match(
            validator=validator,
            generated_answer=generated_answer,
            expected_answer=expected_answer,
        )

    def normalize_test_case(self, test_case):
        if isinstance(test_case, RAGTestCase):
            return (
                test_case.query,
                test_case.expected_answer,
                test_case.expected_evidence,
            )

        return test_case

    def interesting_cases(self, df):
        if "generative_case_type" not in df.columns:
            return df

        return df[
            df["generative_case_type"] != "ideal_supported_answer"
        ]
    # summarize eval row
    def summarize_rows(self,rows):
        """
        Summarize evaluation rows produced by evaluate_rag().

        This turns per-query debug/evaluation results into stage-wise metrics.
        """

        if not rows:
            return self.empty_summary()

        n = len(rows)

        failure_counts = (
            pd.Series([row["failure_stage"] for row in rows])
            .value_counts()
        )

        safety_issue_counts = self.count_answer_safety_issues(rows)

        summary = {
            "num_cases": n,
            "answer_accuracy": self.mean_row_value(rows, "answer_pass"),
            "evidence_accuracy": self.mean_row_value(rows, "evidence_pass"),
            "overall_accuracy": self.mean_row_value(rows, "overall_pass"),
            "retrieval_recall": self.mean_row_value(
                rows,
                "gold_in_retrieved_candidates",
            ),
            "reranked_chunk_recall": self.mean_row_value(
                rows,
                "gold_in_reranked_chunks",
            ),
            "sentence_recall": self.mean_row_value(
                rows,
                "found_in_sentence_candidates",
            ),
            "sentence_mrr": self.mean_row_value(rows, "reciprocal_rank"),
            "false_abstention_count": safety_issue_counts[
                "false_abstention_count"
            ],
            "unsupported_answer_count": safety_issue_counts[
                "unsupported_answer_count"
            ],
            "failure_counts": failure_counts,
        }

        summary.update(
            self.build_generative_summary(rows)
        )

        return summary
        
    def run_generative_evaluation(
        self,
        rag,
        question,
        evidence_list,
        expected_answer,
    ):
        if self.generative_mode == "mock":
            generative_result = (
                rag.generative_evidence_answerer.answer_mock_from_evidence(
                    question=question,
                    evidence_sentences=evidence_list,
                )
            )
        elif self.generative_mode == "real":
            generative_result = (
                rag.generative_evidence_answerer.answer_real_from_evidence(
                    question=question,
                    evidence_sentences=evidence_list,
                )
            )
        else:
            raise ValueError(f"Unknown generative_mode: {self.generative_mode}")

        validation = generative_result.validation

        generative_answer_pass = self.generative_answer_matches(
            rag=rag,
            generated_answer=generative_result.answer,
            expected_answer=expected_answer,
        )

        return {
            "generative_result": generative_result,
            "validation": validation,
            "generative_answer_pass": generative_answer_pass,
        }
    def compute_retrieval_diagnostics(
        self,
        rag,
        debug,
        expected_evidence,
    ):
        # -------------------------------------------------
        # Stage 1: retrieved candidates before reranking
        # -------------------------------------------------
        retrieved_candidates = debug.get("retrieved_candidates", [])

        retrieved_rank, retrieved_match = self.find_rank_in_debug_items(
            retrieved_candidates,
            expected_evidence,
            text_key="text_preview",
        )

        gold_in_retrieved_candidates = retrieved_rank is not None

        # -------------------------------------------------
        # Stage 2: reranked chunks
        # -------------------------------------------------
        top_chunks = debug.get("top_chunks", [])

        reranked_rank, reranked_match = self.find_rank_in_debug_items(
            top_chunks,
            expected_evidence,
            text_key="text_preview",
        )

        gold_in_reranked_chunks = reranked_rank is not None

        # -------------------------------------------------
        # Stage 3: sentence candidates
        # -------------------------------------------------
        sentence_rank, matched_sentence = self.find_evidence_rank(
            getattr(rag, "top_sentence_candidates", []),
            expected_evidence,
        )

        found_in_sentence_candidates = sentence_rank is not None
        reciprocal_rank = 0.0 if sentence_rank is None else 1.0 / sentence_rank

        return {
            "gold_in_retrieved_candidates": gold_in_retrieved_candidates,
            "retrieved_rank": retrieved_rank,
            "retrieved_match": retrieved_match,
            "gold_in_reranked_chunks": gold_in_reranked_chunks,
            "reranked_rank": reranked_rank,
            "reranked_match": reranked_match,
            "found_in_sentence_candidates": found_in_sentence_candidates,
            "sentence_rank": sentence_rank,
            "reciprocal_rank": reciprocal_rank,
            "matched_sentence": matched_sentence,
        }
    def build_evaluation_row(
        self,
        query,
        expected_answer,
        expected_evidence,
        result,
        evidence_list,
        generative_result,
        validation,
        generative_answer_pass,
        retrieval_diagnostics,
    ):
        answer = result.answer or ""
        first_evidence = evidence_list[0] if evidence_list else ""

        answer_pass = self.contains_text(answer, expected_answer)
        evidence_pass = self.contains_text(first_evidence, expected_evidence)
        overall_pass = answer_pass and evidence_pass

        retrieved_match = retrieval_diagnostics["retrieved_match"] or ""
        reranked_match = retrieval_diagnostics["reranked_match"] or ""
        matched_sentence = retrieval_diagnostics["matched_sentence"] or ""

        row = {
            "query": query,

            "expected_answer_contains": expected_answer,
            "answer": answer,
            "answer_pass": answer_pass,
            "expected_answer": expected_answer,
            "expected_evidence": expected_evidence,

            "expected_evidence_contains": expected_evidence,
            "first_evidence": first_evidence[:250],
            "evidence_pass": evidence_pass,
            "generative_answer_pass": generative_answer_pass,
            "generative_evidence": generative_result.evidence,
            "generative_prompt": generative_result.prompt,

            # Retrieval diagnostics
            "gold_in_retrieved_candidates": retrieval_diagnostics[
                "gold_in_retrieved_candidates"
            ],
            "retrieved_rank": retrieval_diagnostics["retrieved_rank"],
            "retrieved_match": retrieved_match[:250],

            # Reranking diagnostics
            "gold_in_reranked_chunks": retrieval_diagnostics[
                "gold_in_reranked_chunks"
            ],
            "reranked_rank": retrieval_diagnostics["reranked_rank"],
            "reranked_match": reranked_match[:250],

            # Sentence-level diagnostics
            "found_in_sentence_candidates": retrieval_diagnostics[
                "found_in_sentence_candidates"
            ],
            "sentence_rank": retrieval_diagnostics["sentence_rank"],
            "reciprocal_rank": retrieval_diagnostics["reciprocal_rank"],
            "matched_sentence": matched_sentence[:250],

            # Final result
            "overall_pass": overall_pass,
            "mode": result.mode,
            "confidence": result.confidence,
        }

        row["generative_answer"] = generative_result.answer
        row["generative_status"] = generative_result.status
        row["generative_semantic_support"] = validation.get("semantic_support")
        row["generative_support_ratio"] = validation.get("support_ratio")
        row["generative_key_terms"] = validation.get("key_terms")
        row["generative_matched_terms"] = validation.get("matched_terms")

        row["failure_stage"] = self.diagnose_failure(row)

        return row

    def print_verbose_failure(
        self,
        query,
        expected_answer,
        expected_evidence,
        row,
        evidence_list,
    ):
        print("=" * 80)
        print("❌ FAILURE")
        print("QUERY:", query)
        print()

        print("EXPECTED ANSWER CONTAINS:", expected_answer)
        print("ANSWER:", row["answer"])
        print("ANSWER PASS:", row["answer_pass"])
        print()

        print("EXPECTED EVIDENCE CONTAINS:", expected_evidence)
        print("FIRST EVIDENCE:", row["first_evidence"])
        print("EVIDENCE PASS:", row["evidence_pass"])
        print()

        print("GOLD IN RETRIEVED CANDIDATES:", row["gold_in_retrieved_candidates"])
        print("RETRIEVED RANK:", row["retrieved_rank"])
        print("RETRIEVED MATCH:", row["retrieved_match"][:300])
        print()

        print("GOLD IN RERANKED CHUNKS:", row["gold_in_reranked_chunks"])
        print("RERANKED RANK:", row["reranked_rank"])
        print("RERANKED MATCH:", row["reranked_match"][:300])
        print()

        print("FOUND IN SENTENCE CANDIDATES:", row["found_in_sentence_candidates"])
        print("SENTENCE RANK:", row["sentence_rank"])
        print("MATCHED SENTENCE:", row["matched_sentence"][:300])
        print()

        print("FAILURE STAGE:", row["failure_stage"])
        print("MODE:", row["mode"])
        print("CONFIDENCE:", row["confidence"])
        print()

        print("ALL FINAL EVIDENCE:")
        for i, ev in enumerate(evidence_list, 1):
            print(f"{i}. {ev[:300]}")

    def evaluate_one_case(self, rag, test_case):
        query, expected_answer, expected_evidence = (
            self.normalize_test_case(test_case)
        )

        result = rag.answer(query, use_cache=self.use_cache)

        debug = result.debug or {}
        evidence_list = result.evidence_sentences or []

        generative_eval = self.run_generative_evaluation(
            rag=rag,
            question=query,
            evidence_list=evidence_list,
            expected_answer=expected_answer,
        )

        generative_result = generative_eval["generative_result"]
        validation = generative_eval["validation"]
        generative_answer_pass = generative_eval["generative_answer_pass"]

        retrieval_diagnostics = self.compute_retrieval_diagnostics(
            rag=rag,
            debug=debug,
            expected_evidence=expected_evidence,
        )

        row = self.build_evaluation_row(
            query=query,
            expected_answer=expected_answer,
            expected_evidence=expected_evidence,
            result=result,
            evidence_list=evidence_list,
            generative_result=generative_result,
            validation=validation,
            generative_answer_pass=generative_answer_pass,
            retrieval_diagnostics=retrieval_diagnostics,
        )

        return {
            "row": row,
            "query": query,
            "expected_answer": expected_answer,
            "expected_evidence": expected_evidence,
            "evidence_list": evidence_list,
        }
    def build_df_from_rows(self, rows):
        df = pd.DataFrame(rows)

        if (
            "generative_answer_pass" in df.columns
            and "generative_status" in df.columns
        ):
            df["generative_case_type"] = df.apply(
                self.classify_generative_case,
                axis=1,
            )

        return df
        
    def empty_summary(self):
        return {
            "num_cases": 0,
            "answer_accuracy": 0.0,
            "evidence_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "retrieval_recall": 0.0,
            "reranked_chunk_recall": 0.0,
            "sentence_recall": 0.0,
            "sentence_mrr": 0.0,
            "false_abstention_count": 0,
            "unsupported_answer_count": 0,
        }
    def mean_row_value(self, rows, key):
        if not rows:
            return 0.0

        return sum(row[key] for row in rows) / len(rows)
        
    def count_answer_safety_issues(self, rows):
        false_abstention_count = 0
        unsupported_answer_count = 0

        abstention_modes = {
            "refusal",
            "abstain",
            "safety_abstain",
        }

        non_answer_modes = abstention_modes | {
            "clarification_needed",
        }

        for row in rows:
            mode = row.get("mode", "")
            answer_pass = row.get("answer_pass", False)
            evidence_pass = row.get("evidence_pass", False)

            # If expected answer/evidence exists but system refuses,
            # it may be a false abstention.
            if mode in abstention_modes and not answer_pass:
                false_abstention_count += 1

            # If system produces an answer but evidence does not pass,
            # this may be unsupported.
            if mode not in non_answer_modes:
                if answer_pass and not evidence_pass:
                    unsupported_answer_count += 1

        return {
            "false_abstention_count": false_abstention_count,
            "unsupported_answer_count": unsupported_answer_count,
        }
    def build_generative_summary(self, rows):
        summary = {}
        n = len(rows)

        if not rows:
            return summary

        if "generative_answer_pass" in rows[0]:
            summary["generative_answer_accuracy"] = (
                sum(row["generative_answer_pass"] for row in rows) / n
            )

        if "generative_status" in rows[0]:
            summary["generative_status_counts"] = (
                pd.Series([row["generative_status"] for row in rows])
                .value_counts()
            )

        return summary
    def expected_answer_is_abstention(self, expected_answer):
        expected_norm = str(expected_answer).lower()

        return (
            "not enough evidence" in expected_norm
            or "insufficient" in expected_norm
        )
    def extract_eval_terms(self, text):
        eval_stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by",
            "was", "were", "is", "are", "be", "been", "being",
            "he", "she", "it", "they", "his", "her", "their",
            "who", "what", "where", "when", "why", "how",
            "did", "does", "do", "had", "has", "have",
            "shakespeare", "william",
        }

        return [
            term
            for term in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
            if term not in eval_stopwords and len(term) > 1
        ]
    def flexible_generative_answer_match(
        self,
        validator,
        generated_answer,
        expected_answer,
    ):
        generated_clean = validator.clean_answer_for_validation(
            generated_answer
        )
        expected_clean = validator.clean_answer_for_validation(
            expected_answer
        )

        expected_terms = self.extract_eval_terms(expected_clean)

        if not expected_terms:
            return False

        generated_lower = generated_clean.lower()

        matched_terms = [
            term
            for term in expected_terms
            if term in generated_lower
        ]

        return len(matched_terms) == len(expected_terms)
        
    def contains_text(self, container, target):
        if not isinstance(container, str) or not isinstance(target, str):
            return False

        return target.lower() in container.lower()
        
    def find_rank_in_debug_items(
        self,
        items,
        expected_text,
        text_key="text_preview",
    ):
        """
        Find 1-based rank of the first debug item whose text contains expected_text.
        """
        if not items or not expected_text:
            return None, ""

        expected = expected_text.lower()

        for rank, item in enumerate(items, start=1):
            text = item.get(text_key, "") or ""

            if expected in text.lower():
                return rank, text

        return None, ""

    def find_evidence_rank(
        self,
        sentence_candidates,
        expected_evidence_contains,
    ):
        """
        Find the 1-based rank of the first sentence candidate containing
        the expected evidence substring.
        """
        if not sentence_candidates or not expected_evidence_contains:
            return None, ""

        expected = expected_evidence_contains.lower()

        for rank, cand in enumerate(sentence_candidates, start=1):
            text = cand.text or ""

            if expected in text.lower():
                return rank, text

        return None, ""

    def diagnose_failure(self, row):
        """
        Diagnose the most likely failure stage from evaluation signals.
        """

        if row["overall_pass"]:
            return "success"

        if not row["gold_in_retrieved_candidates"]:
            return "retrieval_recall"

        if not row["gold_in_reranked_chunks"]:
            return "reranking_or_top_k_cutoff"

        if not row["found_in_sentence_candidates"]:
            return "sentence_candidate_generation"

        if not row["evidence_pass"]:
            return "sentence_scoring_or_evidence_selection"

        if not row["answer_pass"]:
            return "answer_extraction"

        return "unknown_or_partial_mismatch"

    def evidence_pass_anywhere(self, row):
        expected = row.get("expected_evidence_contains", "")

        candidates = [
            row.get("first_evidence", ""),
            row.get("matched_sentence", ""),
            row.get("retrieved_match", ""),
            row.get("reranked_match", ""),
        ]

        combined = " ".join(
            str(x)
            for x in candidates
            if x is not None
        )

        return self.contains_text(combined, expected)
