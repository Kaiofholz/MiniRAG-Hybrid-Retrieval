def assert_smoke_tests(rag, evaluator_mock):
    single_result = rag.answer(
        "Who were Shakespeare's twins?",
        use_cache=False,
    )

    assert single_result.answer is not None
    assert "Hamnet" in single_result.answer
    assert "Judith" in single_result.answer
    assert single_result.supported is True

    dashboard_mock = evaluator_mock.dashboard(rag)
    summary = dashboard_mock.summary

    assert summary["num_cases"] == 18
    assert summary["answer_accuracy"] == 1.0
    assert summary["evidence_accuracy"] == 1.0
    assert summary["overall_accuracy"] == 1.0

    comparison_result = rag.answer(
        "Compare Shakespeare and John Shakespeare.",
        use_cache=False,
    )

    assert comparison_result.answer is not None
    assert isinstance(comparison_result.answer, str)
    assert len(comparison_result.answer.strip()) > 0

    return {
        "single_query_answer": single_result.answer,
        "single_query_mode": single_result.mode,
        "mock_summary": summary,
        "comparison_answer": comparison_result.answer,
    }
def assert_real_baseline(dashboard_real):
    summary = dashboard_real.summary

    assert summary["num_cases"] == 18
    assert summary["answer_accuracy"] == 1.0
    assert summary["evidence_accuracy"] == 1.0
    assert summary["overall_accuracy"] == 1.0
    assert summary["failure_counts"].get("success", 0) == 18

    return summary
def assert_scoring_regression_tests(rag):
    rag.clear_cache()
    rag.clear_llm_cache()

    # 1. Twins query should not be hijacked by birth-date evidence
    twins = rag.answer(
        "Who were Shakespeare's twins?",
        use_cache=False,
    )

    assert twins.answer is not None
    assert "Hamnet" in twins.answer
    assert "Judith" in twins.answer
    assert twins.supported is True

    twins_evidence_text = " ".join(twins.evidence_sentences)
    assert "Hamnet" in twins_evidence_text
    assert "Judith" in twins_evidence_text

    # 2. Birth-date query should still use date evidence
    born_when = rag.answer(
        "When was Shakespeare born?",
        use_cache=False,
    )

    assert born_when.answer is not None
    assert "1564" in born_when.answer
    assert born_when.supported is True

    # 3. Birthplace query should not be treated as birth-date query
    born_where = rag.answer(
        "Where was Shakespeare born?",
        use_cache=False,
    )

    assert born_where.answer is not None
    assert "Stratford-upon-Avon" in born_where.answer
    assert born_where.supported is True

    # 4. Direct intent checks on SentenceScorer
    scorer = rag.sentence_scorer

    assert scorer.is_birth_date_question("When was Shakespeare born?") is True
    assert scorer.is_birth_date_question("Where was Shakespeare born?") is False

    return {
        "twins_answer": twins.answer,
        "born_when_answer": born_when.answer,
        "born_where_answer": born_where.answer,
    }
