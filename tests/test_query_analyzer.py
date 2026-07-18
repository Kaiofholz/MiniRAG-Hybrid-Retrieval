from minirag.extraction import QueryAnalyzer


def test_query_analyzer_detects_birthplace_question():
    analyzer = QueryAnalyzer()

    result = analyzer.analyze("Where was Shakespeare born?")

    assert result["question_type"] == "where"
    assert result["expected_answer_type"] == "location"
    assert result["relation"] == "birthplace"
    assert result["target_entity"] == "shakespeare"


def test_query_analyzer_detects_profession_question():
    analyzer = QueryAnalyzer()

    result = analyzer.analyze("What was Shakespeare's profession?")

    assert result["question_type"] == "what"
    assert result["expected_answer_type"] == "profession"
    assert result["relation"] == "profession"
    assert result["target_entity"] == "shakespeare"


def test_query_analyzer_detects_person_question():
    analyzer = QueryAnalyzer()

    result = analyzer.analyze("Who were Shakespeare's twins?")

    assert result["question_type"] == "who"
    assert result["expected_answer_type"] == "person"
    assert result["target_entity"] == "shakespeare"
