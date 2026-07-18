from minirag.extraction import BasicAnswerExtractor, TextPatternExtractor


def test_text_pattern_extractor_extracts_person_name():
    extractor = TextPatternExtractor()

    result = extractor.extract_person_name(
        "William Shakespeare was born in Stratford-upon-Avon."
    )

    assert result == "William Shakespeare"


def test_basic_answer_extractor_extracts_location_answer():
    extractor = BasicAnswerExtractor()

    result = extractor.extract_location_answer(
        question="Where was Shakespeare born?",
        text="William Shakespeare was born in Stratford-upon-Avon.",
    )

    assert result == "Stratford-upon-Avon"


def test_basic_answer_extractor_extracts_number_answer():
    extractor = BasicAnswerExtractor()

    result = extractor.extract_number_answer(
        question="What is the battery voltage?",
        text="The battery voltage is 51.2 V.",
    )

    assert result == "51.2"
