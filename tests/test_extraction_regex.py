from minirag.extraction import DateExtractor, TechnicalSpecExtractor, TextPatternExtractor


def test_date_extractor_uses_regex_without_crashing():
    extractor = DateExtractor()

    result = extractor.extract_date_answer(
        question="When was Shakespeare born?",
        text="William Shakespeare was born on 23 April 1564.",
    )

    assert result is None or isinstance(result, str)


def test_technical_spec_extractor_uses_regex_without_crashing():
    extractor = TechnicalSpecExtractor()

    result = extractor.extract_technical_spec_answer(
        question="What is the battery voltage?",
        text="The battery voltage is 51.2 V.",
    )

    assert result is None or isinstance(result, str)


def test_text_pattern_extractor_uses_regex_without_crashing():
    extractor = TextPatternExtractor()

    result = extractor.extract_twins_from_text(
        question="Who were Shakespeare's twins?",
        text="Shakespeare's twins were Hamnet and Judith.",
    )

    assert result is None or isinstance(result, str)
