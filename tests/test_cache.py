from minirag.cache import AnswerCache, LLMCache


def test_answer_cache_normalizes_question_key():
    cache = AnswerCache()

    fake_result = {"answer": "Hamnet and Judith"}

    cache.save("Who were Shakespeare's twins?", fake_result)

    assert cache.get("who were shakespeare's twins?") == fake_result
    assert cache.get("  Who were Shakespeare's twins?  ") == fake_result


def test_llm_cache_uses_prompt_text():
    cache = LLMCache()

    cache.save("Prompt text", "Generated answer")

    assert cache.get("Prompt text") == "Generated answer"
    assert len(cache) == 1
