import pytest

from minirag.cache import LLMCache
from minirag.generation import RealLLMGenerator


def test_real_llm_generator_calls_llm_and_caches_answer():
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "  Generated answer  "

    cache = LLMCache()
    generator = RealLLMGenerator(
        llm_generate_fn=fake_llm,
        llm_cache=cache,
    )

    first = generator.generate("Prompt text")
    second = generator.generate("Prompt text")

    assert first == "Generated answer"
    assert second == "Generated answer"
    assert calls == ["Prompt text"]
    assert len(cache) == 1


def test_real_llm_generator_rejects_missing_llm_function():
    generator = RealLLMGenerator(
        llm_generate_fn=None,
        llm_cache=LLMCache(),
    )

    with pytest.raises(ValueError):
        generator.generate("Prompt text")


def test_real_llm_generator_rejects_non_string_prompt():
    generator = RealLLMGenerator(
        llm_generate_fn=lambda prompt: "answer",
        llm_cache=LLMCache(),
    )

    with pytest.raises(TypeError):
        generator.generate({"not": "a string"})
