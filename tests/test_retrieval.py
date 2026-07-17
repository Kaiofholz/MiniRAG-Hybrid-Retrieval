from minirag.retrieval import RetrieverWrapper


def test_retriever_wrapper_passes_query_and_kwargs():
    calls = []

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return [("mock chunk", 1.0, 42)]

    retriever = RetrieverWrapper(fake_search)

    result = retriever.search("Who was Shakespeare?", top_k=3)

    assert result == [("mock chunk", 1.0, 42)]
    assert calls == [("Who was Shakespeare?", {"top_k": 3})]
