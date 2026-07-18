from minirag.answering import MiniRAG
from minirag.retrieval import RetrieverWrapper


def fake_dense_search(query, **kwargs):
    return [
        ("William Shakespeare was born in Stratford-upon-Avon.", 0.9, 1),
        ("William Shakespeare was an English playwright and poet.", 0.7, 2),
    ]


def fake_bm25_search(query, **kwargs):
    return [
        ("William Shakespeare was born in Stratford-upon-Avon.", 10.0, 1),
        ("John Shakespeare was an alderman and a successful glover.", 8.0, 3),
    ]


def main():
    rag = MiniRAG(
        dense_retriever=RetrieverWrapper(fake_dense_search),
        bm25_retriever=RetrieverWrapper(fake_bm25_search),
        cross_encoder=None,
        retrieval_fusion="rrf",
    )

    question = "Where was Shakespeare born?"
    result = rag.answer(question, use_cache=False, debug=False)

    print("Question:", result.question)
    print("Answer:", result.answer)
    print("Supported:", result.supported)
    print("Confidence:", result.confidence)
    print("Mode:", result.mode)
    print("Evidence:")
    for sentence in result.evidence_sentences:
        print("-", sentence)


if __name__ == "__main__":
    main()
