def build_minirag(llm_generate_fn=None):
    char_tokenizer = CharTokenizerAdapter(stoi, itos)
    small_lm_adapter = SmallLMAdapter(model_transformer, char_tokenizer, device)
    
    return MiniRAG(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        cross_encoder=reranker,
        spec_records=spec_records,
        small_lm=small_lm_adapter,
        tokenizer=char_tokenizer,
        llm_generate_fn=llm_generate_fn,
        top_n_retrieval=30,
        top_n_rerank=20,
        top_k_evidence=3,
        evidence_threshold=0.5,
        debug=False,
    )

