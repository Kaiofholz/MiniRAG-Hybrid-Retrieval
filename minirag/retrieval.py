class RetrieverWrapper:
    def __init__(self, search_func):
        self.search_func = search_func
    def search(self, question, **kwargs):
        # We accept 'top_k' here so the code doesn't crash, 
        # but we only pass 'question' to your original function.
        return self.search_func(question, **kwargs)

class RetrievalEngine:
    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        top_n_retrieval,
        retrieval_fusion="rrf",
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.top_n_retrieval = top_n_retrieval
        self.retrieval_fusion = retrieval_fusion

    def retrieve_dense(self, query, k=10, **kwargs):
        top_k = k or self.top_n_retrieval
        return self.dense_retriever.search(query, top_k=top_k)

    def retrieve_bm25(self, query, k=10, **kwargs):
        top_k = k or self.top_n_retrieval
        return self.bm25_retriever.search(query, top_k=top_k)

    def union_candidates(self, dense_results, bm25_results):
        merged = {}

        for chunk_text, score, chunk_id in dense_results:
            merged[chunk_id] = chunk_text

        for chunk_text, score, chunk_id in bm25_results:
            merged[chunk_id] = chunk_text

        return [(chunk_text, chunk_id) for chunk_id, chunk_text in merged.items()]

    def retrieve_candidates(self, question: str) -> List[RetrievedChunk]:
        result = self.retrieve_candidates_with_details(question)
        return result.retrieved

    def retrieve_candidates_with_details(self, question: str) -> RetrievalResult:
        dense_results = []
        bm25_results = []

        if self.dense_retriever is not None:
            dense_results = self.dense_retriever.search(
                question,
                top_k=self.top_n_retrieval,
            )

        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.search(
                question,
                top_k=self.top_n_retrieval,
            )

        if self.retrieval_fusion == "rrf":
            retrieved = self.rrf_fuse(
                bm25_results=bm25_results,
                dense_results=dense_results,
                rrf_k=60,
                top_n=self.top_n_retrieval,
            )

            return RetrievalResult(
                dense_results=dense_results,
                bm25_results=bm25_results,
                retrieved=retrieved,
            )

        merged = {}

        for item in dense_results:
            cid = item[2]

            if cid not in merged:
                merged[cid] = RetrievedChunk(
                    chunk_id=cid,
                    text=item[0],
                    source=None,
                    dense_score=float(item[1]),
                )
            else:
                merged[cid].dense_score = float(item[1])

        for item in bm25_results:
            cid = item[2]

            if cid not in merged:
                merged[cid] = RetrievedChunk(
                    chunk_id=cid,
                    text=item[0],
                    source=None,
                    bm25_score=float(item[1]),
                )
            else:
                merged[cid].bm25_score = float(item[1])

        retrieved = list(merged.values())

        return RetrievalResult(
            dense_results=dense_results,
            bm25_results=bm25_results,
            retrieved=retrieved,
        )

    def rrf_fuse(self, bm25_results, dense_results, rrf_k: int = 60, top_n: int = None):
        fused = {}

        def add_results(results, source_name):
            for rank, item in enumerate(results, start=1):
                text = item[0]
                score = float(item[1])
                chunk_id = int(item[2])

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": text,
                        "rrf_score": 0.0,
                        "dense_score": 0.0,
                        "bm25_score": 0.0,
                        "dense_rank": None,
                        "bm25_rank": None,
                        "sources": [],
                    }

                fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)
                fused[chunk_id]["sources"].append(source_name)

                if source_name == "dense":
                    fused[chunk_id]["dense_score"] = score
                    fused[chunk_id]["dense_rank"] = rank

                if source_name == "bm25":
                    fused[chunk_id]["bm25_score"] = score
                    fused[chunk_id]["bm25_rank"] = rank

        add_results(dense_results, "dense")
        add_results(bm25_results, "bm25")

        fused_items = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        if top_n is not None:
            fused_items = fused_items[:top_n]

        results = []

        for item in fused_items:
            chunk = RetrievedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                source=None,
                dense_score=item["dense_score"],
                bm25_score=item["bm25_score"],
            )

            # Extra debug attributes
            chunk.rrf_score = item["rrf_score"]
            chunk.dense_rank = item["dense_rank"]
            chunk.bm25_rank = item["bm25_rank"]
            chunk.retriever = "+".join(item["sources"])

            results.append(chunk)

        return results

        
    def print_rrf_debug(self, question: str, n: int = 10):
        dense_results = []
        bm25_results = []

        if self.dense_retriever is not None:
            dense_results = self.dense_retriever.search(
                question,
                top_k=self.top_n_retrieval
            )

        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.search(
                question,
                top_k=self.top_n_retrieval
            )

        fused = self.rrf_fuse(
            bm25_results=bm25_results,
            dense_results=dense_results,
            rrf_k=60,
            top_n=n,
        )
        if debug:
            print("\n=== RRF Fused Candidates ===")

            lines = [
                f"{i}. chunk={c.chunk_id} | "
                f"rrf={c.rrf_score:.4f} | "
                f"dense_rank={c.dense_rank} | "
                f"bm25_rank={c.bm25_rank} | "
                f"source={c.retriever} | "
                f"text={c.text[:120]}"
                for i, c in enumerate(fused, start=1)
            ]

            print("\n".join(lines))

class ChunkReranker:
    def __init__(self, cross_encoder_predict_fn, top_n_rerank):
        self.cross_encoder_predict_fn = cross_encoder_predict_fn
        self.top_n_rerank = top_n_rerank
        
    def rerank_chunks(self, question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not chunks:
            return []

        if self.cross_encoder_predict_fn is None:
            # fallback: simple hybrid if no cross-encoder
            for ch in chunks:
                ch.rerank_score = 0.5 * ch.dense_score + 0.5 * ch.bm25_score
            return sorted(chunks, key=lambda x: x.rerank_score, reverse=True)

        pairs = [(question, ch.text) for ch in chunks]
        scores = self.cross_encoder_predict_fn(pairs)

        for ch, score in zip(chunks, scores):
            ch.rerank_score = float(score)

        chunks = sorted(chunks, key=lambda x: x.rerank_score, reverse=True)
        return chunks[:self.top_n_rerank]
