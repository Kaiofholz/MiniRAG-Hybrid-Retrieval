class CharTokenizerAdapter:
    def __init__(self, stoi, itos, unknown_token_id=None):
        self.stoi = stoi
        self.itos = itos
        self.unknown_token_id = unknown_token_id

    def encode(self, text: str):
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                if self.unknown_token_id is not None:
                    ids.append(self.unknown_token_id)
                # otherwise skip unknown chars
        return ids

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i in self.itos)

    def chunk_text(text, chunk_size=200, overlap=50):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

chunks = chunk_text(text, chunk_size=500, overlap=150)

print("Number of chunks:", len(chunks))
print("\nExample chunk:\n")
print(chunks[0])

def simple_tokenize(text):
    return re.findall(r"\w+", text.lower())

tokenized_chunks = [simple_tokenize(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

reranker = CrossEncoder("/kaggle/input/models/johnsonhk88/cross-encoderms-marco-minilm-l-6-v2/transformers/v1/1/ms-marco-MiniLM-L-6-v2")
