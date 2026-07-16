class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, return_weights=False):
        B, T, C = x.shape

        K = self.key(x)                                    # (B, T, head_size)
        Q = self.query(x)                                  # (B, T, head_size)
        V = self.value(x)                                  # (B, T, head_size)

        scores = Q @ K.transpose(-2, -1)                   # (B, T, T)
        scores = scores / math.sqrt(K.shape[-1])
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)                # (B, T, T)
        weights = self.dropout(weights)
        out = weights @ V                                  # (B, T, head_size)

        if return_weights:
            return out, weights
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(n_embd, head_size, block_size)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self, x, return_weights=False):
        if return_weights:
            head_outputs = []
            head_weights = []

            for h in self.heads:
                out, weights = h(x, return_weights=True)
                head_outputs.append(out)
                head_weights.append(weights)

            out = torch.cat(head_outputs, dim=-1)          # (B, T, n_embd)
            out = self.proj(out)                           # (B, T, n_embd)
            weights = torch.stack(head_weights, dim=1)     # (B, n_heads, T, T)
            return out, weights

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)  
        return out
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x, return_weights=False):
        if return_weights:
            attn_out, weights = self.sa(self.ln1(x), return_weights=True)
            x = x + attn_out
            x = x + self.ffwd(self.ln2(x))
            return x, weights

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
      
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_heads, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_heads, block_size) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, return_weights=False):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb

        all_weights = []

        for i, block in enumerate(self.blocks):
            if return_weights and i == 0:
                x, weights = block(x, return_weights=True)
                all_weights.append(weights)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        if return_weights:
            return logits, loss, all_weights
        return logits, loss

    
    
    
    def generate(self, idx, max_new_tokens, block_size):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
      
class SmallLMAdapter:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        input_ids,
        max_new_tokens=64,
        temperature=1.0,
        top_k=None,
    ):
        """
        input_ids: list[int]
        returns: list[int]
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # If your original model.generate only accepts (idx, max_new_tokens),
            # start with that first.
            out = self.model.generate(idx, max_new_tokens=max_new_tokens, block_size=128)

        # convert from shape (1, T) to python list
        return out[0].tolist()

class SmallLMGenerator:
    def __init__(self, small_lm=None, tokenizer=None):
        self.small_lm = small_lm
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 20,
        debug: bool = False,
    ) -> str:
        if self.small_lm is None or self.tokenizer is None:
            return ""

        input_ids = self.tokenizer.encode(prompt)

        output_ids = self.small_lm.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        full_text = self.tokenizer.decode(output_ids)

        if debug:
            print("PROMPT START:", repr(prompt[:100]))
            print("FULL START:", repr(full_text[:100]))
            print("STARTSWITH:", full_text.startswith(prompt))

        # remove prompt echo
        if "answer:" in full_text:
            answer = full_text.split("answer:")[-1].strip()
        elif full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            answer = full_text.strip()

        # keep only first line
        answer = answer.split("\n")[0].strip()

        return answer
