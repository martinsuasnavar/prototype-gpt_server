import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    BLOCK_SIZE
)

from tokenizer import vocab_size


# =========================================================
# ATTENTION HEAD
# =========================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            EMBED_DIM,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            EMBED_DIM,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            EMBED_DIM,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(BLOCK_SIZE, BLOCK_SIZE)
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = (
            q @ k.transpose(-2, -1)
        ) * (k.shape[-1] ** -0.5)

        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(
            weights,
            dim=-1
        )

        v = self.value(x)

        out = weights @ v

        return out


# =========================================================
# MULTI HEAD ATTENTION
# =========================================================

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = EMBED_DIM // NUM_HEADS

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(NUM_HEADS)
        ])

        self.proj = nn.Linear(
            EMBED_DIM,
            EMBED_DIM
        )

    def forward(self, x):

        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        out = self.proj(out)

        return out


# =========================================================
# FEED FORWARD
# =========================================================

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                EMBED_DIM * 4
            ),

            nn.ReLU(),

            nn.Linear(
                EMBED_DIM * 4,
                EMBED_DIM
            )
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# TRANSFORMER BLOCK
# =========================================================

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = MultiHeadAttention()

        self.ffwd = FeedForward()

        self.ln1 = nn.LayerNorm(
            EMBED_DIM
        )

        self.ln2 = nn.LayerNorm(
            EMBED_DIM
        )

    def forward(self, x):

        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
        )

        return x


# =========================================================
# MINIGPT
# =========================================================

class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            EMBED_DIM
        )

        self.blocks = nn.Sequential(
            *[
                Block()
                for _ in range(NUM_LAYERS)
            ]
        )

        self.ln_f = nn.LayerNorm(
            EMBED_DIM
        )

        self.lm_head = nn.Linear(
            EMBED_DIM,
            vocab_size
        )

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        tok_emb = self.token_embedding(
            idx
        )

        pos_emb = self.position_embedding(
            torch.arange(
                T,
                device=idx.device
            )
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits_flat = logits.view(
                B * T,
                C
            )

            targets_flat = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens
    ):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -BLOCK_SIZE:]

            logits, _ = self(
                idx_cond
            )

            logits = logits[:, -1, :]

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (
                    idx,
                    next_token
                ),
                dim=1
            )

        return idx