"""
Stage 4: custom encoder-decoder Transformer (CLAUDE.md 4.6, 4.7).

Architecture:
    input tokens -> embedding -> sinusoidal positional encoding
      -> TransformerEncoder  (multi-head SELF-attention)
      -> memory
      -> TransformerDecoder  (masked self-attention + CROSS-attention + FFN)
      -> linear projection -> softmax over vocab

All three attention flavours the project must demonstrate are present:
  1. encoder self-attention
  2. decoder masked (causal) self-attention
  3. decoder cross-attention over the encoder memory

Each is standard scaled dot-product attention run in `nhead` parallel heads:
    Q = XWq, K = XWk, V = XWv
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    MultiHead(X)     = Concat(head_1..head_h) Wo

Self-test:
    python model.py
"""

import math

import torch
import torch.nn as nn

from vocab import BOS_ID, EOS_ID, PAD_ID


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (CLAUDE.md 4.7).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Chosen over learned embeddings: no extra parameters, no length-generalization
    surprises, and the closed form goes straight onto a presentation slide.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class CommentaryTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.config = dict(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, max_len=max_len,
        )

        # Shared embedding: src and tgt use the same word-level vocabulary, and the
        # entity tags must mean the same thing on both sides.
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, dropout, max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_encoder_layers, norm=nn.LayerNorm(d_model)
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_decoder_layers, norm=nn.LayerNorm(d_model)
        )

        self.out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()

    @staticmethod
    def causal_mask(size, device):
        """True = blocked. Stops the decoder seeing future target tokens."""
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
        )

    def encode(self, src, src_pad_mask):
        x = self.pos(self.embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_key_padding_mask=src_pad_mask)

    def decode(self, tgt_in, memory, src_pad_mask, tgt_pad_mask):
        y = self.pos(self.embed(tgt_in) * math.sqrt(self.d_model))
        return self.decoder(
            y,
            memory,
            tgt_mask=self.causal_mask(tgt_in.size(1), tgt_in.device),
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

    def forward(self, src, tgt_in):
        """src: (B, S) source ids. tgt_in: (B, T) target ids shifted right."""
        src_pad_mask = src.eq(PAD_ID)
        tgt_pad_mask = tgt_in.eq(PAD_ID)
        memory = self.encode(src, src_pad_mask)
        h = self.decode(tgt_in, memory, src_pad_mask, tgt_pad_mask)
        return self.out(h)  # (B, T, vocab)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=120, banned=None):
        """Greedy autoregressive decoding (CLAUDE.md 4.9). No top-k/top-p/beam.

        <BOS> -> predict -> append -> predict -> ... -> <EOS>
        The decoder only ever attends to tokens it has already produced.

        `banned`: optional (B, vocab) bool mask, True = forbidden for that row.
        Used for GROUNDED CONSTRAINED DECODING (CLAUDE.md 4.14): entity tags the
        input never established are driven to -inf BEFORE the argmax, so an
        ungrounded placeholder is unreachable by construction rather than
        filtered out afterwards.
        """
        self.eval()
        src_pad_mask = src.eq(PAD_ID)
        memory = self.encode(src, src_pad_mask)

        B = src.size(0)
        ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            h = self.decode(ys, memory, src_pad_mask, ys.eq(PAD_ID))
            logits = self.out(h[:, -1])                  # (B, vocab)
            if banned is not None:
                logits = logits.masked_fill(banned, float("-inf"))
            nxt = logits.argmax(-1)                      # (B,)
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt.eq(EOS_ID)
            if bool(done.all()):
                break
        return ys

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _self_test():
    """Stage 4 gate: real tensors through every path, with shapes and a loss."""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V, B, S, T = 1176, 4, 24, 18

    model = CommentaryTransformer(V).to(dev)
    print(f"device            : {dev}")
    print(f"config            : {model.config}")
    print(f"trainable params  : {model.n_params():,}")

    src = torch.randint(4, V, (B, S), device=dev)
    tgt = torch.randint(4, V, (B, T), device=dev)
    src[0, -6:] = PAD_ID          # exercise the padding masks
    tgt[1, -4:] = PAD_ID
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

    memory = model.encode(src, src.eq(PAD_ID))
    logits = model(src, tgt_in)
    print(f"\nsrc               : {tuple(src.shape)}")
    print(f"tgt_in            : {tuple(tgt_in.shape)}")
    print(f"memory (encoder)  : {tuple(memory.shape)}")
    print(f"logits            : {tuple(logits.shape)}   expect ({B}, {T - 1}, {V})")

    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    loss = crit(logits.reshape(-1, V), tgt_out.reshape(-1))
    print(f"loss              : {loss.item():.4f}  (random init ~ ln({V}) = "
          f"{math.log(V):.4f})")
    assert torch.isfinite(loss), "loss is not finite"

    loss.backward()
    gnorm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"grad norm         : {gnorm:.4f}")

    cm = model.causal_mask(5, dev)
    print(f"\ncausal mask (True = blocked, strictly upper triangular):\n{cm.int()}")

    ys = model.greedy_decode(src[:2], max_len=8)
    print(f"\ngreedy_decode out : {tuple(ys.shape)}  first row: {ys[0].tolist()}")
    assert ys[0, 0].item() == BOS_ID

    print("\nSTAGE 4 GATE: PASS")


if __name__ == "__main__":
    _self_test()
