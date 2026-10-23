"""
BDH (Baby Dragon Hatchling, Pathway, arXiv 2509.26507) adapted to grounded
data-to-text commentary.

BDH is a decoder-only language model, so the encoder-decoder task is posed as a
prefix language model over one sequence:

    <input tokens> <BOS> <target tokens> <EOS> <PAD> ...

Padding only ever sits at the END. BDH's attention is causal (each position sees
strictly earlier positions), so trailing padding can never influence a real token
and no padding mask is needed. Loss is computed on target tokens + <EOS> only,
exactly the positions the Transformer is scored on, so validation losses are
directly comparable.

The upstream implementation (third_party/bdh/bdh.py) is imported unchanged. This
subclass re-states its forward pass only to (a) drop the built-in loss, which would
also score the input and the padding, and (b) optionally record the sparse,
positive neuron activations for the interpretability analysis.

Greedy decoding uses the same grounding mask as the Transformer (CLAUDE.md 4.14):
banned ids get logit -inf before the argmax.
"""

import importlib.util
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from vocab import BOS_ID, EOS_ID, PAD_ID

# Load bdh.py by file path. Putting third_party/bdh on sys.path would let its
# train.py shadow this project's train.py.
_spec = importlib.util.spec_from_file_location(
    "bdh_upstream", Path(__file__).resolve().parent / "third_party" / "bdh" / "bdh.py")
_bdh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bdh)
BDH, BDHConfig = _bdh.BDH, _bdh.BDHConfig


class BDHCommentary(BDH):
    def __init__(self, vocab_size, n_layer=6, n_embd=256, n_head=4,
                 mlp_internal_dim_multiplier=32, dropout=0.1):
        super().__init__(BDHConfig(
            n_layer=n_layer, n_embd=n_embd, n_head=n_head,
            mlp_internal_dim_multiplier=mlp_internal_dim_multiplier,
            dropout=dropout, vocab_size=vocab_size,
        ))
        self.hparams = dict(
            vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
            mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, dropout=dropout,
        )

    @property
    def n_neurons(self):
        """Neurons per layer = n_head * N (the sparse latent width)."""
        C = self.config
        return C.n_embd * C.mlp_internal_dim_multiplier

    def forward(self, idx, capture=None, last_only=False):
        """Upstream BDH.forward without the loss. If `capture` is a list, one dict
        per layer is appended: x_sparse / y_sparse as (B, T, n_neurons), i.e. the
        ReLU (sparse, positive) neuron activations, before dropout.
        last_only=True returns (B, vocab) logits for the final position only, which
        is all greedy decoding needs."""
        C = self.config
        B, T = idx.size()
        D, nh = C.n_embd, C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.ln(self.embed(idx).unsqueeze(1))                 # B, 1, T, D
        for _ in range(C.n_layer):
            x_sparse = F.relu(x @ self.encoder)                   # B, nh, T, N
            # BDH attention has no softmax: it sums unbounded products of positive
            # activations over N dims and T positions, which overflows fp16 (NaN
            # after one epoch on a T4). Run this step in fp32 under any autocast.
            with torch.autocast(x.device.type, enabled=False):
                q = x_sparse.float()                              # Q is K upstream
                yKV = self.attn(Q=q, K=q, V=x.float())
            yKV = self.ln(yKV)
            y_sparse = F.relu(yKV @ self.encoder_v)               # B, nh, T, N
            xy_sparse = x_sparse * y_sparse
            if capture is not None:
                capture.append({
                    "x_sparse": x_sparse.detach().transpose(1, 2).reshape(B, T, nh * N),
                    "y_sparse": y_sparse.detach().transpose(1, 2).reshape(B, T, nh * N),
                })
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            x = self.ln(x + self.ln(yMLP))
        if last_only:
            return x[:, 0, -1] @ self.lm_head                     # B, vocab
        return x.view(B, T, D) @ self.lm_head                     # B, T, vocab

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def greedy_decode(self, srcs, max_len, banned=None, batch_size=128,
                      amp_dtype=torch.bfloat16):
        """Greedy decoding for a list of source id lists (no padding).

        Sources are grouped by length so every row in a batch shares one prefix
        length and no padding enters the prefix. Returns, per source, the
        generated ids starting with <BOS> (same convention as the Transformer's
        greedy_decode), with <PAD> after <EOS>.

        Also accepts a right-padded (B, S) tensor, exactly like
        CommentaryTransformer.greedy_decode, and then returns a padded (B, T)
        tensor, so code written for the Transformer runs BDH unchanged.
        """
        if torch.is_tensor(srcs):
            rows = [[t for t in r.tolist() if t != PAD_ID] for r in srcs]
            outs = self.greedy_decode(rows, max_len, banned, batch_size, amp_dtype)
            ys = torch.full((len(outs), max(len(o) for o in outs)), PAD_ID,
                            dtype=torch.long, device=srcs.device)
            for i, o in enumerate(outs):
                ys[i, :len(o)] = torch.tensor(o, device=srcs.device)
            return ys
        self.eval()
        device = self.lm_head.device
        out = [None] * len(srcs)
        by_len = defaultdict(list)
        for i, s in enumerate(srcs):
            by_len[len(s)].append(i)
        for L, idxs in by_len.items():
            for k in range(0, len(idxs), batch_size):
                ii = idxs[k:k + batch_size]
                seq = torch.tensor([srcs[i] + [BOS_ID] for i in ii], device=device)
                ban = banned[ii] if banned is not None else None
                done = torch.zeros(len(ii), dtype=torch.bool, device=device)
                for _ in range(max_len):
                    with torch.autocast("cuda", dtype=amp_dtype or torch.float32,
                                        enabled=device.type == "cuda"
                                        and amp_dtype is not None):
                        logits = self(seq, last_only=True).float()
                    if ban is not None:
                        logits = logits.masked_fill(ban, float("-inf"))
                    nxt = logits.argmax(-1)
                    nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
                    seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
                    done |= nxt.eq(EOS_ID)
                    if bool(done.all()):
                        break
                for j, i in enumerate(ii):
                    out[i] = seq[j, L:].tolist()
        return out
