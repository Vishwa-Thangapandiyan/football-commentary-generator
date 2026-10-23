"""
Do individual neurons stand for football concepts? BDH vs Transformer.

For every example in the official VALID split (test stays untouched), the model
reads the structured input and we record its neuron activations at the moment
generation starts (the <BOS> position):

  BDH          x_sparse and y_sparse of every layer (sparse, positive by design)
  Transformer  decoder feed-forward hidden units after ReLU, first decoding step
  BDH, random  same architecture, untrained: the chance-level control

For each neuron and each concept we compute one-vs-rest AUROC: how well that single
neuron's activation separates examples with the concept from those without.

Concepts:
  event        the event type. It is literally in the input (<EVT> pass), so this
               is the EASY case: a neuron can solve it by detecting one input token.
  plan_*       facts about the commentary the model is ABOUT to write that are not
               in the input: will it mention a corner kick (non-corner events
               only), a scoreline, the referee. A neuron that separates these
               encodes a plan, not a copied token.

Neuron counts differ (BDH 8,192 per layer, Transformer 1,024), and a max over more
neurons is higher by chance. So best-neuron scores for BDH are also reported on
random subsets of 1,024 neurons.

    python analyze_neurons.py --bdh checkpoints/bdh_v2_best.pt \
        --transformer checkpoints/v2_best.pt --data data/processed_v2.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

from bdh_model import BDHCommentary
from generate import load_model
from train import MAX_SRC
from train_bdh import load_bdh
from vocab import BOS_ID, PAD_ID

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MIN_POS = 30            # a concept needs at least this many positive examples
MONO_HI, MONO_LO = 0.95, 0.75

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ concepts ---

def concepts(rows):
    """name -> (bool tensor over rows, row mask of rows the concept is defined on)."""
    n = len(rows)
    out = {}
    labels = Counter(r["label"] for r in rows)
    everyone = torch.ones(n, dtype=torch.bool)
    for lab, c in labels.items():
        if c >= MIN_POS:
            out[f"event:{lab}"] = (torch.tensor([r["label"] == lab for r in rows]), everyone)
    tgt = [r["target"].lower() for r in rows]
    non_corner = torch.tensor([r["label"] != "corner" for r in rows])
    out["plan:mentions_corner"] = (torch.tensor(["corner" in t for t in tgt]), non_corner)
    out["plan:mentions_score"] = (
        torch.tensor([bool(re.search(r"\d+\s*:\s*\d+", t)) for t in tgt]), everyone)
    out["plan:mentions_referee"] = (
        torch.tensor(["<referee>" in t or "referee" in t for t in tgt]), everyone)
    return {k: v for k, v in out.items() if int(v[0][v[1]].sum()) >= MIN_POS}


# --------------------------------------------------------------- activations ---

@torch.no_grad()
def bdh_acts(model, vocab, rows, device, batch=128):
    """layer name -> (n, neurons) fp16 activations at the <BOS> position."""
    acts = {}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        seqs = [vocab.encode(r["input"])[:MAX_SRC] + [BOS_ID] for r in chunk]
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), PAD_ID, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
        pos = torch.tensor([len(s) - 1 for s in seqs], device=device)
        cap = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            model(x.to(device), capture=cap)          # trailing PAD cannot affect <BOS>
        ar = torch.arange(len(seqs), device=device)
        for L, d in enumerate(cap):
            for kind in ("x_sparse", "y_sparse"):
                acts.setdefault(f"L{L}.{kind}", []).append(
                    d[kind][ar, pos].float().half().cpu())
    return {k: torch.cat(v) for k, v in acts.items()}


@torch.no_grad()
def transformer_acts(model, vocab, rows, device, batch=128):
    """Decoder feed-forward hidden units (post-ReLU) at the first decoding step."""
    store = {}
    hooks = [layer.linear1.register_forward_hook(
        lambda m, inp, out, L=L: store.__setitem__(L, torch.relu(out[:, 0])))
        for L, layer in enumerate(model.decoder.layers)]
    acts = {}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        srcs = [vocab.encode(r["input"])[:MAX_SRC] for r in chunk]
        S = max(len(s) for s in srcs)
        src = torch.full((len(srcs), S), PAD_ID, dtype=torch.long)
        for j, s in enumerate(srcs):
            src[j, :len(s)] = torch.tensor(s)
        src = src.to(device)
        pad = src.eq(PAD_ID)
        memory = model.encode(src, pad)
        bos = torch.full((len(srcs), 1), BOS_ID, dtype=torch.long, device=device)
        model.decode(bos, memory, pad, bos.eq(PAD_ID))
        for L, a in store.items():
            acts.setdefault(f"L{L}.ffn", []).append(a.float().half().cpu())
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in acts.items()}


# -------------------------------------------------------------------- scores ---

def auroc(A, Y):
    """A: (n, N) non-negative activations. Y: (n, C) bool. Returns (C, N) AUROC.
    Tie-aware for the many exact zeros that ReLU produces."""
    n = A.size(0)
    order = A.argsort(0)
    ranks = torch.empty_like(A)
    ranks.scatter_(0, order, torch.arange(1, n + 1, device=A.device, dtype=A.dtype)
                   .unsqueeze(1).expand_as(A))
    zeros = A.eq(0)
    n0 = zeros.sum(0, keepdim=True).to(A.dtype)
    ranks = torch.where(zeros, (n0 + 1) / 2, ranks)       # zeros sort first
    Yf = Y.to(A.dtype)
    npos = Yf.sum(0).unsqueeze(1)
    nneg = n - npos
    S = Yf.T @ ranks
    return (S - npos * (npos + 1) / 2) / (npos * nneg)


def score_layer(A, cons, device, subsample=None, seed=0):
    A = A.to(device).float()
    if subsample and A.size(1) > subsample:
        g = torch.Generator().manual_seed(seed)
        A = A[:, torch.randperm(A.size(1), generator=g)[:subsample].to(device)]
    res = {}
    event_names = [k for k in cons if k.startswith("event:")]
    # Events share one row set, so score them together; plan concepts each have
    # their own row mask.
    if event_names:
        Y = torch.stack([cons[k][0] for k in event_names], 1).to(device)
        auc = auroc(A, Y)
        for k, a in zip(event_names, auc):
            res[k] = a
    for k in cons:
        if k.startswith("plan:"):
            y, rowmask = cons[k]
            rm = rowmask.to(device)
            res[k] = auroc(A[rm], y.to(device)[rm].unsqueeze(1))[0]
    return res, A


def summarize(name, acts, cons, device, subsample=None, n_sub=5):
    out = {"model": name, "layers": {}}
    for layer, A in acts.items():
        full, Ad = score_layer(A, cons, device)
        active = Ad.gt(0).float()
        events = [k for k in full if k.startswith("event:")]
        E = torch.stack([full[k] for k in events]) if events else None
        mono = 0
        if E is not None:
            hi = (E >= MONO_HI).sum(0)
            second = E.topk(2, dim=0).values[1] if E.size(0) > 1 else torch.zeros(E.size(1), device=device)
            mono = int(((hi == 1) & (second < MONO_LO)).sum())
        best = {k: round(float(v.max()), 4) for k, v in full.items()}
        rec = {
            "neurons": A.size(1),
            "frac_active_per_example": round(float(active.mean()), 4),
            "dead_neurons": int(active.sum(0).eq(0).sum()),
            "best_auroc": best,
            "monosemantic_event_neurons": mono,
            "monosemantic_frac": round(mono / A.size(1), 5),
        }
        if subsample and A.size(1) > subsample:
            subs = [score_layer(A, cons, device, subsample, seed=s)[0] for s in range(n_sub)]
            rec[f"best_auroc_sub{subsample}"] = {
                k: round(sum(float(s[k].max()) for s in subs) / n_sub, 4) for k in full}
        out["layers"][layer] = rec
        del Ad
        torch.cuda.empty_cache()
    return out


def table(results, cons):
    keys = list(cons)
    for r in results:
        print(f"\n=== {r['model']} ===")
        print(f"{'layer':<14}{'neurons':>8}{'active':>8}{'mono':>6}  "
              f"{'mean best AUROC (events)':>26}{'mean best AUROC (plans)':>25}")
        for layer, rec in r["layers"].items():
            b = rec.get("best_auroc_sub1024", rec["best_auroc"])
            ev = [b[k] for k in keys if k.startswith("event:")]
            pl = [b[k] for k in keys if k.startswith("plan:")]
            tag = " (1024-subset)" if "best_auroc_sub1024" in rec else ""
            print(f"{layer:<14}{rec['neurons']:>8}{rec['frac_active_per_example']:>8.3f}"
                  f"{rec['monosemantic_event_neurons']:>6}  "
                  f"{sum(ev)/max(len(ev),1):>26.3f}{sum(pl)/max(len(pl),1):>25.3f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdh", default="checkpoints/bdh_v2_best.pt")
    ap.add_argument("--transformer", default="checkpoints/v2_best.pt")
    ap.add_argument("--transformer-vocab", default="data/vocab_v2.json")
    ap.add_argument("--data", default="data/processed_v2.json")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=128, help="lower if the GPU is shared")
    ap.add_argument("--out", default="outputs/neurons_v2.json")
    args = ap.parse_args()
    if args.split == "test":
        sys.exit("analysis runs on valid; the test split is reserved for final metrics")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))[args.split]
    if args.limit:
        rows = rows[:args.limit]
    cons = concepts(rows)
    print(f"{len(rows):,} {args.split} examples | concepts: "
          + ", ".join(f"{k}({int(v[0][v[1]].sum())})" for k, v in cons.items()))

    results = []
    bdh, vocab, ck = load_bdh(args.bdh, device)
    results.append(summarize(f"BDH trained (epoch {ck['epoch']})",
                             bdh_acts(bdh, vocab, rows, device, args.batch), cons, device, subsample=1024))
    torch.manual_seed(0)
    rnd = BDHCommentary(**{**ck["config"], "dropout": 0.0}).to(device).eval()
    results.append(summarize("BDH random init (control)",
                             bdh_acts(rnd, vocab, rows, device, args.batch), cons, device, subsample=1024))
    del bdh, rnd
    tf, tvocab, tck = load_model(args.transformer, device, vocab_path=args.transformer_vocab)
    results.append(summarize(f"Transformer (epoch {tck['epoch']})",
                             transformer_acts(tf, tvocab, rows, device, args.batch), cons, device))

    table(results, cons)
    Path(args.out).write_text(json.dumps(
        {"split": args.split, "n": len(rows),
         "concepts": {k: int(v[0][v[1]].sum()) for k, v in cons.items()},
         "results": results}, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
