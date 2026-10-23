"""
Train and evaluate the BDH commentary model, like-for-like with the Transformer.

Same data files, same vocabulary, same official splits, same truncation limits
(MAX_SRC / MAX_TGT from train.py), same 10-epoch budget with a checkpoint every
epoch, same BLEU-4 and grounding audit from generate.py. Only the architecture
differs.

    python train_bdh.py overfit --data data/processed_v2.json --vocab data/vocab_v2.json
    python train_bdh.py train   --data data/processed_v2.json --vocab data/vocab_v2.json --tag bdh_v2
    python train_bdh.py eval    --ckpt checkpoints/bdh_v2_best.pt --data data/processed_v2.json

Precision: bf16 autocast, as in the upstream BDH training script. Learning rate
1e-3 and weight decay 0.1 are also the upstream BDH defaults.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bdh_model import BDHCommentary
from generate import build_banned_mask, corpus_bleu4, hallucination_audit
from train import MAX_EPOCH_SECONDS, MAX_EPOCHS, MAX_SRC, MAX_TGT
from vocab import BOS_ID, EOS_ID, PAD_ID, Vocab

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
OUT = ROOT / "outputs"
IGNORE = -100

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class PrefixLMDataset(Dataset):
    """input <BOS> target <EOS>; labels only on target tokens and <EOS>."""

    def __init__(self, examples, vocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        src = self.vocab.encode(ex["input"])[:MAX_SRC]
        tgt = self.vocab.encode(ex["target"])[: MAX_TGT - 2]
        seq = src + [BOS_ID] + tgt + [EOS_ID]
        x, y = seq[:-1], seq[1:]
        # Position j predicts seq[j+1]. Positions before the <BOS> predict input
        # tokens (or the <BOS> itself) and are not part of the task.
        y = [IGNORE] * len(src) + y[len(src):]
        return torch.tensor(x), torch.tensor(y)


def collate_lm(batch):
    xs, ys = zip(*batch)
    T = max(len(x) for x in xs)
    x = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
    y = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    for i, (a, b) in enumerate(zip(xs, ys)):
        x[i, : len(a)] = a
        y[i, : len(b)] = b
    return x, y


AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def run_epoch(model, loader, opt, device, train=True, clip=1.0, amp="bf16",
              scaler=None):
    """amp: bf16 (upstream default; needs Ampere+ such as the RTX 4060), fp16 with
    a GradScaler (for a Tesla T4, which only emulates bf16), or fp32."""
    model.train() if train else model.eval()
    dtype = AMP_DTYPES[amp]
    total_loss = total_tok = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train), torch.autocast(
                "cuda", dtype=dtype or torch.float32,
                enabled=device == "cuda" and dtype is not None):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=IGNORE)
        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
        n = int(y.ne(IGNORE).sum())
        total_loss += loss.item() * n
        total_tok += n
    return total_loss / max(total_tok, 1)


def save_ckpt(path, model, vocab, epoch, tr_loss, va_loss, meta):
    torch.save({
        "arch": "bdh",
        "model_state": model.state_dict(),
        "config": model.hparams,
        "vocab_itos": vocab.itos,
        "vocab_target_ids": sorted(vocab.target_ids) if vocab.target_ids else None,
        "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
        "max_src": MAX_SRC, "max_tgt": MAX_TGT, "meta": meta,
    }, path)


def load_bdh(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = Vocab(ck["vocab_itos"], ck.get("vocab_target_ids"))
    model = BDHCommentary(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, vocab, ck


def build_model(vocab, args):
    return BDHCommentary(
        len(vocab), n_layer=args.layers, n_embd=args.d_model, n_head=args.heads,
        mlp_internal_dim_multiplier=args.mult, dropout=args.dropout)


def overfit(args):
    """Gate: the adapter must be able to memorise a handful of examples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    vocab = Vocab.load(args.vocab)
    ex = data["train"][: args.n]
    loader = DataLoader(PrefixLMDataset(ex, vocab), batch_size=16, shuffle=True,
                        collate_fn=collate_lm)
    args.dropout = 0.0
    model = build_model(vocab, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"BDH OVERFIT | {len(ex)} examples | params {model.n_params():,} | "
          f"ln(V)={math.log(len(vocab)):.3f}")
    for step in range(1, args.steps + 1):
        loss = run_epoch(model, loader, opt, device)
        if step == 1 or step % 25 == 0:
            print(f"  step {step:4d}  loss {loss:.4f}")
    srcs = [vocab.encode(e["input"])[:MAX_SRC] for e in ex[:2]]
    outs = model.greedy_decode(srcs, MAX_TGT)
    exact = 0
    for e, o in zip(ex[:2], outs):
        gen = vocab.decode(o)
        exact += gen == vocab.decode(vocab.encode(e["target"]))
        print(f"\n  REF : {e['target'][:110]}\n  GEN : {gen[:110]}")
    ok = loss < 0.5 and exact == 2
    print(f"\nfinal loss {loss:.4f}, exact reproductions {exact}/2 -> "
          f"OVERFIT GATE: {'PASS' if ok else 'FAIL'}")
    return ok


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    vocab = Vocab.load(args.vocab)
    tr = DataLoader(PrefixLMDataset(data["train"], vocab), batch_size=args.batch_size,
                    shuffle=True, collate_fn=collate_lm)
    va = DataLoader(PrefixLMDataset(data["valid"], vocab), batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate_lm)
    model = build_model(vocab, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda") if args.amp == "fp16" else None
    CKPT.mkdir(exist_ok=True)
    print(f"BDH | train/valid {len(data['train']):,}/{len(data['valid']):,} "
          f"(official splits, test untouched) | vocab {len(vocab):,} | "
          f"params {model.n_params():,} | neurons/layer {model.n_neurons:,} | "
          f"seed {args.seed}")

    meta = {"data": args.data, "seed": args.seed, "lr": args.lr, "wd": args.wd,
            "batch_size": args.batch_size, "amp": args.amp,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    history, best, reason = [], float("inf"), f"completed {MAX_EPOCHS} epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, tr, opt, device, train=True, amp=args.amp,
                            scaler=scaler)
        if not math.isfinite(tr_loss):
            raise SystemExit(f"STOPPED: non-finite training loss ({tr_loss}) in "
                             f"epoch {epoch}; numeric overflow, not continuing")
        elapsed = time.time() - t0
        va_loss = run_epoch(model, va, opt, device, train=False, amp=args.amp)
        m = {**meta, "epoch_seconds": round(elapsed, 1)}
        save_ckpt(CKPT / f"{args.tag}_epoch{epoch:02d}.pt", model, vocab, epoch,
                  tr_loss, va_loss, m)
        if va_loss < best:
            best = va_loss
            save_ckpt(CKPT / f"{args.tag}_best.pt", model, vocab, epoch, tr_loss,
                      va_loss, m)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "val_loss": round(va_loss, 4), "epoch_seconds": round(elapsed, 1),
                        "val_ppl": round(math.exp(min(va_loss, 20)), 2)})
        print(f"epoch {epoch:2d}/{MAX_EPOCHS}  train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"{elapsed:6.1f}s{'  *best' if va_loss == best else ''}", flush=True)
        if elapsed > MAX_EPOCH_SECONDS:
            reason = f"STOPPED: epoch {epoch} took {elapsed:.1f}s > {MAX_EPOCH_SECONDS}s cap"
            print(reason)
            break
    OUT.mkdir(exist_ok=True)
    (OUT / f"history_{args.tag}.json").write_text(json.dumps(
        {"history": history, "stopped_reason": reason, "params": model.n_params(),
         "config": model.hparams, **meta}, indent=2), encoding="utf-8")
    print(f"{reason} | best val {best:.4f} -> {CKPT / f'{args.tag}_best.pt'}")


@torch.no_grad()
def evaluate(args):
    """Official test split: BLEU-4 + grounding audit, constrained and unconstrained."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Leave headroom for the desktop and the driver: an unbounded run took the
        # whole machine down once (NVRM out-of-memory).
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    model, vocab, ck = load_bdh(args.ckpt, device)
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))[args.split]
    if args.limit:
        rows = rows[: args.limit]
    srcs = [vocab.encode(e["input"])[:MAX_SRC] for e in rows]

    # Decoded ids are appended to a cache chunk by chunk, so a crash resumes where
    # it stopped instead of starting over.
    cache = OUT / f"eval_cache_{Path(args.ckpt).stem}_{args.split}.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["i"]] = r
    t0 = time.time()
    with cache.open("a", encoding="utf-8") as fh:
        for start in range(0, len(rows), args.chunk):
            idx = [i for i in range(start, min(start + args.chunk, len(rows)))
                   if i not in done]
            if not idx:
                continue
            banned = build_banned_mask([rows[i] for i in idx], vocab, device)
            sub = [srcs[i] for i in idx]
            con = model.greedy_decode(sub, ck["max_tgt"], banned=banned,
                                      batch_size=args.batch,
                                      amp_dtype=AMP_DTYPES[args.amp])
            raw = model.greedy_decode(sub, ck["max_tgt"], banned=None,
                                      batch_size=args.batch,
                                      amp_dtype=AMP_DTYPES[args.amp])
            for i, c, r in zip(idx, con, raw):
                done[i] = {"i": i, "con": c, "raw": r}
                fh.write(json.dumps(done[i]) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            print(f"  decoded {len(done):,}/{len(rows):,}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    con = [done[i]["con"] for i in range(len(rows))]
    raw = [done[i]["raw"] for i in range(len(rows))]
    recs = [{
        "label": e["label"],
        "grounding": {k: v["name"] for k, v in e["grounding"].items()},
        "reference_tagged": e["target"],
        "generated_tagged": vocab.decode(c),
        "generated_tagged_raw_debug": vocab.decode(r),
    } for e, c, r in zip(rows, con, raw)]
    bleu, prec = corpus_bleu4([r["generated_tagged"] for r in recs],
                              [r["reference_tagged"] for r in recs])
    aud = hallucination_audit(recs)
    raw_aud = hallucination_audit(recs, key="generated_tagged_raw_debug")
    res = {
        "model": "BDH", "checkpoint": str(args.ckpt), "epoch": ck["epoch"],
        "val_loss": round(ck["val_loss"], 4), "params": model.n_params(),
        "split": args.split, "n_examples": len(recs), "bleu4": round(bleu, 2),
        "precisions": [round(p * 100, 1) for p in prec],
        "ungrounded_constrained_pct": round(aud["ungrounded_rate"], 3),
        "ungrounded_unconstrained_pct": round(raw_aud["ungrounded_rate"], 3),
        "raw_entity_tags_leaked": aud["raw_tags"],
        "decode_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    out = OUT / f"eval_{Path(args.ckpt).stem}_{args.split}.json"
    out.write_text(json.dumps({"summary": res, "records": recs}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["overfit", "train", "eval"])
    ap.add_argument("--data", default="data/processed_v2.json")
    ap.add_argument("--vocab", default="data/vocab_v2.json")
    ap.add_argument("--ckpt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="bdh_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mult", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--amp", choices=list(AMP_DTYPES), default="bf16",
                    help="bf16 on Ampere+ (RTX 4060), fp16 on a Tesla T4")
    ap.add_argument("--batch", type=int, default=32, help="decode batch size (eval)")
    ap.add_argument("--chunk", type=int, default=512, help="rows per cached eval chunk")
    ap.add_argument("--mem-fraction", type=float, default=0.6,
                    help="cap on this process's share of GPU memory (eval)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    if args.stage == "overfit":
        sys.exit(0 if overfit(args) else 1)
    elif args.stage == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
