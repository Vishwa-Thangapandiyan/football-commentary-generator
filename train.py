"""
Stage 5-6: overfit sanity check, then the real training run.

Training budget is a HARD limit (CLAUDE.md 4.12):
  * max 10 epochs
  * max 5 minutes per epoch -> if one epoch exceeds it, STOP and report
  * checkpoint after EVERY completed epoch
We are deliberately NOT training to convergence.

Split policy is binding (CLAUDE.md 4.10): train on the official train split,
validate on the official valid split, never touch test.

Usage:
    python train.py overfit --data data/processed_dev.json --n 32
    python train.py train   --data data/processed.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from download_and_parse import tokenize
from model import CommentaryTransformer
from vocab import BOS_ID, EOS_ID, PAD_ID, Vocab

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"

MAX_SRC, MAX_TGT = 32, 96          # covers 100% / 99.94% of the dev slice
MAX_EPOCHS = 10                    # CLAUDE.md 4.12
MAX_EPOCH_SECONDS = 300            # CLAUDE.md 4.12

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class CommentaryDataset(Dataset):
    def __init__(self, examples, vocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        src = self.vocab.encode(ex["input"])[:MAX_SRC]
        tgt = self.vocab.encode(ex["target"])[: MAX_TGT - 2]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor([BOS_ID] + tgt + [EOS_ID], dtype=torch.long),
        )


def collate(batch):
    srcs, tgts = zip(*batch)
    S = max(len(s) for s in srcs)
    T = max(len(t) for t in tgts)
    src = torch.full((len(batch), S), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : len(s)] = s
        tgt[i, : len(t)] = t
    return src, tgt


def load(data_path, vocab_path):
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    vocab = Vocab.load(vocab_path)
    return data, vocab


def run_epoch(model, loader, crit, opt, device, train=True, clip=1.0):
    model.train() if train else model.eval()
    total_loss = total_tok = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.set_grad_enabled(train):
            logits = model(src, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        n = int(tgt_out.ne(PAD_ID).sum())
        total_loss += loss.item() * n
        total_tok += n
    return total_loss / max(total_tok, 1)


def save_ckpt(path, model, vocab, epoch, train_loss, val_loss, meta=None):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config,
            "vocab_itos": vocab.itos,      # vocab travels with the checkpoint
            "vocab_target_ids": (
                sorted(vocab.target_ids) if vocab.target_ids is not None else None
            ),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "max_src": MAX_SRC,
            "max_tgt": MAX_TGT,
            "meta": meta or {},
        },
        path,
    )


# ------------------------------------------------------------- stage 5 gate ---


def overfit(data_path, vocab_path, n=32, steps=300, lr=3e-4):
    """A correct seq2seq must be able to memorize a handful of examples.
    If loss does not collapse here, the bug is in masking/shifting - not the data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, vocab = load(data_path, vocab_path)
    examples = data["train"][:n]

    ds = CommentaryDataset(examples, vocab)
    loader = DataLoader(ds, batch_size=min(16, n), shuffle=True, collate_fn=collate)
    model = CommentaryTransformer(len(vocab), dropout=0.0).to(device)  # no dropout
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"OVERFIT TEST | {n} examples | vocab {len(vocab):,} | {device}")
    print(f"random-init reference loss = ln({len(vocab)}) = {math.log(len(vocab)):.3f}\n")
    first = None
    for step in range(1, steps + 1):
        loss = run_epoch(model, loader, crit, opt, device, train=True)
        if first is None:
            first = loss
        if step % 25 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss:.4f}")
    print(f"\nfirst {first:.4f} -> final {loss:.4f}")

    ok = loss < 0.5
    print("STAGE 5 GATE:", "PASS" if ok else "FAIL")
    if not ok:
        print("Loss did not collapse. Debug masking / shifting BEFORE scaling.")
        return False

    # Show it really memorized: greedy decode two training examples.
    src, _ = collate([ds[0], ds[1]])
    out = model.greedy_decode(src.to(device), max_len=MAX_TGT)
    for i in range(2):
        print(f"\n  [{i}] INPUT     : {examples[i]['input'][:110]}")
        print(f"      REFERENCE : {examples[i]['target'][:110]}")
        print(f"      GENERATED : {vocab.decode(out[i].tolist())[:110]}")

    CKPT.mkdir(exist_ok=True)
    path = CKPT / "overfit.pt"
    save_ckpt(path, model, vocab, 0, loss, float("nan"),
              {"note": f"stage-5 overfit on {n} examples", "steps": steps})
    print(f"\ncheckpoint -> {path}")
    return True


# ------------------------------------------------------------------ stage 6 ---


def train(data_path, vocab_path, batch_size=32, lr=3e-4, tag="full"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, vocab = load(data_path, vocab_path)
    train_ex, val_ex = data["train"], data["valid"]
    if not val_ex:
        raise SystemExit(
            "Official valid split is empty - download must finish first. "
            "Never substitute a slice of train (CLAUDE.md rule 11)."
        )

    tr_loader = DataLoader(
        CommentaryDataset(train_ex, vocab), batch_size=batch_size,
        shuffle=True, collate_fn=collate, drop_last=False,
    )
    va_loader = DataLoader(
        CommentaryDataset(val_ex, vocab), batch_size=batch_size,
        shuffle=False, collate_fn=collate,
    )

    model = CommentaryTransformer(len(vocab)).to(device)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    CKPT.mkdir(exist_ok=True)

    print(f"device        : {device}")
    print(f"train / valid : {len(train_ex):,} / {len(val_ex):,} examples "
          f"(official splits, test untouched)")
    print(f"vocab         : {len(vocab):,}   params: {model.n_params():,}")
    print(f"budget        : max {MAX_EPOCHS} epochs, max "
          f"{MAX_EPOCH_SECONDS}s per epoch, checkpoint every epoch\n")

    history, best = [], float("inf")
    stopped_reason = f"completed {MAX_EPOCHS} epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, tr_loader, crit, opt, device, train=True)
        elapsed = time.time() - t0
        va_loss = run_epoch(model, va_loader, crit, opt, device, train=False)

        path = CKPT / f"{tag}_epoch{epoch:02d}.pt"
        save_ckpt(path, model, vocab, epoch, tr_loss, va_loss,
                  {"epoch_seconds": elapsed, "data": str(data_path)})
        if va_loss < best:
            best = va_loss
            save_ckpt(CKPT / f"{tag}_best.pt", model, vocab, epoch, tr_loss, va_loss,
                      {"epoch_seconds": elapsed, "data": str(data_path)})
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
             "epoch_seconds": round(elapsed, 1),
             "val_ppl": round(math.exp(min(va_loss, 20)), 2)}
        )
        print(f"epoch {epoch:2d}/{MAX_EPOCHS}  train {tr_loss:.4f}  val {va_loss:.4f}"
              f"  ppl {math.exp(min(va_loss, 20)):7.2f}  {elapsed:6.1f}s"
              f"  -> {path.name}{'  *best' if va_loss == best else ''}")

        # HARD BUDGET CHECK - CLAUDE.md 4.12
        if elapsed > MAX_EPOCH_SECONDS:
            stopped_reason = (
                f"STOPPED: epoch {epoch} took {elapsed:.1f}s, over the "
                f"{MAX_EPOCH_SECONDS}s per-epoch cap"
            )
            print(f"\n{'!' * 70}\n{stopped_reason}\n"
                  f"Checkpoint for epoch {epoch} is saved. Awaiting direction - "
                  f"not continuing, not shrinking the model or the data.\n{'!' * 70}")
            break

    (ROOT / "outputs" / f"history_{tag}.json").write_text(
        json.dumps({"history": history, "stopped_reason": stopped_reason}, indent=2),
        encoding="utf-8",
    )
    print(f"\n{stopped_reason}")
    print(f"best val loss : {best:.4f}  -> {CKPT / f'{tag}_best.pt'}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["overfit", "train"])
    ap.add_argument("--data", default=str(DATA / "processed_dev.json"))
    ap.add_argument("--vocab", default=str(DATA / "vocab.json"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    if args.stage == "overfit":
        ok = overfit(args.data, args.vocab, n=args.n, steps=args.steps, lr=args.lr)
        sys.exit(0 if ok else 1)
    train(args.data, args.vocab, batch_size=args.batch_size, lr=args.lr, tag=args.tag)


if __name__ == "__main__":
    main()
