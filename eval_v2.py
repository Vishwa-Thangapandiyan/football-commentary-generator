"""
V2 evaluation, and a like-for-like comparison against V1.

Both models are scored on the SAME 7,826 official test examples with the SAME
reference targets. Only the encoder input differs: V1 sees `<EVT> <no_event>` for
62% of rows, V2 sees the action subtype. That is precisely the thing under test -
does finer conditioning produce better text?

    python eval_v2.py                # full: BLEU + grounding, both models
    python eval_v2.py --probes-only  # just the targeted pass/shot/save probes

Read-only with respect to V1: never writes processed.json, vocab.json or
full_best.pt. Writes only outputs/eval_v2_*.json.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from generate import (
    build_banned_mask, corpus_bleu4, hallucination_audit, load_model, restore_names,
)
from train import CommentaryDataset, collate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

V1 = dict(name="V1", ckpt="checkpoints/full_best.pt",
          data="data/processed.json", vocab="data/vocab.json")
V2 = dict(name="V2", ckpt="checkpoints/v2_best.pt",
          data="data/processed_v2.json", vocab="data/vocab_v2.json")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@torch.no_grad()
def decode_all(cfg, split="test", batch=64, limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, ck = load_model(cfg["ckpt"], device, vocab_path=cfg["vocab"])
    rows = json.loads(Path(cfg["data"]).read_text(encoding="utf-8"))[split]
    if limit:
        rows = rows[:limit]
    ds = CommentaryDataset(rows, vocab)

    recs = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        src, _ = collate([ds[j] for j in range(i, min(i + batch, len(rows)))])
        src = src.to(device)
        banned = build_banned_mask(chunk, vocab, device)
        con = model.greedy_decode(src, max_len=ck["max_tgt"], banned=banned)
        raw = model.greedy_decode(src, max_len=ck["max_tgt"], banned=None)
        for ex, c, r in zip(chunk, con, raw):
            recs.append({
                "label": ex["label"],
                "grounding": {k: v["name"] for k, v in ex["grounding"].items()},
                "reference_tagged": ex["target"],
                "generated_tagged": vocab.decode(c.tolist()),
                "generated_tagged_raw_debug": vocab.decode(r.tolist()),
            })
        if (i + batch) % 1024 == 0:
            print(f"    {min(i+batch, len(rows))}/{len(rows)}", flush=True)
    return recs, ck


def score(cfg, limit=None):
    print(f"\n=== {cfg['name']} : {cfg['ckpt']} ===", flush=True)
    recs, ck = decode_all(cfg, limit=limit)
    bleu, prec = corpus_bleu4([r["generated_tagged"] for r in recs],
                              [r["reference_tagged"] for r in recs])
    aud = hallucination_audit(recs)
    raw = hallucination_audit(recs, key="generated_tagged_raw_debug")
    res = {
        "model": cfg["name"], "checkpoint": cfg["ckpt"], "epoch": ck["epoch"],
        "val_loss": round(ck["val_loss"], 4), "n_examples": len(recs),
        "bleu4": round(bleu, 2),
        "precisions": [round(p * 100, 1) for p in prec],
        "ungrounded_constrained_pct": round(aud["ungrounded_rate"], 3),
        "ungrounded_unconstrained_pct": round(raw["ungrounded_rate"], 3),
        "raw_entity_tags_leaked": aud["raw_tags"],
    }
    print(f"  BLEU-4 {res['bleu4']}   p1-4 {res['precisions']}")
    print(f"  ungrounded: constrained {res['ungrounded_constrained_pct']}%  "
          f"unconstrained {res['ungrounded_unconstrained_pct']}%")
    return res, recs


# ------------------------------------------------------------------ probes ---

PROBES = [
    ("pass  (2 players)", "pass", 2),
    ("cross (1 player)", "cross", 1),
    ("shot  (1 player)", "shot", 1),
    ("save  (shooter+keeper)", "save", 2),
    ("dribble (1 player)", "dribble", 1),
    ("foul  (1 player)", "foul", 1),
    ("offside (1 player)", "offside", 1),
    ("soccer-ball (2 players)", "soccer-ball", 2),
]
NAMES = ["Ander Herrera", "Ashley Young", "David de Gea"]


def make_example(label, n_players, half=1, minute=20):
    parts = [f"<EVT> {label}", f"<HALF> {half}", f"<MIN> {minute}",
             f"<IMP> {'1' if label in ('soccer-ball',) else '0'}",
             "<HOME> manchester united", "<AWAY> arsenal"]
    grounding = {}
    for k in range(1, n_players + 1):
        tag = f"<PLAYER_{k}>"
        parts.append(f"<P> {tag} home")
        grounding[tag] = {"name": NAMES[k - 1], "side": "home",
                          "shirt": None, "hash": None}
    return {"input": " ".join(parts), "target": "", "grounding": grounding}


@torch.no_grad()
def run_probes(cfg, n_variations=2):
    from sampling_experiment import sample_decode
    import random
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, ck = load_model(cfg["ckpt"], device, vocab_path=cfg["vocab"])
    print(f"\n=== TARGETED PROBES ({cfg['name']}) ===")
    out = []
    for title, label, n in PROBES:
        # Hyphenated labels ("soccer-ball") tokenize into several tokens, so a
        # whole-string vocab lookup is a false negative. Check the pieces.
        from download_and_parse import tokenize as _tok
        if any(t not in vocab.stoi for t in _tok(label)):
            print(f"\n  {title}: label '{label}' not representable - skipped")
            continue
        ex = make_example(label, n)
        ds = CommentaryDataset([ex], vocab)
        src, _ = collate([ds[0]])
        src = src.to(device)
        banned = build_banned_mask([ex], vocab, device)
        g = restore_names(
            vocab.decode(model.greedy_decode(
                src, max_len=ck["max_tgt"], banned=banned)[0].tolist()),
            ex["grounding"])
        print(f"\n  {title}")
        print(f"    input  : {ex['input']}")
        print(f"    GREEDY : {g}")
        vs = []
        for _ in range(n_variations):
            sd = random.randrange(2**31 - 1)
            t = restore_names(
                vocab.decode(sample_decode(model, src, banned, ck["max_tgt"],
                                           0.9, 0.9, sd)[0].tolist()),
                ex["grounding"])
            vs.append(t)
            print(f"    VAR    : {t}")
        out.append({"probe": title, "label": label, "n_players": n,
                    "input": ex["input"], "greedy": g, "variations": vs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    probes = run_probes(V2)
    (OUT / "eval_v2_probes.json").write_text(
        json.dumps(probes, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.probes_only:
        print(f"\nwrote {OUT/'eval_v2_probes.json'}")
        return

    r2, _ = score(V2, limit=args.limit)
    r1, _ = score(V1, limit=args.limit)
    (OUT / "eval_v2_compare.json").write_text(
        json.dumps({"v2": r2, "v1": r1, "probes": probes}, indent=2),
        encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"{'metric':<34}{'V1':>13}{'V2':>13}")
    print(f"{'BLEU-4 (7,826 official test)':<34}{r1['bleu4']:>13}{r2['bleu4']:>13}")
    print(f"{'val loss':<34}{r1['val_loss']:>13}{r2['val_loss']:>13}")
    print(f"{'ungrounded, constrained %':<34}"
          f"{r1['ungrounded_constrained_pct']:>13}{r2['ungrounded_constrained_pct']:>13}")
    print(f"{'ungrounded, unconstrained %':<34}"
          f"{r1['ungrounded_unconstrained_pct']:>13}{r2['ungrounded_unconstrained_pct']:>13}")
    print("=" * 62)
    print(f"\nwrote {OUT/'eval_v2_compare.json'}")


if __name__ == "__main__":
    main()
