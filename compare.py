"""
BDH vs Transformer, side by side, on the same examples and the same grounding mask.

    python compare.py                       # 8 test events, one per event type
    python compare.py --n 20 --split valid
    python compare.py --full                # whole split: BLEU-4 + grounding audit
                                            # for both, then `python metrics.py`

Both checkpoints load through generate.load_model, which dispatches on the
checkpoint's architecture, and decode greedily with the same constrained-decoding
mask, so the only difference between the two columns is the architecture.

--full decodes every example twice per model. On a laptop GPU that is a long,
hot run; the paper numbers were produced on Colab (colab_seeds.py).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from generate import build_banned_mask, load_model, restore_names
from train import CommentaryDataset, collate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS = {
    "BDH": "checkpoints/bdh_v3_best.pt",
    "Transformer": "checkpoints/transformer_v3_best.pt",
}


def pick(rows, n):
    """One example per event type first (with players), then fill in order."""
    picked, seen = [], set()
    for want in ("soccer-ball", "shot", "y-card", "corner", "substitution", "foul",
                 "offside", "save", "<no_event>", "cross"):
        for r in rows:
            if r["label"] == want and want not in seen and r["grounding"]:
                picked.append(r)
                seen.add(want)
                break
    for r in rows:
        if len(picked) >= n:
            break
        if r not in picked:
            picked.append(r)
    return picked[:n]


@torch.no_grad()
def side_by_side(args, device):
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))[args.split]
    ex = pick(rows, args.n)
    gens = {}
    for name, ckpt in MODELS.items():
        model, vocab, ck = load_model(ckpt, device)
        ds = CommentaryDataset(ex, vocab)
        src, _ = collate([ds[i] for i in range(len(ex))])
        out = model.greedy_decode(src.to(device), max_len=ck["max_tgt"],
                                  banned=build_banned_mask(ex, vocab, device))
        gens[name] = [restore_names(vocab.decode(o.tolist()), e["grounding"])
                      for o, e in zip(out, ex)]
        del model
    for i, e in enumerate(ex):
        print(f"\n[{i + 1}] event={e['label']}  {e['gameTime']}")
        print(f"  INPUT       : {e['input']}")
        print(f"  REFERENCE   : {e['reference_real']}")
        for name in MODELS:
            print(f"  {name:<12}: {gens[name][i]}")


def full(args):
    from eval_v2 import score
    results = {}
    for name, ckpt in MODELS.items():
        res, recs = score(dict(name=name, ckpt=ckpt, data=args.data, vocab=None))
        results[name] = res
        path = OUT / f"compare_{name.lower()}_{args.split}.json"
        path.write_text(json.dumps({"summary": res, "records": recs}, indent=1,
                                   ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {path}")
    print(f"\n{'metric':<34}" + "".join(f"{n:>14}" for n in MODELS))
    for key in ("bleu4", "val_loss", "ungrounded_constrained_pct",
                "ungrounded_unconstrained_pct"):
        print(f"{key:<34}" + "".join(f"{results[n][key]:>14}" for n in MODELS))
    print("\nfull metric set:  python metrics.py outputs/compare_*_" + args.split + ".json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_v3.json")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.full:
        full(args)
    else:
        side_by_side(args, device)


if __name__ == "__main__":
    main()
