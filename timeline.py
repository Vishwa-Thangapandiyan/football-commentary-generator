"""
Rich commentary timeline: every annotated event of one half, narrated by V2.

    SoccerNet annotation (event subtype + grounded players)
        -> existing structured input
        -> V2 Transformer, greedy decoding, grounding mask active
        -> timestamped commentary line

Every line is generated at runtime. Nothing is copied from the reference text; the
reference is carried alongside only so the two can be compared.

    python timeline.py                       # demo match, half 1
    python timeline.py --half 2
    python timeline.py --variation           # also sample an alternative per line
    python timeline.py --game-index 84       # pick the match by test-split index

V1 is untouched: this reads checkpoints/v2_best.pt and data/processed_v2.json only.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

from generate import build_banned_mask, load_model, restore_names
from train import CommentaryDataset, collate

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed_v2.json"
CKPT = ROOT / "checkpoints" / "v2_best.pt"
VOCAB = ROOT / "data" / "vocab_v2.json"
OUT = ROOT / "outputs"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def secs(game_time):
    m = re.match(r"(\d+) - (\d+):(\d+)", game_time)
    return int(m.group(2)) * 60 + int(m.group(3)) if m else 0


@torch.no_grad()
def build(rows, variation=False, n_var=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, ck = load_model(str(CKPT), device, vocab_path=str(VOCAB))
    from sampling_experiment import sample_decode

    out = []
    for ex in rows:
        ds = CommentaryDataset([ex], vocab)
        src, _ = collate([ds[0]])
        src = src.to(device)
        banned = build_banned_mask([ex], vocab, device)
        tagged = vocab.decode(
            model.greedy_decode(src, max_len=ck["max_tgt"], banned=banned)[0].tolist())

        rec = {
            "game_time": ex["gameTime"],
            "t_seconds": secs(ex["gameTime"]),
            "event": ex["label"],
            "from_v1_label": ex.get("label_v1"),
            "players": {k: v["name"] for k, v in ex["grounding"].items()},
            "structured_input": ex["input"],
            "commentary_tagged": tagged,
            "commentary": restore_names(tagged, ex["grounding"]),
            "reference": restore_names(ex["target"], ex["grounding"]),
        }
        if variation:
            rec["variations"] = []
            for _ in range(n_var):
                sd = random.randrange(2 ** 31 - 1)
                v = sample_decode(model, src, banned, ck["max_tgt"], 0.9, 0.9, sd)
                rec["variations"].append({
                    "seed": sd,
                    "commentary": restore_names(
                        vocab.decode(v[0].tolist()), ex["grounding"]),
                })
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--game-index", type=int, default=84,
                    help="index into the test split identifying the match")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variation", action="store_true")
    ap.add_argument("--n-var", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show-reference", action="store_true")
    args = ap.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))[args.split]
    game = data[args.game_index]["game"]
    rows = sorted(
        [e for e in data if e["game"] == game
         and e["gameTime"].startswith(f"{args.half} -")],
        key=lambda e: secs(e["gameTime"]),
    )
    if not rows:
        sys.exit(f"no annotations for half {args.half} of that game")

    title = game.split("\\")[-1]
    print(f"{title}   |   half {args.half}   |   {len(rows)} events")
    print(f"generated live by V2 (greedy{', + sampling' if args.variation else ''})\n")
    print("=" * 78)

    recs = build(rows, variation=args.variation, n_var=args.n_var)
    for r in recs:
        mm, ss = divmod(r["t_seconds"], 60)
        who = ", ".join(v for v in r["players"].values() if v) or "-"
        print(f"\n{mm:02d}:{ss:02d}  [{r['event']}]  {who}")
        print(f"   {r['commentary']}")
        if args.show_reference:
            print(f"   ref: {r['reference'][:150]}")
        for v in r.get("variations", []):
            print(f"   var: {v['commentary']}")

    out = Path(args.out) if args.out else OUT / f"timeline_half{args.half}.json"
    out.write_text(json.dumps({
        "game": game, "half": args.half, "n_events": len(recs),
        "model": "checkpoints/v2_best.pt",
        "note": "every commentary line generated at runtime by the Transformer; "
                "reference kept only for comparison",
        "events": recs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'=' * 78}\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
