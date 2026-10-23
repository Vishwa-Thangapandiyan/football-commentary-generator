"""
Paper metrics from saved evaluation files. CPU only; never touches a GPU.

Every eval file (train_bdh.py eval, colab_seeds.py) stores all generated test
sentences under "records". From those this computes:

  BLEU-4      the project's corpus BLEU (generate.corpus_bleu4), for continuity
  METEOR, ROUGE-L, CIDEr   pycocoevalcap scorers, the SoccerNet-Caption metric set
  length      output / reference tokens (BLEU's brevity penalty is driven by this)
  distinct-n  share of unique n-grams across all outputs (diversity)
  copied      share of outputs identical to some TRAIN target (memorisation)

All scorers see the same pre-tokenised text (download_and_parse.tokenize, entity
placeholders kept as single tokens). The PTB tokenizer is deliberately not used:
it splits placeholders into pieces and inflates n-gram overlap.

    python metrics.py outputs/seeds/eval_*_v3_*.json
    python metrics.py --compare A.json B.json      # paired bootstrap on BLEU-4
"""

import argparse
import glob
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from download_and_parse import tokenize
from generate import corpus_bleu4

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = d.get("records")
    if not recs:
        raise SystemExit(f"{path}: no 'records' (older eval file without all outputs)")
    return d.get("summary", {}), recs


def coco(hyps, refs):
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    H = {i: [" ".join(tokenize(h))] for i, h in enumerate(hyps)}
    R = {i: [" ".join(tokenize(r))] for i, r in enumerate(refs)}
    return {
        "METEOR": 100 * Meteor().compute_score(R, H)[0],
        "ROUGE_L": 100 * Rouge().compute_score(R, H)[0],
        "CIDEr": 100 * Cider().compute_score(R, H)[0],
    }


def distinct(hyps, n):
    grams = Counter()
    for h in hyps:
        t = tokenize(h)
        grams.update(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    return len(grams) / max(sum(grams.values()), 1)


def train_targets(data_path):
    d = json.loads(Path(data_path).read_text(encoding="utf-8"))
    return {" ".join(tokenize(r["target"])) for r in d["train"]}


def score_file(path, train_set=None):
    summary, recs = load(path)
    hyps = [r["generated_tagged"] for r in recs]
    refs = [r["reference_tagged"] for r in recs]
    bleu, prec = corpus_bleu4(hyps, refs)
    hl = sum(len(tokenize(h)) for h in hyps)
    rl = sum(len(tokenize(r)) for r in refs)
    out = {"file": Path(path).name, "n": len(recs), "BLEU4": bleu,
           "p1-4": [100 * p for p in prec], "len_ratio": hl / rl,
           **coco(hyps, refs),
           "distinct1": 100 * distinct(hyps, 1), "distinct2": 100 * distinct(hyps, 2)}
    if train_set is not None:
        out["copied_pct"] = 100 * sum(" ".join(tokenize(h)) in train_set
                                      for h in hyps) / len(hyps)
    return out


# ------------------------------------------------------- paired bootstrap ---

def sent_stats(hyp, ref):
    """Per-sentence BLEU sufficient statistics: clipped/total n-gram counts, lengths."""
    h, r = tokenize(hyp), tokenize(ref)
    row = [len(h), len(r)]
    for n in range(1, 5):
        hn = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        row += [sum(min(c, rn[g]) for g, c in hn.items()), sum(hn.values())]
    return row


def bleu_from(stats):
    c, r = stats[0], stats[1]
    ps = [stats[2 + 2 * k] / stats[3 + 2 * k] if stats[3 + 2 * k] else 0 for k in range(4)]
    if min(ps) <= 0:
        return 0.0
    bp = 1.0 if c > r else math.exp(1 - r / max(c, 1))
    return 100 * bp * math.exp(sum(math.log(p) for p in ps) / 4)


def paired_bootstrap(path_a, path_b, n_boot=1000, seed=0):
    """Koehn (2004): resample test sentences with replacement; how often is A > B?"""
    _, ra = load(path_a)
    _, rb = load(path_b)
    assert [r["reference_tagged"] for r in ra] == [r["reference_tagged"] for r in rb], \
        "systems must be scored on the same references in the same order"
    sa = [sent_stats(r["generated_tagged"], r["reference_tagged"]) for r in ra]
    sb = [sent_stats(r["generated_tagged"], r["reference_tagged"]) for r in rb]
    tot = lambda rows, idx: [sum(rows[i][k] for i in idx) for k in range(10)]
    full = range(len(sa))
    base_a, base_b = bleu_from(tot(sa, full)), bleu_from(tot(sb, full))
    rng = random.Random(seed)
    wins, deltas = 0, []
    for _ in range(n_boot):
        idx = [rng.randrange(len(sa)) for _ in full]
        d = bleu_from(tot(sa, idx)) - bleu_from(tot(sb, idx))
        deltas.append(d)
        wins += d > 0
    deltas.sort()
    lo, hi = deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot) - 1]
    print(f"A {Path(path_a).name}: BLEU {base_a:.2f}")
    print(f"B {Path(path_b).name}: BLEU {base_b:.2f}")
    print(f"A - B = {base_a - base_b:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"A wins {wins}/{n_boot}  (p = {1 - wins / n_boot:.3f} one-sided)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--train-data", default=None,
                    help="processed_*.json, to measure copying of train targets")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.compare:
        paired_bootstrap(*args.compare)
        return
    files = [f for pat in args.files for f in sorted(glob.glob(pat))]
    train_set = train_targets(args.train_data) if args.train_data else None
    rows = [score_file(f, train_set) for f in files]
    cols = ["BLEU4", "METEOR", "ROUGE_L", "CIDEr", "len_ratio", "distinct2", "copied_pct"]
    print(f"{'file':<36}" + "".join(f"{c:>11}" for c in cols))
    for r in rows:
        print(f"{r['file'][:35]:<36}" + "".join(
            f"{r[c]:>11.2f}" if c in r else f"{'-':>11}" for c in cols))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
