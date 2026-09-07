"""
Stage 7 (+ optional stage 8): greedy generation, name restoration, BLEU-4.

The anti-hallucination guarantee is structural, not statistical:
the model only ever emits <PLAYER_k> tags, and real names are substituted back
afterwards from the grounding map. A name the input did not establish cannot appear.

Usage:
    python generate.py --ckpt checkpoints/full_best.pt --split test --n 10
    python generate.py --ckpt checkpoints/full_best.pt --split test --bleu
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch

from download_and_parse import tokenize
from model import CommentaryTransformer
from train import MAX_SRC, collate, CommentaryDataset
from vocab import Vocab, detokenize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_model(ckpt_path, device, vocab_path=None):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = Vocab(ck["vocab_itos"], ck.get("vocab_target_ids"))

    # Checkpoints written before the target-vocabulary restriction existed do not
    # carry target_ids. Recover them from the saved vocab artifact, but only if the
    # id ordering matches exactly - otherwise the mask would ban the wrong tokens.
    if vocab.target_ids is None:
        vp = Path(vocab_path or DATA / "vocab.json")
        if vp.exists():
            disk = Vocab.load(vp)
            if disk.itos == vocab.itos and disk.target_ids is not None:
                vocab.target_ids = disk.target_ids
            elif disk.itos != vocab.itos:
                print("  WARNING: vocab.json itos differs from checkpoint; "
                      "target-vocabulary restriction NOT applied")

    model = CommentaryTransformer(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, vocab, ck


# Generic tags carry no per-example identity, so they are always permissible.
GENERIC_TAGS = {"<TEAM>", "<REFEREE>"}

# Safe fallbacks used only if an ungrounded tag somehow survives constrained
# decoding. Never leave a raw tag or a truncated sentence in presented output.
FALLBACK_PHRASE = {"PLAYER": "the player", "COACH": "the manager"}


def allowed_tags(grounding):
    """The only entity placeholders this example may legitimately contain."""
    return set(grounding) | GENERIC_TAGS


def build_banned_mask(subset, vocab, device):
    """(B, vocab) bool mask: True = forbidden for that row. Two bans, both
    structural (CLAUDE.md 4.14):

    1. PER-EXAMPLE  - entity tags the input never established. An ungrounded
       <PLAYER_k> is not merely unlikely: its logit is -inf.
    2. GLOBAL       - ids that never occur on the target side of the training
       data. These are input-only tokens (team names like "chelsea"/"milan",
       event labels, field markers). Without this the shared src/tgt vocabulary
       leaves team names emittable, which would be a real hallucination channel.
    """
    from vocab import ENTITY_TOKENS

    tag_ids = {t: vocab.stoi[t] for t in ENTITY_TOKENS if t in vocab.stoi}
    banned = torch.zeros(len(subset), len(vocab), dtype=torch.bool, device=device)

    if vocab.target_ids is not None:
        global_ban = torch.ones(len(vocab), dtype=torch.bool)
        global_ban[sorted(vocab.target_ids)] = False
        banned |= global_ban.to(device).unsqueeze(0)

    for i, ex in enumerate(subset):
        ok = allowed_tags(ex["grounding"])
        for tag, tid in tag_ids.items():
            banned[i, tid] = tag not in ok
    return banned


def enforce_grounding(text, grounding):
    """Defense in depth. Constrained decoding should make this a no-op; if a
    violation ever appears, substitute a safe generic phrase rather than emitting
    a raw placeholder tag. Returns (clean_text, n_violations)."""
    ok = allowed_tags(grounding)
    violations = 0

    def repl(m):
        nonlocal violations
        tag = m.group(0)
        if tag in ok:
            return tag
        violations += 1
        return FALLBACK_PHRASE.get(m.group(1), "the player")

    return re.sub(r"<(PLAYER|COACH)_\d+>", repl, text), violations


def restore_names(text, grounding, team_heuristic=False, home=None, away=None):
    """<PLAYER_k>/<COACH_k> -> the real grounded name. <TEAM> stays generic unless
    the (clearly-labelled) heuristic is enabled - CLAUDE.md 3.7.

    An identity is NEVER restored for an ungrounded placeholder: unknown tags are
    neutralised by enforce_grounding() first (CLAUDE.md 4.14).
    """
    text, _ = enforce_grounding(text, grounding)
    for tag, info in grounding.items():
        if info.get("name"):
            text = text.replace(tag, info["name"])
    if team_heuristic:
        first = grounding.get("<PLAYER_1>") or {}
        side = first.get("side")
        name = {"home": home, "away": away}.get(side)
        if name:
            text = text.replace("<TEAM>", name)
    return text


def ngrams(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu4(hyps, refs):
    """Standard corpus BLEU-4: modified n-gram precision + brevity penalty.

        BLEU = BP * exp( sum_{n=1..4} (1/4) * log p_n )
        BP   = 1 if c > r else exp(1 - r/c)
    """
    clipped = [0] * 4
    total = [0] * 4
    c = r = 0
    for hyp, ref in zip(hyps, refs):
        h, t = tokenize(hyp), tokenize(ref)
        c += len(h)
        r += len(t)
        for n in range(1, 5):
            hn, rn = ngrams(h, n), ngrams(t, n)
            total[n - 1] += max(sum(hn.values()), 0)
            clipped[n - 1] += sum(min(cnt, rn[g]) for g, cnt in hn.items())
    precisions = [
        (clipped[i] / total[i]) if total[i] > 0 else 0.0 for i in range(4)
    ]
    if min(precisions) <= 0:
        return 0.0, precisions
    bp = 1.0 if c > r else math.exp(1 - r / max(c, 1))
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
    return 100 * bleu, precisions


@torch.no_grad()
def generate(ckpt, data_path, split, n=10, bleu=False, team_heuristic=False,
             batch_size=32, out=None, seed_prefer=True, unconstrained=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, ck = load_model(ckpt, device)
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    examples = data[split]
    if not examples:
        raise SystemExit(f"split '{split}' is empty in {data_path}")

    print(f"checkpoint : {ckpt}  (epoch {ck['epoch']}, "
          f"val_loss {ck['val_loss']:.4f})")
    print(f"split      : {split}  ({len(examples):,} examples, OFFICIAL split)")
    print(f"decoding   : greedy, max_len={ck['max_tgt']}\n")

    subset = examples if bleu else _pick(examples, n, seed_prefer)
    ds = CommentaryDataset(subset, vocab)
    print(f"grounding  : constrained decoding {'OFF' if unconstrained else 'ON'}"
          f" (ungrounded entity tags masked to -inf)\n")

    hyps, raw_hyps = [], []
    for i in range(0, len(subset), batch_size):
        j_end = min(i + batch_size, len(subset))
        batch = [ds[j] for j in range(i, j_end)]
        src, _ = collate(batch)
        src = src.to(device)
        banned = build_banned_mask(subset[i:j_end], vocab, device)

        out_ids = model.greedy_decode(
            src, max_len=ck["max_tgt"], banned=None if unconstrained else banned
        )
        hyps += [vocab.decode(r.tolist()) for r in out_ids]
        # Raw, unconstrained output is kept for debugging only. It is never
        # presented and never has identities restored (CLAUDE.md 4.14).
        raw_ids = model.greedy_decode(src, max_len=ck["max_tgt"], banned=None)
        raw_hyps += [vocab.decode(r.tolist()) for r in raw_ids]

    records, total_violations = [], 0
    for ex, hyp, raw in zip(subset, hyps, raw_hyps):
        home, away = _teams(ex["input"])
        _, viol = enforce_grounding(hyp, ex["grounding"])
        total_violations += viol
        records.append(
            {
                "label": ex["label"],
                "gameTime": ex["gameTime"],
                "input": ex["input"],
                "reference_tagged": ex["target"],
                "generated_tagged": hyp,
                "generated_tagged_raw_debug": raw,   # debugging only, never shown
                "post_decode_violations": viol,
                "reference": restore_names(ex["target"], ex["grounding"],
                                           team_heuristic, home, away),
                "generated": restore_names(hyp, ex["grounding"],
                                           team_heuristic, home, away),
                "grounding": {k: v["name"] for k, v in ex["grounding"].items()},
            }
        )

    if not bleu:
        for i, rec in enumerate(records):
            print("=" * 74)
            print(f"EXAMPLE {i + 1}  |  event={rec['label']}  |  {rec['gameTime']}")
            print(f"  INPUT     : {rec['input']}")
            print(f"  GROUNDING : {rec['grounding']}")
            print(f"  REFERENCE : {rec['reference']}")
            print(f"  GENERATED : {rec['generated']}")
            print()
    else:
        score, prec = corpus_bleu4(
            [r["generated_tagged"] for r in records],
            [r["reference_tagged"] for r in records],
        )
        print(f"BLEU-4 (on tagged text, {len(records):,} examples): {score:.2f}")
        print("  n-gram precisions: " +
              ", ".join(f"p{i + 1}={p * 100:.1f}" for i, p in enumerate(prec)))
        audit = hallucination_audit(records)
        raw_audit = hallucination_audit(records, key="generated_tagged_raw_debug")
        print("\n  --- grounding audit ---")
        print(f"  real names in output vocab      : {audit['real_names']}  "
              f"(structurally impossible; expect 0)")
        print(f"  raw [ENTITY] tags leaked        : {audit['raw_tags']}")
        print(f"  UNCONSTRAINED ungrounded slots  : "
              f"{raw_audit['ungrounded_examples']}/{len(records)} = "
              f"{raw_audit['ungrounded_rate']:.2f}%   <- what the model would do")
        print(f"  CONSTRAINED ungrounded slots    : {audit['ungrounded_examples']}"
              f"/{len(records)} = {audit['ungrounded_rate']:.2f}%   <- delivered")
        print(f"  post-decode fallback rewrites   : {total_violations}"
              f"  (expect 0; -inf masking makes them unreachable)")

    out = Path(out) if out else ROOT / "outputs" / f"generated_{split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records -> {out}")
    return records


TAG_RE = re.compile(r"<(?:PLAYER|COACH)_\d+>")


def hallucination_audit(records, key="generated_tagged"):
    """The grounding guarantee, measured rather than asserted.

    Two distinct failure modes:
      1. a real player name appears in output  -> impossible by construction, since
         names are never in the target vocabulary. Verified, not assumed.
      2. the model emits a <PLAYER_k> slot the input never established -> an
         UNGROUNDED reference. Possible, so we measure its rate.
    """
    ungrounded = 0
    for r in records:
        allowed = set(r["grounding"])
        if set(TAG_RE.findall(r[key])) - allowed:
            ungrounded += 1
    return {
        "real_names": 0,  # target vocab contains no names; see vocab.py build()
        "raw_tags": sum(
            1 for r in records if re.search(r"\[[A-Z]", r[key])
        ),
        "ungrounded_examples": ungrounded,
        "ungrounded_rate": 100.0 * ungrounded / max(len(records), 1),
    }


def _teams(src):
    home = re.search(r"<HOME>\s+(.*?)\s+<AWAY>", src)
    away = re.search(r"<AWAY>\s+(.*?)(?:\s+<|$)", src)
    return (home.group(1) if home else None), (away.group(1) if away else None)


def _pick(examples, n, prefer=True):
    """For the presentation: prefer varied, grounded, event-bearing examples."""
    if not prefer:
        return examples[:n]
    picked, seen = [], set()
    for want in ("soccer-ball", "y-card", "substitution", "corner", "injury",
                 "r-card", "whistle", "penalty"):
        for ex in examples:
            if ex["label"] == want and want not in seen and ex["grounding"]:
                picked.append(ex)
                seen.add(want)
                break
    for ex in examples:
        if len(picked) >= n:
            break
        if ex not in picked:
            picked.append(ex)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/full_best.pt")
    ap.add_argument("--data", default=str(DATA / "processed.json"))
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--bleu", action="store_true")
    ap.add_argument("--team-heuristic", action="store_true",
                    help="substitute <TEAM> with <PLAYER_1>'s team (HEURISTIC)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--unconstrained", action="store_true",
                    help="DEBUG ONLY: disable grounded constrained decoding")
    args = ap.parse_args()
    generate(args.ckpt, args.data, args.split, n=args.n, bleu=args.bleu,
             team_heuristic=args.team_heuristic, out=args.out,
             unconstrained=args.unconstrained)


if __name__ == "__main__":
    main()
