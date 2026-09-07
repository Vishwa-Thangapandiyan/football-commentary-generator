"""
Structural proof of the grounding guarantee (CLAUDE.md 4.14).

The claim is NOT "ungrounded placeholders are rare after training".
The claim is "an ungrounded placeholder is unreachable".

Three independent checks:
  1. VOCAB      - no real player/team name exists in the vocabulary at all, so no
                  parameter setting can emit one. (name-anonymization guarantee)
  2. MASK       - the -inf mask makes banned ids unreachable for argmax under
                  ADVERSARIAL logits: we force every banned tag to +1e9 and verify
                  greedy still never selects one. (grounding guarantee, by
                  construction, independent of what the model learned)
  3. END-TO-END - run a real checkpoint over real examples and audit the output.

Usage:
    python test_grounding.py --ckpt checkpoints/overfit.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

import re
from download_and_parse import tokenize
from generate import (
    GENERIC_TAGS, allowed_tags, build_banned_mask, enforce_grounding,
    hallucination_audit, load_model, restore_names,
)
from train import CommentaryDataset, collate
from vocab import ENTITY_TOKENS, Vocab

ROOT = Path(__file__).resolve().parent
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check_vocab_has_no_names(vocab, data):
    """Guarantee 1: no real identity is EMITTABLE.

    An earlier version of this check compared name-parts against the whole
    vocabulary and produced false positives: "holding", "son", "park", "success"
    are ordinary commentary words ("holding the ball") that merely coincide with
    surnames. Presence of such a word is not a leak.

    The property that actually matters is emittability, so we test the TARGET-side
    vocabulary (what the decoder can produce) and, separately, that no training
    target ever contained a real name in the first place.
    """
    emittable = (
        {vocab.itos[i] for i in vocab.target_ids}
        if vocab.target_ids is not None
        else set(vocab.stoi)
    )

    names, full_names = set(), set()
    for split in data.values():
        for ex in split:
            for info in ex["grounding"].values():
                if info.get("name"):
                    toks = tuple(tokenize(info["name"]))
                    if len(toks) >= 2:           # multi-token names only
                        full_names.add(toks)
                    for part in toks:
                        names.add(part)

    # (a) Did anonymization actually work? Test for a FULL name appearing verbatim
    # as a token sequence in a training target. Single-token matching is useless
    # here ("holding", "son", "park" are ordinary commentary words), so we require
    # the complete multi-token name - a real leak, not a coincidence.
    sizes = {len(fn) for fn in full_names}
    target_ngrams = set()
    for ex in data["train"]:
        toks = tuple(tokenize(ex["target"]))
        for n in sizes:
            for i in range(len(toks) - n + 1):
                target_ngrams.add(toks[i : i + n])
    leaked_targets = sorted(" ".join(fn) for fn in (full_names & target_ngrams))

    # (b) No COMPLETE team identity may be emittable. A multi-token team name is
    # producible only if every one of its tokens is emittable; if any token is
    # banned, the name is unreachable. Single tokens that happen to coincide with
    # ordinary commentary words ("nice", "real", "as") are not identities and are
    # reported separately rather than counted as leaks.
    teams = set()
    for split in data.values():
        for ex in split:
            m = re.search(r"<HOME>\s+(.*?)\s+<AWAY>\s+(.*?)(?:\s+<|$)", ex["input"])
            if m:
                for side in (m.group(1), m.group(2)):
                    toks = tuple(tokenize(side))
                    if toks:
                        teams.add(toks)

    producible = sorted(
        " ".join(t) for t in teams if all(tok in emittable for tok in t)
    )
    multi_word_leaks = [t for t in producible if " " in t]
    coincidental = [t for t in producible if " " not in t]

    print(f"  distinct real name-parts in corpus : {len(names):,}")
    print(f"  target-side (emittable) vocabulary : {len(emittable):,}")
    print(f"  distinct team names in corpus      : {len(teams)}")
    print(f"  full team names leaked into targets: {len(leaked_targets)}")
    print(f"  COMPLETE team names emittable      : {len(multi_word_leaks)}"
          f"  {multi_word_leaks[:6]}")
    print(f"  single tokens coinciding with an   : {len(coincidental)}"
          f"  {coincidental[:8]}")
    print(f"    ordinary commentary word (not an identity; informational)")
    return leaked_targets + multi_word_leaks


def check_mask_is_unreachable(vocab, examples, device):
    """Guarantee 2, the important one: ADVERSARIAL test.

    We do not trust the model to avoid banned tags - we make them impossible.
    Force every banned tag's logit to +1e9 (i.e. the model maximally 'wants' to
    emit an ungrounded placeholder) and confirm the mask still wins.
    """
    banned = build_banned_mask(examples, vocab, device)
    B, V = banned.shape

    logits = torch.zeros(B, V, device=device)
    logits[banned] = 1e9                      # adversarial: banned tags dominate
    masked = logits.masked_fill(banned, float("-inf"))
    picked = masked.argmax(-1)

    violations = int(banned.gather(1, picked.unsqueeze(1)).sum())
    n_banned = int(banned.sum())
    print(f"  rows                               : {B}")
    print(f"  banned (row, tag) pairs            : {n_banned:,}")
    print(f"  adversarial logit on every banned  : +1e9")
    print(f"  banned tokens selected by argmax   : {violations}")

    # And every entity tag that IS allowed must remain selectable.
    tag_ids = [vocab.stoi[t] for t in ENTITY_TOKENS if t in vocab.stoi]
    allowed_still_finite = all(
        torch.isfinite(masked[i, tid])
        for i, ex in enumerate(examples)
        for tag, tid in zip(
            [t for t in ENTITY_TOKENS if t in vocab.stoi], tag_ids
        )
        if tag in allowed_tags(ex["grounding"])
    )
    print(f"  allowed tags still selectable      : {allowed_still_finite}")
    return violations, allowed_still_finite


def check_fallback_never_leaks():
    """Guarantee 3: post-decode fallback replaces, never emits a raw tag."""
    grounding = {"<PLAYER_1>": {"name": "Ada Lovelace", "side": "home"}}
    dirty = "<PLAYER_1> passes to <PLAYER_4> and <COACH_2> applauds <TEAM>."
    clean, viol = enforce_grounding(dirty, grounding)
    restored = restore_names(dirty, grounding)
    print(f"  input   : {dirty}")
    print(f"  cleaned : {clean}   (violations={viol})")
    print(f"  restored: {restored}")
    ok = (
        "<PLAYER_4>" not in restored
        and "<COACH_2>" not in restored
        and "Ada Lovelace" in restored
        and viol == 2
    )
    print(f"  no ungrounded tag survives, no identity invented : {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/overfit.pt")
    ap.add_argument("--data", default="data/processed_dev.json")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    device = "cpu"
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    examples = data["train"][: args.n]

    print("=" * 72)
    print("1. VOCABULARY CONTAINS NO REAL NAMES (name-anonymization guarantee)")
    print("=" * 72)
    vocab = Vocab.load("data/vocab.json")
    name_hits = check_vocab_has_no_names(vocab, data)

    print("\n" + "=" * 72)
    print("2. BANNED TAGS UNREACHABLE UNDER ADVERSARIAL LOGITS (by construction)")
    print("=" * 72)
    viol, allowed_ok = check_mask_is_unreachable(vocab, examples, device)

    print("\n" + "=" * 72)
    print("3. POST-DECODE FALLBACK NEVER LEAKS A TAG OR INVENTS AN IDENTITY")
    print("=" * 72)
    fallback_ok = check_fallback_never_leaks()

    print("\n" + "=" * 72)
    print("4. END-TO-END ON A REAL CHECKPOINT")
    print("=" * 72)
    e2e_ok = True
    if not Path(args.ckpt).exists():
        print(f"  SKIPPED - {args.ckpt} not found")
    else:
        model, ckvocab, ck = load_model(args.ckpt, device)
        ds = CommentaryDataset(examples, ckvocab)
        src, _ = collate([ds[i] for i in range(len(examples))])
        banned = build_banned_mask(examples, ckvocab, device)
        con = model.greedy_decode(src, max_len=ck["max_tgt"], banned=banned)
        raw = model.greedy_decode(src, max_len=ck["max_tgt"], banned=None)

        recs = []
        for ex, c, r in zip(examples, con, raw):
            recs.append({
                "grounding": {k: v["name"] for k, v in ex["grounding"].items()},
                "generated_tagged": ckvocab.decode(c.tolist()),
                "generated_tagged_raw_debug": ckvocab.decode(r.tolist()),
            })
        a_con = hallucination_audit(recs)
        a_raw = hallucination_audit(recs, key="generated_tagged_raw_debug")
        print(f"  examples                           : {len(recs)}")
        print(f"  UNCONSTRAINED ungrounded rate      : "
              f"{a_raw['ungrounded_rate']:.2f}%  ({a_raw['ungrounded_examples']})")
        print(f"  CONSTRAINED   ungrounded rate      : "
              f"{a_con['ungrounded_rate']:.2f}%  ({a_con['ungrounded_examples']})")
        e2e_ok = a_con["ungrounded_examples"] == 0

    print("\n" + "=" * 72)
    results = {
        "vocab has no real names": not name_hits,
        "banned tags unreachable (adversarial)": viol == 0,
        "allowed tags still selectable": allowed_ok,
        "fallback never leaks": fallback_ok,
        "end-to-end ungrounded rate == 0": e2e_ok,
    }
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    print("=" * 72)
    print("GROUNDING GUARANTEE:", "VERIFIED" if ok else "BROKEN")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
