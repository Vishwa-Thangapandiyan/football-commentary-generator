"""
Stage 3: word-level vocabulary (CLAUDE.md 4.5).

No BPE / SentencePiece: small corpus, fast to build, trivial to debug, and easy to
explain in the presentation.

CRITICAL (CLAUDE.md rule 2 + 11):
  * The vocabulary is fitted on the OFFICIAL TRAIN SPLIT ONLY. Fitting on valid/test
    would leak.
  * Real player and team names never reach the target side, so they can never be
    generated. The entity tags are forced into the vocab so they are never <UNK>.

Usage:
    python vocab.py build --data data/processed_dev.json
    python vocab.py test  --data data/processed_dev.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from download_and_parse import MAX_PLAYERS, tokenize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# Forced into the vocab regardless of frequency, so grounding tags are never <UNK>.
ENTITY_TOKENS = (
    [f"<PLAYER_{i}>" for i in range(1, MAX_PLAYERS + 1)]
    + [f"<COACH_{i}>" for i in range(1, 4)]
    + ["<TEAM>", "<REFEREE>"]
)
FIELD_TOKENS = [
    "<EVT>", "<HALF>", "<MIN>", "<IMP>", "<HOME>", "<AWAY>", "<P>", "<C>",
    "<no_event>",
]

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Vocab:
    def __init__(self, itos, target_ids=None):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        # Ids that legitimately occur on the TARGET side of the training data.
        # Everything else (team names, event labels, field markers) is input-only
        # and is masked out at decode time - see CLAUDE.md 4.14.
        self.target_ids = set(target_ids) if target_ids is not None else None

    def __len__(self):
        return len(self.itos)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text)]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids, strip_special=True):
        toks = []
        for i in ids:
            t = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if strip_special and t in (PAD, BOS, EOS):
                continue
            toks.append(t)
        return detokenize(toks)

    def save(self, path):
        blob = {"itos": self.itos}
        if self.target_ids is not None:
            blob["target_ids"] = sorted(self.target_ids)
        Path(path).write_text(
            json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(blob["itos"], blob.get("target_ids"))


def detokenize(tokens):
    """Re-join word-level tokens into readable text (punctuation reattached).

    Note: the tokenizer regex `[A-Za-z0-9']+` keeps contractions whole ("side's",
    "haven't"), so a token that STARTS with an apostrophe is an opening quote
    ("'kick and rush'") and must keep its leading space. Special-casing it as a
    contraction suffix corrupted 12 round-trips on the full corpus.
    """
    out = ""
    no_space_before = set(".,!?;:)%")
    no_space_after = set("(")
    for t in tokens:
        if not out:
            out = t
        elif t in no_space_before:
            out += t
        elif out[-1] in no_space_after:
            out += t
        else:
            out += " " + t
    return out


def build(data_path, min_freq=2, out=None):
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    train = data["train"]
    if not train:
        raise SystemExit(f"No train examples in {data_path}")

    counter = Counter()
    for ex in train:  # TRAIN SPLIT ONLY - CLAUDE.md rule 11
        counter.update(tokenize(ex["input"]))
        counter.update(tokenize(ex["target"]))

    itos = [PAD, BOS, EOS, UNK]
    forced = [t for t in ENTITY_TOKENS + FIELD_TOKENS if t not in itos]
    itos += forced
    seen = set(itos)
    for tok, freq in counter.most_common():
        if freq >= min_freq and tok not in seen:
            itos.append(tok)
            seen.add(tok)

    # Target-side vocabulary: what the decoder is allowed to emit at all.
    # Input-only tokens (team names, event labels, field markers) are excluded so
    # a team name cannot be produced - structurally, not just by lack of training
    # signal (CLAUDE.md 4.14).
    tgt_counter = Counter()
    for ex in train:
        tgt_counter.update(tokenize(ex["target"]))
    vocab = Vocab(itos)
    target_ids = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
    target_ids |= {vocab.stoi[t] for t in ENTITY_TOKENS if t in vocab.stoi}
    target_ids |= {vocab.stoi[t] for t in tgt_counter if t in vocab.stoi}
    vocab.target_ids = target_ids
    input_only = len(vocab) - len(target_ids)
    print(f"target-side ids: {len(target_ids):,}  "
          f"(input-only tokens masked at decode: {input_only})")

    out = Path(out) if out else DATA / "vocab.json"
    vocab.save(out)

    # <UNK> rate on train, measured in tokens
    total = sum(counter.values())
    unk = sum(f for t, f in counter.items() if t not in vocab.stoi)
    print(f"corpus         : {len(train)} train examples, {total:,} tokens")
    print(f"distinct types : {len(counter):,}")
    print(f"min_freq       : {min_freq}")
    print(f"VOCAB SIZE     : {len(vocab):,}  (incl. {len(forced)} forced special)")
    print(f"<UNK> rate     : {unk:,}/{total:,} = {100.0 * unk / total:.2f}% of tokens")
    print(f"saved -> {out}")

    # Guarantee: no real player/team surname can be emitted as a name token because
    # targets only ever contain tags. Verify no bracket tag survived into the vocab.
    leaked = [t for t in vocab.itos if t.startswith("[")]
    print(f"leaked raw entity tags in vocab: {len(leaked)} {leaked[:5]}")
    return vocab


def test(data_path, vocab_path=None):
    """Stage 3 gate: text -> ids -> text round-trip must be identity."""
    vocab = Vocab.load(vocab_path or DATA / "vocab.json")
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    train = data["train"]

    print(f"vocab size {len(vocab):,}\n")
    ok = fail = 0
    for ex in train:
        for field in ("input", "target"):
            toks = tokenize(ex[field])
            if any(t not in vocab.stoi for t in toks):
                continue  # contains <UNK>; identity is not expected
            round_trip = vocab.decode(vocab.encode(ex[field]))
            if tokenize(round_trip) == toks:
                ok += 1
            else:
                fail += 1
                if fail <= 3:
                    print(f"MISMATCH\n  in : {toks}\n  out: {tokenize(round_trip)}")
    print(f"ROUND-TRIP (UNK-free fields): {ok} identical, {fail} mismatched")

    ex = next((e for e in train if e["label"] == "soccer-ball"), train[0])
    print(f"\n--- worked example (label={ex['label']}) ---")
    ids = vocab.encode(ex["target"], add_bos=True, add_eos=True)
    print(f"target : {ex['target'][:150]}")
    print(f"ids[:16]: {ids[:16]}")
    print(f"decoded : {vocab.decode(ids)[:150]}")
    tags = [t for t in ENTITY_TOKENS if t in vocab.stoi]
    print(f"\nentity tags present in vocab: {len(tags)}/{len(ENTITY_TOKENS)}")
    print(f"  ids: {[(t, vocab.stoi[t]) for t in tags[:5]]}")
    assert fail == 0, "round-trip is not identity - fix before proceeding"
    print("\nSTAGE 3 GATE: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "test"])
    ap.add_argument("--data", default=str(DATA / "processed_dev.json"))
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.stage == "build":
        build(args.data, min_freq=args.min_freq, out=args.out)
    else:
        test(args.data, args.vocab)


if __name__ == "__main__":
    main()
