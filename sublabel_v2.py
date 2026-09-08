"""
V2 data build: split the single <no_event> class into action subtypes.

Reads  data/processed.json   (V1, never written to)
Writes data/processed_v2.json

Only rows whose label is <no_event> are touched. Every existing event class
(soccer-ball, corner, y-card, substitution, injury, ...) is preserved exactly.

WHY: <no_event> is 62% of training data and conflates passes, crosses, shots,
fouls, offsides, clearances, saves and dribbles under one label. Identical inputs
for different situations force the model to mode-collapse onto the most frequent
template ("is adjudged offside"). Splitting the label gives the encoder something
to condition on.

HOW: priority-ordered regex over the TARGET commentary text; first family wins.
The order encodes what a sentence is *about*, because nearly every sentence is a
chain (cross -> cleared -> corner):

  offside > foul > save > shot > cross > dribble > clearance > pass > _other

  offside   an offside call voids everything before it
  foul      a foul stops play; the preceding action is background
  save      the resolution of a shot; ranked ABOVE shot deliberately
  shot      highest-intent attacking action
  cross     "a cross which is cleared" is about the cross
  dribble   a run/take-on that is not already a shot or cross
  clearance defence-only sentences
  pass      least specific, most often incidental to something else

    python sublabel_v2.py            # writes data/processed_v2.json
    python sublabel_v2.py --dry-run  # distribution only, writes nothing

Does not modify V1: processed.json, vocab.json, full_best.pt are opened read-only
or not at all.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "processed.json"
DST = ROOT / "data" / "processed_v2.json"
NO_EVENT = "<no_event>"

# Priority-ordered. First family whose any-pattern matches assigns the subtype.
RULES = [
    ("offside", [
        r"\boffside\b", r"\bflag(?:ged)? up against\b", r"\bflag goes up\b",
        r"\braises? (?:his |the )?flag\b", r"\bflag(?:ged)? for offside\b",
    ]),
    ("foul", [
        r"\bfoul(?:s|ed)?\b", r"\bfree kick\b",
        r"\bpenalis(?:e|ed)\b|\bpenaliz(?:e|ed)\b",
        r"\bpenalty kick\b|\bawarded a penalty\b|\b(?:takes?|taking|going to take) the penalty\b",
        r"\bpulls? the (?:shirt|jersey)\b", r"\btrips?\b",
        r"\bbr(?:ought|ings|ing) .{0,20}down\b",
        r"\bcareless with his (?:tackle|challenge)\b",
        r"\btoo (?:forceful|aggressive|fierce|rough|careless) (?:with|in)\b",
        r"\brough (?:challenge|tackle)\b", r"\bslide tackle\b",
        r"\btackl(?:e|ed|ing)\b", r"\bhacks? down\b", r"\bwarning\b",
        r"\byellow card\b|\bred card\b|\bshown? a card\b",
        r"\bholds? (?:his|the) opponent\b",
        r"\bkicks? (?:an? )?opponent'?s? (?:leg|legs|shin)\b",
        r"\bimmediately stops play\b",
    ]),
    ("save", [
        r"\bsaves?\b", r"\bsaving\b",
        r"\bpulls? off (?:a |an )?(?:fine |great |brilliant |glorious |decent |excellent |superb )?save\b",
        r"\bbrilliant reflexes\b|\bquick reflexes\b",
        r"\bgoalkeeper .{0,25}(?:gathers|catches|holds|denies|stops)\b",
        r"\bpunches? the ball away\b", r"\btips? (?:it|the ball) (?:over|wide|away)\b",
        r"\bcomfortably gathers?\b", r"\bdives? (?:superbly|well|to) (?:to )?(?:stop|deny)\b",
    ]),
    ("shot", [
        r"\bshoots?\b|\bshot\b", r"\beffort\b", r"\bstrik(?:e|es|ing)\b",
        r"\b(?:low |quick )?drive\b", r"\bpiledriver\b", r"\bsnap shot\b", r"\bheader\b",
        r"\bgoes (?:just |narrowly )?(?:wide|over)\b", r"\bcrossbar\b|\bover the bar\b",
        r"(?:hits?|onto) the (?:post|bar)\b", r"\bsmashes? (?:the ball|against)\b",
        r"\bunleash(?:es)? a (?:shot|strike)\b", r"\bskies the ball\b", r"\bfires? the ball\b",
        r"\bgoal-bound\b", r"\bwide of the (?:left|right) post\b", r"\bpulls the trigger\b",
        r"\bfinish(?:es|ing)? (?:from|was|is)\b", r"\bsquanders? a great opportunity\b",
    ]),
    ("cross", [
        r"\bcross(?:es|ing)?\b", r"\bwhips? (?:the |it )?(?:ball )?in\b",
        r"\bdelivers? a (?:cross|ball)\b", r"\bcrosses into the box\b",
        r"\bflights? in the cross\b", r"\bswings? (?:it|the ball) in\b",
    ]),
    # Added for V2: a run / take-on that is not already a shot or a cross.
    ("dribble", [
        r"\bdribbl", r"\bskips? past\b", r"\bslalom", r"\bsolo run\b",
        r"\bruns? (?:with the ball|at the defence|towards goal|into the box)\b",
        r"\bbeats? (?:his|the) (?:man|marker|opponent)\b",
        r"\bzig-?zags?\b", r"\bjinks?\b", r"\bweaves?\b",
        r"\bbreaks? (?:free|away|clear)\b", r"\bsurges? (?:forward|past)\b",
        r"\braces? (?:towards|into|clear)\b", r"\btakes? on (?:his|an|the) (?:man|opponent|defender)\b",
    ]),
    ("clearance", [
        r"\bclear(?:s|ed|ance)?\b", r"\baverts? the threat\b",
        r"\bfoot in to clear\b", r"\bheads? (?:it |the ball )?clear\b",
    ]),
    ("pass", [
        r"\bpass(?:es|ed)?\b", r"\bthrough ball\b", r"\blong ball\b", r"\bchip pass\b",
        r"\bsends? the ball\b", r"\bplays? a ball\b", r"\bone-two\b",
        r"\bslips? the ball through\b",
    ]),
]
COMPILED = [(name, [re.compile(p, re.I) for p in pats]) for name, pats in RULES]
RESIDUAL = "other-play"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def subtype_of(target: str) -> str:
    for name, pats in COMPILED:
        if any(p.search(target) for p in pats):
            return name
    return RESIDUAL


def relabel(ex: dict) -> dict:
    """Return a copy with the new label reflected in BOTH `label` and `input`."""
    if ex["label"] != NO_EVENT:
        return ex
    new = subtype_of(ex["target"])
    out = dict(ex)
    out["label"] = new
    out["label_v1"] = NO_EVENT                       # provenance, for auditing
    # The encoder input carries the label; it must change too.
    out["input"] = re.sub(r"^<EVT>\s+\S+", f"<EVT> {new}", ex["input"], count=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(DST))
    args = ap.parse_args()

    src = Path(args.src)
    if Path(args.out).resolve() == src.resolve():
        sys.exit("refusing to overwrite the V1 data file")

    data = json.loads(src.read_text(encoding="utf-8"))
    out, stats = {}, {}
    for split, rows in data.items():
        new_rows = [relabel(e) for e in rows]
        out[split] = new_rows
        stats[split] = Counter(
            e["label"] for e in new_rows if e.get("label_v1") == NO_EVENT
        )

    tr = stats["train"]
    total = sum(tr.values())
    print(f"<no_event> rows reclassified (train): {total:,}\n")
    print(f"{'subtype':<12} {'train':>7} {'%':>6} {'valid':>7} {'test':>7}")
    for name, n in tr.most_common():
        print(f"{name:<12} {n:>7,} {100*n/total:>5.1f}% "
              f"{stats['valid'][name]:>7,} {stats['test'][name]:>7,}")

    # Sanity: existing classes must be untouched.
    v1 = Counter(e["label"] for e in data["train"] if e["label"] != NO_EVENT)
    v2 = Counter(e["label"] for e in out["train"] if e.get("label_v1") != NO_EVENT)
    print(f"\npre-existing classes preserved exactly: {v1 == v2}  ({len(v1)} classes)")
    print(f"total train rows unchanged: {len(data['train']) == len(out['train'])}")

    ex = next(e for e in out["train"] if e.get("label_v1") == NO_EVENT
              and e["label"] == "pass" and len(e["grounding"]) >= 2)
    print(f"\nworked example (pass, 2 players):")
    print(f"  input : {ex['input']}")
    print(f"  target: {ex['target'][:120]}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
