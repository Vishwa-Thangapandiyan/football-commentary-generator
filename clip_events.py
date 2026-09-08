"""
Canonical structured events for one clip, then commentary from the existing model.

    cached Spivak confidence timeline
        -> peak picking inside the clip window
        -> CANONICAL EVENT JSON  (the stable interface for anything downstream)
        -> spot_to_commentary.build_input()   (unchanged structured-input builder)
        -> generate.load_model + greedy_decode (unchanged Transformer)
        -> timestamped commentary JSON

Reuses the existing inference path exactly. Does not modify the Transformer, the
checkpoint, training, evaluation, or the JSON demo. Reads the cached confidence
timeline so TensorFlow is not needed at all.

    python clip_events.py
    python clip_events.py --start 1710 --end 1890 --threshold 0.30

Player identity is NOT recoverable from video, so every event here has zero player
slots. Labels that always carried a player in training (y-card, substitution,
penalty, r-card) therefore generate degraded text; each event is flagged with
`text_quality` so callers can filter instead of discovering it at the podium.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from spot_to_commentary import (
    SPIVAK_CLASSES, SPIVAK_TO_LABEL, IMPORTANT, build_input, pick_peaks,
)

ROOT = Path(__file__).resolve().parent
CONF = ROOT / "outputs" / "spivak_confidence_half1.npy"
CKPT = ROOT / "checkpoints" / "full_best.pt"
OUT = ROOT / "outputs"

CLIP_START, CLIP_END = 28 * 60 + 30, 31 * 60 + 30
HOME, AWAY = "Manchester United", "Arsenal"
HALF = 1

# Labels that read correctly with no player slot (they have player-less examples in
# the training data). Everything else degrades - see CLAUDE.md 4.15.
CLEAN_WITHOUT_PLAYER = {"soccer-ball", "corner", "whistle", "time"}

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_canonical_events(conf, start_s, end_s, threshold):
    """Spivak peaks -> canonical event records. This JSON is the stable contract."""
    events = []
    for i, ev in enumerate(
        [e for e in pick_peaks(conf, threshold) if start_s <= e["t"] <= end_s], 1
    ):
        label = ev["label"]
        minute, second = int(ev["t"] // 60), int(ev["t"] % 60)
        events.append({
            "event_id": f"E{i:02d}",
            "t_seconds": round(ev["t"], 2),
            "game_time": f"{HALF} - {minute:02d}:{second:02d}",
            "half": HALF,
            "minute": minute,
            "spivak_class": ev["spivak_class"],
            "detector_confidence": round(ev["confidence"], 3),
            "event": label,                       # our Transformer's label space
            "important": label in IMPORTANT,
            "home": HOME,
            "away": AWAY,
            "players": [],                        # not recoverable from video
            "text_quality": ("clean" if label in CLEAN_WITHOUT_PLAYER
                             else "degraded_no_player"),
        })
    return events


@torch.no_grad()
def generate_commentary(events):
    """Existing path only: build_input -> load_model -> greedy_decode."""
    from generate import build_banned_mask, load_model, restore_names
    from train import CommentaryDataset, collate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, ck = load_model(str(CKPT), device)
    out = []
    for ev in events:
        src_text = build_input({"t": ev["t_seconds"], "label": ev["event"]},
                               ev["home"], ev["away"], half=ev["half"])
        ex = {"input": src_text, "target": "", "grounding": {}}
        ds = CommentaryDataset([ex], vocab)
        src, _ = collate([ds[0]])
        banned = build_banned_mask([ex], vocab, device)
        ids = model.greedy_decode(src.to(device), max_len=ck["max_tgt"], banned=banned)
        tagged = vocab.decode(ids[0].tolist())
        out.append({
            **ev,
            "structured_input": src_text,
            "commentary_tagged": tagged,
            "commentary": restore_names(tagged, {}),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=CLIP_START)
    ap.add_argument("--end", type=float, default=CLIP_END)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--only-clean", action="store_true",
                    help="emit only events that read correctly without a player")
    args = ap.parse_args()

    if not CONF.exists():
        sys.exit(f"missing {CONF} - run spot_to_commentary.py once to create it")
    conf = np.load(CONF)

    events = build_canonical_events(conf, args.start, args.end, args.threshold)
    if args.only_clean:
        events = [e for e in events if e["text_quality"] == "clean"]
    if not events:
        sys.exit("no events detected in that window at that threshold")

    a, b = int(args.start), int(args.end)
    clip = f"{a//60}:{a%60:02d}-{b//60}:{b%60:02d}"
    print(f"clip {clip}  |  threshold {args.threshold}  |  "
          f"{len(events)} canonical events\n")

    results = generate_commentary(events)

    payload = {
        "clip": clip,
        "half": HALF,
        "teams": {"home": HOME, "away": AWAY},
        "source": "Spivak action spotting over SoccerNet ResNET features "
                  "(cached confidence timeline)",
        "detector_threshold": args.threshold,
        "generator": "project Transformer, greedy decoding, grounding mask active",
        "player_identity": "not recoverable from video - all events have zero "
                           "player slots",
        "n_events": len(results),
        "events": results,
    }
    ep = OUT / "clip_events.json"
    cp = OUT / "clip_commentary.json"
    ep.write_text(json.dumps({**payload, "events": events}, indent=2), encoding="utf-8")
    cp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for r in results:
        flag = "" if r["text_quality"] == "clean" else "   [DEGRADED - no player slot]"
        print(f"[{r['minute']:02d}:{int(r['t_seconds'])%60:02d}]  "
              f"{r['spivak_class']:14s} conf={r['detector_confidence']:.2f} "
              f"-> {r['event']}{flag}")
        print(f"        INPUT : {r['structured_input']}")
        print(f"        OUTPUT: {r['commentary']}\n")

    print(f"canonical events   -> {ep.relative_to(ROOT)}")
    print(f"timestamped output -> {cp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
