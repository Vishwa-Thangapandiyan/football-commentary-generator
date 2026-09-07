"""
PHASE 2 bridge: video features -> Spivak action spotting -> structured event
-> EXISTING Transformer (untouched) -> commentary.

    video frames (SoccerNet ResNET_TF2 features, 2 fps, 2048-d)
        -> Spivak confidence model (pretrained, Yahoo, CC-BY-4.0)
        -> per-class confidence timeline (17 SoccerNet-v2 actions)
        -> peak picking -> event class + timestamp
        -> our existing structured input format
        -> our Transformer, greedy decoding, constrained grounding
        -> commentary

Nothing here trains, modifies or re-saves the Transformer, its checkpoint, its
vocabulary, or the evaluation path. Inference only, greedy only.

    python spot_to_commentary.py
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
GAME = (ROOT / "data" / "caption-2024" / "england_epl" / "2014-2015" /
        "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal")
SPIVAK = (ROOT / "models" / "spivak" / "models" /
          "spotting_test_resnet_normalized_confidence_zoo_lr1e-3_dwd2e-4_sr0.5_mu2.0" /
          "best_model")
NORMALIZER = ROOT / "models" / "spivak" / "models" / "resnet_normalizer.pkl"

# Spivak / SoccerNet-v2 action classes, in the model's output order.
# Confirmed empirically: index 2 (Goal) peaks at the known goal.
SPIVAK_CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target",
    "Shots off target", "Clearance", "Ball out of play", "Throw-in", "Foul",
    "Indirect free-kick", "Direct free-kick", "Corner", "Yellow card", "Red card",
    "Yellow->red card",
]

# Spivak class -> the event label our Transformer was trained on.
# Only classes with a genuine counterpart are mapped. Everything else is dropped
# rather than forced into <no_event>, which mode-collapses (CLAUDE.md 4.15).
SPIVAK_TO_LABEL = {
    "Goal": "soccer-ball",
    "Substitution": "substitution",
    "Corner": "corner",
    "Yellow card": "y-card",
    "Red card": "r-card",
    "Penalty": "penalty",
    "Yellow->red card": "yr-card",
}
IMPORTANT = {"soccer-ball", "y-card", "r-card", "penalty", "yr-card", "substitution"}

FPS = 2.0

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def spot(features_path, chunk=224, batch=8):
    """ResNET features -> (T, 17) confidence timeline."""
    import tensorflow as tf

    feats = np.load(features_path)
    scaler = pickle.load(open(NORMALIZER, "rb"))
    X = scaler.transform(feats).astype("float32")

    fn = tf.saved_model.load(str(SPIVAK)).signatures["serving_default"]
    n = int(np.ceil(len(X) / chunk))
    pad = np.zeros((n * chunk - len(X), X.shape[1]), dtype="float32")
    Xc = np.concatenate([X, pad]).reshape(n, chunk, X.shape[1], 1)

    outs = []
    for i in range(0, n, batch):
        r = fn(main_input=tf.constant(Xc[i:i + batch]))["confidence_logits"].numpy()
        outs.append(r)
    logits = np.concatenate(outs, 0).reshape(n * chunk, len(SPIVAK_CLASSES))[:len(X)]
    return 1.0 / (1.0 + np.exp(-logits))


def pick_peaks(conf, threshold=0.30, min_gap_s=20.0):
    """Greedy non-maximum suppression over the confidence timeline."""
    events = []
    gap = int(min_gap_s * FPS)
    for c, name in enumerate(SPIVAK_CLASSES):
        label = SPIVAK_TO_LABEL.get(name)
        if label is None:
            continue
        s = conf[:, c]
        taken = []
        for i in np.argsort(-s):
            if s[i] < threshold:
                break
            if all(abs(i - j) > gap for j in taken):
                taken.append(int(i))
                events.append({
                    "t": i / FPS, "spivak_class": name, "label": label,
                    "confidence": float(s[i]),
                })
    return sorted(events, key=lambda e: e["t"])


def build_input(ev, home, away, half=1):
    """Our existing structured format. NOTE: no <P> player slots - player identity
    is not recoverable from video (no jersey OCR / roster linkage)."""
    minute = int(ev["t"] // 60)
    imp = "1" if ev["label"] in IMPORTANT else "0"
    return (f"<EVT> {ev['label']} <HALF> {half} <MIN> {minute} <IMP> {imp} "
            f"<HOME> {home.lower()} <AWAY> {away.lower()}")


@torch.no_grad()
def generate(events, home, away, ckpt="checkpoints/full_best.pt"):
    """Existing Transformer, greedy, constrained. Untouched."""
    from generate import build_banned_mask, load_model, restore_names
    from train import CommentaryDataset, collate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, ck = load_model(ckpt, device)
    out = []
    for ev in events:
        src_text = build_input(ev, home, away)
        ex = {"input": src_text, "target": "", "grounding": {}}
        ds = CommentaryDataset([ex], vocab)
        src, _ = collate([ds[0]])
        banned = build_banned_mask([ex], vocab, device)
        ids = model.greedy_decode(src.to(device), max_len=ck["max_tgt"], banned=banned)
        out.append({**ev, "input": src_text,
                    "commentary": restore_names(vocab.decode(ids[0].tolist()), {})})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(GAME / "1_ResNET_TF2.npy"))
    ap.add_argument("--home", default="Manchester United")
    ap.add_argument("--away", default="Arsenal")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--only", default=None,
                    help="comma-separated labels to keep, e.g. "
                         "'soccer-ball,corner'. Cards and substitutions generate "
                         "broken text without a player slot (CLAUDE.md 4.15), so "
                         "use this rather than raising --threshold, which cannot "
                         "separate them by confidence.")
    args = ap.parse_args()

    print("1) ResNET features -> Spivak action spotting ...")
    conf = spot(args.features)
    print(f"   confidence timeline: {conf.shape}  ({conf.shape[0]/FPS/60:.1f} min)")

    print(f"2) peak picking (threshold={args.threshold}) ...")
    events = pick_peaks(conf, args.threshold)
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        before = len(events)
        events = [e for e in events if e["label"] in keep]
        print(f"   filtered to {sorted(keep)}: {before} -> {len(events)}")
    print(f"   {len(events)} events mapped to Transformer labels")

    print("3) structured input -> existing Transformer (greedy) ...\n")
    for r in generate(events, args.home, args.away):
        mm, ss = int(r["t"] // 60), int(r["t"] % 60)
        print(f"  [{mm:02d}:{ss:02d}]  {r['spivak_class']:14s} conf={r['confidence']:.2f}"
              f"  -> {r['label']}")
        print(f"            INPUT : {r['input']}")
        print(f"            OUTPUT: {r['commentary']}\n")


if __name__ == "__main__":
    main()
