"""
The polished end-to-end demo.

    224p broadcast video  ->  YOLO person boxes (canvas overlay, precomputed)
    Spivak action spotting ->  VIDEO-DERIVED events   (no player identity)
    SoccerNet annotations  ->  ANNOTATION-DERIVED events (fine-grained + players)
                           ->  canonical JSON
                           ->  V2 Transformer, greedy or sampled, at runtime
                           ->  timestamped commentary

    python demo_final.py     ->  http://127.0.0.1:8020

THE TWO EVENT SOURCES ARE DELIBERATELY KEPT SEPARATE AND LABELLED AS SUCH:

  VIDEO-DERIVED   Spivak action spotting run over SoccerNet ResNET features of this
                  video. Genuinely inferred from pixels. Coarse (goal/corner/card)
                  and carries NO player identity, because identity is not
                  recoverable from video - so these lines have no names.

  ANNOTATION      SoccerNet's human commentary annotations, re-labelled by V2 into
                  fine-grained actions (pass/shot/cross/foul/...). These carry
                  grounded players, so these lines DO have names. They are NOT
                  detected from the video.

Commentary for both is generated at runtime by the V2 Transformer. Nothing is
replayed. YOLO locates people only - it never identifies anyone.

V1 is untouched; this reads checkpoints/v2_best.pt and data/vocab_v2.json.
"""

import json
import random
import re
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch

from generate import build_banned_mask, load_model, restore_names
from sampling_experiment import sample_decode
from spot_to_commentary import build_input as build_video_input
from train import CommentaryDataset, collate

ROOT = Path(__file__).resolve().parent
PAGE_FILE = ROOT / "web" / "final_demo.html"
VIDEO = ROOT / "outputs" / "web" / "half1.mp4"
MKV = (ROOT / "data" / "caption-2024" / "england_epl" / "2014-2015" /
       "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal" / "1_224p.mkv")
SPIVAK = ROOT / "outputs" / "spivak_detections_half1.json"
YOLO_BOXES = ROOT / "outputs" / "yolo_boxes_half1.json"
DATA_V2 = ROOT / "data" / "processed_v2.json"
CKPT = ROOT / "checkpoints" / "v2_best.pt"
VOCAB = ROOT / "data" / "vocab_v2.json"

PORT = 8020
HOME, AWAY = "Manchester United", "Arsenal"
REF_IDX = 84
# Video-derived events carry NO players. Only labels that actually have
# player-less training support may be narrated. Measured: 0 of 812 `soccer-ball`
# training examples have zero players, so a player-less goal is fully
# out-of-distribution and produces broken text ("the rebound fell to block and
# headed low"). We therefore SHOW the goal detection but do not narrate it.
VIDEO_SHOW = {"soccer-ball", "corner"}
# Video gives the event but never WHO. Rather than emitting a player-less input
# (out-of-distribution -> broken text), we supply ANONYMOUS slots: the model fills
# <PLAYER_k> and we deliberately DO NOT resolve them to names. The slot COUNT is a
# prior from the event type, not something detected - stated as such in the UI.
VIDEO_SLOTS = {"soccer-ball": 2, "corner": 1, "y-card": 1, "r-card": 1,
               "substitution": 2, "penalty": 1, "yr-card": 1}
TEMPERATURE, TOP_P = 1.0, 0.9      # 0.9 was too peaked; 1.0 gives real variety

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ensure_video():
    if VIDEO.exists():
        return
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    print("remuxing mkv -> mp4 ...")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(MKV), "-c", "copy",
                    "-movflags", "+faststart", str(VIDEO)], check=True)


def secs(gt):
    m = re.match(r"(\d+) - (\d+):(\d+)", gt)
    return int(m.group(2)) * 60 + int(m.group(3)) if m else 0


ensure_video()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading V2 on {DEVICE} ...")
MODEL, VOCAB_OBJ, CK = load_model(str(CKPT), DEVICE, vocab_path=str(VOCAB))
print(f"  epoch {CK['epoch']}, val_loss {CK['val_loss']:.4f}, "
      f"vocab {len(VOCAB_OBJ):,}")

# ---- build the merged event list -------------------------------------------
EVENTS = []

# 1. VIDEO-DERIVED: Spivak detections, no players.
for e in json.loads(SPIVAK.read_text(encoding="utf-8")):
    if e["label"] not in VIDEO_SHOW:
        continue
    n_slots = VIDEO_SLOTS.get(e["label"], 1)
    src_text = build_video_input({"t": e["t"], "label": e["label"]},
                                 HOME, AWAY, half=1)
    grounding = {}
    for k in range(1, n_slots + 1):
        tag = f"<PLAYER_{k}>"
        src_text += f" <P> {tag} unknown"
        grounding[tag] = {"name": None, "side": "unknown",
                          "shirt": None, "hash": None}   # name=None -> stays a tag
    EVENTS.append({
        "source": "video",
        "t": round(e["t"], 2),
        "event": e["label"],
        "spivak_class": e["spivak_class"],
        "confidence": round(e["confidence"], 3),
        "players": {},
        "player_slots": n_slots,
        "slots_note": f"{n_slots} anonymous player slot(s), count assumed from the "
                      f"event type. Identity is NOT detected - the output keeps "
                      f"<PLAYER_k> placeholders rather than inventing names.",
        "narratable": True,
        "no_narration_reason": None,
        "input": src_text,
        "_grounding": grounding,
    })

# 2. ANNOTATION-DERIVED: SoccerNet annotations, fine-grained, WITH players.
_d = json.loads(DATA_V2.read_text(encoding="utf-8"))["test"]
_game = _d[REF_IDX]["game"]
for ex in _d:
    if ex["game"] != _game or not ex["gameTime"].startswith("1 -"):
        continue
    EVENTS.append({
        "source": "annotation",
        "narratable": True,
        "no_narration_reason": None,
        "t": secs(ex["gameTime"]),
        "event": ex["label"],
        "from_v1_label": ex.get("label_v1"),
        "confidence": None,
        "players": {k: v["name"] for k, v in ex["grounding"].items()},
        "input": ex["input"],
        "_grounding": ex["grounding"],
        "reference": restore_names(ex["target"], ex["grounding"]),
    })

EVENTS.sort(key=lambda e: (e["t"], e["source"]))
for i, e in enumerate(EVENTS):
    e["id"] = i
print(f"events: {sum(1 for e in EVENTS if e['source']=='video')} video-derived + "
      f"{sum(1 for e in EVENTS if e['source']=='annotation')} annotation-derived")

YOLO_DATA = (json.loads(YOLO_BOXES.read_text(encoding="utf-8"))
             if YOLO_BOXES.exists() else None)
print(f"yolo overlay: {'loaded' if YOLO_DATA else 'NOT PRECOMPUTED (run precompute_yolo.py)'}")


@torch.no_grad()
def generate(idx, mode="greedy"):
    ev = EVENTS[idx]
    grounding = ev.get("_grounding", {})
    ex = {"input": ev["input"], "target": "", "grounding": grounding}
    ds = CommentaryDataset([ex], VOCAB_OBJ)
    src, _ = collate([ds[0]])
    src = src.to(DEVICE)
    banned = build_banned_mask([ex], VOCAB_OBJ, DEVICE)

    t0 = time.perf_counter()
    seed = None
    if mode == "variation":
        # The distribution is peaked, so a draw can land on the greedy mode. Retry
        # a few times so "Generate Variation" actually shows something different.
        base = VOCAB_OBJ.decode(
            MODEL.greedy_decode(src, max_len=CK["max_tgt"], banned=banned)[0].tolist())
        for _ in range(4):
            seed = random.randrange(2 ** 31 - 1)
            ids = sample_decode(MODEL, src, banned, CK["max_tgt"],
                                TEMPERATURE, TOP_P, seed)
            if VOCAB_OBJ.decode(ids[0].tolist()) != base:
                break
    else:
        ids = MODEL.greedy_decode(src, max_len=CK["max_tgt"], banned=banned)
    ms = (time.perf_counter() - t0) * 1000

    tagged = VOCAB_OBJ.decode(ids[0].tolist())
    return {
        "id": idx, "source": ev["source"], "t": ev["t"], "event": ev["event"],
        "spivak_class": ev.get("spivak_class"), "confidence": ev.get("confidence"),
        "players": ev["players"], "input": ev["input"],
        "player_slots": ev.get("player_slots"), "slots_note": ev.get("slots_note"),
        "canonical": {k: v for k, v in ev.items() if not k.startswith("_")},
        "commentary_tagged": tagged,
        "commentary": restore_names(tagged, grounding),
        "mode": mode, "ms": round(ms, 1), "seed": seed,
        "temperature": TEMPERATURE if mode == "variation" else None,
        "top_p": TOP_P if mode == "variation" else None,
    }


def _warmup():
    if EVENTS:
        t0 = time.perf_counter()
        generate(0)
        print(f"warmup decode: {(time.perf_counter()-t0)*1000:.0f} ms")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _video(self):
        size = VIDEO.stat().st_size
        rng = self.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(VIDEO, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    return
                left -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                if not PAGE_FILE.exists():
                    self._send(500, f"missing {PAGE_FILE}", "text/plain")
                    return
                self._send(200, PAGE_FILE.read_text(encoding="utf-8"),
                           "text/html; charset=utf-8")
            elif u.path == "/video":
                self._video()
            elif u.path == "/api/events":
                pub = [{k: v for k, v in e.items() if not k.startswith("_")}
                       for e in EVENTS]
                self._send(200, json.dumps(pub), "application/json; charset=utf-8")
            elif u.path == "/api/yolo":
                if YOLO_DATA is None:
                    self._send(200, json.dumps({"boxes": {}, "missing": True}),
                               "application/json")
                else:
                    self._send(200, json.dumps(YOLO_DATA), "application/json")
            elif u.path == "/api/generate":
                q = parse_qs(u.query)
                idx = int(q.get("idx", ["0"])[0])
                mode = q.get("mode", ["greedy"])[0]
                if not 0 <= idx < len(EVENTS):
                    raise ValueError("index out of range")
                if not EVENTS[idx].get("narratable", True):
                    raise ValueError("this event is displayed as a detection only")
                if mode not in ("greedy", "variation"):
                    raise ValueError("mode must be greedy or variation")
                self._send(200, json.dumps(generate(idx, mode), ensure_ascii=False),
                           "application/json; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")
        except Exception as exc:
            try:
                self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                           "application/json")
            except Exception:
                pass

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    _warmup()
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Demo ready at {url}   (Ctrl+C to stop)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped.")
