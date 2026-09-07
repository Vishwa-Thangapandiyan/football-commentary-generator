"""
JSON demo: a user types a football EVENT as JSON (no video, no action-spotting),
and the EXISTING trained Transformer turns it into commentary, live.

    python demo_json.py
    -> http://127.0.0.1:8010

What is precomputed vs live:
  * PRECOMPUTED - nothing. There is no video and no cached detections here.
  * LIVE        - every commentary line, for every request. The structured input
    string is built from the user's JSON on each call, the grounding mask is
    rebuilt on each call, and the Transformer runs greedy (or sampled) decoding
    at request time. The response carries the real, measured decode time.

Nothing here trains or modifies the Transformer, its checkpoint or its vocabulary.
This file owns exactly one thing: demo_json.py. The page it serves,
web/json_demo.html, is owned by another agent and is only ever read, never
written, by this file.
"""

import json
import random
import sys
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from generate import build_banned_mask, load_model, restore_names
from sampling_experiment import sample_decode      # isolated, reusable sampler
from train import CommentaryDataset, collate

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "full_best.pt"
DATA = ROOT / "data" / "processed.json"
PAGE_PATH = ROOT / "web" / "json_demo.html"
PORT = 8010

# Mirrors download_and_parse.MAX_PLAYERS (6 player slots) and the coach range
# vocab.py forces into the vocabulary (<COACH_1> .. <COACH_3>). Kept as local
# constants so this file does not need to import the SoccerNet-downloading
# module just for two integers.
MAX_PLAYERS = 6
MAX_COACHES = 3
VALID_SIDES = {"home", "away", "unknown"}

SAMPLING_TEMPERATURE = 0.9
SAMPLING_TOP_P = 0.9

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class ValidationError(Exception):
    """A problem with the request itself (-> HTTP 400), as opposed to an
    unexpected internal failure (-> HTTP 500)."""


# ------------------------------------------------------------- model load ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading Transformer on {DEVICE} ...")
MODEL, VOCAB, CK = load_model(str(CKPT), DEVICE)
print(f"checkpoint loaded: epoch {CK['epoch']}, val_loss {CK['val_loss']:.4f}, "
      f"vocab {len(VOCAB):,}")


def _load_event_types():
    """Real event labels, taken from the OFFICIAL TRAIN split actually on disk -
    never invented. <no_event> sorts last, purely cosmetic."""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    labels = {ex["label"] for ex in data["train"]}
    return sorted(labels, key=lambda s: (s == "<no_event>", s))


EVENT_TYPES = _load_event_types()
print(f"event types loaded from data/processed.json (train split): {EVENT_TYPES}")

SCHEMA_EXAMPLE = {
    "events": [
        {
            "event": "soccer-ball",
            "half": 1,
            "minute": 29,
            "important": True,
            "home": "Manchester United",
            "away": "Arsenal",
            "players": [
                {"name": "Ander Herrera", "side": "home"},
                {"name": "Ashley Young", "side": "home"},
            ],
            "coaches": [],
        }
    ],
    "mode": "greedy",
}
SCHEMA = {"event_types": EVENT_TYPES, "example": SCHEMA_EXAMPLE}


# --------------------------------------------------------------- request parsing ---


def _validate_person(raw, kind, idx):
    """kind is 'player' or 'coach'. Accepts either {"name": ..., "side": ...}
    or a bare name string (side then defaults to 'unknown')."""
    if isinstance(raw, str):
        name, side = raw, "unknown"
    elif isinstance(raw, dict):
        name = raw.get("name")
        side = raw.get("side", "unknown")
        if side is None:
            side = "unknown"
    else:
        raise ValidationError(f"{kind} #{idx + 1} must be an object or a string")

    if not isinstance(name, str) or not name.strip():
        raise ValidationError(f"{kind} #{idx + 1} is missing a non-empty 'name'")

    side = str(side).strip().lower()
    if side not in VALID_SIDES:
        raise ValidationError(
            f"{kind} #{idx + 1} has invalid 'side' {side!r}; "
            f"must be 'home', 'away' or 'unknown'"
        )
    return {"name": name.strip(), "side": side}


def _validate_event(raw, idx):
    where = f"event #{idx + 1}"
    if not isinstance(raw, dict):
        raise ValidationError(f"{where} must be a JSON object")

    event = raw.get("event")
    if not isinstance(event, str) or not event.strip():
        raise ValidationError(f"{where} is missing the required field 'event'")
    event = event.strip()
    if event not in EVENT_TYPES:
        raise ValidationError(
            f"{where} has unknown event type {event!r}; "
            f"must be one of {EVENT_TYPES}"
        )

    half = raw.get("half", 1)
    if isinstance(half, bool) or not isinstance(half, int):
        raise ValidationError(f"{where}: 'half' must be an integer")
    if not (1 <= half <= 3):
        raise ValidationError(f"{where}: 'half' must be between 1 and 3")

    minute = raw.get("minute", 0)
    if isinstance(minute, bool) or not isinstance(minute, int):
        raise ValidationError(f"{where}: 'minute' must be an integer")
    if not (0 <= minute <= 60):
        raise ValidationError(f"{where}: 'minute' must be between 0 and 60")

    important_raw = raw.get("important", 0)
    important = 1 if important_raw in (1, True, "1", "true", "True") else 0

    home = raw.get("home", "home team")
    away = raw.get("away", "away team")
    if not isinstance(home, str) or not isinstance(away, str):
        raise ValidationError(f"{where}: 'home' and 'away' must be strings")

    players_raw = raw.get("players", [])
    if not isinstance(players_raw, list):
        raise ValidationError(f"{where}: 'players' must be a list")
    if len(players_raw) > MAX_PLAYERS:
        raise ValidationError(
            f"{where}: too many players ({len(players_raw)}), max {MAX_PLAYERS}"
        )
    players = [_validate_person(p, "player", i) for i, p in enumerate(players_raw)]

    coaches_raw = raw.get("coaches", [])
    if not isinstance(coaches_raw, list):
        raise ValidationError(f"{where}: 'coaches' must be a list")
    if len(coaches_raw) > MAX_COACHES:
        raise ValidationError(
            f"{where}: too many coaches ({len(coaches_raw)}), max {MAX_COACHES}"
        )
    coaches = [_validate_person(c, "coach", i) for i, c in enumerate(coaches_raw)]

    return {
        "event": event, "half": half, "minute": minute, "important": important,
        "home": home, "away": away, "players": players, "coaches": coaches,
    }


def _normalize_body(body):
    """Accepts, per the API contract:
      {"events": [...], "mode": "greedy"|"variation"}   (the primary shape)
      [ {...}, {...} ]                                   (a bare array)
      { ...single event fields... }                      (a bare single event)
    Returns (validated_events, mode).
    """
    if body is None:
        raise ValidationError("empty request body")

    mode = "greedy"
    if isinstance(body, list):
        raw_events = body
    elif isinstance(body, dict):
        if "events" in body:
            raw_events = body["events"]
            mode = body.get("mode", "greedy")
        elif "event" in body:
            raw_events = [body]
            mode = body.get("mode", "greedy")
        else:
            raise ValidationError(
                "request body must contain an 'events' list, or be a single "
                "event object with an 'event' field, or be a bare array of "
                "event objects"
            )
    else:
        raise ValidationError("request body must be a JSON object or array")

    if not isinstance(raw_events, list):
        raise ValidationError("'events' must be a list")
    if not raw_events:
        raise ValidationError("'events' list is empty")
    if mode not in ("greedy", "variation"):
        raise ValidationError(f"unknown mode {mode!r}; must be 'greedy' or 'variation'")

    events = [_validate_event(e, i) for i, e in enumerate(raw_events)]
    return events, mode


# ------------------------------------------------------------------ generation ---


def _build_structured(ev):
    """The exact structured-input string form the model was trained on
    (download_and_parse.build_input): <EVT> ... <HALF> ... <MIN> ... <IMP> ...
    <HOME> ... <AWAY> ... then <P> <PLAYER_k> side / <C> <COACH_k> side per
    entity, in the order given (that order IS the grounding, CLAUDE.md 4.3)."""
    parts = [
        "<EVT>", ev["event"],
        "<HALF>", str(ev["half"]),
        "<MIN>", str(ev["minute"]),
        "<IMP>", str(ev["important"]),
        "<HOME>", ev["home"].strip().lower(),
        "<AWAY>", ev["away"].strip().lower(),
    ]
    grounding = {}
    for i, p in enumerate(ev["players"], start=1):
        tag = f"<PLAYER_{i}>"
        parts += ["<P>", tag, p["side"]]
        grounding[tag] = {"name": p["name"], "side": p["side"], "shirt": None, "hash": None}
    for i, c in enumerate(ev["coaches"], start=1):
        tag = f"<COACH_{i}>"
        parts += ["<C>", tag, c["side"]]
        grounding[tag] = {"name": c["name"], "side": c["side"], "shirt": None, "hash": None}
    return " ".join(parts), grounding


@torch.no_grad()
def generate_one(ev, mode):
    """Build the structured input from the validated event, run ONE live decode
    (greedy by default, sampled for 'variation'), and restore real names. The
    grounding mask is active in both modes - CLAUDE.md 4.14."""
    src_text, grounding = _build_structured(ev)
    ex = {"input": src_text, "target": "", "grounding": grounding}
    ds = CommentaryDataset([ex], VOCAB)
    src, _ = collate([ds[0]])
    src = src.to(DEVICE)
    banned = build_banned_mask([ex], VOCAB, DEVICE)

    seed = temperature = top_p = None
    if mode == "variation":
        seed = random.randrange(2 ** 31 - 1)
        temperature, top_p = SAMPLING_TEMPERATURE, SAMPLING_TOP_P
        t0 = time.perf_counter()
        ids = sample_decode(MODEL, src, banned, CK["max_tgt"], temperature, top_p, seed)
        ms = (time.perf_counter() - t0) * 1000
    else:
        t0 = time.perf_counter()
        ids = MODEL.greedy_decode(src, max_len=CK["max_tgt"], banned=banned)
        ms = (time.perf_counter() - t0) * 1000

    tagged = VOCAB.decode(ids[0].tolist())
    commentary = restore_names(tagged, grounding)

    return {
        "commentary": commentary,
        "commentary_tagged": tagged,
        "structured_input": src_text,
        "mode": mode,
        "ms": round(ms, 1),
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "grounding": {tag: info["name"] for tag, info in grounding.items()},
        "event": ev["event"],
        "half": ev["half"],
        "minute": ev["minute"],
        # Not part of the minimal contract, but the frontend's card styling
        # keys off it (important border/badge) and we already have it for free.
        "important": bool(ev["important"]),
    }


def _warmup():
    """First CUDA decode costs ~2-3 s of kernel warmup. Pay it once at startup
    so the first real request from the page is fast."""
    warm_event = {
        "event": EVENT_TYPES[0], "half": 1, "minute": 1, "important": 0,
        "home": "home team", "away": "away team", "players": [], "coaches": [],
    }
    t0 = time.perf_counter()
    generate_one(warm_event, "greedy")
    print(f"warmup decode: {(time.perf_counter() - t0) * 1000:.0f} ms "
          f"(subsequent calls are much faster)")


_warmup()


# ------------------------------------------------------------------------ HTTP ---


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _send_text(self, code, text, ctype="text/plain; charset=utf-8"):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _serve_index(self):
        if PAGE_PATH.exists():
            self._send_text(200, PAGE_PATH.read_text(encoding="utf-8"),
                             "text/html; charset=utf-8")
        else:
            self._send_text(
                200,
                "web/json_demo.html not found.\n\n"
                "The backend API is up regardless:\n"
                "  GET  /api/schema\n"
                "  POST /api/generate   body: "
                '{"events": [...], "mode": "greedy"|"variation"}\n',
            )

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                self._serve_index()
            elif u.path == "/api/schema":
                self._send_json(200, SCHEMA)
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw.strip() else None
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": f"malformed JSON: {exc}"})
                return

            try:
                events, mode = _normalize_body(body)
            except ValidationError as ve:
                self._send_json(400, {"error": str(ve)})
                return

            results = [generate_one(ev, mode) for ev in events]
            self._send_json(200, {"results": results})
        except Exception as exc:
            traceback.print_exc()
            try:
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  JSON commentary demo ready at {url}   (Ctrl+C to stop)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped.")
