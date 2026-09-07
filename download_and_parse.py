"""
Stage 1-2: download SoccerNet caption-2024 labels, then parse them into
grounded (input, target) training pairs.

Official train/valid/test splits are preserved exactly. Never mixed, never altered.

Usage:
    python download_and_parse.py download                 # all 471 games
    python download_and_parse.py parse                    # all downloaded games
    python download_and_parse.py parse --limit 20         # 20-game dev slice
    python download_and_parse.py inspect                  # print worked examples

See CLAUDE.md section 3 for the verified schema this relies on.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# --- CLAUDE.md 3.2: no-op SoccerNet's google-analytics call. It fires AFTER the
# file is written and raises SSLError here, killing the loop on a success. ---
import SoccerNet.Downloader as _DL

_DL.report = lambda *a, **k: None
from SoccerNet.Downloader import SoccerNetDownloader, getListGames  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "caption-2024"
SPLITS = ("train", "valid", "test")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- download ---


def game_dir(game: str) -> Path:
    """getListGames returns backslash-separated relative paths (Windows-style)."""
    return RAW / Path(game.replace("\\", os.sep))


def download_all(splits=SPLITS):
    RAW.mkdir(parents=True, exist_ok=True)
    dl = SoccerNetDownloader(LocalDirectory=str(RAW))

    manifest = {}
    stats = Counter()
    for split in splits:
        games = getListGames(split, task="caption")
        manifest[split] = games
        print(f"\n=== {split}: {len(games)} games ===", flush=True)

        for i, game in enumerate(games, 1):
            target = game_dir(game) / "Labels-caption.json"
            if target.exists() and target.stat().st_size > 0:
                stats["cached"] += 1
                continue
            try:
                dl.downloadGame(
                    game=game, files=["Labels-caption.json"], spl=split, verbose=False
                )
                stats["downloaded"] += 1
            except Exception as exc:  # per-game isolation; failures are reported
                stats["failed"] += 1
                print(f"  FAIL [{split} {i}/{len(games)}] {game}: {exc}", flush=True)
            if i % 25 == 0:
                print(f"  {split} {i}/{len(games)}  {dict(stats)}", flush=True)

    (DATA / "splits.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nDOWNLOAD DONE: {dict(stats)}")
    print(f"split manifest -> {DATA / 'splits.json'}")
    if stats["failed"]:
        print(f"WARNING: {stats['failed']} games failed. Re-run to retry.")
    return stats


# ------------------------------------------------------------------- parse ---

# CLAUDE.md 3.6/3.7: entity placeholders as they really appear in `identified`.
RE_PLAYER = re.compile(r"\[PLAYER_([A-Za-z0-9]+)\]")
RE_COACH = re.compile(r"\[COACH_([A-Za-z0-9]+)\]")
RE_TEAM = re.compile(r"\[TEAM[^\]]*\]")
RE_REFEREE = re.compile(r"\[REFEREE[^\]]*\]")
RE_ANY_TAG = re.compile(r"\[[A-Z]+[^\]]*\]")

MAX_PLAYERS = 6  # slots per annotation; overflow annotations are dropped
NO_EVENT = "<no_event>"


def build_hash_index(labels: dict) -> dict:
    """hash -> {name, side, role}. Covers players AND coaches (CLAUDE.md 3.6)."""
    index = {}
    lineup = labels.get("lineup") or {}
    for side in ("home", "away"):
        block = lineup.get(side) or {}
        for role in ("players", "coach"):
            for person in block.get(role) or []:
                if isinstance(person, dict) and person.get("hash"):
                    index[person["hash"]] = {
                        "name": person.get("long_name")
                        or person.get("name")
                        or person.get("short_name")
                        or "",
                        "short": person.get("name") or person.get("short_name") or "",
                        "side": side,
                        "role": "player" if role == "players" else "coach",
                        "shirt": person.get("shirt_number") or "",
                    }
    return index


def parse_game_time(game_time: str):
    """'2 - 49:59' -> (half=2, minute=49). Half may be 1, 2 or 3 (CLAUDE.md 3.4)."""
    m = re.match(r"\s*(\d+)\s*-\s*(\d+):(\d+)", game_time or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def tokenize(text: str):
    """Word-level: lowercase, keep <TAGS> intact, split punctuation off."""
    text = text.strip()
    out = []
    for chunk in re.findall(r"<[A-Z_0-9]+>|[A-Za-z0-9']+|[^\sA-Za-z0-9]", text):
        out.append(chunk if chunk.startswith("<") else chunk.lower())
    return out


def anonymize(identified: str, hash_index: dict):
    """
    `identified` -> target text with positional tags, plus the grounding map.

    [PLAYER_<hash>] -> <PLAYER_k>, k by order of first mention (CLAUDE.md 3.6)
    [COACH_<hash>]  -> <COACH_k>
    [TEAM_]         -> <TEAM>      (hash is always empty; CLAUDE.md 3.7)
    [REFEREE]       -> <REFEREE>
    """
    player_map, coach_map = {}, {}
    unresolved = 0

    def sub_player(m):
        nonlocal unresolved
        h = m.group(1)
        if h not in player_map:
            player_map[h] = f"<PLAYER_{len(player_map) + 1}>"
            if h not in hash_index:
                unresolved += 1
        return player_map[h]

    def sub_coach(m):
        h = m.group(1)
        if h not in coach_map:
            coach_map[h] = f"<COACH_{len(coach_map) + 1}>"
        return coach_map[h]

    text = RE_PLAYER.sub(sub_player, identified)
    text = RE_COACH.sub(sub_coach, text)
    text = RE_TEAM.sub("<TEAM>", text)
    text = RE_REFEREE.sub("<REFEREE>", text)
    # Any residual bracket tag would leak an unknown entity into the vocab.
    leftovers = RE_ANY_TAG.findall(text)

    grounding = {}
    for h, tag in player_map.items():
        info = hash_index.get(h)
        grounding[tag] = {
            "hash": h,
            "name": info["name"] if info else None,
            "side": info["side"] if info else None,
            "shirt": info["shirt"] if info else None,
        }
    for h, tag in coach_map.items():
        info = hash_index.get(h)
        grounding[tag] = {
            "hash": h,
            "name": info["name"] if info else None,
            "side": info["side"] if info else None,
            "shirt": None,
        }
    return text, grounding, unresolved, leftovers


def build_input(labels: dict, ann: dict, grounding: dict) -> str:
    """Flat structured context string (CLAUDE.md 4.3)."""
    half, minute = parse_game_time(ann.get("gameTime", ""))
    event = ann.get("label") or NO_EVENT
    home = (labels.get("gameHomeTeam") or "").lower()
    away = (labels.get("gameAwayTeam") or "").lower()

    parts = [
        "<EVT>", event,
        "<HALF>", str(half if half is not None else 0),
        "<MIN>", str(minute if minute is not None else 0),
        "<IMP>", "1" if ann.get("important") else "0",
        "<HOME>", home,
        "<AWAY>", away,
    ]
    # Player slots in first-mention order. This is the content plan, not leakage
    # (CLAUDE.md 4.3) - it is what makes identity hallucination impossible.
    for tag in sorted(
        [t for t in grounding if t.startswith("<PLAYER_")],
        key=lambda t: int(t.split("_")[1].rstrip(">")),
    ):
        parts += ["<P>", tag, grounding[tag]["side"] or "unknown"]
    for tag in sorted(
        [t for t in grounding if t.startswith("<COACH_")],
        key=lambda t: int(t.split("_")[1].rstrip(">")),
    ):
        parts += ["<C>", tag, grounding[tag]["side"] or "unknown"]
    return " ".join(parts)


def parse_split(split: str, games, limit=None):
    examples = []
    stats = Counter()
    for game in games[: limit if limit else None]:
        path = game_dir(game) / "Labels-caption.json"
        if not path.exists():
            stats["missing_file"] += 1
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))
        hash_index = build_hash_index(labels)
        stats["games"] += 1

        for ann in labels.get("annotations", []):
            stats["annotations"] += 1
            identified = (ann.get("identified") or "").strip()
            if not identified:
                stats["skip_empty"] += 1
                continue

            target, grounding, unresolved, leftovers = anonymize(identified, hash_index)
            stats["player_refs"] += sum(
                1 for t in grounding if t.startswith("<PLAYER_")
            )
            stats["unresolved_hashes"] += unresolved
            if leftovers:
                stats["skip_leftover_tag"] += 1
                continue
            if sum(1 for t in grounding if t.startswith("<PLAYER_")) > MAX_PLAYERS:
                stats["skip_too_many_players"] += 1
                continue

            src = build_input(labels, ann, grounding)
            examples.append(
                {
                    "game": game,
                    "split": split,
                    "gameTime": ann.get("gameTime"),
                    "label": ann.get("label") or NO_EVENT,
                    "input": src,
                    "target": target,
                    "grounding": grounding,
                    "reference_real": ann.get("description"),
                }
            )
            stats["kept"] += 1
    return examples, stats


def parse_all(limit=None, out_name=None):
    manifest_path = DATA / "splits.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {s: getListGames(s, task="caption") for s in SPLITS}

    all_examples = {}
    for split in SPLITS:
        examples, stats = parse_split(split, manifest[split], limit=limit)
        all_examples[split] = examples
        res = stats["player_refs"] - stats["unresolved_hashes"]
        rate = 100.0 * res / stats["player_refs"] if stats["player_refs"] else 100.0
        print(f"\n=== {split} ===")
        print(f"  games parsed        : {stats['games']}")
        print(f"  annotations seen    : {stats['annotations']}")
        print(f"  examples kept       : {stats['kept']}")
        print(f"  hash resolution     : {res}/{stats['player_refs']} = {rate:.2f}%")
        print(f"  skipped (empty)     : {stats['skip_empty']}")
        print(f"  skipped (odd tag)   : {stats['skip_leftover_tag']}")
        print(f"  skipped (>{MAX_PLAYERS} players): {stats['skip_too_many_players']}")
        if stats["missing_file"]:
            print(f"  MISSING label files : {stats['missing_file']}")

    out = DATA / (out_name or ("processed_dev.json" if limit else "processed.json"))
    out.write_text(
        json.dumps(all_examples, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    total = sum(len(v) for v in all_examples.values())
    print(f"\nWROTE {total} examples -> {out}")
    return all_examples


def inspect(path=None, n=3):
    path = Path(path) if path else (DATA / "processed_dev.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    train = data["train"]
    print(f"{path.name}: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()))

    picks = [e for e in train if e["label"] == "soccer-ball"][:1]
    picks += [e for e in train if len(e["grounding"]) >= 2 and e not in picks][:1]
    picks += [e for e in train if e not in picks][: max(0, n - len(picks))]

    for i, e in enumerate(picks):
        print(f"\n{'=' * 70}\nEXAMPLE {i}  |  label={e['label']}  {e['gameTime']}")
        print(f"  REAL (never used as target):\n    {e['reference_real'][:200]}")
        print(f"  INPUT :\n    {e['input']}")
        print(f"  TARGET:\n    {e['target'][:240]}")
        print("  GROUNDING:")
        for tag, g in e["grounding"].items():
            print(f"    {tag:<12} -> {g['name']}  ({g['side']}, #{g['shirt']})")
        print(f"  TOKENS(target)[:14]: {tokenize(e['target'])[:14]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "parse", "inspect"])
    ap.add_argument("--limit", type=int, default=None,
                    help="games per split (dev slice)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--path", default=None)
    args = ap.parse_args()

    if args.stage == "download":
        download_all()
    elif args.stage == "parse":
        parse_all(limit=args.limit, out_name=args.out)
    else:
        inspect(args.path)


if __name__ == "__main__":
    main()
