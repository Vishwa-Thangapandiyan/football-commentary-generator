"""
V3 data: event context from SoccerNet-v2 action annotations, NOT from the reference.

V2 split <no_event> into pass/shot/foul/... with keyword rules over the TARGET
commentary, on every split. That leaks the answer into the input. V3 replaces it
with an independent signal: the human action-spotting annotations of SoccerNet-v2
(Labels-v2.json: 17 classes such as Shots on target, Foul, Corner, Offside, each
with a timestamp and a team). They were annotated separately from the commentary,
and they use the class set an action-spotting model (e.g. Spivak) predicts, so
the same input format can later be filled from video.

For each commentary annotation we attach the SoccerNet-v2 actions of the same
half that fall inside a window [t - BEFORE, t + AFTER] around its timestamp. The
window is chosen on TRAIN only, by how well the joined actions agree with the
commentary's own labelled events (corner, goal, card, substitution, ...); valid
and test never influence it.

    python build_v3.py download          # Labels-v2.json for all 471 games
    python build_v3.py window            # pick the window on train, report agreement
    python build_v3.py build             # write data/processed_v3.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "caption-2024"
SPLITS = ("train", "valid", "test")
MAX_ACTIONS = 4          # most recent actions kept in the input

# SoccerNet-Caption labels with an unambiguous SoccerNet-v2 counterpart, used only
# to choose and audit the alignment window.
CAPTION_TO_V2 = {
    "corner": "Corner",
    "soccer-ball": "Goal",
    "y-card": "Yellow card",
    "r-card": "Red card",
    "yr-card": "Yellow->red card",
    "substitution": "Substitution",
    "penalty": "Penalty",
}

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def game_path(game):
    """splits.json stores Windows-style paths; always rebuild with '/'."""
    return game.replace("\\", "/")


def secs(game_time):
    """'2 - 49:59' -> (2, 2999)."""
    m = re.match(r"\s*(\d+)\s*-\s*(\d+):(\d+)", game_time or "")
    return (int(m.group(1)), int(m.group(2)) * 60 + int(m.group(3))) if m else (None, None)


# ---------------------------------------------------------------- download ---

def download():
    import SoccerNet.Downloader as DL
    DL.report = lambda *a, **k: None   # CLAUDE.md 3.2: analytics call crashes
    from SoccerNet.Downloader import SoccerNetDownloader

    from concurrent.futures import ThreadPoolExecutor

    manifest = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    todo = [(s, g) for s in SPLITS for g in manifest[s]]
    stats = Counter()

    def one(item):
        split, game = item
        target = RAW / game_path(game) / "Labels-v2.json"
        if target.exists() and target.stat().st_size > 0:
            return "cached"
        try:
            SoccerNetDownloader(LocalDirectory=str(RAW)).downloadGame(
                game=game_path(game), files=["Labels-v2.json"], spl=split,
                verbose=False)
            return "downloaded" if target.exists() else "missing_after"
        except Exception as exc:
            print(f"  FAIL {split} {game}: {exc}", flush=True)
            return "failed"

    # Network-bound: 8 parallel requests instead of one at a time.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, res in enumerate(pool.map(one, todo), 1):
            stats[res] += 1
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} {dict(stats)}", flush=True)
    print(f"DOWNLOAD DONE: {dict(stats)} of {len(todo)} games")


# --------------------------------------------------------------------- join ---

def load_actions(game):
    p = RAW / game_path(game) / "Labels-v2.json"
    if not p.exists():
        return None
    out = []
    for a in json.loads(p.read_text(encoding="utf-8"))["annotations"]:
        half, t = secs(a["gameTime"])
        if half is not None:
            out.append((half, t, a["label"], a.get("team", "not applicable")))
    return sorted(out, key=lambda x: (x[0], x[1]))


def window_actions(actions, half, t, before, after):
    """SoccerNet-v2 actions of the same half in [t - before, t + after], oldest
    first, keeping the MAX_ACTIONS closest to the commentary timestamp."""
    hits = [a for a in actions if a[0] == half and t - before <= a[1] <= t + after]
    hits.sort(key=lambda a: abs(a[1] - t))
    return sorted(hits[:MAX_ACTIONS], key=lambda a: a[1])


def agreement(rows, actions_by_game, before, after):
    """Of the commentary rows whose own label maps to a v2 class, the share whose
    window contains that class. Also returns the share of rows with any action."""
    hit = tot = nonempty = 0
    for r in rows:
        acts = actions_by_game.get(r["game"])
        if acts is None:
            continue
        half, t = secs(r["gameTime"])
        w = window_actions(acts, half, t, before, after)
        nonempty += bool(w)
        want = CAPTION_TO_V2.get(r["label"])
        if want:
            tot += 1
            hit += any(a[2] == want for a in w)
    return hit / max(tot, 1), tot, nonempty / max(len(rows), 1)


def load_v1():
    return json.loads((DATA / "processed.json").read_text(encoding="utf-8"))


def choose_window():
    data = load_v1()
    games = {r["game"] for s in SPLITS for r in data[s]}
    acts = {g: load_actions(g) for g in games}
    missing = sum(v is None for v in acts.values())
    print(f"games with Labels-v2.json: {len(games) - missing}/{len(games)}")
    print(f"\nwindow search on TRAIN only ({len(data['train']):,} rows)")
    print(f"{'before':>7}{'after':>7}{'agreement':>11}{'rows w/ action':>16}")
    grid = []
    for before in (10, 20, 30, 45, 60, 90):
        for after in (0, 5, 10, 20, 30):
            agr, n, cov = agreement(data["train"], acts, before, after)
            grid.append((before, after, agr, n))
            print(f"{before:>7}{after:>7}{agr:>11.3f}{cov:>16.3f}")
    # Agreement only grows with width, but wider windows pull in unrelated
    # actions. Rule: the NARROWEST window reaching 95% of the best agreement.
    top = max(g[2] for g in grid)
    before, after, agr, n = min((g for g in grid if g[2] >= 0.95 * top),
                                key=lambda g: (g[0] + g[1], g[0]))
    print(f"\nbest agreement on the grid {top:.3f}; rule: narrowest window >= 95% of it")
    print(f"CHOSEN on train: before={before}s after={after}s "
          f"(agreement {agr:.3f} on {n:,} labelled rows)")
    for s in ("valid", "test"):
        a, m, cov = agreement(data[s], acts, before, after)
        print(f"  held-out check {s:<5}: agreement {a:.3f} on {m:,} rows, "
              f"{cov:.3f} of rows have an action")
    (DATA / "v3_window.json").write_text(json.dumps(
        {"before": before, "after": after, "train_agreement": round(agr, 4),
         "max_actions": MAX_ACTIONS, "chosen_on": "train"}, indent=2), encoding="utf-8")
    return before, after


# -------------------------------------------------------------------- build ---

def act_tag(label):
    """'Shots on target' -> '<A_SHOTS_ON_TARGET>'. Upper-case <...> tags survive
    tokenize() as ONE token; a hyphenated word would split into five."""
    return "<A_" + re.sub(r"[^A-Z0-9]+", "_",
                          label.upper().replace("->", " TO ")).strip("_") + ">"


def time_bucket(dt):
    """Seconds from the commentary timestamp -> one word."""
    if dt > 3:
        return "next"
    if dt >= -3:
        return "now"
    return "recent" if dt >= -15 else "earlier"


def build():
    cfg = json.loads((DATA / "v3_window.json").read_text(encoding="utf-8"))
    before, after = cfg["before"], cfg["after"]
    data = load_v1()
    acts = {}
    out, stats = {}, Counter()
    for split in SPLITS:
        rows = []
        for r in data[split]:
            if r["game"] not in acts:
                acts[r["game"]] = load_actions(r["game"]) or []
            half, t = secs(r["gameTime"])
            w = window_actions(acts[r["game"]], half, t, before, after)
            parts = ["<ACT>"]
            for _, at, label, team in w:
                side = team if team in ("home", "away") else "none"
                parts += [act_tag(label), side, time_bucket(at - t)]
            if not w:
                parts.append("<NO_ACTION>")
                stats[f"{split}_no_action"] += 1
            # Keep V1's input exactly (the commentary's own label is dataset
            # metadata, not derived from the text) and append the v2 actions.
            new = dict(r)
            new["input"] = r["input"] + " " + " ".join(parts)
            new["v2_actions"] = [{"label": l, "team": tm, "dt": at - t}
                                 for _, at, l, tm in w]
            rows.append(new)
        out[split] = rows
        stats[f"{split}_rows"] = len(rows)
    (DATA / "processed_v3.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # Ablation: drop the commentary's own <EVT> label, so the only event signal is
    # the independent SoccerNet-v2 actions. V1 inputs start "<EVT> <label> ...".
    strict = {s: [dict(r, input=r["input"].split(" ", 2)[2]) for r in rows]
              for s, rows in out.items()}
    assert all(not r["input"].startswith("<EVT>") and "<EVT>" not in r["input"]
               for rows in strict.values() for r in rows)
    (DATA / "processed_v3_noevt.json").write_text(
        json.dumps(strict, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {DATA / 'processed_v3_noevt.json'} (ablation: no <EVT> label)")
    print(f"window before={before}s after={after}s, max {MAX_ACTIONS} actions")
    for s in SPLITS:
        n = stats[f"{s}_rows"]
        print(f"  {s:<5}: {n:,} rows, {stats[f'{s}_no_action'] / n:.1%} without any action")
    ex = next(r for r in out["train"] if r["label"] == "<no_event>" and r["v2_actions"])
    print(f"\nexample (<no_event> row)\n  INPUT : {ex['input']}\n  TARGET: {ex['target']}")
    print(f"\nwrote {DATA / 'processed_v3.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "window", "build"])
    args = ap.parse_args()
    {"download": download, "window": choose_window, "build": build}[args.stage]()


if __name__ == "__main__":
    main()
