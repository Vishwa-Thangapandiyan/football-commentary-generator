"""
Download SoccerNet broadcast video for the demo clip (NDA-gated files).

PASSWORD HANDLING - read this before changing anything:

  * The password is read ONLY from the environment variable SOCCERNET_PASSWORD.
  * It is never prompted for, never written to disk, never logged, and never
    passed as a command-line argument (argv is visible in shell history and to
    other processes).
  * Any exception text is scrubbed before printing, so a traceback cannot leak it.
  * This script deliberately REFUSES to prompt interactively. A prompt run through
    a non-TTY harness can silently fall back to echoing input.

Set the variable in YOUR OWN terminal, then run this yourself. See the bottom of
this file for the exact commands.

Default target is the demo example we already have generated commentary for:
    game : england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal
    event: soccer-ball (goal), half 1, 29:51 - Ander Herrera, assist Ashley Young
    file : 1_224p.mkv   (first half only - the goal is in the first half)
"""

import argparse
import os
import sys
from pathlib import Path

# CLAUDE.md 3.2: no-op the google-analytics call, which fires AFTER the file is
# written and raises SSLError in this environment.
import SoccerNet.Downloader as _DL

_DL.report = lambda *a, **k: None
from SoccerNet.Downloader import SoccerNetDownloader, getListGames  # noqa: E402

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "caption-2024"

DEFAULT_GAME = (
    "england_epl\\2014-2015\\2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"
)
DEFAULT_SPLIT = "test"

ENV_VAR = "SOCCERNET_PASSWORD"


def scrub(text, secret):
    """Remove the password from any string before it is printed."""
    if not secret:
        return text
    return str(text).replace(secret, "<REDACTED>")


def get_password():
    pw = os.environ.get(ENV_VAR)
    if not pw:
        print(
            f"ERROR: environment variable {ENV_VAR} is not set.\n\n"
            f"This script will not prompt for a password by design.\n"
            f"Set it in your own PowerShell terminal, then re-run:\n\n"
            f'    $env:{ENV_VAR} = "<your NDA password>"\n'
            f"    python download_video.py\n",
            file=sys.stderr,
        )
        sys.exit(2)
    if pw.strip() != pw:
        print(f"WARNING: {ENV_VAR} has leading/trailing whitespace - stripping it.",
              file=sys.stderr)
    return pw.strip()


def main():
    ap = argparse.ArgumentParser(
        description="Download one SoccerNet video file (password via env var only)."
    )
    ap.add_argument("--game", default=DEFAULT_GAME,
                    help="game path as returned by getListGames")
    ap.add_argument("--split", default=DEFAULT_SPLIT,
                    choices=["train", "valid", "test", "challenge"])
    ap.add_argument("--half", default="1", choices=["1", "2", "both"],
                    help="which half to fetch (default: 1)")
    ap.add_argument("--res", default="224p", choices=["224p", "720p"],
                    help="224p is much smaller; use it unless you need 720p")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file already exists")
    ap.add_argument("--list-games", action="store_true",
                    help="print the games in the split and exit (no password used)")
    args = ap.parse_args()

    if args.list_games:
        for g in getListGames(args.split, task="caption"):
            print(g)
        return

    # NOTE: password is fetched only after --list-games, so listing needs no secret.
    password = get_password()

    halves = ["1", "2"] if args.half == "both" else [args.half]
    files = [f"{h}_{args.res}.mkv" for h in halves]

    target_dir = RAW / Path(args.game.replace("\\", os.sep))
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"game   : {args.game}")
    print(f"split  : {args.split}")
    print(f"files  : {files}")
    print(f"dest   : {target_dir}")
    print(f"auth   : {ENV_VAR} is set ({len(password)} chars, value never printed)\n")

    dl = SoccerNetDownloader(LocalDirectory=str(RAW))
    dl.password = password

    for fname in files:
        dest = target_dir / fname
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            mb = dest.stat().st_size / 1e6
            print(f"SKIP {fname} - already present ({mb:.1f} MB). "
                  f"Use --force to re-download, or delete it if it is a partial file.")
            continue

        print(f"downloading {fname} ... (large file; this can take a long while)")
        try:
            dl.downloadGame(
                game=args.game, files=[fname], spl=args.split, verbose=True
            )
        except KeyboardInterrupt:
            print(f"\nInterrupted. Partial file may remain at {dest} - "
                  f"delete it before retrying.")
            sys.exit(130)
        except Exception as exc:
            # Scrub before printing: never let the secret reach stdout/stderr.
            print(f"\nFAILED on {fname}: "
                  f"{type(exc).__name__}: {scrub(exc, password)}", file=sys.stderr)
            print("If this is a 401/403, the password or the NDA access is the "
                  "likely cause. If it is an SSL error on google-analytics, the "
                  "telemetry no-op above did not apply.", file=sys.stderr)
            sys.exit(1)

        if dest.exists():
            print(f"OK {fname} -> {dest.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"WARNING: {fname} reported success but is not on disk.")

    print("\nDone.")
    print("Reminder: clear the secret from your shell when finished:")
    print(f'    Remove-Item Env:\\{ENV_VAR}')


if __name__ == "__main__":
    main()
