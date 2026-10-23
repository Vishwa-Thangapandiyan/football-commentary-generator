"""
Local watcher for a colab_seeds.py run: pull every result as soon as its job is
DONE, then stop the Colab session so it does not bill while idle.

Colab can reclaim a VM without warning (it happened mid-run once), so nothing is
left on the VM longer than one polling interval.

    python colab_watch.py -s a100 --last eval_tf_v3_noevt_a100_s0

Exits (and runs `colab stop`) when the --last job is DONE, any job FAILED, or the
session disappears.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = "/content/proj"


def colab(*args, timeout=300):
    try:
        r = subprocess.run(["colab", *args], capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 1, "timeout"


def pull(session, remote, local):
    local = ROOT / local
    local.parent.mkdir(parents=True, exist_ok=True)
    rc, out = colab("download", "-s", session, f"{REMOTE}/{remote}", str(local))
    return rc == 0 and "Downloaded" in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--session", required=True)
    ap.add_argument("--last", required=True, help="job name whose DONE ends the run")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--no-stop", action="store_true")
    args = ap.parse_args()

    got = set()
    status_local = ROOT / "outputs" / "seeds" / f"colab_status_{args.session}.txt"
    end_reason = None
    while end_reason is None:
        rc, out = colab("sessions", timeout=120)
        if args.session not in out:
            end_reason = "SESSION LOST"
            break
        if pull(args.session, "outputs/colab_status.txt", status_local.relative_to(ROOT)):
            lines = status_local.read_text().splitlines()
            for line in lines:
                m = re.search(r"(DONE|FAILED) (\S+)", line)
                if not m or m.group(2) in got:
                    continue
                state, job = m.groups()
                if state == "DONE":
                    files = []
                    if job.startswith("eval_"):
                        tag = job[len("eval_"):]
                        files.append((f"outputs/eval_{tag}_best_test.json",
                                      f"outputs/seeds/eval_{tag}.json"))
                    else:
                        files += [(f"outputs/history_{job}.json",
                                   f"outputs/seeds/history_{job}.json"),
                                  (f"checkpoints/{job}_best.pt",
                                   f"checkpoints/seeds/{job}_best.pt")]
                    files.append((f"outputs/colab_{job}.log", f"outputs/seeds/colab_{job}.log"))
                    ok = all(pull(args.session, r, l) for r, l in files)
                    print(f"{time.strftime('%H:%M:%S')} {job}: "
                          f"{'pulled' if ok else 'PULL FAILED, will retry'}", flush=True)
                    if ok:
                        got.add(job)
                else:
                    pull(args.session, f"outputs/colab_{job}.log",
                         f"outputs/seeds/colab_{job}.log")
                    got.add(job)
                    end_reason = f"FAILED {job}"
            if args.last in got and end_reason is None:
                end_reason = f"DONE (last job {args.last})"
        if end_reason is None:
            time.sleep(args.interval)

    print(f"{time.strftime('%H:%M:%S')} watcher ending: {end_reason}", flush=True)
    if not args.no_stop and end_reason != "SESSION LOST":
        print(colab("stop", "-s", args.session)[1].strip().splitlines()[-1], flush=True)
    sys.exit(0 if end_reason.startswith("DONE") else 1)


if __name__ == "__main__":
    main()
