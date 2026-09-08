"""
Multi-object tracking over the selected clip: persistent PLAYER_X track IDs.

    YOLOv8n person detections -> ByteTrack -> persistent track ids -> timeline JSON

NO identification of any kind. Every track is PLAYER_X, an anonymous, persistent
handle for "the same detected person across consecutive frames". YOLO and ByteTrack
know nothing about who anyone is, which team they are on, or whether they are a
player at all (officials and crowd are also detected as `person`).

Broadcast football is cut between cameras every few seconds. A cut destroys visual
continuity, so a tracker cannot carry an identity across it. This script detects
shot boundaries and reports tracks per shot, which is the honest unit of
persistence, rather than pretending ids survive a cut.

    python track_players.py                 # track + timeline + sample video
    python track_players.py --no-video      # timeline only (faster)

Does not modify the Transformer, training code, checkpoint, or the JSON demo.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEO = (ROOT / "data" / "caption-2024" / "england_epl" / "2014-2015" /
         "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal" / "1_224p.mkv")
OUT = ROOT / "outputs" / "player_id"
CLIP_START, CLIP_END = 28 * 60 + 30, 31 * 60 + 30

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def colour(tid):
    """Stable, distinct colour per track id."""
    rng = np.random.default_rng(int(tid) * 9781 + 17)
    c = rng.integers(80, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])


def is_cut(prev, cur, threshold=0.55):
    """Shot-boundary detection by HSV histogram correlation between frames."""
    if prev is None:
        return False
    h1 = cv2.calcHist([cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)], [0, 1], None,
                      [50, 60], [0, 180, 0, 256])
    h2 = cv2.calcHist([cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)], [0, 1], None,
                      [50, 60], [0, 180, 0, 256])
    cv2.normalize(h1, h1); cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) < threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=CLIP_START)
    ap.add_argument("--end", type=float, default=CLIP_END)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min-frames", type=int, default=5,
                    help="drop tracks shorter than this (detection flicker)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")
    OUT.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    n_frames = int((args.end - args.start) * fps)

    per_frame = []          # [(t, [(tid, bbox, conf), ...]), ...]
    shot_of_frame = []
    shot = 0
    prev = None
    print(f"tracking {n_frames} frames ({args.start/60:.1f}-{args.end/60:.1f} min) ...")

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = args.start + i / fps

        if is_cut(prev, frame):
            shot += 1
            # A cut breaks visual continuity: reset so ids are not carried across
            # unrelated camera angles, which would be a false claim of identity.
            model.predictor.trackers[0].reset()
        prev = frame

        r = model.track(frame, classes=[0], conf=args.conf, persist=True,
                        tracker="bytetrack.yaml", verbose=False)[0]
        rows = []
        if r.boxes.id is not None:
            for tid, box, cf in zip(r.boxes.id.tolist(), r.boxes.xyxy.tolist(),
                                    r.boxes.conf.tolist()):
                rows.append((int(tid), [round(v, 1) for v in box], round(float(cf), 3)))
        per_frame.append((round(t, 2), rows))
        shot_of_frame.append(shot)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_frames} frames, {shot+1} shots so far")

    cap.release()

    # ---- aggregate tracks, keyed by (shot, track id) --------------------------
    agg = defaultdict(lambda: {"frames": [], "confs": [], "boxes": []})
    for (t, rows), sh in zip(per_frame, shot_of_frame):
        for tid, box, cf in rows:
            k = (sh, tid)
            agg[k]["frames"].append(t)
            agg[k]["confs"].append(cf)
            agg[k]["boxes"].append(box)

    tracks = []
    for (sh, tid), v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[1]["frames"][0])):
        if len(v["frames"]) < args.min_frames:
            continue
        hs = [b[3] - b[1] for b in v["boxes"]]
        tracks.append({
            "player": None,                       # filled in below, ordered by time
            "track_id": tid,
            "shot": sh,
            "first_seen_s": v["frames"][0],
            "last_seen_s": v["frames"][-1],
            "duration_s": round(v["frames"][-1] - v["frames"][0], 2),
            "n_frames": len(v["frames"]),
            "mean_confidence": round(float(np.mean(v["confs"])), 3),
            "mean_box_height_px": round(float(np.mean(hs)), 1),
            "first_bbox": v["boxes"][0],
            "last_bbox": v["boxes"][-1],
        })
    for i, tr in enumerate(tracks, 1):
        tr["player"] = f"PLAYER_{i}"

    shots = []
    for sh in sorted(set(shot_of_frame)):
        idx = [i for i, s in enumerate(shot_of_frame) if s == sh]
        shots.append({
            "shot": sh,
            "start_s": per_frame[idx[0]][0],
            "end_s": per_frame[idx[-1]][0],
            "duration_s": round(per_frame[idx[-1]][0] - per_frame[idx[0]][0], 2),
            "n_frames": len(idx),
            "n_tracks": sum(1 for t in tracks if t["shot"] == sh),
        })

    durs = [t["duration_s"] for t in tracks] or [0]
    payload = {
        "video": str(VIDEO.relative_to(ROOT)).replace("\\", "/"),
        "clip": f"{int(args.start)//60}:{int(args.start)%60:02d}-"
                f"{int(args.end)//60}:{int(args.end)%60:02d}",
        "fps": fps,
        "frames_processed": len(per_frame),
        "detector": f"YOLOv8n person, conf>={args.conf}",
        "tracker": "ByteTrack (ultralytics bytetrack.yaml), reset at each shot cut",
        "identity_note": "PLAYER_X is an anonymous persistent track handle. "
                         "No identification, no jersey OCR, no names. Detections "
                         "may include match officials and crowd.",
        "persistence_unit": "a track id is only meaningful WITHIN one shot; "
                            "broadcast cuts end all tracks",
        "summary": {
            "shots_detected": len(shots),
            "tracks_kept": len(tracks),
            "min_track_frames": args.min_frames,
            "median_track_duration_s": round(float(np.median(durs)), 2),
            "longest_track_s": round(max(durs), 2),
            "mean_tracks_per_shot": round(len(tracks) / max(len(shots), 1), 2),
        },
        "shots": shots,
        "tracks": tracks,
    }
    jp = OUT / "tracking_timeline.json"
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nshots detected      : {len(shots)}")
    print(f"tracks kept (>={args.min_frames}f): {len(tracks)}")
    print(f"median duration     : {payload['summary']['median_track_duration_s']}s")
    print(f"longest track       : {payload['summary']['longest_track_s']}s")
    print(f"timeline JSON       : {jp.relative_to(ROOT)}")

    # ---- annotated sample video: the longest shot ----------------------------
    if not args.no_video and shots:
        best = max(shots, key=lambda s: s["n_frames"])
        s0, s1 = best["start_s"], best["end_s"]
        sel = [(t, rows) for t, rows in per_frame if s0 <= t <= s1]
        sc = args.scale
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_MSEC, s0 * 1000)
        ok, f0 = cap.read()
        h, w = f0.shape[:2]
        vp = OUT / "tracking_sample.mp4"
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (w * sc, h * sc))
        cap.set(cv2.CAP_PROP_POS_MSEC, s0 * 1000)
        name_of = {(t["shot"], t["track_id"]): t["player"] for t in tracks}
        for t, rows in sel:
            ok, frame = cap.read()
            if not ok:
                break
            big = cv2.resize(frame, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
            for tid, box, cf in rows:
                nm = name_of.get((best["shot"], tid))
                if nm is None:
                    continue
                c = colour(tid)
                p1 = (int(box[0] * sc), int(box[1] * sc))
                p2 = (int(box[2] * sc), int(box[3] * sc))
                cv2.rectangle(big, p1, p2, c, 2)
                (tw, th), _ = cv2.getTextSize(nm, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ly = p1[1] - 5 if p1[1] - th - 8 > 0 else p1[1] + th + 8
                cv2.rectangle(big, (p1[0] - 2, ly - th - 5),
                              (p1[0] + tw + 4, ly + 4), c, -1)
                cv2.putText(big, nm, (p1[0] + 1, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (15, 15, 15), 1, cv2.LINE_AA)
            mm, ss = int(t // 60), int(t % 60)
            cv2.putText(big, f"shot {best['shot']}  {mm:02d}:{ss:02d}  "
                             f"persistent track ids", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(big)
        cap.release(); vw.release()
        print(f"sample video        : {vp.relative_to(ROOT)}  "
              f"(shot {best['shot']}, {best['duration_s']}s, "
              f"{best['n_tracks']} tracks)")


if __name__ == "__main__":
    main()
