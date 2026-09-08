"""
Player-marking experiment (STEP 1 of manual player identification).

Extracts one representative frame from the selected clip, runs the project's
existing YOLOv8n on it, and produces:
  1. an annotated image with each detection numbered PLAYER_1..PLAYER_N
  2. a JSON file with the exact YOLO bounding boxes (native frame coordinates)
  3. a labelling list to be filled in by a human

IMPORTANT: YOLO's `person` class locates PEOPLE. It knows nothing about identity,
team, or role. Every name in the output must come from a human, not from this
script. Detections may include match officials and crowd members.

Does not touch the Transformer, generate.py, training, the checkpoint, or the demos.

    python mark_players.py
    python mark_players.py --time 1873 --scale 3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
VIDEO = (ROOT / "data" / "caption-2024" / "england_epl" / "2014-2015" /
         "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal" / "1_224p.mkv")
OUT = ROOT / "outputs" / "player_id"

# Selected clip: 28:30 - 31:30 (3 min) around the goal. Default frame is 3 s in -
# the opening seconds are a wide shot where players are only ~20 px tall.
CLIP_START, CLIP_END = 28 * 60 + 30, 31 * 60 + 30
DEFAULT_T = 28 * 60 + 33

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=DEFAULT_T,
                    help="seconds into the half (default 28:33)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--scale", type=int, default=3,
                    help="upscale factor for the ANNOTATED IMAGE ONLY; "
                         "JSON coordinates always stay in native frame space")
    args = ap.parse_args()

    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")
    OUT.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read a frame at {args.time}s")

    h, w = frame.shape[:2]
    frame_no = int(round(args.time * fps))

    from ultralytics import YOLO
    det = YOLO("yolov8n.pt")(frame, classes=[0], conf=args.conf, verbose=False)[0]

    # Number left-to-right by box centre, which is how a person reads the image.
    boxes = sorted(
        [(list(map(float, b)), float(c))
         for b, c in zip(det.boxes.xyxy.tolist(), det.boxes.conf.tolist())],
        key=lambda t: (t[0][0] + t[0][2]) / 2,
    )

    s = args.scale
    canvas = cv2.resize(frame, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
    players = {}

    for i, ((x1, y1, x2, y2), conf) in enumerate(boxes, 1):
        tag = f"PLAYER_{i}"
        players[tag] = {
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "confidence": round(conf, 3),
        }
        p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
        cv2.rectangle(canvas, p1, p2, (0, 235, 255), 2)

        # Label sits above the box, or just below its top edge when there is no room.
        label = tag
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        ly = p1[1] - 8 if p1[1] - th - 12 > 0 else p1[1] + th + 12
        lx = min(p1[0], canvas.shape[1] - tw - 10)
        cv2.rectangle(canvas, (lx - 5, ly - th - 7), (lx + tw + 5, ly + 6),
                      (0, 235, 255), -1)
        cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (20, 20, 20), 2, cv2.LINE_AA)

    mm, ss = int(args.time // 60), int(args.time % 60)
    img_path = OUT / f"frame_{mm:02d}-{ss:02d}_marked.jpg"
    cv2.imwrite(str(img_path), canvas)

    payload = {
        "video": str(VIDEO.relative_to(ROOT)).replace("\\", "/"),
        "clip": f"{CLIP_START // 60}:{CLIP_START % 60:02d}-{CLIP_END // 60}:{CLIP_END % 60:02d}",
        "frame": frame_no,
        "timestamp_seconds": args.time,
        "timestamp": f"{mm:02d}:{ss:02d} (first half)",
        "frame_width": w,
        "frame_height": h,
        "coordinate_space": "native frame pixels (398x224) - NOT the upscaled image",
        "annotated_image_scale": s,
        "detector": "YOLOv8n, COCO class 0 (person), conf>=%.2f" % args.conf,
        "detector_note": "person locations only - no identity, team or role",
        "num_detections": len(players),
        "players": players,
    }
    json_path = OUT / f"frame_{mm:02d}-{ss:02d}_boxes.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"video      : {payload['video']}")
    print(f"clip       : {payload['clip']}")
    print(f"frame      : {frame_no}  ({payload['timestamp']})")
    print(f"frame size : {w} x {h}   (annotated image upscaled {s}x -> {w*s} x {h*s})")
    print(f"detections : {len(players)}")
    print(f"image      : {img_path.relative_to(ROOT)}")
    print(f"json       : {json_path.relative_to(ROOT)}")
    print("\nLabel these from the image (YOLO does not know who they are):")
    for tag, v in players.items():
        x1, y1, x2, y2 = v["bbox"]
        print(f"  {tag} = ?            # conf {v['confidence']:.2f}, "
              f"bbox [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], "
              f"{x2-x1:.0f}x{y2-y1:.0f}px")


if __name__ == "__main__":
    main()
