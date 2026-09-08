"""
Precompute YOLO person boxes for the demo half, so the browser can draw them as a
canvas overlay synced to video time without running inference live.

Sampled at 5 fps (every 5th frame of 25 fps) - enough for a smooth overlay, and
1/5 the size of per-frame data.

    python precompute_yolo.py

Writes outputs/yolo_boxes_half1.json:
    {"fps_sampled": 5, "frame_w": 398, "frame_h": 224,
     "boxes": {"612.4": [[x1,y1,x2,y2,conf], ...], ...}}   # key = seconds

YOLO locates PEOPLE. It does not identify players, teams, or roles, and it also
detects officials and crowd. Nothing here claims otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
VIDEO = (ROOT / "data" / "caption-2024" / "england_epl" / "2014-2015" /
         "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal" / "1_224p.mkv")
OUT = ROOT / "outputs" / "yolo_boxes_half1.json"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="sample every Nth frame")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_f = int(args.end * fps) if args.end else total
    start_f = int(args.start * fps)

    print(f"{w}x{h} @ {fps:.0f}fps, frames {start_f}-{end_f}, stride {args.stride} "
          f"-> {(end_f-start_f)//args.stride} samples")

    boxes, i, done = {}, start_f, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    while i < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (i - start_f) % args.stride == 0:
            r = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
            rows = [[round(x1), round(y1), round(x2), round(y2), round(float(c), 2)]
                    for (x1, y1, x2, y2), c in
                    zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
            if rows:
                boxes[f"{i / fps:.2f}"] = rows
            done += 1
            if done % 500 == 0:
                print(f"  {done} samples, t={i/fps/60:.1f} min", flush=True)
        i += 1
    cap.release()

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "video": str(VIDEO.relative_to(ROOT)).replace("\\", "/"),
        "fps_source": fps, "stride": args.stride,
        "fps_sampled": fps / args.stride,
        "frame_w": w, "frame_h": h,
        "detector": f"YOLOv8n COCO person, conf>={args.conf}",
        "note": "person locations only - no identity, team or role; includes "
                "officials and crowd",
        "n_sampled_frames": done,
        "boxes": boxes,
    }, separators=(",", ":")), encoding="utf-8")
    mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({mb:.1f} MB, "
          f"{len(boxes)} frames with detections)")


if __name__ == "__main__":
    main()
