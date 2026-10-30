"""Live prototype: laptop webcam (or a video file) -> privacy pipeline -> masked window/recording.

    python app/run_camera.py --source 0 --profile retail --record out.mp4 --log run.csv

Keys:  g  toggle ground truth ("a PIN is being entered right now") for the evaluation log
       q  quit

Only masked frames are ever shown or written. The CSV log has one row per frame with stage
timings, mask count, whether PIN masking was active and the ground-truth flag, which the
notebook turns into frame-level recall / false-mask rate / FPS for your own recordings.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_cctv import PROFILES, PrivacyPipeline  # noqa: E402
from privacy_cctv.edge_camera import EdgeCamera, MaskedRecorder, hud  # noqa: E402

STAGES = ("prep", "pose", "body_parts", "devices", "hands", "hands_roi", "devices_roi", "fuse", "mask", "total")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--profile", default="all", choices=sorted(PROFILES))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--record", type=Path)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cfg = PROFILES[args.profile].with_(video_mode=True, zoom_crops=1)
    pipeline = PrivacyPipeline(cfg)
    recorder = MaskedRecorder(args.record, 15, (args.width, args.height)) if args.record else None
    log_file = args.log.open("w", newline="") if args.log else None
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(["frame", "masks", "pin_active", "ground_truth_pin", "fault", *STAGES])

    ground_truth = False
    fps, last = 0.0, time.perf_counter()
    try:
        with EdgeCamera(source, pipeline, args.width, args.height) as cam:
            for i, r in enumerate(cam.frames()):
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now
                if log:
                    log.writerow([i, len(r.regions), int(r.pin_active), int(ground_truth), r.fault or "",
                                  *(round(r.timings_ms.get(s, 0.0), 2) for s in STAGES)])
                if recorder:
                    recorder.write(r)
                if args.no_window:
                    continue
                frame = hud(r, fps)
                if ground_truth:
                    cv2.putText(frame, "GT: PIN", (frame.shape[1] - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                cv2.imshow("privacy-cctv (masked output)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("g"):
                    ground_truth = not ground_truth
    finally:
        if recorder:
            recorder.close()
        if log_file:
            log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
