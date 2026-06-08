#!/usr/bin/env python3
"""
Live lane detection from a plain USB / UVC camera (or any video file).

No ROS. Just OpenCV VideoCapture -> LaneDetector -> overlay. This is the
fastest way to sanity-check a profile on whatever webcam is on your desk before
moving to the ZED + ROS path.

    # webcam index 0
    python adapters/usb_cam.py --device 0 --config configs/default.yaml
    # an explicit device node or a recorded clip
    python adapters/usb_cam.py --device /dev/video2
    python adapters/usb_cam.py --device clip.mp4 --out _artifacts/usb

Keys (display mode): q = quit, r = reset temporal/adaptive state,
                     space = pause, s = save current panel.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lane_cv import LaneConfig, LaneDetector  # noqa: E402


def _open(device, width, height):
    # Integer string -> webcam index; otherwise a path / device node.
    src = int(device) if str(device).isdigit() else device
    cap = cv2.VideoCapture(src)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _panel(det, bgr, result, fps):
    overlay = det.draw_overlay(bgr, result)
    cand = cv2.cvtColor(result.candidate, cv2.COLOR_GRAY2BGR)
    lane = cv2.cvtColor(result.lane_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(overlay, f"{fps:4.1f} FPS  segs:{len(result.segments)}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(cand, "candidate", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(lane, "lane_mask", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return np.hstack([overlay, cand, lane])


def main():
    ap = argparse.ArgumentParser(description="Live lane_cv on a USB camera / video.")
    ap.add_argument("--device", default="0", help="webcam index, /dev/videoN, or a video path")
    ap.add_argument("--config", default="", help="profile YAML (default: built-in)")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--out", default="", help="dir to save panels (optional)")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=unlimited)")
    args = ap.parse_args()

    cfg = LaneConfig.from_yaml(args.config) if args.config else LaneConfig()
    det = LaneDetector(cfg)

    cap = _open(args.device, args.width, args.height)
    if not cap.isOpened():
        ap.error(f"could not open camera/video: {args.device!r} "
                 f"(try a different index, or check `ls /dev/video*`)")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
    paused, i, t_prev, fps = False, 0, time.time(), 0.0
    while True:
        if not paused:
            ok, bgr = cap.read()
            if not ok:
                print("[usb_cam] stream ended / read failed.")
                break
            result = det.process(bgr)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - t_prev))
            t_prev = now
            panel = _panel(det, bgr, result, fps)
            if args.out:
                cv2.imwrite(os.path.join(args.out, f"frame{i:05d}.png"), panel)
            i += 1
            if args.max_frames and i >= args.max_frames:
                break

        if not args.no_display:
            cv2.imshow("lane_cv / usb", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                det.reset()
            if key == ord(" "):
                paused = not paused
            if key == ord("s") and args.out:
                cv2.imwrite(os.path.join(args.out, f"saved{i:05d}.png"), panel)

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"[usb_cam] processed {i} frames.")


if __name__ == "__main__":
    main()
