#!/usr/bin/env python3
"""
Interactive trackbar tuner for a lane_cv profile.

    python tools/tune.py --config configs/default.yaml --input frame.jpg

Drag the sliders until the lanes survive and the road specks/cracks don't, then
press 's' to print the updated YAML (redirect to a new profile). The two sliders
that matter most for the speck problem are `min_elongation` and `min_area`.
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lane_cv import LaneConfig, LaneDetector  # noqa: E402

WIN = "lane_cv tuner"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--input", required=True, help="image to tune against")
    args = ap.parse_args()

    bgr = cv2.imread(args.input)
    if bgr is None:
        ap.error(f"could not read image: {args.input}")
    cfg = LaneConfig.from_yaml(args.config) if args.config else LaneConfig()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("white_v_min", WIN, cfg.white.v_min, 255, lambda v: None)
    cv2.createTrackbar("white_s_max", WIN, cfg.white.s_max, 255, lambda v: None)
    cv2.createTrackbar("adapt_k_x10", WIN, int(cfg.adaptive.k * 10), 80, lambda v: None)
    cv2.createTrackbar("min_area", WIN, cfg.line.min_area, 2000, lambda v: None)
    cv2.createTrackbar("min_elong_x10", WIN, int(cfg.line.min_elongation * 10), 200, lambda v: None)
    cv2.createTrackbar("max_fill_x100", WIN, int(cfg.line.max_fill * 100), 100, lambda v: None)

    print("[tune] keys: s=print YAML, q=quit")
    while True:
        cfg.white.v_min = cv2.getTrackbarPos("white_v_min", WIN)
        cfg.white.s_max = cv2.getTrackbarPos("white_s_max", WIN)
        cfg.adaptive.k = cv2.getTrackbarPos("adapt_k_x10", WIN) / 10.0
        cfg.line.min_area = cv2.getTrackbarPos("min_area", WIN)
        cfg.line.min_elongation = cv2.getTrackbarPos("min_elong_x10", WIN) / 10.0
        cfg.line.max_fill = cv2.getTrackbarPos("max_fill_x100", WIN) / 100.0

        det = LaneDetector(cfg)            # fresh state each frame (temporal off effectively)
        result = det.process(bgr)
        view = det.draw_overlay(bgr, result)
        cv2.putText(view, f"segments: {len(result.segments)}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(WIN, view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            try:
                import yaml
                print("\n# ---- tuned profile ----")
                print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
            except Exception:
                print(cfg.to_dict())

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
