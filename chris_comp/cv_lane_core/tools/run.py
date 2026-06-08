#!/usr/bin/env python3
"""
Run the lane_cv pipeline on a single image, a single video, or a whole folder
containing images AND/OR videos (mixed is fine; --recursive descends subfolders).

Examples:
    python tools/run.py --config configs/default.yaml --input frame.jpg
    python tools/run.py --config configs/default.yaml --input clip.mp4 --out out/
    python tools/run.py --config configs/default.yaml --input my_testset/ --out out/
    python tools/run.py --input my_testset/ --recursive --no-display --dump-masks

Folder behaviour: still images are each judged independently (the detector's
temporal state is reset between them); each video resets at its start and then
accumulates temporal votes within that clip.

Writes a side-by-side overlay (and the raw masks with --dump-masks) so you can
see exactly which specks the geometric filter removed between `candidate` and
`lane_mask`.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

# Allow running from the repo without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lane_cv import LaneConfig, LaneDetector  # noqa: E402

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_VID_EXT = (".mp4", ".avi", ".mov", ".mkv")


def load_config(path):
    if not path:
        return LaneConfig()
    return LaneConfig.from_yaml(path)


def make_view(det, bgr, result, mode):
    """Render the frame for display / saving.
    mode 'overlay' -> just the input with lanes drawn on it.
    mode 'panel'   -> three-up: overlay | candidate (colour) | lane_mask (filtered)."""
    overlay = det.draw_overlay(bgr, result)
    cv2.putText(overlay, f"segments: {len(result.segments)}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if mode == "overlay":
        return overlay
    cand = cv2.cvtColor(result.candidate, cv2.COLOR_GRAY2BGR)
    lane = cv2.cvtColor(result.lane_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(cand, "candidate (color)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(lane, "lane_mask (shape-filtered)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return np.hstack([overlay, cand, lane])


def process_frame(det, bgr, args, name, idx):
    result = det.process(bgr)
    view = make_view(det, bgr, result, args.view)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        cv2.imwrite(os.path.join(args.out, f"{name}_{args.view}.png"), view)
        if args.dump_masks:
            cv2.imwrite(os.path.join(args.out, f"{name}_lane.png"), result.lane_mask)
            cv2.imwrite(os.path.join(args.out, f"{name}_candidate.png"), result.candidate)
    if not args.no_display:
        cv2.imshow("lane_cv", view)
        key = cv2.waitKey(0 if args.step else 1) & 0xFF
        if key == ord("q"):
            return False
    return True


def process_image(det, path, args):
    """One still image. Reset first so temporal state never leaks BETWEEN
    unrelated images (each photo is judged on its own)."""
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"[run] skipping unreadable image: {path}")
        return True
    det.reset()
    name = os.path.splitext(os.path.basename(path))[0]
    return process_frame(det, bgr, args, name, 0)


def _video_out_path(args, stem, multi):
    """Resolve where this clip's output video should go.
    --video-out as a file path -> use directly (single clip); otherwise treat it
    as a directory and write '<stem>_lanes.mp4' inside it."""
    vo = args.video_out
    if not vo:
        return None
    if vo.lower().endswith(_VID_EXT) and not multi:
        os.makedirs(os.path.dirname(os.path.abspath(vo)), exist_ok=True)
        return vo
    os.makedirs(vo, exist_ok=True)
    return os.path.join(vo, f"{stem}_lanes.mp4")


def process_video(det, path, args, multi=False):
    """One video. Reset first so the previous clip's frames don't vote here;
    temporal voting then accumulates WITHIN this clip as intended. If
    --video-out is set, the processed frames are written to a playable video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[run] skipping unreadable video: {path}")
        return True
    det.reset()
    stem = os.path.splitext(os.path.basename(path))[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = _video_out_path(args, stem, multi)
    writer = None
    i = 0
    keep_going = True
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        result = det.process(bgr)
        view = make_view(det, bgr, result, args.view)

        if out_path is not None:
            if writer is None:                 # open lazily — view size known now
                h, w = view.shape[:2]
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(view)
        if args.out:                           # optional per-frame PNGs
            os.makedirs(args.out, exist_ok=True)
            cv2.imwrite(os.path.join(args.out, f"{stem}_f{i:05d}_{args.view}.png"), view)
        if not args.no_display:
            cv2.imshow("lane_cv", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                keep_going = False
                break
        i += 1
        if i % 200 == 0:
            print(f"[run] {stem}: {i} frames...")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[run] wrote {out_path}  ({i} frames @ {fps:.0f} fps, view='{args.view}')")
    else:
        print(f"[run] {stem}: {i} frames")
    return keep_going


def gather(inp, recursive):
    """Return (images, videos) found at a path: a file -> one entry; a
    directory -> all images and videos inside (optionally recursing)."""
    if os.path.isfile(inp):
        low = inp.lower()
        if low.endswith(_VID_EXT):
            return [], [inp]
        return [inp], []
    pattern = "**/*" if recursive else "*"
    files = sorted(glob.glob(os.path.join(inp, pattern), recursive=recursive))
    images = [f for f in files if f.lower().endswith(_IMG_EXT)]
    videos = [f for f in files if f.lower().endswith(_VID_EXT)]
    return images, videos


def main():
    ap = argparse.ArgumentParser(description="Run lane_cv on an image, a video, "
                                             "or a folder of images and/or videos.")
    ap.add_argument("--config", default="", help="path to a profile YAML")
    ap.add_argument("--input", required=True, help="image, video, or directory")
    ap.add_argument("--out", default="", help="directory to write per-frame PNGs (optional)")
    ap.add_argument("--video-out", default="",
                    help="write processed frames to a video file (single clip) "
                         "or directory (one '<name>_lanes.mp4' per input video)")
    ap.add_argument("--view", choices=["panel", "overlay"], default="panel",
                    help="'panel' = input|candidate|lane_mask side-by-side (best for "
                         "seeing the CV work); 'overlay' = lanes drawn on the input only")
    ap.add_argument("--recursive", action="store_true", help="descend into subfolders")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--step", action="store_true", help="wait for key each frame")
    ap.add_argument("--dump-masks", action="store_true")
    args = ap.parse_args()

    det = LaneDetector(load_config(args.config))
    images, videos = gather(args.input, args.recursive)
    if not images and not videos:
        ap.error(f"no images or videos found at: {args.input}")
    print(f"[run] {len(images)} image(s), {len(videos)} video(s)")

    multi_vid = len(videos) > 1
    for path in images:
        if not process_image(det, path, args):
            break
    else:  # only reached if the image loop wasn't broken out of
        for path in videos:
            if not process_video(det, path, args, multi=multi_vid):
                break

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
