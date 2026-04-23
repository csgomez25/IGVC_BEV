#!/usr/bin/env python3
"""
HSV Calibrator for IGVC AutoNav Segmentation
============================================
Interactive tool to tune the white (lane line) and orange (barrel) HSV
thresholds used by avl_bev_perception/seg_inference.py.

Usage:
  # On a saved still:
  python3 calibrate_hsv.py path/to/igvc_scene.jpg

  # On a live ZED feed via ROS 2 (requires rclpy + cv_bridge installed):
  python3 calibrate_hsv.py --topic /zed_front/zed_node/rgb/image_rect_color

Workflow at the venue:
  1. Save a frame from each camera that has both white lines and an orange
     barrel visible. (`ros2 run image_view image_saver` works.)
  2. Run this tool against each frame, adjust the H/S/V trackbars until
     only the lines / only the barrel are highlighted.
  3. Copy the final values into config/bev_config.yaml under
     segmentation.hsv.white.* and segmentation.hsv.orange.* and rebuild
     (or re-source if you used --symlink-install).

Controls:
  - 'w'  show only the white (lane line) mask
  - 'o'  show only the orange (barrel) mask
  - 'b'  show both masks together (default)
  - 'p'  print the current YAML snippet to terminal
  - 'q' / ESC  quit
"""

import argparse
import sys

import cv2
import numpy as np


WINDOW_IMG = 'IGVC HSV Calibrator'
WINDOW_TRACK = 'Thresholds'

DEFAULTS = {
    'white':  dict(h_min=0,  h_max=179, s_min=0,   s_max=60,  v_min=180, v_max=255),
    'orange': dict(h_min=5,  h_max=20,  s_min=130, s_max=255, v_min=100, v_max=255),
}


def make_trackbars():
    cv2.namedWindow(WINDOW_TRACK)

    def add(name, default, maxval):
        cv2.createTrackbar(name, WINDOW_TRACK, default, maxval, lambda x: None)

    # White
    add('W H min', DEFAULTS['white']['h_min'], 179)
    add('W H max', DEFAULTS['white']['h_max'], 179)
    add('W S min', DEFAULTS['white']['s_min'], 255)
    add('W S max', DEFAULTS['white']['s_max'], 255)
    add('W V min', DEFAULTS['white']['v_min'], 255)
    add('W V max', DEFAULTS['white']['v_max'], 255)
    # Orange
    add('O H min', DEFAULTS['orange']['h_min'], 179)
    add('O H max', DEFAULTS['orange']['h_max'], 179)
    add('O S min', DEFAULTS['orange']['s_min'], 255)
    add('O S max', DEFAULTS['orange']['s_max'], 255)
    add('O V min', DEFAULTS['orange']['v_min'], 255)
    add('O V max', DEFAULTS['orange']['v_max'], 255)


def read_thresholds():
    def g(n): return cv2.getTrackbarPos(n, WINDOW_TRACK)
    return {
        'white': dict(
            h_min=g('W H min'), h_max=g('W H max'),
            s_min=g('W S min'), s_max=g('W S max'),
            v_min=g('W V min'), v_max=g('W V max'),
        ),
        'orange': dict(
            h_min=g('O H min'), h_max=g('O H max'),
            s_min=g('O S min'), s_max=g('O S max'),
            v_min=g('O V min'), v_max=g('O V max'),
        ),
    }


def apply_thresholds(bgr, thr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    w = thr['white']
    o = thr['orange']
    w_mask = cv2.inRange(
        hsv,
        np.array([w['h_min'], w['s_min'], w['v_min']], dtype=np.uint8),
        np.array([w['h_max'], w['s_max'], w['v_max']], dtype=np.uint8),
    )
    o_mask = cv2.inRange(
        hsv,
        np.array([o['h_min'], o['s_min'], o['v_min']], dtype=np.uint8),
        np.array([o['h_max'], o['s_max'], o['v_max']], dtype=np.uint8),
    )
    return w_mask, o_mask


def render(bgr, w_mask, o_mask, mode='both'):
    out = bgr.copy()
    if mode in ('white', 'both'):
        out[w_mask > 0] = (255, 255, 255)
    if mode in ('orange', 'both'):
        out[o_mask > 0] = (0, 140, 255)
    return out


def print_yaml(thr):
    print()
    print('# Paste into config/bev_config.yaml under segmentation.hsv:')
    for color in ('white', 'orange'):
        print(f'        {color}:')
        for key in ('h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max'):
            print(f'          {key}: {thr[color][key]}')
    print()


# -------- File mode ----------------------------------------------------------

def run_file(path):
    bgr = cv2.imread(path)
    if bgr is None:
        print(f'Could not load image: {path}')
        sys.exit(1)
    bgr = cv2.resize(bgr, (960, int(bgr.shape[0] * 960 / bgr.shape[1])))

    make_trackbars()
    mode = 'both'
    while True:
        thr = read_thresholds()
        w_mask, o_mask = apply_thresholds(bgr, thr)
        cv2.imshow(WINDOW_IMG, render(bgr, w_mask, o_mask, mode))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('w'):
            mode = 'white'
        elif k == ord('o'):
            mode = 'orange'
        elif k == ord('b'):
            mode = 'both'
        elif k == ord('p'):
            print_yaml(thr)

    print_yaml(read_thresholds())
    cv2.destroyAllWindows()


# -------- Live ROS topic mode -----------------------------------------------

def run_topic(topic):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except ImportError:
        print('rclpy / cv_bridge not available. Either source ROS 2 first, '
              'or run with a saved image instead of --topic.')
        sys.exit(1)

    rclpy.init()
    bridge = CvBridge()

    class Sub(Node):
        def __init__(self):
            super().__init__('hsv_calibrator')
            self.latest = None
            self.create_subscription(Image, topic, self._cb, 5)

        def _cb(self, msg):
            try:
                self.latest = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().warn(f'cv_bridge: {e}')

    sub = Sub()
    make_trackbars()
    mode = 'both'

    print(f'Subscribing to {topic} ... press q/ESC in image window to quit.')
    try:
        while rclpy.ok():
            rclpy.spin_once(sub, timeout_sec=0.05)
            if sub.latest is None:
                continue
            bgr = sub.latest
            thr = read_thresholds()
            w_mask, o_mask = apply_thresholds(bgr, thr)
            cv2.imshow(WINDOW_IMG, render(bgr, w_mask, o_mask, mode))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('w'):
                mode = 'white'
            elif k == ord('o'):
                mode = 'orange'
            elif k == ord('b'):
                mode = 'both'
            elif k == ord('p'):
                print_yaml(thr)
    finally:
        print_yaml(read_thresholds())
        cv2.destroyAllWindows()
        sub.destroy_node()
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image', nargs='?', help='Path to a still image to calibrate against')
    ap.add_argument('--topic', help='ROS 2 image topic to subscribe to instead')
    args = ap.parse_args()

    if args.topic:
        run_topic(args.topic)
    elif args.image:
        run_file(args.image)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
