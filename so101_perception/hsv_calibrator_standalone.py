"""Standalone HSV calibrator for SO-101 — no ROS, no Docker.

Talks directly to the RealSense via pyrealsense2 and shows OpenCV windows
with live trackbars.  Use this to find good HSV ranges for each of your 6
objects, then save them all to a YAML file at the end.

Requirements (install once on host):
    pip install pyrealsense2 opencv-python numpy pyyaml

Usage:
    python3 hsv_calibrator_standalone.py
    python3 hsv_calibrator_standalone.py --output ./objects_hsv.yaml
    python3 hsv_calibrator_standalone.py --width 640 --height 480

Keys (focus must be on an OpenCV window):
    s   save current trackbar HSV range to the selected class slot
    r   reset (clear) the currently selected class slot
    n   next class slot
    p   previous class slot
    w   write all captured ranges to YAML
    q   quit
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 is not installed.")
    print("Install it with:  pip install pyrealsense2")
    sys.exit(1)


DEFAULT_LABELS: List[str] = [
    "red_heart_bear",
    "blue_dragon",
    "cereal_box",
    "object_4",
    "object_5",
    "object_6",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="objects_hsv.yaml",
                   help="YAML output path (default: ./objects_hsv.yaml).")
    p.add_argument("--width",  type=int, default=640,
                   help="Color stream width (default 640).")
    p.add_argument("--height", type=int, default=480,
                   help="Color stream height (default 480).")
    p.add_argument("--fps",    type=int, default=30,
                   help="Color stream FPS (default 30).")
    p.add_argument("--min-area", type=int, default=500,
                   help="Minimum contour area in pixels (default 500).")
    return p.parse_args()


def noop(_v):
    """OpenCV trackbar callback (we poll values in the loop)."""
    return None


def make_windows(win_main: str, win_mask: str) -> None:
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_mask, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("class", win_main, 0, 5,   noop)
    cv2.createTrackbar("H min", win_main, 0, 179, noop)
    cv2.createTrackbar("H max", win_main, 179, 179, noop)
    cv2.createTrackbar("S min", win_main, 80, 255, noop)
    cv2.createTrackbar("S max", win_main, 255, 255, noop)
    cv2.createTrackbar("V min", win_main, 50, 255, noop)
    cv2.createTrackbar("V max", win_main, 255, 255, noop)


def read_trackbars(win_main: str):
    cid  = cv2.getTrackbarPos("class", win_main)
    h_lo = cv2.getTrackbarPos("H min", win_main)
    h_hi = cv2.getTrackbarPos("H max", win_main)
    s_lo = cv2.getTrackbarPos("S min", win_main)
    s_hi = cv2.getTrackbarPos("S max", win_main)
    v_lo = cv2.getTrackbarPos("V min", win_main)
    v_hi = cv2.getTrackbarPos("V max", win_main)
    return cid, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)


def compute_mask(hsv: np.ndarray, lo, hi) -> np.ndarray:
    """HSV mask with hue wrap-around (H_min > H_max enables wrap)."""
    h_lo, s_lo, v_lo = lo
    h_hi, s_hi, v_hi = hi
    if h_lo <= h_hi:
        mask = cv2.inRange(hsv,
                           np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                           np.array([h_hi, s_hi, v_hi], dtype=np.uint8))
    else:
        m1 = cv2.inRange(hsv,
                         np.array([0,    s_lo, v_lo], dtype=np.uint8),
                         np.array([h_hi, s_hi, v_hi], dtype=np.uint8))
        m2 = cv2.inRange(hsv,
                         np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                         np.array([179,  s_hi, v_hi], dtype=np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def annotate(bgr: np.ndarray, mask: np.ndarray, cid: int, label: str,
             lo, hi, captured, min_area: int) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]

    # Outline largest contour for visual feedback.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(out, (cx, cy), 6, (0, 255, 255), -1)

    # Status bar at the top.
    cv2.rectangle(out, (0, 0), (w, 70), (0, 0, 0), thickness=-1)
    cv2.putText(out, f"Class {cid}: {label}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        out,
        f"HSV: [{lo[0]},{lo[1]},{lo[2]}] -> [{hi[0]},{hi[1]},{hi[2]}]",
        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
    )

    # Capture status strip at the bottom.
    cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 0), thickness=-1)
    for i in range(6):
        colour = (0, 200, 0) if captured[i] is not None else (90, 90, 90)
        tag = "OK" if captured[i] is not None else "--"
        if i == cid:
            cv2.rectangle(out, (10 + i * 90, h - 25),
                          (95 + i * 90, h - 5), (0, 255, 255), 1)
        cv2.putText(out, f"{i}:{tag}",
                    (15 + i * 90, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return out


def write_yaml(output_path: str, captured, labels, min_area: int) -> int:
    classes_out = []
    for i, entry in enumerate(captured):
        if entry is None:
            classes_out.append({
                "label":     labels[i],
                "enabled":   False,
                "hsv_lower": [0, 0, 0],
                "hsv_upper": [179, 255, 255],
                "min_contour_area": int(min_area),
            })
        else:
            classes_out.append({**entry, "enabled": True})

    payload = {
        "object_classifier": {
            "ros__parameters": {
                "calibrated_at": datetime.now().isoformat(timespec="seconds"),
                "classes": classes_out,
            }
        }
    }

    try:
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
    except OSError:
        pass

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=None)
    return sum(1 for c in captured if c is not None)


def main():
    args = parse_args()

    # ── Start RealSense color stream ──────────────────────────────────────────
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"ERROR: Could not start RealSense pipeline: {e}")
        print("Check: is the camera plugged into a USB 3 port?")
        print("       is realsense-viewer or another process using it?")
        sys.exit(1)

    print(f"RealSense started at {args.width}x{args.height} @ {args.fps} fps")
    print(f"Output will be written to: {os.path.abspath(args.output)}")
    print()
    print("Trackbars:")
    print("  class  — pick slot 0-5")
    print("  H/S/V min/max — adjust until the mask isolates the object")
    print()
    print("Keys (focus on an OpenCV window):")
    print("  s = save current HSV to the selected slot")
    print("  r = reset (clear) the selected slot")
    print("  n / p = next / previous slot")
    print("  w = write YAML")
    print("  q = quit")
    print()

    # ── Set up OpenCV windows ─────────────────────────────────────────────────
    win_main = "HSV Calibrator (image)"
    win_mask = "HSV Calibrator (mask)"
    make_windows(win_main, win_mask)

    captured: List[Optional[Dict]] = [None] * 6
    labels = DEFAULT_LABELS

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            bgr = np.asanyarray(color_frame.get_data())

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            cid, lo, hi = read_trackbars(win_main)
            cid = max(0, min(5, cid))
            mask = compute_mask(hsv, lo, hi)

            annotated = annotate(bgr, mask, cid, labels[cid], lo, hi,
                                 captured, args.min_area)
            masked = cv2.bitwise_and(bgr, bgr, mask=mask)

            cv2.imshow(win_main, annotated)
            cv2.imshow(win_mask, masked)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            ch = chr(key).lower() if key < 128 else ""

            if ch == "s":
                captured[cid] = {
                    "label":     labels[cid],
                    "hsv_lower": [int(lo[0]), int(lo[1]), int(lo[2])],
                    "hsv_upper": [int(hi[0]), int(hi[1]), int(hi[2])],
                    "min_contour_area": int(args.min_area),
                }
                print(f"Saved class {cid} ({labels[cid]}): "
                      f"lower={list(lo)} upper={list(hi)}")
            elif ch == "r":
                captured[cid] = None
                print(f"Reset class {cid} ({labels[cid]}).")
            elif ch == "n":
                cv2.setTrackbarPos("class", win_main, (cid + 1) % 6)
            elif ch == "p":
                cv2.setTrackbarPos("class", win_main, (cid - 1) % 6)
            elif ch == "w":
                n_ok = write_yaml(args.output, captured, labels, args.min_area)
                print(f"Wrote {n_ok}/6 calibrated classes to "
                      f"{os.path.abspath(args.output)}")
            elif ch == "q":
                print("Quit.")
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()