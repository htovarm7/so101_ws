"""Standalone zone-detector test using a webcam.

Replicates the exact HSV + morphology pipeline from
  so101_perception/so101_perception/zone_detector.py
and shows every intermediate step with cv2.imshow.

No ROS, no Docker required.

Usage
-----
  python3 test_zone_detection.py                   # camera 0, default thresholds
  python3 test_zone_detection.py --camera 1        # use a different index
  python3 test_zone_detection.py --tune            # show HSV trackbar window
  python3 test_zone_detection.py --min-area 2000   # lower area threshold

Keys
----
  q / ESC   quit
  s         print current HSV ranges to stdout
  w         overwrite so101_perception/config/zones_hsv.yaml with current values
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np


# Path to the live config consumed by the ROS node (relative to this script).
YAML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "so101_perception",
    "config",
    "zones_hsv.yaml",
)


# ---------------------------------------------------------------------------
#  Zone specs — mirrors zone_detector.py ROS parameter defaults
# ---------------------------------------------------------------------------

ZONES: list[dict] = [
    {
        "name": "zone_a",
        "label": "ZONE A (pink)",
        "lower": [128, 158, 42],
        "upper": [176, 221, 243],
        "min_area": 88,  # estimated from slider position
        "bgr": (255, 0, 255),  # magenta — same as node
        "wrap": False,
    },
    {
        "name": "zone_b",
        "label": "ZONE B (orange)",
        "lower": [0, 127, 215],
        "upper": [99, 198, 255],
        "min_area": 255,  # slider appears maxed
        "bgr": (0, 140, 255),  # orange in BGR
        "wrap": False,
    },
]


# ---------------------------------------------------------------------------
#  Core detection helpers — identical logic to zone_detector.py
# ---------------------------------------------------------------------------


def _mask_for(hsv: np.ndarray, zone: dict) -> np.ndarray:
    lo = np.array(zone["lower"], dtype=np.uint8)
    hi = np.array(zone["upper"], dtype=np.uint8)
    if zone["wrap"]:
        lo1 = np.array([0, lo[1], lo[2]], dtype=np.uint8)
        hi1 = np.array([hi[0], hi[1], hi[2]], dtype=np.uint8)
        lo2 = np.array([lo[0], lo[1], lo[2]], dtype=np.uint8)
        hi2 = np.array([179, hi[1], hi[2]], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
    else:
        mask = cv2.inRange(hsv, lo, hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_contour(
    mask: np.ndarray, min_area: int
) -> Tuple[Optional[np.ndarray], float, Optional[Tuple[int, int]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, area, None
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None, area, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return c, area, (cx, cy)


# ---------------------------------------------------------------------------
#  Trackbar helpers (--tune mode)
# ---------------------------------------------------------------------------

TUNE_WIN = "HSV Tuner"


def _setup_trackbars(zones: list[dict]) -> None:
    cv2.namedWindow(TUNE_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNE_WIN, 500, 400)
    for z in zones:
        prefix = z["name"]
        lo, hi = z["lower"], z["upper"]
        for i, ch in enumerate(("H_lo", "S_lo", "V_lo")):
            cv2.createTrackbar(
                f"{prefix} {ch}",
                TUNE_WIN,
                lo[i],
                179 if ch == "H_lo" else 255,
                lambda _: None,
            )
        for i, ch in enumerate(("H_hi", "S_hi", "V_hi")):
            cv2.createTrackbar(
                f"{prefix} {ch}",
                TUNE_WIN,
                hi[i],
                179 if ch == "H_hi" else 255,
                lambda _: None,
            )
        cv2.createTrackbar(
            f"{prefix} min_area", TUNE_WIN, z["min_area"], 50000, lambda _: None
        )


def _read_trackbars(zones: list[dict]) -> None:
    for z in zones:
        prefix = z["name"]
        z["lower"] = [
            cv2.getTrackbarPos(f"{prefix} H_lo", TUNE_WIN),
            cv2.getTrackbarPos(f"{prefix} S_lo", TUNE_WIN),
            cv2.getTrackbarPos(f"{prefix} V_lo", TUNE_WIN),
        ]
        z["upper"] = [
            cv2.getTrackbarPos(f"{prefix} H_hi", TUNE_WIN),
            cv2.getTrackbarPos(f"{prefix} S_hi", TUNE_WIN),
            cv2.getTrackbarPos(f"{prefix} V_hi", TUNE_WIN),
        ]
        z["min_area"] = max(1, cv2.getTrackbarPos(f"{prefix} min_area", TUNE_WIN))


# ---------------------------------------------------------------------------
#  Composite frame builder
# ---------------------------------------------------------------------------


def _build_composite(
    raw: np.ndarray,
    debug: np.ndarray,
    masks: list[np.ndarray],
    zones: list[dict],
    target_w: int = 1280,
) -> np.ndarray:
    """Build a 2×2 grid: [raw | debug] / [mask_a_coloured | mask_b_coloured]."""
    h, w = raw.shape[:2]
    cell_w = target_w // 2
    cell_h = int(h * cell_w / w)

    def _resize(img):
        return cv2.resize(img, (cell_w, cell_h))

    top = np.hstack([_resize(raw), _resize(debug)])

    mask_panels = []
    for z, mask in zip(zones, masks):
        coloured = np.zeros((h, w, 3), dtype=np.uint8)
        coloured[mask > 0] = z["bgr"]
        # overlay label
        cv2.putText(
            coloured, z["label"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, z["bgr"], 2
        )
        mask_panels.append(_resize(coloured))

    while len(mask_panels) < 2:
        mask_panels.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    bottom = np.hstack(mask_panels[:2])
    return np.vstack([top, bottom])


# ---------------------------------------------------------------------------
#  Simulated ROS topic printer
# ---------------------------------------------------------------------------

_last_print: dict[str, float] = {}


def _print_topic(
    zone_name: str, detected: bool, area: float, centroid: Optional[Tuple[int, int]]
) -> None:
    now = time.monotonic()
    if now - _last_print.get(zone_name, 0) < 0.5:  # throttle to 2 Hz
        return
    _last_print[zone_name] = now

    topic = f"/zone_detector/{zone_name}"
    if detected and centroid:
        cx, cy = centroid
        print(
            f"[PUB]  {topic:<30s}  area={int(area):<6d}  "
            f"pixel_centroid=({cx:4d}, {cy:4d})  "
            f"(3-D needs depth — no RealSense here)"
        )
    else:
        print(f"[    ] {topic:<30s}  not published  (best_area={int(area):<6d})")


# ---------------------------------------------------------------------------
#  YAML writer — preserves comments, only patches HSV fields
# ---------------------------------------------------------------------------


def _overwrite_yaml(zones: list[dict], yaml_path: str) -> None:
    """Update hsv_lower / hsv_upper / min_contour_area in zones_hsv.yaml in-place.

    Reads the file line-by-line so all comments and unrelated parameters are
    kept exactly as they were.
    """
    try:
        with open(yaml_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: YAML not found at {yaml_path}", file=sys.stderr)
        return

    zone_by_name = {z["name"]: z for z in zones}
    current_zone: Optional[dict] = None
    current_zone_indent = 0
    result: list[str] = []

    for line in lines:
        raw = line.rstrip("\n")
        stripped = raw.lstrip()

        # Preserve blank lines and comments unchanged.
        if not stripped or stripped.startswith("#"):
            result.append(line)
            continue

        indent = len(raw) - len(stripped)

        # Leaving a zone block when we reach a line at the same or shallower
        # indent as the zone header (and it isn't the header itself).
        if (
            current_zone is not None
            and indent <= current_zone_indent
            and stripped != f"{current_zone['name']}:"
        ):
            current_zone = None

        # Detect zone block header.
        for name, z in zone_by_name.items():
            if stripped == f"{name}:":
                current_zone = z
                current_zone_indent = indent
                break

        # Patch known fields while inside a zone block.
        if current_zone is not None and stripped != f"{current_zone['name']}:":
            z = current_zone
            if stripped.startswith("hsv_lower:"):
                line = " " * indent + f"hsv_lower: {z['lower']}\n"
            elif stripped.startswith("hsv_upper:"):
                line = " " * indent + f"hsv_upper: {z['upper']}\n"
            elif stripped.startswith("min_contour_area:"):
                line = " " * indent + f"min_contour_area: {z['min_area']}\n"

        result.append(line)

    with open(yaml_path, "w") as f:
        f.writelines(result)

    print(f"\nSaved → {yaml_path}")
    for z in zones:
        print(
            f"  {z['name']}: hsv_lower={z['lower']}  "
            f"hsv_upper={z['upper']}  min_contour_area={z['min_area']}"
        )
    print()


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="cv2.VideoCapture index (default: 0)"
    )
    parser.add_argument(
        "--min-area", type=int, default=None, help="Override min_area for both zones"
    )
    parser.add_argument("--tune", action="store_true", help="Show HSV trackbar window")
    args = parser.parse_args()

    if args.min_area is not None:
        for z in ZONES:
            z["min_area"] = args.min_area

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.camera}", file=sys.stderr)
        sys.exit(1)

    # Try to get a decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    main_win = "Zone Detector Test  [q/ESC=quit | s=print HSV | w=save YAML]"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(main_win, 1280, 720)

    if args.tune:
        _setup_trackbars(ZONES)

    print(f"\nOpened camera {args.camera}")
    print(f"{'Zone':<10}  {'HSV lower':<20}  {'HSV upper':<20}  {'min_area'}")
    for z in ZONES:
        print(
            f"{z['name']:<10}  {str(z['lower']):<20}  {str(z['upper']):<20}  {z['min_area']}"
        )
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: failed to read frame — camera disconnected?", file=sys.stderr)
            break

        if args.tune:
            _read_trackbars(ZONES)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        debug = frame.copy()
        masks: list[np.ndarray] = []

        for idx, zone in enumerate(ZONES):
            mask = _mask_for(hsv, zone)
            masks.append(mask)

            contour, area, centroid = _largest_contour(mask, zone["min_area"])
            detected = contour is not None and centroid is not None

            # ── annotate debug overlay (mirrors node's drawing code) ──────
            y_text = 30 + idx * 30
            if detected:
                cx_px, cy_px = centroid
                cv2.drawContours(debug, [contour], -1, zone["bgr"], 2)
                cv2.circle(debug, (cx_px, cy_px), 8, zone["bgr"], -1)
                label = f"{zone['label']}  A={int(area)}  px=({cx_px},{cy_px})"
                cv2.putText(
                    debug,
                    label,
                    (cx_px + 10, max(cy_px - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    zone["bgr"],
                    2,
                )
                cv2.putText(
                    debug,
                    f"DETECTED  {zone['label']}",
                    (12, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    zone["bgr"],
                    2,
                )
            else:
                cv2.putText(
                    debug,
                    f"-- {zone['label']}  (best area={int(area)})",
                    (12, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (160, 160, 160),
                    2,
                )

            _print_topic(zone["name"], detected, area, centroid if detected else None)

        composite = _build_composite(frame, debug, masks, ZONES)
        cv2.imshow(main_win, composite)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("s"):
            print("\n── Current HSV ranges ──")
            for z in ZONES:
                print(
                    f"  {z['name']}: lower={z['lower']}  upper={z['upper']}  "
                    f"min_area={z['min_area']}"
                )
            print()
        if key == ord("w"):
            _overwrite_yaml(ZONES, YAML_PATH)

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
