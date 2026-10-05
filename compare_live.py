"""Side-by-side: team's rule-based detector (left) vs YOLO (right), on anything you point it at.

Usage:
  python compare_live.py                      # laptop webcam (index 0)
  python compare_live.py --source 1           # another camera
  python compare_live.py --source photo.jpg   # one image
  python compare_live.py --source my_folder   # every image in a folder (any key = next)
  python compare_live.py --source clip.mp4    # a video
Options: --weights yolo11n.pt   --conf 0.5
Keys: q / Esc = quit, s = save screenshot, any other key = next image (folder mode)
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from detectors import YoloDetector, rules_detect, to_robot_resolution

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def draw(img, dets, color, name, ms):
    out = img.copy()
    for x, y, w, h, s in dets:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        tag = "STOP" if s >= 1.0 else f"STOP {s:.2f}"
        cv2.putText(out, tag, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (30, 30, 30), -1)
    verdict = "STOP SIGN" if dets else "no sign"
    cv2.putText(out, f"{name}: {verdict}   {ms:.0f} ms", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def frames(source):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f, False
    elif os.path.isdir(source):
        for p in sorted(glob.glob(os.path.join(source, "*"))):
            if p.lower().endswith(IMG_EXT):
                yield cv2.imread(p), True
    elif source.lower().endswith(IMG_EXT):
        yield cv2.imread(source), True
    else:
        cap = cv2.VideoCapture(source)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    yolo = YoloDetector(args.weights, conf=args.conf)

    for frame, still in frames(args.source):
        img = to_robot_resolution(frame)
        t = time.perf_counter(); r = rules_detect(img); tr = 1000 * (time.perf_counter() - t)
        t = time.perf_counter(); y = yolo(img); ty = 1000 * (time.perf_counter() - t)
        grid = np.hstack([draw(img, r, (0, 140, 255), "Rules", tr),
                          draw(img, y, (255, 120, 40), yolo.name, ty)])
        cv2.imshow("Rules (left) vs YOLO (right)", grid)
        key = cv2.waitKey(0 if still else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            name = time.strftime("compare_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, grid)
            print("saved", name)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
