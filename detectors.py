"""Thin wrappers so both detectors take the same input and return the same output.

Every detector returns a list of detections: (x, y, w, h, score) in pixels of the
image it was given. score is the YOLO confidence, or 1.0 for the rule-based detector
(it has no notion of confidence: a blob either passes every rule or it doesn't).
"""
import os

import cv2

import stop_sign_detector as ssd   # the team's detector, copied unchanged from ECE49022Team22/testing

ROBOT_WIDTH = 640                  # the robot camera runs at 640x480


def to_robot_resolution(img, width=ROBOT_WIDTH):
    """Downscale an image so its width is 640 px, like a frame from the robot's camera."""
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def rules_detect(img, min_area=ssd.MIN_AREA):
    """The team's HSV-red + octagon-shape rules, single frame (no 3-frame confirmation)."""
    dets, _ = ssd.detect_stop_signs(img, min_area)
    return [(*d["bbox"], 1.0) for d in dets]


def stop_class_id(names):
    """Index of the stop-sign class: 11 ("stop sign") in the pretrained COCO models,
    0 ("stop_sign") in our fine-tuned models. Found by name so both work."""
    for i, n in (names.items() if isinstance(names, dict) else enumerate(names)):
        if "".join(ch for ch in n.lower() if ch.isalnum()) == "stopsign":
            return int(i)
    raise ValueError(f"model has no stop-sign class: {names}")


class YoloDetector:
    """Returns stop-sign boxes only (any other class, e.g. do_not_enter, is ignored)."""

    def __init__(self, weights="yolo26n.pt", conf=0.05, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf           # keep low here; the evaluation applies the real threshold
        self.imgsz = imgsz
        self.name = os.path.basename(weights).rsplit(".", 1)[0]
        self.stop_id = stop_class_id(self.model.names)

    def __call__(self, img):
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf,
                               classes=[self.stop_id], verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(c)))
        return out
