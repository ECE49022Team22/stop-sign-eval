"""Thin wrappers so both detectors take the same input and return the same output.

Every detector returns a list of detections: (x, y, w, h, score) in pixels of the
image it was given. score is the YOLO confidence, or 1.0 for the rule-based detector
(it has no notion of confidence: a blob either passes every rule or it doesn't).
"""
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


COCO_STOP_SIGN = 11                # class index of "stop sign" in the pretrained COCO models


class YoloDetector:
    def __init__(self, weights="yolo26n.pt", conf=0.05, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf           # keep low here; the evaluation applies the real threshold
        self.imgsz = imgsz
        self.name = weights.rsplit(".", 1)[0]

    def __call__(self, img):
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf,
                               classes=[COCO_STOP_SIGN], verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(c)))
        return out
