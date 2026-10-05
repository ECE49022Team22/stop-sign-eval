"""Build a controlled test set by pasting real stop-sign photos onto real street scenes.

Why: the real photos only say "has a sign / doesn't". To answer "how far away can
it see a sign?" and "what happens at an angle, in the dark, behind a pole?" we
need images where the sign's exact size and position are known. So we cut 15 real
stop signs out of the positive photos and paste them, at controlled sizes and
conditions, onto the 100 street photos that have no stop sign.

Groups written to data/synthetic/labels.json (box = exact sign box; lure_box = where
the look-alike was pasted in a hard negative):
  size_sweep      sign widths 12-128 px in a 640-px-wide frame (= distance sweep)
  condition       angled / rotated / dark / overexposed / motion blur / occluded,
                  each at 40 px and 80 px sign width
  hard_negative   red look-alikes with no stop sign: "do not enter"-style discs,
                  plain red discs, red-bordered triangles

Usage:  python make_synthetic.py        (run get_data.py first)
"""
import json
import os
import random

import cv2
import numpy as np

from detectors import to_robot_resolution

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "data", "real")
OUT = os.path.join(HERE, "data", "synthetic")
SEED = 49022

# Hand-checked sign boxes (x, y, w, h) in the 640-px-wide version of each positive photo.
SIGN_SOURCES = {
    "2.jpg": (372, 63, 147, 139), "3.jpg": (459, 67, 163, 160), "5.jpg": (292, 91, 182, 161),
    "8.jpg": (189, 82, 236, 216), "9.jpg": (38, 68, 125, 131), "10.jpg": (83, 100, 130, 126),
    "11.jpg": (129, 32, 123, 127), "16.jpg": (86, 26, 181, 182), "20.jpg": (528, 50, 93, 95),
    "29.jpg": (261, 96, 331, 301), "44.jpg": (166, 75, 93, 92), "58.jpg": (267, 143, 150, 167),
    "65.jpg": (330, 233, 172, 179), "69.jpg": (97, 57, 164, 170), "89.jpg": (396, 50, 180, 188),
}

SIZES = [12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 100, 128]
PER_SIZE = 40
CONDITIONS = ["angled", "rotated", "dark", "overexposed", "motion_blur", "occluded"]
CONDITION_SIZES = [40, 80]
PER_CONDITION = 40
NEG_TYPES = ["no_entry_disc", "red_disc", "red_triangle"]
NEG_SIZES = [40, 80]
PER_NEG = 30


def cut_out_sign(img, box):
    """Return the sign as BGRA: convex hull of its red area, grown to include the white rim."""
    x, y, w, h = box
    p = int(0.08 * max(w, h))
    x0, y0 = max(x - p, 0), max(y - p, 0)
    crop = img[y0:y + h + p, x0:x + w + p].copy()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 60, 40), (12, 255, 255)) |
           cv2.inRange(hsv, (155, 60, 40), (179, 255, 255)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = cv2.findNonZero((lab == biggest).astype(np.uint8))
    mask = np.zeros(red.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    k = max(3, int(0.05 * w)) | 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bx, by, bw, bh = cv2.boundingRect(mask)
    return np.dstack([crop, mask])[by:by + bh, bx:bx + bw]


def warp_sign(sign, condition, rng):
    h, w = sign.shape[:2]
    if condition == "angled":                       # ~55 deg yaw: narrower, far edge shorter
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dw, s = 0.55 * w, 0.1 * h
        dst = np.float32([[0, 0], [dw, s], [dw, h - s], [0, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(sign, M, (int(dw) + 1, h))
    if condition == "rotated":                      # bent post / tilted camera
        ang = rng.choice([-1, 1]) * rng.uniform(12, 20)
        d = int(np.hypot(w, h)) + 2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[:, 2] += [(d - w) / 2, (d - h) / 2]
        out = cv2.warpAffine(sign, M, (d, d))
        bx, by, bw, bh = cv2.boundingRect(out[:, :, 3])
        return out[by:by + bh, bx:bx + bw]
    return sign


def paste(bg, sign_bgra, width, rng, pole=True):
    """Alpha-blend the sign at `width` px wide; returns the exact box (x, y, w, h)."""
    H, W = bg.shape[:2]
    sh, sw = sign_bgra.shape[:2]
    w = width
    h = max(1, round(sh * width / sw))
    s = cv2.resize(sign_bgra, (w, h), interpolation=cv2.INTER_AREA)
    x = rng.randint(4, W - w - 4)
    y = rng.randint(int(0.08 * H), max(int(0.08 * H) + 1, int(0.6 * H) - h))
    if pole:                                        # a grey post under the sign
        pw = max(1, w // 14)
        cx = x + w // 2
        cv2.rectangle(bg, (cx - pw // 2, y + h // 2), (cx + pw // 2, min(H - 1, y + h + 3 * h)),
                      (110, 110, 115), -1)
    a = cv2.GaussianBlur(s[:, :, 3], (3, 3), 0).astype(np.float32)[..., None] / 255
    roi = bg[y:y + h, x:x + w].astype(np.float32)
    bg[y:y + h, x:x + w] = (a * s[:, :, :3] + (1 - a) * roi).astype(np.uint8)
    return [int(x), int(y), int(w), int(h)]


def apply_scene_condition(img, condition, box, rng):
    if condition == "dark":                         # dusk / heavy shade
        return np.clip(img.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)
    if condition == "overexposed":                  # sun behind camera, washed out
        f = img.astype(np.float32) / 255
        return np.clip(255 * (0.35 + 0.75 * f ** 0.6), 0, 255).astype(np.uint8)
    if condition == "motion_blur":                  # robot turning while capturing
        k = np.zeros((11, 11), np.float32)
        k[5, :] = 1 / 11
        return cv2.filter2D(img, -1, k)
    if condition == "occluded":                     # pole / tree trunk / pedestrian in front
        x, y, w, h = box
        bw = max(2, int(0.3 * w))
        bx = x + rng.randint(int(0.2 * w), int(0.5 * w))
        img = img.copy()
        cv2.rectangle(img, (bx, y - h // 2), (bx + bw, y + h + h // 2), (70, 80, 75), -1)
        return img
    return img


def draw_look_alike(kind, width):
    """Generic red objects that are not stop signs (drawn at 4x then downsampled)."""
    S = 4 * width
    img = np.zeros((S, S, 4), np.uint8)
    c = (S // 2, S // 2)
    red = (40, 30, 200, 255)
    white = (245, 245, 245, 255)
    if kind == "no_entry_disc":
        cv2.circle(img, c, S // 2 - 2, white, -1, cv2.LINE_AA)
        cv2.circle(img, c, int(S * 0.46), red, -1, cv2.LINE_AA)
        cv2.rectangle(img, (int(S * 0.2), int(S * 0.42)), (int(S * 0.8), int(S * 0.58)), white, -1)
    elif kind == "red_disc":
        cv2.circle(img, c, S // 2 - 2, red, -1, cv2.LINE_AA)
    elif kind == "red_triangle":
        pts = np.int32([[S // 2, 2], [S - 2, S - 2], [2, S - 2]])
        cv2.fillPoly(img, [pts], red, cv2.LINE_AA)
        inner = np.int32([[S // 2, int(S * 0.3)], [int(S * 0.78), int(S * 0.86)], [int(S * 0.22), int(S * 0.86)]])
        cv2.fillPoly(img, [inner], white, cv2.LINE_AA)
    return img


def finish(img):
    """Mild blur + JPEG so pasted signs aren't unnaturally crisp."""
    return cv2.GaussianBlur(img, (0, 0), 0.6)


def main():
    rng = random.Random(SEED)
    labels_real = json.load(open(os.path.join(REAL, "labels.json")))
    load = lambda f: to_robot_resolution(cv2.imread(os.path.join(REAL, "images", f)))
    backgrounds = [load(l["file"]) for l in labels_real if not l["has_sign"]]
    signs = {f: cut_out_sign(load(f), b) for f, b in SIGN_SOURCES.items()}
    names = list(signs)

    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "sign_cutouts"), exist_ok=True)
    for f, s in signs.items():
        cv2.imwrite(os.path.join(OUT, "sign_cutouts", f.rsplit(".", 1)[0] + ".png"), s)

    labels = []

    def save(img, **meta):
        fname = f"{len(labels):04d}_{meta['group']}.jpg"
        cv2.imwrite(os.path.join(OUT, "images", fname), finish(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
        labels.append({"file": fname, **meta})

    for size in SIZES:
        for _ in range(PER_SIZE):
            bg, src = rng.choice(backgrounds).copy(), rng.choice(names)
            box = paste(bg, signs[src], size, rng)
            save(bg, group="size_sweep", has_sign=True, size=size, condition="frontal", box=box, sign_src=src)

    for cond in CONDITIONS:
        for size in CONDITION_SIZES:
            for _ in range(PER_CONDITION):
                bg, src = rng.choice(backgrounds).copy(), rng.choice(names)
                sign = warp_sign(signs[src], cond, rng)
                # for "angled", size = width of the un-rotated sign (same distance as frontal)
                w = round(size * 0.55) if cond == "angled" else size
                box = paste(bg, sign, w, rng)
                img = apply_scene_condition(bg, cond, box, rng)
                save(img, group="condition", has_sign=True, size=size, condition=cond, box=box, sign_src=src)

    for kind in NEG_TYPES:
        for size in NEG_SIZES:
            for _ in range(PER_NEG):
                bg = rng.choice(backgrounds).copy()
                lure = paste(bg, draw_look_alike(kind, size), size, rng)
                save(bg, group="hard_negative", has_sign=False, size=size, condition=kind, box=None,
                     lure_box=lure, sign_src=None)

    json.dump(labels, open(os.path.join(OUT, "labels.json"), "w"), indent=1)
    from collections import Counter
    print(f"synthetic set: {len(labels)} images", dict(Counter(l["group"] for l in labels)))


if __name__ == "__main__":
    main()
