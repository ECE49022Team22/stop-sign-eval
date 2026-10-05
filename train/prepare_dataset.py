"""Turn whatever sign datasets you attached into one YOLO training set.

Classes in the output:
  0 stop_sign      (stop signs)
  1 do_not_enter   (do-not-enter / no-entry signs) - only if the inputs contain enough of them

Every image whose signs are something else (speed limit, yield, ...) is kept as an
unlabelled "background" image, which teaches the model those are NOT stop signs.

Formats found automatically anywhere under each --inputs folder:
  * LISA Traffic Signs   a CSV whose header has "Annotation tag" (allAnnotations.csv)
  * Pascal VOC           *.xml files with <object><name> + <bndbox> (e.g. Kaggle andrewmvd/road-sign-detection)
  * YOLO                 a data.yaml with class names next to images/ + labels/ folders
                         (Roboflow exports, and the output of fetch_open_images.py)

Images that look like the evaluation photos (data/real) are skipped, so the
before/after comparison stays fair.

Usage:
  python train/prepare_dataset.py --inputs /kaggle/input /tmp/work/open_images --out /tmp/work/finetune
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import random
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MIN_DNE = 20          # below this many do-not-enter boxes, train stop_sign only


# ------------------------------------------------------------------ class mapping
def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


STOP_NAMES = {"stop", "stopsign", "regulatorystopg1", "r11"}
DNE_NAMES = {"donotenter", "noentry", "regulatorynoentryg1", "r51", "wrongway"}


def map_class(name):
    n = norm(name)
    if n in STOP_NAMES:
        return "stop_sign"
    if n in DNE_NAMES or "donotenter" in n or "noentry" in n:
        return "do_not_enter"
    return None            # any other sign: image is still used, box is dropped


# ------------------------------------------------------------------ readers
# Each reader yields records: {"img": path, "boxes": [(source_class, x1, y1, x2, y2)], "group": str, "source": str}
# Coordinates are in pixels of the original image.

def read_lisa(csv_path):
    base = os.path.dirname(csv_path)
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
        delim = ";" if head.count(";") > head.count(",") else ","
        f.seek(0)
        rows = list(csv.DictReader(f, delimiter=delim))
    by_img = collections.OrderedDict()
    for r in rows:
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        fn = r.get("Filename")
        if not fn:
            continue
        try:
            box = (r["Annotation tag"], float(r["Upper left corner X"]), float(r["Upper left corner Y"]),
                   float(r["Lower right corner X"]), float(r["Lower right corner Y"]))
        except (KeyError, ValueError):
            continue
        rec = by_img.setdefault(fn, {"img": os.path.join(base, fn), "boxes": [],
                                     "group": "lisa:" + (r.get("Origin file") or os.path.dirname(fn)),
                                     "source": "LISA"})
        rec["boxes"].append(box)
    for rec in by_img.values():
        if os.path.exists(rec["img"]):
            yield rec


def read_voc(xml_path, image_index):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return
    if root.tag != "annotation":
        return
    fn = (root.findtext("filename") or "").strip()
    img = image_index.get(os.path.basename(fn).lower()) or image_index.get(
        os.path.splitext(os.path.basename(xml_path))[0].lower())
    if not img:
        return
    boxes = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        if bb is None:
            continue
        try:
            boxes.append((o.findtext("name"), float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                          float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
        except (TypeError, ValueError):
            continue
    yield {"img": img, "boxes": boxes, "group": "voc:" + group_from_name(img), "source": "VOC:" + top_folder(xml_path)}


def read_yolo(yaml_path):
    names = parse_yaml_names(yaml_path)
    if not names:
        return
    root = os.path.dirname(yaml_path)
    for img in glob.glob(os.path.join(root, "**", "images", "**", "*"), recursive=True):
        if not img.lower().endswith(IMG_EXT):
            continue
        lab = os.path.splitext(img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))[0] + ".txt"
        if not os.path.exists(lab):
            continue
        im = cv2.imread(img)
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = []
        for line in open(lab):
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            boxes.append((names.get(c, str(c)), (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        yield {"img": img, "boxes": boxes, "group": "yolo:" + group_from_name(img),
               "source": "YOLO:" + top_folder(yaml_path), "size": (W, H)}


def parse_yaml_names(path):
    try:
        import yaml
        d = yaml.safe_load(open(path))
        n = d.get("names")
        return dict(enumerate(n)) if isinstance(n, list) else {int(k): v for k, v in (n or {}).items()}
    except Exception:
        return {}


def group_from_name(path):
    """Frames cut from one video (e.g. clip07_frame_0042.jpg) should stay in the same split,
    or validation would just re-test near-identical frames. Group by the name minus its frame
    number when the rest still identifies a clip (contains a digit); otherwise each image is
    its own group (road12.png, IMG_1234.jpg)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.split(".rf.")[0]                       # Roboflow adds .rf.<hash>
    prefix = re.sub(r"[_\-]?\d+$", "", stem)
    return prefix if prefix != stem and re.search(r"\d", prefix) else stem


def top_folder(path):
    parts = os.path.normpath(path).split(os.sep)
    return parts[3] if len(parts) > 4 and parts[1] == "kaggle" else os.path.basename(os.path.dirname(path))


# ------------------------------------------------------------------ dedupe vs evaluation photos
def dhash(img, n=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (n + 1, n), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).tobytes()


def hamming(a, b):
    return bin(int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).count("1")


def eval_hashes():
    hs = []
    for p in glob.glob(os.path.join(REPO, "data", "real", "images", "*")):
        im = cv2.imread(p)
        if im is not None:
            hs.append(dhash(im))
    return hs


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="folders to search for datasets")
    ap.add_argument("--out", required=True, help="output folder (YOLO format + data.yaml)")
    ap.add_argument("--val", type=float, default=0.15, help="share of image groups held out for validation")
    ap.add_argument("--max-background", type=int, default=1500, help="cap on unlabelled background images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 1. find datasets
    files = []
    for root in args.inputs:
        if os.path.isdir(root):
            files += glob.glob(os.path.join(root, "**", "*"), recursive=True)
    image_index = {os.path.basename(f).lower(): f for f in files if f.lower().endswith(IMG_EXT)}
    records, found = [], collections.Counter()
    for f in files:
        fl = f.lower()
        if fl.endswith(".csv"):
            with open(f, encoding="utf-8", errors="ignore") as fh:
                if "annotation tag" in fh.readline().lower():
                    recs = list(read_lisa(f))
                    records += recs
                    found[f"LISA csv {f} ({len(recs)} images)"] += 1
        elif fl.endswith(".xml"):
            recs = list(read_voc(f, image_index))
            records += recs
            found["VOC xml files"] += len(recs)
        elif os.path.basename(fl) in ("data.yaml", "dataset.yaml"):
            recs = list(read_yolo(f))
            records += recs
            found[f"YOLO {f} ({len(recs)} images)"] += 1
    # the same image can be listed twice (e.g. LISA's per-folder csvs + allAnnotations.csv)
    uniq = {}
    for r in records:
        uniq.setdefault(os.path.realpath(r["img"]), r)
    records = list(uniq.values())
    print("Found:")
    for k, v in found.items():
        print(f"  {k}" + (f": {v}" if k == "VOC xml files" else ""))
    if not records:
        raise SystemExit("No labelled images found. Attach a dataset (see train/README.md) and check --inputs.")

    # 2. map classes
    mapping = collections.Counter()
    for r in records:
        for b in r["boxes"]:
            mapping[(b[0], map_class(b[0]))] += 1
    print("\nClass mapping (source name -> training class, boxes):")
    for (src, dst), n in sorted(mapping.items(), key=lambda x: (x[0][1] is None, -x[1])):
        print(f"  {src!s:28s} -> {dst or '(background)':14s} {n}")
    n_dne = sum(n for (s, d), n in mapping.items() if d == "do_not_enter")
    classes = ["stop_sign", "do_not_enter"] if n_dne >= MIN_DNE else ["stop_sign"]
    if len(classes) == 1:
        print(f"\nOnly {n_dne} do-not-enter boxes found (< {MIN_DNE}): training stop_sign only; "
              "do-not-enter images are kept as background (= 'not a stop sign').")
    cid = {c: i for i, c in enumerate(classes)}

    # 3. split by group, skip near-duplicates of the evaluation photos
    ev = eval_hashes()
    rng = random.Random(args.seed)
    is_pos = [any(map_class(b[0]) in cid for b in r["boxes"]) for r in records]
    pos_per_group = collections.Counter(r["group"] for r, p in zip(records, is_pos) if p)
    groups = sorted({r["group"] for r in records})
    rng.shuffle(groups)
    val_groups, val_pos, target = set(), 0, args.val * sum(pos_per_group.values())
    for g in groups:                       # fill validation group by group up to ~15% of positives
        if val_pos >= target:
            break
        if pos_per_group[g] and val_pos + pos_per_group[g] <= max(target * 1.5, 1):
            val_groups.add(g)
            val_pos += pos_per_group[g]
    positives = [r for r, p in zip(records, is_pos) if p]
    background = [r for r, p in zip(records, is_pos) if not p]
    rng.shuffle(background)
    background = background[:args.max_background]

    stats = collections.Counter()
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
    for r in positives + background:
        im = cv2.imread(r["img"])
        if im is None:
            stats["unreadable"] += 1
            continue
        if ev:
            h = dhash(im)
            if any(hamming(h, e) <= 4 for e in ev):
                stats["skipped: looks like an evaluation photo"] += 1
                continue
        H, W = im.shape[:2]
        lines = []
        for name, x1, y1, x2, y2 in r["boxes"]:
            c = map_class(name)
            if c not in cid:
                continue
            x1, x2 = sorted((max(0, min(W, x1)), max(0, min(W, x2))))
            y1, y2 = sorted((max(0, min(H, y1)), max(0, min(H, y2))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{cid[c]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            stats[f"boxes {c}"] += 1
        split = "val" if r["group"] in val_groups else "train"
        tag = hashlib.md5(r["img"].encode()).hexdigest()[:10]
        name = f"{re.sub(r'[^A-Za-z0-9]+', '_', r['source'])[:20]}_{tag}.jpg"
        cv2.imwrite(os.path.join(args.out, "images", split, name), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        open(os.path.join(args.out, "labels", split, name[:-4] + ".txt"), "w").write("\n".join(lines))
        stats[f"{split} images"] += 1
        stats[f"{split} {'positive' if lines else 'background'}"] += 1
        stats[f"source {r['source']}"] += 1

    yaml_text = (f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n"
                 "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    open(os.path.join(args.out, "data.yaml"), "w").write(yaml_text)
    summary = {"classes": classes, "stats": dict(stats),
               "class_mapping": [{"source": s, "class": d, "boxes": n} for (s, d), n in mapping.items()]}
    json.dump(summary, open(os.path.join(args.out, "dataset_summary.json"), "w"), indent=1)
    print("\nDataset written to", args.out)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    if stats["boxes stop_sign"] < 100:
        print("\nWARNING: fewer than 100 stop-sign boxes; the fine-tuned model may end up worse than the pretrained one.")


if __name__ == "__main__":
    main()
