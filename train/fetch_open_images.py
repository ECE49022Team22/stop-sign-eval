"""Download stop-sign photos + boxes from Google's Open Images V7 into a YOLO folder.

License: Open Images annotations are CC BY 4.0, images are listed as CC BY 2.0.
Only the annotation CSVs (streamed, not stored) and the selected images are downloaded.

Usage:
  python train/fetch_open_images.py --out /tmp/work/open_images --max-images 800

The output folder is picked up by prepare_dataset.py as a YOLO dataset.
If anything fails (no internet, URL moved), it prints a warning and exits 0 so a
Kaggle run keeps going with the other datasets.
"""
import argparse
import concurrent.futures as cf
import csv
import io
import os
import sys
import urllib.request

CLASS_DESCRIPTIONS = [
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
]
BOX_CSVS = {   # split -> candidate URLs (small splits first; train is ~2 GB and only streamed if needed)
    "validation": ["https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"],
    "test": ["https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"],
    "train": ["https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"],
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
FALLBACK_MID = "/m/02pv19"       # "Stop sign"


def open_url(urls):
    for u in urls:
        try:
            return urllib.request.urlopen(u, timeout=60), u
        except Exception as e:
            print(f"  could not open {u}: {e}")
    return None, None


def find_mid(name):
    r, _ = open_url(CLASS_DESCRIPTIONS)
    if r is None:
        return FALLBACK_MID
    for row in csv.reader(io.TextIOWrapper(r, encoding="utf-8")):
        if len(row) >= 2 and row[1].strip().lower() == name.lower():
            return row[0]
    return FALLBACK_MID


def collect_boxes(mid, max_images):
    images = {}                                  # (split, id) -> [(x1, y1, x2, y2) relative]
    for split, urls in BOX_CSVS.items():
        if len(images) >= max_images:
            break
        r, u = open_url(urls)
        if r is None:
            continue
        print(f"  scanning {u} ...")
        reader = csv.DictReader(io.TextIOWrapper(r, encoding="utf-8"))
        for row in reader:
            if row["LabelName"] != mid or row.get("IsDepiction") == "1" or row.get("IsGroupOf") == "1":
                continue
            key = (split, row["ImageID"])
            if key not in images and len(images) >= max_images:
                break
            images.setdefault(key, []).append(tuple(float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax")))
        r.close()
        print(f"  {len(images)} images so far")
    return images


def download(item, out):
    (split, image_id), boxes = item
    dst = os.path.join(out, "images", f"oi_{image_id}.jpg")
    if not os.path.exists(dst):
        try:
            with urllib.request.urlopen(IMAGE_URL.format(split=split, image_id=image_id), timeout=60) as r:
                data = r.read()
            open(dst, "wb").write(data)
        except Exception:
            return False
    lines = [f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}" for x1, y1, x2, y2 in boxes]
    open(os.path.join(out, "labels", f"oi_{image_id}.txt"), "w").write("\n".join(lines))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=800)
    ap.add_argument("--class-name", default="Stop sign")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    try:
        mid = find_mid(args.class_name)
        print(f"Open Images class '{args.class_name}' = {mid}")
        items = collect_boxes(mid, args.max_images)
        with cf.ThreadPoolExecutor(16) as ex:
            ok = sum(ex.map(lambda it: download(it, args.out), items.items()))
        open(os.path.join(args.out, "data.yaml"), "w").write(
            f"path: {os.path.abspath(args.out)}\ntrain: images\nval: images\nnames:\n  0: {args.class_name}\n")
        print(f"Open Images: downloaded {ok} of {len(items)} images to {args.out}")
    except Exception as e:                       # never break the whole Kaggle run
        print(f"WARNING: Open Images download failed ({e}); continuing without it.", file=sys.stderr)


if __name__ == "__main__":
    main()
