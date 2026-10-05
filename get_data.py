"""Download the real-photo test set and write data/real/labels.json.

Source: github.com/mbasilyan/Stop-Sign-Detection ("Stop Sign Dataset" folder):
197 street photos, 97 with a stop sign (listed in labels.tsv) and 100 without.
Not part of COCO, so YOLO has not seen these exact images during training.

Usage:  python get_data.py
"""
import io
import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile

REPO = "https://github.com/mbasilyan/Stop-Sign-Detection"
ZIP = "https://codeload.github.com/mbasilyan/Stop-Sign-Detection/zip/refs/heads/master"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "_download")
OUT = os.path.join(HERE, "data", "real")


def download():
    if os.path.isdir(RAW):
        return
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO, RAW], check=True)
    except (OSError, subprocess.CalledProcessError):
        print("git not available, downloading zip instead...")
        with urllib.request.urlopen(ZIP) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(RAW + "_zip")
        inner = os.path.join(RAW + "_zip", os.listdir(RAW + "_zip")[0])
        shutil.move(inner, RAW)
        shutil.rmtree(RAW + "_zip")


def main():
    download()
    src = os.path.join(RAW, "Stop Sign Dataset")
    lines = open(os.path.join(src, "labels.tsv")).read().splitlines()[1:]
    positives = {l.split("\t")[0] for l in lines if l.strip()}
    files = sorted((f for f in os.listdir(src) if re.match(r"^\d+\.", f)),
                   key=lambda f: int(f.split(".")[0]))
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    labels = []
    for f in files:
        shutil.copy(os.path.join(src, f), os.path.join(OUT, "images", f))
        labels.append({"file": f, "has_sign": f in positives})
    json.dump(labels, open(os.path.join(OUT, "labels.json"), "w"), indent=1)
    n_pos = sum(l["has_sign"] for l in labels)
    print(f"real set: {len(labels)} images ({n_pos} with a stop sign, {len(labels) - n_pos} without)")


if __name__ == "__main__":
    main()
