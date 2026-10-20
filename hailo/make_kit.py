#!/usr/bin/env python3
"""Build a kit for compiling a YOLO model to a Hailo-8L HEF on an x86_64 Linux PC.

    .venv/bin/python hailo/make_kit.py                       # yolo26n.pt, ~1,050 calibration images
    .venv/bin/python hailo/make_kit.py --weights best.pt     # a fine-tuned model instead
    .venv/bin/python hailo/make_kit.py --camera-frames 300   # more frames from the robot's camera

Hailo compilation needs the Hailo Dataflow Compiler, which only runs on Linux x86_64, so it can't
happen on the Pi. This collects what that PC needs into ~/hailo_export_kit.tar.gz:

  weights        the .pt to compile
  calib/         calibration images (Hailo recommends >= 1,024, in-domain beats out-of-domain):
                 frames from the robot's own camera + data/real + a slice of data/synthetic
  calib.yaml     dataset file for `yolo export ... data=calib.yaml`
  export.sh      sets up a venv with the same Ultralytics version as the Pi, installs the DFC
                 wheel you downloaded, compiles, and packs the result
  README.md      the steps

The kit contains the third-party photos from data/ (no license), so keep it out of git.
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOME = os.path.expanduser("~")


def capture(n, interval, out):
    sys.path.insert(0, os.path.join(HOME, "testing"))
    from latest_camera import LatestFrameCamera
    import cv2
    cam = LatestFrameCamera(0, 640, 480, 30)
    saved = 0
    try:
        for _ in range(10):                                # let auto-exposure settle
            cam.read()
        while saved < n:
            frame, _ = cam.read()
            if frame is None:
                continue
            cv2.imwrite(os.path.join(out, f"camera_{saved:04d}.jpg"), frame)
            saved += 1
            time.sleep(interval)
    finally:
        cam.release()
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.path.join(REPO, "yolo26n.pt"))
    ap.add_argument("--camera-frames", type=int, default=150, help="frames to capture from the robot's camera (0 = none)")
    ap.add_argument("--interval", type=float, default=0.3, help="seconds between captured frames")
    ap.add_argument("--synthetic", type=int, default=700, help="images taken from data/synthetic")
    ap.add_argument("--out", default=os.path.join(HOME, "hailo_export_kit"))
    a = ap.parse_args()

    from ultralytics import YOLO, __version__ as ul_version
    names = YOLO(a.weights).names
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    img_dir = os.path.join(a.out, "calib", "images")
    os.makedirs(img_dir)
    os.makedirs(os.path.join(a.out, "calib", "labels"))   # empty: calibration needs images only

    counts = {}
    if a.camera_frames:
        print(f"capturing {a.camera_frames} frames from the robot's camera ...", flush=True)
        counts["robot camera"] = capture(a.camera_frames, a.interval, img_dir)
    real = sorted(os.listdir(os.path.join(REPO, "data/real/images")))
    for f in real:
        shutil.copy(os.path.join(REPO, "data/real/images", f), os.path.join(img_dir, "real_" + f))
    counts["data/real"] = len(real)
    syn = sorted(os.listdir(os.path.join(REPO, "data/synthetic/images")))
    random.Random(0).shuffle(syn)
    for f in syn[:a.synthetic]:
        shutil.copy(os.path.join(REPO, "data/synthetic/images", f), os.path.join(img_dir, "syn_" + f))
    counts["data/synthetic"] = min(a.synthetic, len(syn))

    stem = os.path.splitext(os.path.basename(a.weights))[0]
    shutil.copy(a.weights, os.path.join(a.out, os.path.basename(a.weights)))
    with open(os.path.join(a.out, "calib.yaml"), "w") as f:
        f.write("# calibration images for Hailo INT8 quantization (labels not needed)\n")
        f.write("path: calib\ntrain: images\nval: images\nnames:\n")
        for i in sorted(names):
            f.write(f"  {i}: {names[i]}\n")

    with open(os.path.join(a.out, "export.sh"), "w") as f:
        f.write(f"""#!/usr/bin/env bash
# Compile {stem}.pt for the Raspberry Pi AI HAT+ (Hailo-8L). Run on Linux x86_64 (Ubuntu 22.04 works).
#   ./export.sh /path/to/hailo_dataflow_compiler-3.33*-linux_x86_64.whl
set -euo pipefail
cd "$(dirname "$0")"
WHL="${{1:?pass the Hailo Dataflow Compiler wheel (Hailo Developer Zone, DFC 3.33 for HailoRT 4.23)}}"
[ "$(uname -m)" = x86_64 ] || {{ echo "needs Linux x86_64 (this is $(uname -m))"; exit 1; }}
PY="${{PYTHON:-python3}}"
"$PY" -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install "ultralytics=={ul_version}"     # same version as the Pi, so its Hailo backend reads the metadata
pip install "$WHL"
sed -i "s|^path: .*|path: $PWD/calib|" calib.yaml   # Ultralytics resolves a relative path against its datasets dir
yolo export model={os.path.basename(a.weights)} format=hailo name=hailo8l imgsz=640 data=calib.yaml
tar czf {stem}_hailo_model.tar.gz {stem}_hailo_model
echo
echo "Done: copy {stem}_hailo_model.tar.gz to the Pi and unpack it in ~/stop-sign-eval/"
""")
    os.chmod(os.path.join(a.out, "export.sh"), 0o755)

    total = sum(counts.values())
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(f"""# Compile {stem} for the AI HAT+ (Hailo-8L)

Built on the Pi by `stop-sign-eval/hailo/make_kit.py` on {time.strftime('%Y-%m-%d')}.
Calibration images: {total} ({', '.join(f'{k} {v}' for k, v in counts.items())}).

Needs a **Linux x86_64** machine (Ubuntu 22.04; WSL2 may work but isn't tested), Python 3.10 recommended,
and about an hour on CPU (a supported NVIDIA GPU makes the optimization step much faster).

1. Register (free) at the Hailo Developer Zone (hailo.ai/developer-zone) and download the **Dataflow Compiler 3.33**
   wheel for Linux x86_64. Ultralytics validated Hailo export with DFC 3.33 + HailoRT 4.23, and the Pi runs HailoRT 4.23.
2. Copy this kit to the PC, unpack it, and run:

       ./export.sh /path/to/hailo_dataflow_compiler-3.33*.whl

3. Copy `{stem}_hailo_model.tar.gz` back to the Pi and unpack it in `~/stop-sign-eval/`
   (you get `~/stop-sign-eval/{stem}_hailo_model/` with the `.hef` and `metadata.yaml`; keep them together).
4. On the Pi: `cd ~/testing/demo && python3 run_demo.py --vision-model {stem}`.

Ultralytics' notes for YOLO26 on Hailo-8L: about 93% of the PyTorch mAP50 is kept after INT8 quantization, ~22 ms per
image, and the confidence threshold should be about 0.05 lower than with PyTorch.
""")
    tar_path = a.out + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as t:
        t.add(a.out, arcname=os.path.basename(a.out))
    print(f"\nkit: {tar_path} ({os.path.getsize(tar_path) / 1e6:.0f} MB)")
    for k, v in counts.items():
        print(f"  {v:5d} calibration images from {k}")
    print(f"  {total:5d} total (Hailo recommends >= 1,024)")


if __name__ == "__main__":
    main()
