# Stop sign detector comparison: team rules vs pretrained YOLO

Everything needed to reproduce the comparison in `REPORT.md`, and to try both detectors
on your own photos or webcam. No Raspberry Pi needed; any laptop works (CPU only).

## Files

| File | What it does |
|---|---|
| `stop_sign_detector.py` | The team's rule-based detector, copied unchanged from `ECE49022Team22/testing` |
| `detectors.py` | Wraps both detectors so they get the same 640-px frame and return the same box format |
| `get_data.py` | Downloads 197 labelled street photos (97 with a stop sign, 100 without) |
| `make_synthetic.py` | Builds 1,180 controlled test images: sign-size (distance) sweep, angle/dark/blur/occlusion, red look-alikes |
| `evaluate.py` | Runs every detector on everything, writes numbers, charts and mistake galleries to `results/` |
| `compare_live.py` | Rules (left) vs YOLO (right) live on your webcam, an image, a folder or a video |
| `results/` | Output from the run described in `REPORT.md` |

## Setup on Windows

Tip: keep the virtual environment **outside OneDrive**, or OneDrive will try to sync thousands of library files.

```bat
cd "C:\Users\jyasu\OneDrive - purdue.edu\Class\Fall 2026\Senior Design\stop_sign_eval"
py -m venv %USERPROFILE%\venvs\stopsign
%USERPROFILE%\venvs\stopsign\Scripts\activate
pip install -r requirements.txt
```

(macOS/Linux: `python3 -m venv ~/venvs/stopsign && source ~/venvs/stopsign/bin/activate`.)

The YOLO weights (`yolo26n.pt`, ~5 MB) download automatically the first time.

## Reproduce the results

```bat
python get_data.py        :: ~140 MB of photos into data\real
python make_synthetic.py  :: 1,180 images into data\synthetic (same every time, fixed seed)
python evaluate.py        :: ~4 min on a laptop CPU; writes results\
```

`python evaluate.py --rescore` re-scores the saved predictions without re-running the models
(useful after changing a threshold such as `YOLO_CONF` or the camera FOV at the top of the file).

## Try it yourself

```bat
python compare_live.py                         :: webcam: hold up a phone showing a stop sign
python compare_live.py --source "C:\path\to\photos"
python compare_live.py --source clip.mp4 --conf 0.5
```

## Testing on your own robot footage (recommended next step)

1. Record video from the robot's camera at its real height, walking toward your actual test sign from ~10 m, plus footage of the route with no sign.
2. Pull frames (e.g. 1 per second) into `data/real/images/` and list them in `data/real/labels.json` as `{"file": "...", "has_sign": true/false}`.
3. Run `python evaluate.py`. The "real photos" numbers are now numbers for *your* camera and sign.
