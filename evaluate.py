"""Run every detector on both test sets, score them, and write results/.

Usage:  python evaluate.py            (after get_data.py and make_synthetic.py)
        python evaluate.py --rescore  (re-score cached predictions without re-running models)
        python evaluate.py --weights runs/best.pt="YOLO26n fine-tuned"   (add your own model(s);
                                       with --rescore only the new model is run)

Outputs (results/):
  predictions.json        every detection from every detector on every image
  summary.json / .md      all the numbers
  chart_*.png             charts used in REPORT.md
  gallery_*.jpg           the images each detector got wrong on the real photos
"""
import argparse
import json
import math
import os
import statistics
import time

import cv2
import numpy as np

from detectors import YoloDetector, rules_detect, to_robot_resolution

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")

YOLO_CONF = 0.25              # Ultralytics' default confidence threshold
IOU_HIT = 0.5                 # a synthetic sign counts as found if a box overlaps it this much
IOU_LURE = 0.3                # a look-alike counts as a false alarm if a box overlaps it this much

# Camera assumption used to turn sign width in pixels into distance (S4 asks for >= 5 m).
CAM_WIDTH_PX, CAM_HFOV_DEG = 640, 70.0     # typical USB webcam; change to your camera's spec
FOCAL_PX = CAM_WIDTH_PX / (2 * math.tan(math.radians(CAM_HFOV_DEG / 2)))
SIGN_WIDTHS_M = {'12" (0.30 m) mini sign': 0.305, '18" (0.46 m)': 0.457,
                 '24" (0.61 m)': 0.610, '30" (0.76 m) standard road sign': 0.762}

DETECTORS = {   # key: (display name, how to build it)
    "yolo26n": ("YOLO26n (pretrained)", lambda: YoloDetector("yolo26n.pt")),
    "yolo11n": ("YOLO11n (pretrained)", lambda: YoloDetector("yolo11n.pt")),
    "rules": ("Rules (team code, as-is)", lambda: rules_detect),
    "rules_a150": ("Rules, min area lowered to 150", lambda: (lambda im: rules_detect(im, min_area=150))),
}
SERIES_COLOR = {"yolo26n": "#2a78d6", "rules": "#eb6834", "yolo11n": "#1baf7a", "rules_a150": "#eda100"}
FT_COLORS = ["#e87ba4", "#4a3aa7", "#008300"]      # fine-tuned models added with --weights
BASE_KEYS = ["yolo26n", "yolo11n", "rules"]         # the three shown in every chart


def add_weights(specs):
    """Register extra YOLO models given as PATH or PATH=Display name."""
    for i, spec in enumerate(specs or []):
        path, _, name = spec.partition("=")
        key = f"yolo_ft{i + 1}"
        DETECTORS[key] = (name or os.path.basename(path), (lambda p=path: YoloDetector(p)))
        SERIES_COLOR[key] = FT_COLORS[i % len(FT_COLORS)]


def chart_keys():
    return BASE_KEYS + [k for k in DETECTORS if k.startswith("yolo_ft")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------- running
def load_sets():
    sets = {}
    for name in ("real", "synthetic"):
        labels = json.load(open(os.path.join(DATA, name, "labels.json")))
        for l in labels:
            l["path"] = os.path.join(DATA, name, "images", l["file"])
        sets[name] = labels
    return sets


def run_all(sets, keys=None):
    preds, timing = {}, {}
    for key, (label, build) in DETECTORS.items():
        if keys is not None and key not in keys:
            continue
        det = build()
        preds[key], times = {}, []
        for set_name, labels in sets.items():
            for l in labels:
                img = to_robot_resolution(cv2.imread(l["path"]))
                t = time.perf_counter()
                out = det(img)
                times.append(1000 * (time.perf_counter() - t))
                preds[key][f"{set_name}/{l['file']}"] = out
        timing[key] = statistics.median(times[5:])      # skip warm-up
        print(f"{label:34s} median {timing[key]:6.1f} ms/frame")
    return preds, timing


# ---------------------------------------------------------------- scoring
def iou(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def kept(dets, key, thr=None):
    if key.startswith("yolo"):
        thr = YOLO_CONF if thr is None else thr
        return [d for d in dets if d[4] >= thr]
    return dets


def rate(hits):
    return sum(hits) / len(hits) if hits else float("nan")


def wilson(k, n, z=1.96):
    """95% confidence interval for a rate k/n (Wilson score)."""
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score(sets, preds):
    S = {}
    for key in DETECTORS:
        P = preds[key]
        s = {}
        real = sets["real"]
        pos = [bool(kept(P[f"real/{l['file']}"], key)) for l in real if l["has_sign"]]
        neg = [bool(kept(P[f"real/{l['file']}"], key)) for l in real if not l["has_sign"]]
        s["real"] = {"detection_rate": rate(pos), "n_pos": len(pos), "hits": sum(pos),
                     "detection_ci95": wilson(sum(pos), len(pos)),
                     "false_alarm_rate": rate(neg), "n_neg": len(neg), "false_alarms": sum(neg),
                     "false_alarm_ci95": wilson(sum(neg), len(neg)),
                     "accuracy": (sum(pos) + len(neg) - sum(neg)) / (len(pos) + len(neg))}
        if key.startswith("yolo"):
            s["real_threshold_sweep"] = {}
            for thr in (0.1, 0.25, 0.4, 0.5, 0.6, 0.7):
                p = [bool(kept(P[f"real/{l['file']}"], key, thr)) for l in real if l["has_sign"]]
                n = [bool(kept(P[f"real/{l['file']}"], key, thr)) for l in real if not l["has_sign"]]
                s["real_threshold_sweep"][thr] = {"detection_rate": rate(p), "false_alarm_rate": rate(n)}

        syn = sets["synthetic"]

        def found(l):
            return any(iou(d, l["box"]) >= IOU_HIT for d in kept(P[f"synthetic/{l['file']}"], key))

        s["size_sweep"] = {}
        for l in syn:
            if l["group"] == "size_sweep":
                s["size_sweep"].setdefault(l["size"], []).append(found(l))
        s["size_sweep"] = {k: rate(v) for k, v in sorted(s["size_sweep"].items())}

        s["condition"] = {}
        for l in syn:
            if l["group"] == "condition":
                s["condition"].setdefault(f"{l['condition']}@{l['size']}", []).append(found(l))
        for l in syn:   # frontal reference at the same two sizes
            if l["group"] == "size_sweep" and l["size"] in (40, 80):
                s["condition"].setdefault(f"frontal@{l['size']}", []).append(found(l))
        s["condition"] = {k: rate(v) for k, v in s["condition"].items()}

        s["hard_negative"] = {}
        for l in syn:
            if l["group"] == "hard_negative":
                fa = any(iou(d, l["lure_box"]) >= IOU_LURE for d in kept(P[f"synthetic/{l['file']}"], key))
                s["hard_negative"].setdefault(l["condition"], []).append(fa)
        s["hard_negative"] = {k: rate(v) for k, v in s["hard_negative"].items()}
        if key.startswith("yolo"):   # same look-alikes with a stricter confidence threshold
            s["hard_negative_conf0.5"] = {}
            for l in syn:
                if l["group"] == "hard_negative":
                    fa = any(iou(d, l["lure_box"]) >= IOU_LURE
                             for d in kept(P[f"synthetic/{l['file']}"], key, 0.5))
                    s["hard_negative_conf0.5"].setdefault(l["condition"], []).append(fa)
            s["hard_negative_conf0.5"] = {k: rate(v) for k, v in s["hard_negative_conf0.5"].items()}
        S[key] = s
    return S


def min_reliable_width(size_rates, target=0.9):
    """Smallest sign width (px) from which detection stays >= target at every larger size."""
    best = None
    for w in sorted(size_rates, reverse=True):
        if size_rates[w] >= target:
            best = w
        else:
            break
    return best


# ---------------------------------------------------------------- charts
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_real(S, path):
    import matplotlib.pyplot as plt
    keys = chart_keys()
    fig, axes = plt.subplots(1, 2, figsize=(10, 1.6 + 0.7 * len(keys)), facecolor=SURFACE)
    for ax, metric, title in ((axes[0], "detection_rate", "Stop signs found (97 photos)  ↑ better"),
                              (axes[1], "false_alarm_rate", "False alarms (100 photos, no sign)  ↓ better")):
        style(ax)
        vals = [100 * S[k]["real"][metric] for k in keys]
        ci = [S[k]["real"]["detection_ci95" if metric == "detection_rate" else "false_alarm_ci95"] for k in keys]
        y = np.arange(len(keys))[::-1]
        ax.barh(y, vals, height=0.55, color=[SERIES_COLOR[k] for k in keys])
        for yi, v, (lo, hi) in zip(y, vals, ci):
            ax.plot([100 * lo, 100 * hi], [yi, yi], color=INK2, linewidth=1)
            ax.text(min(v, 100) + 2, yi + 0.3, f"{v:.0f}%", va="center", color=INK, fontsize=11)
        ax.set_yticks(y, [DETECTORS[k][0] for k in keys], color=INK, fontsize=10)
        ax.set_xlim(0, 112)
        ax.set_xlabel("% of images  (thin line = 95% confidence range)", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    for ax in axes[1:]:
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_size(S, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    style(ax)
    keys = ["rules_a150", "rules", "yolo11n", "yolo26n"] + chart_keys()[3:]     # drawn back to front
    for k in keys:
        xs = list(S[k]["size_sweep"])
        ys = [100 * S[k]["size_sweep"][x] for x in xs]
        ax.plot(xs, ys, color=SERIES_COLOR[k], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                linestyle="--" if k == "rules_a150" else "-", label=DETECTORS[k][0])
    ax.axhline(90, color=INK2, linewidth=1, linestyle=":")
    ax.text(129, 91.5, "S4 target: 90%", color=INK2, fontsize=9, ha="right")
    for name, W in (('12" sign', 0.305), ('30" sign', 0.762)):
        px = FOCAL_PX * W / 5.0
        ax.axvline(px, color=GRID, linewidth=1.2)
        ax.text(px + 1, 3, f"{name}\nat 5 m\n≈{px:.0f} px", color=INK2, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ticks = [12, 16, 20, 24, 32, 40, 48, 64, 80, 100, 128]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(11, 140)
    ax.set_ylim(0, 104)
    ax.set_xlabel("Sign width in a 640×480 frame (pixels)   ←  farther away     closer  →", color=INK2)
    ax.set_ylabel("Signs found (%)", color=INK2)
    ax.set_title("Detection rate vs. sign size (40 pasted signs per size)", color=INK, fontsize=12, loc="left")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1, 0.58), labelcolor=INK)
    ax.text(130, 18, "(lines overlap where they\nshare a value, e.g. both YOLOs\nat 100%, both rule versions ≥ 40 px)",
            color=INK2, fontsize=8, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_conditions(S, path):
    import matplotlib.pyplot as plt
    conds = ["frontal", "angled", "rotated", "dark", "overexposed", "motion_blur", "occluded"]
    keys = chart_keys()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, sharey=True)
    for ax, size in zip(axes, (40, 80)):
        style(ax)
        x = np.arange(len(conds))
        w = 0.8 / len(keys) - 0.02
        for i, k in enumerate(keys):
            v = [100 * S[k]["condition"][f"{c}@{size}"] for c in conds]
            xs = x + (i - (len(keys) - 1) / 2) * (w + 0.02)
            ax.bar(xs, v, width=w, color=SERIES_COLOR[k], label=DETECTORS[k][0] if size == 40 else None)
            for xi, vi in zip(xs, v):
                if vi < 5:
                    ax.text(xi, 1.5, "0", ha="center", color=SERIES_COLOR[k], fontsize=9, fontweight="bold")
        ax.set_xticks(x, [c.replace("_", "\n") for c in conds], color=INK, fontsize=9)
        ax.set_title(f"Sign {size} px wide", color=INK, fontsize=11, loc="left")
        ax.axhline(90, color=INK2, linewidth=1, linestyle=":")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Signs found (%)", color=INK2)
    fig.suptitle("Detection rate by condition (40 images per bar)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK, ncol=min(3, len(keys)))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- galleries
def gallery(sets, preds, key, path, title):
    real = sets["real"]
    missed = [l for l in real if l["has_sign"] and not kept(preds[key][f"real/{l['file']}"], key)]
    false = [l for l in real if not l["has_sign"] and kept(preds[key][f"real/{l['file']}"], key)]
    tiles = []
    for l, kind in [(l, "MISSED") for l in missed] + [(l, "FALSE ALARM") for l in false]:
        img = to_robot_resolution(cv2.imread(l["path"]))
        for d in kept(preds[key][f"real/{l['file']}"], key):
            x, y, w, h = d[:4]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        img = cv2.resize(img, (320, round(img.shape[0] * 320 / img.shape[1])))
        img = cv2.copyMakeBorder(img, 0, max(0, 240 - img.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:240]
        col = (0, 140, 255) if kind == "MISSED" else (0, 0, 220)
        cv2.rectangle(img, (0, 0), (320, 22), col, -1)
        cv2.putText(img, f"{kind}  {l['file']}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return 0, 0
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 40))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    head = np.full((40, grid.shape[1], 3), 30, np.uint8)
    cv2.putText(head, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(missed), len(false)


def gallery_lures(sets, preds, key, path, n_per_type=4):
    """Examples of red look-alikes the detector called a stop sign (crop around the lure)."""
    tiles = []
    for kind in ("no_entry_disc", "red_disc", "red_triangle"):
        hits = []
        for l in sets["synthetic"]:
            if l["group"] == "hard_negative" and l["condition"] == kind:
                ds = [d for d in kept(preds[key][f"synthetic/{l['file']}"], key) if iou(d, l["lure_box"]) >= IOU_LURE]
                hits.append((l, ds))
        flagged = [h for h in hits if h[1]][:n_per_type]
        clean = [h for h in hits if not h[1]][:max(0, n_per_type - len(flagged))]
        for l, ds in flagged + clean:
            img = cv2.imread(l["path"])
            x, y, w, h = l["lure_box"]
            p = int(1.5 * w)
            crop = img[max(0, y - p):y + h + p, max(0, x - p):x + w + p].copy()
            ox, oy = max(0, x - p), max(0, y - p)
            for d in ds:
                cv2.rectangle(crop, (d[0] - ox, d[1] - oy), (d[0] + d[2] - ox, d[1] + d[3] - oy), (0, 0, 255), 2)
            crop = cv2.resize(crop, (220, 220))
            txt = f"called STOP ({max(d[4] for d in ds):.2f})" if ds and key.startswith("yolo") else ("called STOP" if ds else "ignored (correct)")
            cv2.rectangle(crop, (0, 0), (220, 20), (0, 0, 200) if ds else (0, 140, 0), -1)
            cv2.putText(crop, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
    grid = np.vstack([np.hstack(tiles[i:i + n_per_type]) for i in range(0, len(tiles), n_per_type)])
    head = np.full((34, grid.shape[1], 3), 30, np.uint8)
    cv2.putText(head, f"{DETECTORS[key][0]} on red look-alikes", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- report table
def write_summary_md(S, timing, path):
    L = []
    pct = lambda v: f"{100 * v:.0f}%"
    L.append("| Detector | Signs found (real) | False alarms (real) | Accuracy | Smallest reliable sign (≥90%) | Time/frame* |")
    L.append("|---|---|---|---|---|---|")
    for k, (name, _) in DETECTORS.items():
        r = S[k]["real"]
        m = min_reliable_width(S[k]["size_sweep"])
        L.append(f"| {name} | {r['hits']}/{r['n_pos']} = {pct(r['detection_rate'])} | "
                 f"{r['false_alarms']}/{r['n_neg']} = {pct(r['false_alarm_rate'])} | {pct(r['accuracy'])} | "
                 f"{f'{m} px' if m else 'never'} | {timing[k]:.0f} ms |")
    L.append("")
    L.append("Sign width → distance (camera assumed 640 px wide, 70° horizontal FOV; f ≈ %.0f px):" % FOCAL_PX)
    L.append("")
    L.append("| Sign | width at 5 m | 8 m | 3 m |")
    L.append("|---|---|---|---|")
    for name, W in SIGN_WIDTHS_M.items():
        L.append(f"| {name} | {FOCAL_PX * W / 5:.0f} px | {FOCAL_PX * W / 8:.0f} px | {FOCAL_PX * W / 3:.0f} px |")
    L.append("")
    L.append("Hard negatives (red look-alikes, 60 each) — share that were called a stop sign:")
    L.append("")
    L.append("| Detector | " + " | ".join(S["rules"]["hard_negative"]) + " |")
    L.append("|---|" + "---|" * len(S["rules"]["hard_negative"]))
    for k, (name, _) in DETECTORS.items():
        L.append(f"| {name} | " + " | ".join(pct(v) for v in S[k]["hard_negative"].values()) + " |")
        if k.startswith("yolo"):
            L.append(f"| {name}, confidence ≥ 0.5 | " + " | ".join(pct(v) for v in S[k]["hard_negative_conf0.5"].values()) + " |")
    L.append("")
    L.append("Conditions (detection rate, 40 images each):")
    L.append("")
    conds = ["frontal", "angled", "rotated", "dark", "overexposed", "motion_blur", "occluded"]
    for size in (40, 80):
        L.append(f"| {size} px sign | " + " | ".join(conds) + " |")
        L.append("|---|" + "---|" * len(conds))
        for k, (name, _) in DETECTORS.items():
            L.append(f"| {name} | " + " | ".join(pct(S[k]["condition"][f"{c}@{size}"]) for c in conds) + " |")
        L.append("")
    L.append("YOLO confidence threshold sweep (real photos):")
    L.append("")
    L.append("| Detector | threshold | found | false alarms |")
    L.append("|---|---|---|---|")
    for k in [k for k in DETECTORS if k.startswith("yolo")]:
        for thr, v in S[k]["real_threshold_sweep"].items():
            L.append(f"| {DETECTORS[k][0]} | {thr} | {pct(v['detection_rate'])} | {pct(v['false_alarm_rate'])} |")
    L.append("")
    L.append(f"*Median time per 640-px frame on the machine that ran this evaluation ({DEVICE}). "
             "Not a Raspberry Pi number.")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def describe_device():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}, PyTorch"
    except Exception:
        pass
    import platform
    return f"CPU: {os.cpu_count()} cores, {platform.machine()}, PyTorch"


DEVICE = describe_device()


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="reuse results/predictions.json")
    ap.add_argument("--weights", action="append", metavar='PATH[="Name"]',
                    help="extra YOLO model to compare (repeatable), e.g. best.pt=\"YOLO26n fine-tuned\"")
    ap.add_argument("--results", default=RES, help="output folder (default: results/)")
    args = ap.parse_args()
    add_weights(args.weights)
    RES = args.results
    os.makedirs(RES, exist_ok=True)
    sets = load_sets()
    cache = os.path.join(RES, "predictions.json")
    preds, timing = {}, {}
    if args.rescore and os.path.exists(cache):
        c = json.load(open(cache))
        preds, timing = c["predictions"], c["timing_ms"]
    missing = [k for k in DETECTORS if k not in preds]
    if missing:
        p, t = run_all(sets, missing)
        preds.update(p)
        timing.update(t)
        json.dump({"predictions": preds, "timing_ms": timing}, open(cache, "w"))
    S = score(sets, preds)
    json.dump({"scores": S, "timing_ms": timing, "focal_px": FOCAL_PX}, open(os.path.join(RES, "summary.json"), "w"),
              indent=1, default=str)
    write_summary_md(S, timing, os.path.join(RES, "summary.md"))
    chart_real(S, os.path.join(RES, "chart_real_photos.png"))
    chart_size(S, os.path.join(RES, "chart_distance.png"))
    chart_conditions(S, os.path.join(RES, "chart_conditions.png"))
    for k in ["rules", "yolo26n"] + chart_keys()[3:]:
        m, f = gallery(sets, preds, k, os.path.join(RES, f"gallery_mistakes_{k}.jpg"),
                       f"{DETECTORS[k][0]}: every mistake on the real photos")
        print(f"{k}: {m} missed, {f} false alarms on real photos")
        gallery_lures(sets, preds, k, os.path.join(RES, f"gallery_lookalikes_{k}.jpg"))
    print(open(os.path.join(RES, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
