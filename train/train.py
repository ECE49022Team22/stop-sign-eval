"""Fine-tune YOLO26n on the prepared dataset, export it for the Pi, and compare it
against the pretrained model with the same tests as REPORT.md.

Usage:
  python train/train.py --data /tmp/work/finetune/data.yaml --out /kaggle/working/stop_sign_finetune

Writes to --out:
  stop_sign_yolo26n_ft.pt        the fine-tuned model (keep this!)
  stop_sign_yolo26n_ft_ncnn_model/  same model exported for the Raspberry Pi (NCNN)
  training/                      Ultralytics training curves, confusion matrix, val examples
  eval/                          the stop-sign-eval comparison (charts, galleries, summary.md)
  FINETUNE_REPORT.md             before/after table in plain language
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FT_NAME = "YOLO26n fine-tuned"


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml from prepare_dataset.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo26n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15, help="stop early if val doesn't improve for this many epochs")
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu for CPU (default: auto)")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    from ultralytics import YOLO
    import torch
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    print(f"Training on {'GPU ' + torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")

    # 1. train ------------------------------------------------------------
    model = YOLO(os.path.join(REPO, args.model) if os.path.exists(os.path.join(REPO, args.model)) else args.model)
    runs = os.path.join(args.out, "_runs")
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                device=device, project=runs, name="finetune", exist_ok=True, seed=0, plots=True,
                workers=4, cache=False)
    run_dir = os.path.join(runs, "finetune")
    best = os.path.join(args.out, "stop_sign_yolo26n_ft.pt")
    shutil.copy(os.path.join(run_dir, "weights", "best.pt"), best)
    # keep the useful training plots, drop the bulky checkpoints
    shutil.copytree(run_dir, os.path.join(args.out, "training"), dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("weights"))
    shutil.rmtree(runs, ignore_errors=True)

    # 2. validation metrics per class --------------------------------------
    ft = YOLO(best)
    m = ft.val(data=args.data, imgsz=args.imgsz, device=device, plots=False, verbose=False,
               project=os.path.join(args.out, "_val"), name="v", exist_ok=True)
    shutil.rmtree(os.path.join(args.out, "_val"), ignore_errors=True)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[ft.names[int(c)]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap}
    json.dump({"per_class": per_class, "mAP50": m.box.map50, "mAP50-95": m.box.map},
              open(os.path.join(args.out, "val_metrics.json"), "w"), indent=1)

    # 3. export for the Pi --------------------------------------------------
    try:
        exported = os.path.abspath(ft.export(format="ncnn", imgsz=args.imgsz))
        dst = os.path.abspath(os.path.join(args.out, "stop_sign_yolo26n_ft_ncnn_model"))
        if exported != dst:                      # Ultralytics normally writes it right next to the .pt
            shutil.rmtree(dst, ignore_errors=True)
            shutil.move(exported, dst)
        print("NCNN model:", dst)
    except Exception as e:
        print(f"WARNING: NCNN export failed ({e}); the .pt file can be exported later on any PC.")

    # 4. same tests as REPORT.md, pretrained vs fine-tuned ---------------------
    eval_dir = os.path.join(args.out, "eval")
    if not args.skip_eval:
        if not os.path.exists(os.path.join(REPO, "data", "real", "labels.json")):
            run([sys.executable, "get_data.py"])
        if not os.path.exists(os.path.join(REPO, "data", "synthetic", "labels.json")):
            run([sys.executable, "make_synthetic.py"])
        run([sys.executable, "evaluate.py", "--results", eval_dir, "--weights", f"{best}={FT_NAME}"])
        write_report(args, eval_dir, per_class, time.time() - t0)


def write_report(args, eval_dir, per_class, seconds):
    S = json.load(open(os.path.join(eval_dir, "summary.json")))["scores"]
    data_summary = {}
    ds_json = os.path.join(os.path.dirname(args.data), "dataset_summary.json")
    if os.path.exists(ds_json):
        data_summary = json.load(open(ds_json))
    pre, ft = S["yolo26n"], S["yolo_ft1"]
    pct = lambda v: f"{100 * v:.0f}%"

    def smallest(s):
        best = None
        for w in sorted(map(int, s), reverse=True):
            if s[str(w)] >= 0.9:
                best = w
            else:
                break
        return f"{best} px" if best else "never"

    rows = [
        ("Stop signs found, 97 real photos", pct(pre["real"]["detection_rate"]), pct(ft["real"]["detection_rate"]), "higher"),
        ("False alarms, 100 real photos without a sign", pct(pre["real"]["false_alarm_rate"]), pct(ft["real"]["false_alarm_rate"]), "lower"),
        ("Smallest sign found ≥ 90% of the time", smallest(pre["size_sweep"]), smallest(ft["size_sweep"]), "smaller"),
    ]
    for kind, label in (("no_entry_disc", '"Do not enter" look-alike called STOP'),
                        ("red_disc", "Plain red disc called STOP"), ("red_triangle", "Red triangle called STOP")):
        rows.append((label + " (conf ≥ 0.25)", pct(pre["hard_negative"][kind]), pct(ft["hard_negative"][kind]), "lower"))
        rows.append((label + " (conf ≥ 0.5)", pct(pre["hard_negative_conf0.5"][kind]), pct(ft["hard_negative_conf0.5"][kind]), "lower"))
    for cond in ("angled", "dark", "motion_blur", "occluded"):
        rows.append((f"Found: {cond.replace('_', ' ')}, 40 px sign", pct(pre["condition"][f"{cond}@40"]),
                     pct(ft["condition"][f"{cond}@40"]), "higher"))

    L = [f"# Fine-tuning result: {FT_NAME}", "",
         f"Run finished in {seconds / 60:.0f} min. Model: `stop_sign_yolo26n_ft.pt` "
         f"({os.path.getsize(os.path.join(args.out, 'stop_sign_yolo26n_ft.pt')) / 1e6:.1f} MB).", "",
         "## Before vs after (same tests as REPORT.md)", "",
         "| Test | Pretrained YOLO26n | Fine-tuned | Better is |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | **{c}** | {d} |" for a, b, c, d in rows]
    L += ["", "## Validation on held-out training-set images", "",
          "| Class | Precision | Recall | mAP50 |", "|---|---|---|---|"]
    L += [f"| {c} | {v['precision']:.2f} | {v['recall']:.2f} | {v['mAP50']:.2f} |" for c, v in per_class.items()]
    if data_summary:
        st = data_summary.get("stats", {})
        L += ["", "## Training data", "",
              f"Classes: {', '.join(data_summary.get('classes', []))}. "
              f"Train images: {st.get('train images', 0)} ({st.get('train positive', 0)} with a sign), "
              f"val images: {st.get('val images', 0)}. "
              f"Stop-sign boxes: {st.get('boxes stop_sign', 0)}, do-not-enter boxes: {st.get('boxes do_not_enter', 0)}.", ""]
        L += [f"- {k[len('source '):]}: {v} images" for k, v in st.items() if k.startswith("source ")]
    L += ["", "Full charts and every mistake: `eval/` (same layout as `results/` in the repo). "
          "Training curves and confusion matrix: `training/`.", ""]
    open(os.path.join(args.out, "FINETUNE_REPORT.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
