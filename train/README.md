# Fine-tuning YOLO26n on Kaggle (runs in the background, laptop can be off)

The goal is a model that finds stop signs at least as well as the pretrained YOLO26n but stops
calling red round signs ("do not enter") stop signs. The notebook trains two classes,
`stop_sign` and `do_not_enter`, whenever the data has enough do-not-enter examples, so the
model learns the difference instead of guessing.

## One-time setup

1. **Phone-verify your Kaggle account** (Settings → Phone verification). Kaggle requires this
   before it lets you use GPUs or internet in a notebook.
2. **Make sure the latest version of this repo is pushed to GitHub.** The notebook downloads the
   scripts from `github.com/ECE49022Team22/stop-sign-eval` when it starts.

## Each training run

1. On kaggle.com: **Create → New Notebook → File → Import Notebook**, then upload
   `train/kaggle_finetune.ipynb` from this folder.
2. In the right-hand panel, under **Session options**:
   - **Accelerator: GPU T4 x2.** Avoid P100: recent PyTorch builds may no longer support it.
   - **Internet: On.**
3. **+ Add Input → Datasets** and attach sign data. Anything in LISA, Pascal VOC or YOLO format
   works, and you can attach several. Good candidates (search on Kaggle):
   - **LISA traffic signs**: US signs from car cameras, with both `stop` and `doNotEnter`
     labelled. This is the most useful one, because it's what teaches the do-not-enter difference.
   - **"Road Sign Detection" (andrewmvd)**: 877 images labelled `stop`, `speedlimit`,
     `crosswalk` and `trafficlight`. The non-stop signs become useful "not a stop sign" examples.
   - **A Roboflow export (YOLO format)** of any stop-sign or do-not-enter dataset. Check its
     license on Roboflow first.
4. **Save Version → Save & Run All (Commit) → Save.** Close the tab or laptop; the run continues
   on Kaggle's servers. A typical run takes 30–90 minutes. Kaggle's free tier allows up to 9 hours
   per run and ~30 GPU hours per week.
5. When it finishes, open the notebook → **Output** tab of that version. Download
   `stop_sign_finetune_results.zip`, or the individual files:

| File | What it is |
|---|---|
| `FINETUNE_REPORT.md` | Before/after table: pretrained vs fine-tuned on the same tests as `REPORT.md` |
| `stop_sign_yolo26n_ft.pt` | The fine-tuned model (~5 MB). **Keep this.** |
| `stop_sign_yolo26n_ft_ncnn_model/` | The same model exported for the Raspberry Pi |
| `eval/` | Charts and mistake galleries, same layout as `results/` |
| `training/` | Training curves, confusion matrix, example predictions |
| `dataset_summary.json` | What data was used, and how each source label was mapped |

If something goes wrong, the notebook's **Log** shows each step. The "Class mapping" table
printed by `prepare_dataset.py` shows exactly which labels it treated as stop / do-not-enter.

## Where to keep the model

Attach `stop_sign_yolo26n_ft.pt` (and a zip of the NCNN folder) to a **GitHub Release** of this
repo, e.g. `v1-finetune`. Keep the model out of git commits: every retrain would add another
~5 MB to the history forever.

## Using the fine-tuned model locally

```bat
python compare_live.py --weights stop_sign_yolo26n_ft.pt
python evaluate.py --rescore --weights stop_sign_yolo26n_ft.pt="YOLO26n fine-tuned"
```

## Running without Kaggle

The same scripts run anywhere with Python and, ideally, a GPU:

```bat
python get_data.py && python make_synthetic.py
python train\fetch_open_images.py --out work\open_images --max-images 800
python train\prepare_dataset.py --inputs path\to\datasets work\open_images --out work\finetune
python train\train.py --data work\finetune\data.yaml --out finetune_results
```
