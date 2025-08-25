# medscan — Hackathon-ready Anomaly Detector

Fast, dirty, and useful: medscan is a lightweight Gradio web UI that loads best-effort PyTorch checkpoints for CT / X-ray / Ultrasound and shows live anomaly detection with a left-side Anomaly Log and a masked preview.

This README is written in a hackathon style — short, actionable, and focused on getting you running fast.

## What you'll find
- `backend/app.py` — Gradio app & robust model loader
- `backend/repair_checkpoints.py` — helper to remap/checkpoint-fix
- `backend/models/` — trained checkpoints (if present). Fixed copies use `*_fixed.pth`
- `result-images/` — demo output + example masked previews (look here first)
- `userImage/` — example baseline images used by the masking routine

## Quick demo (2-minute setup)
Open a PowerShell terminal in the repository root and run:

```powershell
# create virtualenv (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
# launch the web UI
python backend\app.py
```

After launch, open the printed URL (http://localhost:<port>) and upload an image. The left column shows the Anomaly Log and a masked preview (from `userImage/` baseline if present). The `result-images/` folder contains example outputs.

## Notes about models
- Model files (.pth/.pt) are expected in `backend/models/`. The loader prefers files named like `ct_model.pth`, `cnn_chestxray.pth`, `ultrasound_model.pth`.
- If checkpoint keys don't match the app's simple architecture, run `python backend/repair_checkpoints.py` to create `<name>_fixed.pth` copies — the app prefers `_fixed.pth` when present.
- Large model files should be stored with Git LFS or hosted outside the repo (recommended).

## How masking works (quick)
- The app compares the uploaded image to a baseline (from `userImage/`) using a simple per-pixel abs-diff on a 128x128 tensor.
- Differences above a threshold produce a mask that overlays red where changes occur — example previews are in `result-images/`.

## Troubleshooting
- If you see Git warnings about line endings or the virtualenv being tracked, ensure `.gitignore` includes `.venv/` and `venv/` and then run:

```powershell
# stop tracking an already-tracked venv without deleting files
git rm -r --cached .venv
git add .gitignore
git commit -m "Ignore virtualenv"
```

- If `backend/models/` isn't uploaded because `*.pth` is ignored, either whitelist the folder in `.gitignore` or use `git lfs` to track large files.

## Screenshots / examples
Two example images are included in `result-images/` for quick reference:

- `result-images/example_comparison.png` — left: baseline, center: new scan, right: detected changes mask (thresholded diff)
- `result-images/ui_screenshot.png` — screenshot of the running Gradio UI showing modality, prediction, and confidence



### Detection Output
Baseline, New Scan, and Detected Changes:

![Comparison Example](result-images/example_comparison.png)

### Web UI Screenshot

![UI Screenshot](result-images/ui_screenshot.png)

