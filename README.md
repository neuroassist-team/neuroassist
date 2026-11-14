# NeuroAssist Streamlit app

## 1. Get the project
```bash
git clone https://github.com/<org>/neuroassist.git
cd neuroassist/app
```

## 2. Create a virtual environment and install
Python 3.10 to 3.12 recommended.
```bash
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\Activate.ps1         # Windows PowerShell
python -m pip install --upgrade pip
# No NVIDIA GPU? install the small CPU build of PyTorch first:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. Add the trained weights
Weights are not in git. Put these two files in `app/checkpoints/`:
`teacher_best.pt` and `student_kd_best.pt`
(from the GitHub Release or the shared Kaggle dataset). Optional: `config.json`, `results_table.csv`.
Or run the app and use the sidebar "Download weights from a GitHub Release", or set
`NEUROASSIST_WEIGHTS_URL=https://github.com/<org>/<repo>/releases/download/<tag>`.

## 4. Run
```bash
streamlit run app.py
```
The sidebar shows "Models loaded" with each model's recorded validation Dice (baseline about 0.833, lightweight about 0.79).
If it does not, the weights were not found.

## Cases
- **Upload 4 NIfTI sequences** (BraTS space 240x240x155, names ending _flair/_t1/_t1ce/_t2)
- **BraTS case folder on disk** (e.g. `.../BraTS20_Training_001`)
- **Cached held-out cases** (`cases/*_img.npy`, from `neuroassist_bundle.zip`)
- **Synthetic phantom** (UI preview only, no AI)

Research prototype. Not a medical device. Not for clinical use.
