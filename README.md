<div align="center">

# NeuroAssist

### Lightweight brain-tumor segmentation for interactive surgical planning

An experimental Streamlit app for exploring brain MRI segmentation, model efficiency, and planning tools.

![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-models-EE4C2C?logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-medical%20imaging-75B900)
![Status](https://img.shields.io/badge/status-research%20prototype-2dd4bf)

[Open the project report](report/index.html)

</div>

---

## What’s inside

- **Compare models:** view baseline and lightweight brain-tumor segmentations alongside the reference labels.
- **Explore a case:** upload four MRI sequences or load a BraTS case archive.
- **Inspect planning tools:** review tumor volumes, adjustable geometric margins, trajectories, 3D views, and model disagreement.
- **Try the demo:** explore the interface with a synthetic phantom even when model weights are not installed. The phantom does not use AI.
- **Read the experiments:** browse the notebooks and the [project report](report/index.html) for methods, results, and limitations.

## Quick start

### 1. Get the code

```bash
git clone https://github.com/neuroassist-team/neuroassist.git
cd neuroassist
```

### 2. Install dependencies

Python 3.10–3.12 is recommended. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the project requirements from the repository root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CPU-only PyTorch installation, install its CPU build before the remaining requirements:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app/app.py
```

## Model weights

The trained checkpoints are not included in this repository. To enable the trained models, provide:

- `teacher_best.pt`
- `student_kd_best.pt`

You can upload both files in the app’s **Model** sidebar. Optional `config.json` and `results_table.csv` files can be uploaded alongside them. Alternatively, place the checkpoints in `app/checkpoints/` or download them from a project release.

Without checkpoints, the app still launches and offers the clearly labelled synthetic phantom for a UI preview.

## Add a case

Choose one of these input options in the app:

- **MRI upload:** provide four NIfTI sequences (`.nii` or `.nii.gz`) named with `_flair`, `_t1`, `_t1ce`, and `_t2` modality suffixes.
- **BraTS case archive:** upload a `.zip` containing a case’s MRI sequences.
- **Local or cached case:** configure the app to use a BraTS case directory or cached held-out case.
- **Synthetic phantom:** use the interface preview; it does not run inference.

## Project layout

| Path | Contents |
| --- | --- |
| [`app/`](app/) | Streamlit application and interactive cover |
| [`notebooks/`](notebooks/) | Training experiments and model comparisons |
| [`report/`](report/) | Project report and supporting figures |

## Safety and scope

> **Research prototype. Not a medical device. Not for clinical use.**
>
> This project has not been clinically validated. Its geometric margins are not anatomical safety margins, and outputs should not be used to guide patient care.
