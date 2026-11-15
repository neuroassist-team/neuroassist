"""
NeuroAssist - lightweight AI surgical planning assistant for brain tumor resection (Streamlit demo)

Run:   streamlit run app.py
Needs: teacher_best.pt + student_kd_best.pt (from the training notebook). Upload them in the sidebar on first run,
       or place them next to this file / in ./checkpoints.
Without checkpoints the app still starts and offers a clearly labelled SYNTHETIC PHANTOM (UI preview only, no AI).

Research prototype. Not a medical device. Not for clinical use.
"""
import itertools, os, re, json, time, tempfile, io, math, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import ndimage as ndi
from skimage import measure
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image                                              # frame decoding + GIF fallback (ships with scikit-image)

st.set_page_config(page_title="NeuroAssist", page_icon='<svg width="128" height="140" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 140"><path d="M35 0C54.33 0 70 15.67 70 35C70 46.6755 69.1491 59.4891 75.8158 69.0741L88.1354 86.7865C89.2561 88.3977 91.3496 88.9349 93.2141 88.3219C95.0357 87.723 96.4307 86.1004 96.4307 84.1829V71.9976C96.4307 63.3733 89.7146 56.2646 84.2334 49.6063C80.9635 45.6343 79 40.5467 79 35C79 22.2975 89.2974 12 102 12C114.703 12 125 22.2975 125 35C125 40.2104 123.267 45.0158 120.347 48.8719C114.943 56.0068 108 63.427 108 72.3775C108 81.4476 114.928 88.9226 121.052 95.6129C125.367 100.327 128 106.606 128 113.5C128 128.136 116.136 140 101.5 140C86.8645 140 75 128.136 75 113.5C75 112.983 75.0148 112.469 75.044 111.96C75.6282 101.766 77.3773 90.3018 71.5513 81.9166L63.0602 69.6955C60.1038 65.4405 54.1897 64.7285 49.4735 66.8739C45.7331 68.5754 42.7588 72.0522 42.7588 76.1614V84.8937C42.7588 92.6171 48.3475 99.3284 51.492 106.383C52.4612 108.557 53 110.966 53 113.5C53 123.165 45.165 131 35.5 131C25.835 131 18 123.165 18 113.5C18 109.823 19.1339 106.412 21.071 103.595C25.2178 97.5667 31 91.4905 31 84.1733C31 75.9682 24.5172 69.3968 17.4248 65.2711C7.00483 59.2097 0 47.9238 0 35C0 15.67 15.67 0 35 0Z" fill="#000000"/></svg>', layout="wide")

# ----------------------------------------------------------------------------------------------------------------------
# Config (must match the notebook; overridden by config.json if the notebook exported one)
# ----------------------------------------------------------------------------------------------------------------------
MODALITIES = ["flair", "t1", "t1ce", "t2"]
MOD_TITLES = {"flair": "T2-FLAIR", "t1": "T1", "t1ce": "T1 post-contrast", "t2": "T2"}
CROP = (slice(56, 184), slice(56, 184), slice(13, 141))          # 240x240x155 -> 128^3
CFG = {"teacher_filters": [32, 64, 128, 256, 320], "student_filters": [8, 16, 32, 64, 128]}
LABEL_NAMES = {1: "Necrotic / non-enhancing core", 2: "Peritumoral edema", 3: "Enhancing tumor"}
LABEL_COLORS = {1: (0.92, 0.18, 0.18), 2: (0.15, 0.80, 0.45), 3: (1.00, 0.85, 0.10)}
AXES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}
SPACING = 1.0                                                      # mm, BraTS is 1 mm isotropic
BASE = Path(__file__).resolve().parent                             # works from any working directory
DEFAULT_BUNDLE = os.environ.get("NEUROASSIST_DIR", str(BASE))
WEIGHT_FILES = ["teacher_best.pt", "student_kd_best.pt"]
# Results measured on the 74 held-out validation patients (Run 1); shown if results_table.csv is not next to the app.
KNOWN_RESULTS = pd.DataFrame({
    "Baseline (teacher)": [16.67, 958.8, 196.7, 10438, 0.761, 0.834, 0.903, 0.833, 4.15, 5.68, 6.06],
    "Lightweight (KD)":   [1.41, 63.0, 39.0, 1837, 0.709, 0.782, 0.882, 0.791, 8.67, 10.21, 8.94]},
    index=["Params (M)", "GFLOPs (128^3)", "GPU FP16 (ms)", "CPU (ms)", "Dice ET", "Dice TC", "Dice WT", "Dice mean",
           "HD95 ET (mm)", "HD95 TC (mm)", "HD95 WT (mm)"])
FLIP_SETS = [c for r in range(4) for c in itertools.combinations((0, 1, 2), r)]

st.markdown("""
<style>
/* NeuroAssist report palette and spacing. Presentation only; no widget or chart logic. */
:root {
  --na-bg: #0b1517;
  --na-surface: #142321;
  --na-ink: #ccd2d5;
  --na-muted: #949fa5;
  --na-heading: #dce2e5;
  --na-teal: #2dd4bf;
  --na-line: rgba(182, 204, 211, .14);
  --na-radius: 24px;
  --na-gutter: clamp(20px, 2.6vw, 52px);
  --na-gap: clamp(16px, 1.6vw, 28px);
  --na-padding: clamp(24px, 2.2vw, 36px);
  --na-glass: linear-gradient(135deg, rgba(49, 66, 63, .40), rgba(15, 28, 29, .62));
}
[data-testid="stApp"], [data-testid="stAppViewContainer"], .stApp {
  background: var(--na-bg);
  color: var(--na-ink);
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
[data-testid="stHeader"] {
  background: rgba(15, 28, 29, .62);
  border-bottom: 1px solid var(--na-line);
  backdrop-filter: blur(28px) saturate(75%);
  -webkit-backdrop-filter: blur(28px) saturate(75%);
}
[data-testid="stMainBlockContainer"], .main > .block-container {
  max-width: none;
  padding: calc(64px + var(--na-padding)) var(--na-gutter) clamp(40px, 6vh, 72px);
}
[data-testid="stSidebar"] {
  background: linear-gradient(160deg, rgba(24, 43, 39, .97), rgba(11, 21, 23, .98));
  border-right: 1px solid var(--na-line);
}
[data-testid="stSidebarUserContent"] { padding: var(--na-padding) 24px; }
[data-testid="stVerticalBlock"] { gap: var(--na-gap); }
[data-testid="stMarkdownContainer"] { color: var(--na-ink); line-height: 1.7; }
[data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4) {
  color: var(--na-heading);
  line-height: 1.2;
  font-weight: 550;
  letter-spacing: -.025em;
  padding-top: 0;
}
[data-testid="stMarkdownContainer"] h1 { font-size: clamp(28px, 3vw, 44px); letter-spacing: -.045em; }
[data-testid="stMarkdownContainer"] h2 { font-size: clamp(26px, 2.4vw, 38px); }
[data-testid="stMarkdownContainer"] h3 { font-size: clamp(20px, 1.65vw, 26px); }
[data-testid="stMarkdownContainer"] h4 { font-size: 1.1rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] :is(h2, h3) { font-size: 1.2rem; color: var(--na-teal); }
[data-testid="stMarkdownContainer"] p { line-height: 1.7; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--na-muted); }
[data-testid="stMarkdownContainer"] a { color: var(--na-teal); text-underline-offset: .2em; }
[data-testid="stMarkdownContainer"] a:hover { color: #9ce4d6; }
[data-testid="stMarkdownContainer"] code { color: #a4c9bd; background: rgba(45, 212, 191, .07); border-radius: 8px; }
.hdr {
  background: var(--na-glass);
  border: 1px solid var(--na-line);
  border-radius: var(--na-radius);
  padding: var(--na-padding);
  margin-bottom: var(--na-gap);
  backdrop-filter: blur(28px);
  -webkit-backdrop-filter: blur(28px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .08), 0 16px 44px rgba(0, 0, 0, .10);
}
.hdr h1 { display: flex; align-items: center; gap: 14px; color: var(--na-heading); margin: 0 0 12px; padding: 0; }
.hdr h1 svg { display: block; width: 44px; height: auto; flex-shrink: 0; }
.hdr p { color: #afc4bb; margin: 0; line-height: 1.6; }
.warn, .pre {
  padding: 14px 18px;
  margin: 0;
  border-radius: var(--na-radius);
  font-size: .875rem;
  line-height: 1.7;
}
.warn { background: rgba(45, 212, 191, .055); border: 1px solid var(--na-line); color: #afc4bb; }
.pre { background: rgba(136, 56, 45, .13); border: 1px solid rgba(230, 147, 125, .24); color: #e2b5a9; }
[data-testid="stExpander"] details,
[data-testid="stFileUploaderDropzone"],
[data-testid="stMetric"] {
  background: var(--na-glass);
  border: 1px solid var(--na-line);
  border-radius: var(--na-radius);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .04);
}
[data-testid="stExpander"] summary { padding: 16px 20px; color: #b8d8cd; }
[data-testid="stExpanderDetails"] { padding: 4px 20px 20px; }
[data-testid="stFileUploaderDropzone"] { padding: var(--na-padding); }
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--na-muted); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { padding: 20px 16px; }
[data-testid="stFileUploaderDropzone"] small { font-size: .75rem; white-space: normal; overflow-wrap: anywhere; }
[data-testid="stMetric"] { padding: 20px 24px; }
[data-testid="stMetricLabel"] { color: var(--na-muted); }
[data-testid="stMetricValue"] { color: var(--na-teal); letter-spacing: -.035em; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p { color: #bccfc6; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div {
  background: rgba(20, 35, 33, .88);
  color: var(--na-ink);
  border-color: var(--na-line);
  border-radius: 14px;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div,
[data-baseweb="textarea"]:focus-within { border-color: var(--na-teal); box-shadow: 0 0 0 1px rgba(45, 212, 191, .18); }
:is([data-testid="stButton"], [data-testid="stDownloadButton"], [data-testid="stFileUploader"]) button {
  border: 1px solid rgba(157, 208, 190, .24);
  border-radius: var(--na-radius);
  background: rgba(45, 212, 191, .07);
  color: #c4d8cf;
  padding: .6rem 1.15rem;
  min-height: 42px;
  transition: background-color .2s ease, border-color .2s ease;
}
:is([data-testid="stButton"], [data-testid="stDownloadButton"], [data-testid="stFileUploader"]) button:hover:not(:disabled) {
  border-color: rgba(45, 212, 191, .65);
  background: rgba(45, 212, 191, .14);
  color: #dce9e3;
}
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--na-teal);
  border-color: var(--na-teal);
  color: #09221d;
  font-weight: 600;
}
button[kind="primary"] p, button[kind="primary"] [data-testid="stMarkdownContainer"],
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] { color: #09221d; }
button:disabled { opacity: .46; cursor: not-allowed; }
[data-baseweb="tab-list"] { gap: 6px; background: rgba(21, 36, 32, .55); padding: 6px; border-radius: var(--na-radius); }
[data-baseweb="tab"] { color: #afc4bb; border-radius: 18px; padding: 12px 18px; }
[data-baseweb="tab"][aria-selected="true"] { background: rgba(45, 212, 191, .11); color: var(--na-teal); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--na-line); border-radius: var(--na-radius); overflow: hidden; }
[data-testid="stTable"] :is(th, td) { border-color: var(--na-line); padding: 14px 18px; }
[data-testid="stTable"] th { color: #a4c9bd; background: rgba(45, 212, 191, .045); }
[data-testid="stAlert"] { border-radius: var(--na-radius); }
[data-testid="stMarkdownContainer"] hr { border-color: var(--na-line); margin: var(--na-gap) 0; }
@media (max-width: 720px) {
  :root { --na-gutter: 16px; --na-gap: 16px; --na-padding: 24px; }
  [data-testid="stMainBlockContainer"], .main > .block-container { padding-top: 88px; }
  [data-testid="stSidebarUserContent"] { padding-inline: 20px; }
  [data-testid="stFileUploaderDropzone"] { padding: 20px; }
  [data-baseweb="tab"] { padding-inline: 12px; }
}
@media (prefers-reduced-motion: reduce) { button { transition: none !important; } }
@supports not (backdrop-filter: blur(20px)) {
  .hdr, [data-testid="stExpander"] details, [data-testid="stMetric"], [data-testid="stFileUploaderDropzone"] { background: #142321; }
}

</style>
<div class="hdr"><h1><svg width="128" height="140" viewBox="0 0 128 140" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M35 0C54.33 0 70 15.67 70 35C70 46.6755 69.1491 59.4891 75.8158 69.0741L88.1354 86.7865C89.2561 88.3977 91.3496 88.9349 93.2141 88.3219C95.0357 87.723 96.4307 86.1004 96.4307 84.1829V71.9976C96.4307 63.3733 89.7146 56.2646 84.2334 49.6063C80.9635 45.6343 79 40.5467 79 35C79 22.2975 89.2974 12 102 12C114.703 12 125 22.2975 125 35C125 40.2104 123.267 45.0158 120.347 48.8719C114.943 56.0068 108 63.427 108 72.3775C108 81.4476 114.928 88.9226 121.052 95.6129C125.367 100.327 128 106.606 128 113.5C128 128.136 116.136 140 101.5 140C86.8645 140 75 128.136 75 113.5C75 112.983 75.0148 112.469 75.044 111.96C75.6282 101.766 77.3773 90.3018 71.5513 81.9166L63.0602 69.6955C60.1038 65.4405 54.1897 64.7285 49.4735 66.8739C45.7331 68.5754 42.7588 72.0522 42.7588 76.1614V84.8937C42.7588 92.6171 48.3475 99.3284 51.492 106.383C52.4612 108.557 53 110.966 53 113.5C53 123.165 45.165 131 35.5 131C25.835 131 18 123.165 18 113.5C18 109.823 19.1339 106.412 21.071 103.595C25.2178 97.5667 31 91.4905 31 84.1733C31 75.9682 24.5172 69.3968 17.4248 65.2711C7.00483 59.2097 0 47.9238 0 35C0 15.67 15.67 0 35 0Z" fill="#ffffff"/>
</svg><span>NeuroAssist</span></h1><p>Pre-operative brain tumor planning · multi-sequence MRI · lightweight distilled 3D network</p></div>
<div class="warn">⚠️ Research prototype for demonstration only. Not a medical device; not validated for clinical decisions.</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------------------------
# Preprocessing (identical to the training notebook)
# ----------------------------------------------------------------------------------------------------------------------
def zscore_nonzero(vol):
    mask = vol != 0
    if mask.sum() == 0:
        return vol
    v = vol[mask]
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = np.clip(v, lo, hi)
    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def preprocess_modalities(vols):
    return np.stack([zscore_nonzero(vols[m])[CROP] for m in MODALITIES]).astype(np.float32)


def preprocess_label(seg):
    seg = np.rint(seg).astype(np.uint8)[CROP]
    seg[seg == 4] = 3
    return seg


def load_nii(path):
    import nibabel as nib
    return nib.load(path).get_fdata(dtype=np.float32)


def find_modality(name):
    m = re.search(r"(flair|t1ce|t1gd|t1|t2)\.nii", name.lower())
    if m:
        return "t1ce" if m.group(1) == "t1gd" else m.group(1)
    return "seg" if "seg" in name.lower() else None


# ----------------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------------
def find_ckpt(bundle, fname):
    for p in _ckpt_candidates(bundle, fname):
        if p.exists() and p.stat().st_size > 1_000_000:           # a tiny file is an un-pulled Git LFS pointer
            return p
    return None


def _ckpt_candidates(bundle, fname):
    return (Path(bundle) / fname, Path(bundle) / "checkpoints" / fname, Path(bundle) / "weights" / fname,
            BASE / fname, BASE / "checkpoints" / fname, BASE.parent / "checkpoints" / fname)


def lfs_pointer_found(bundle):
    return any(p.exists() and p.stat().st_size <= 1_000_000 for f in WEIGHT_FILES for p in _ckpt_candidates(bundle, f))


@st.cache_resource(show_spinner="Loading models…")
def load_models(bundle):
    cfg_p = Path(bundle) / "config.json"
    if cfg_p.exists():
        CFG.update(json.load(open(cfg_p)))
    tp, sp = find_ckpt(bundle, "teacher_best.pt"), find_ckpt(bundle, "student_kd_best.pt")
    if tp is None or sp is None:
        return None
    import torch
    from monai.networks.nets import DynUNet
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def build(filters, res):
        n = len(filters)
        return DynUNet(3, 4, 4, [3] * n, [1] + [2] * (n - 1), [2] * (n - 1), filters=list(filters),
                       res_block=res, deep_supervision=False)

    out = {"device": dev}
    for key, p, f, res in (("teacher", tp, CFG["teacher_filters"], True), ("student", sp, CFG["student_filters"], False)):
        m = build(f, res)
        ck = torch.load(p, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"])
        out[key] = m.to(dev).eval()
        out[key + "_params"] = sum(q.numel() for q in m.parameters())
        out.setdefault("info", {})[key] = {"file": str(p), "epoch": ck.get("epoch"), "val": ck.get("val")}
        with torch.no_grad():                                      # self-test: weights must give a 4-class output
            o = m(torch.zeros(1, 4, 32, 32, 32, device=dev))
            assert tuple(o.shape) == (1, 4, 32, 32, 32), f"unexpected output {tuple(o.shape)} for {key}"
    if dev == "cuda":                                              # warm-up so reported latency is realistic
        with torch.no_grad():
            for k in ("teacher", "student"):
                out[k](torch.zeros(1, 4, 128, 128, 128, device=dev))
    return out


def run_model(models, key, img, tta=False):
    """-> (softmax probs (4,128,128,128) float32, latency ms)."""
    import torch
    import torch.nn.functional as F
    dev, m = models["device"], models[key]
    x = torch.from_numpy(img).float().unsqueeze(0).to(dev)
    sets = FLIP_SETS if tta else [()]
    with torch.no_grad():
        if dev == "cuda": torch.cuda.synchronize()
        t0, acc = time.perf_counter(), None
        for fs in sets:
            dims = [d + 2 for d in fs]
            with torch.autocast(dev, enabled=dev == "cuda"):
                lg = m(torch.flip(x, dims) if dims else x)
            p = F.softmax(lg.float(), 1)
            if dims:
                p = torch.flip(p, dims)
            acc = p if acc is None else acc + p
        if dev == "cuda": torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
    return (acc / len(sets))[0].cpu().numpy(), ms


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic phantom (UI preview only - NO AI involved)
# ----------------------------------------------------------------------------------------------------------------------
def make_phantom(seed=0):
    rng, n = np.random.RandomState(seed), 128
    zz, yy, xx = np.mgrid[:n, :n, :n].astype(np.float32)
    brain = (((zz - 64) / 52) ** 2 + ((yy - 64) / 58) ** 2 + ((xx - 62) / 46) ** 2) < 1
    r = np.sqrt((zz - 70) ** 2 + ((yy - 52) * 1.1) ** 2 + ((xx - 66) * 0.9) ** 2)
    gt = np.zeros((n, n, n), np.uint8)
    gt[(r < 24) & brain] = 2; gt[(r < 15) & brain] = 3; gt[(r < 11) & brain] = 1
    img = np.zeros((4, n, n, n), np.float32)
    for i in range(4):
        img[i] = brain * rng.randn(n, n, n).astype(np.float32) * 0.25
    img[0] += (gt > 0) * 1.5; img[1] += (gt == 1) * -0.8; img[2] += (gt == 3) * 2.0; img[3] += (gt > 0) * 1.2
    img = ndi.gaussian_filter(img, (0, 0.8, 0.8, 0.8)) * brain
    return img.astype(np.float32), gt


def phantom_probs(gt):
    oh = np.stack([(gt == c).astype(np.float32) for c in range(4)])
    p_t = ndi.gaussian_filter(oh, (0, 1.5, 1.5, 1.5))
    p_s = ndi.gaussian_filter(np.roll(oh, 1, axis=1), (0, 2.2, 2.2, 2.2))
    return p_t / p_t.sum(0, keepdims=True), p_s / p_s.sum(0, keepdims=True)


# ----------------------------------------------------------------------------------------------------------------------
# Surgical-planning math (same as the notebook)
# ----------------------------------------------------------------------------------------------------------------------
def tumor_report(seg, spacing=SPACING):
    vox = spacing ** 3 / 1000.0
    et, ncr, ed = seg == 3, seg == 1, seg == 2
    tc, wt = et | ncr, seg > 0
    rep = {"Enhancing tumor ET (mL)": et.sum() * vox, "Necrotic core (mL)": ncr.sum() * vox,
           "Edema (mL)": ed.sum() * vox, "Tumor core TC (mL)": tc.sum() * vox, "Whole tumor WT (mL)": wt.sum() * vox}
    if wt.any():
        co = np.argwhere(wt)
        rep["centroid"] = tuple(int(v) for v in co.mean(0).round())
        rep["bbox_mm"] = tuple(int(v) for v in (co.max(0) - co.min(0) + 1) * spacing)
        diam = 0.0
        for z in np.unique(co[:, 2]):
            lab, _ = ndi.label(wt[:, :, z])
            for rp in measure.regionprops(lab):
                ml = rp.axis_major_length if hasattr(rp, "axis_major_length") else rp.major_axis_length
                diam = max(diam, ml * spacing)
        rep["max_axial_diameter_mm"] = round(diam, 1)
        rep["necrosis_fraction_of_TC"] = float(ncr.sum() / max(tc.sum(), 1))
    return rep


def tc_distance(seg, spacing=SPACING):
    tgt = (seg == 1) | (seg == 3)
    return ndi.distance_transform_edt(~tgt, sampling=spacing).astype(np.float32) if tgt.any() else None


def margin_table(seg, brain, dist, margins=(5, 10, 15), spacing=SPACING):
    vox = spacing ** 3 / 1000.0
    tgt = (seg == 1) | (seg == 3)
    rows = []
    for m in margins:
        ring = (dist <= m) & ~tgt & brain
        rows.append({"Margin (mm)": m, "Extra tissue in margin (mL)": round(ring.sum() * vox, 1),
                     "of which edema (mL)": round((ring & (seg == 2)).sum() * vox, 1)})
    return pd.DataFrame(rows)


def trajectory_report(entry, seg, brain, dist, spacing=SPACING):
    tgt = np.argwhere((seg == 1) | (seg == 3))
    if len(tgt) == 0 or dist is None:
        return None
    c, e = tgt.mean(0), np.asarray(entry, float)
    L = np.linalg.norm(c - e) * spacing
    n = max(int(L / spacing) + 1, 2)
    pts = np.linspace(e, c, n).round().astype(int).clip(0, np.array(seg.shape) - 1)
    labs, step = seg[pts[:, 0], pts[:, 1], pts[:, 2]], L / (n - 1)
    return {"Straight-line length to tumor-core centroid (mm)": round(L, 1),
            "Path through brain tissue (mm)": round(brain[pts[:, 0], pts[:, 1], pts[:, 2]].sum() * step, 1),
            "…through edema (mm)": round((labs == 2).sum() * step, 1),
            "…through tumor core (mm)": round(np.isin(labs, (1, 3)).sum() * step, 1),
            "Entry point to nearest tumor-core voxel (mm)": round(float(dist[tuple(e.round().astype(int).clip(0, 127))]), 1)}


def default_entry(seg, brain):
    tgt = np.argwhere((seg == 1) | (seg == 3))
    if len(tgt) == 0:
        return [64, 64, 100]
    cx, cy, _ = tgt.mean(0).round().astype(int)
    col = np.where(brain[cx, cy])[0]
    return [int(cx), int(cy), int(col.max() if len(col) else 120)]


def js_divergence_map(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(a + eps) - np.log(b + eps))).sum(0)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).astype(np.float32)


def entropy_map(p, eps=1e-8):
    return (-(p * np.log(p + eps)).sum(0)).astype(np.float32)


def mesh_from_mask(mask, step=2, sigma=1.0):
    if mask.sum() < 50:
        return None
    m = np.pad(ndi.gaussian_filter(mask.astype(np.float32), sigma), 1)
    try:
        v, f, _, _ = measure.marching_cubes(m, level=0.5, step_size=step)
    except Exception:
        return None
    return v - 1, f


def make_3d_figure(seg, brain, dist=None, margin_mm=0, traj=None, title=""):
    fig = go.Figure()

    def add(mask, name, color, opacity, step=2, sigma=1.0):
        mm = mesh_from_mask(mask, step, sigma)
        if mm is not None:
            v, f = mm
            fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color,
                                    opacity=opacity, name=name, showlegend=True,
                                    lighting=dict(ambient=0.55, diffuse=0.7, specular=0.2)))

    add(brain, "Brain", "#b0b0b0", 0.06, step=4, sigma=2.0)
    add(seg == 2, "Edema", "#26cc73", 0.22)
    add(seg == 1, "Necrotic / non-enh. core", "#eb2e2e", 0.95)
    add(seg == 3, "Enhancing tumor", "#ffd91a", 0.95)
    if margin_mm and dist is not None:
        add(dist <= margin_mm, f"{margin_mm} mm safety margin", "#00e5ff", 0.14, step=3, sigma=1.5)
    if traj is not None:
        e, c = traj
        fig.add_trace(go.Scatter3d(x=[e[0], c[0]], y=[e[1], c[1]], z=[e[2], c[2]], mode="lines+markers",
                                   line=dict(color="#ff9f1c", width=8), marker=dict(size=5), name="Planned trajectory"))
    ax = dict(showbackground=False, showticklabels=False, title="", showgrid=False, zeroline=False)
    fig.update_layout(template="plotly_dark", title=title, height=640, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data"), legend=dict(bgcolor="rgba(0,0,0,.3)"))
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------------------------------------------------
def take(vol, axis, idx):
    return np.take(vol, int(idx), axis=axis)


def best_slice(seg, axis=2):
    other = tuple(a for a in range(3) if a != axis)
    s = (seg > 0).sum(axis=other)
    return int(np.argmax(s)) if s.max() > 0 else seg.shape[axis] // 2


def gray_norm(sl):
    nz = sl[sl != 0]
    if nz.size == 0:
        return np.zeros_like(sl, dtype=np.float32)
    lo, hi = np.percentile(nz, [1, 99])
    return np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def overlay_rgb(base2d, seg2d=None, alpha=0.5):
    g = gray_norm(base2d)
    rgb = np.stack([g, g, g], -1)
    if seg2d is not None:
        for c, col in LABEL_COLORS.items():
            m = seg2d == c
            rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(col)
    return np.transpose(rgb, (1, 0, 2))


def draw(ax, base2d, seg2d=None, title="", alpha=0.5, dist2d=None, margin=0):
    ax.imshow(overlay_rgb(base2d, seg2d, alpha), origin="lower")
    if dist2d is not None and margin > 0 and dist2d.min() <= margin:
        ax.contour(dist2d.T, levels=[margin], colors=["cyan"], linewidths=1.4)
    ax.set_title(title, fontsize=10, color="white"); ax.axis("off")


def legend(fig):
    fig.legend(handles=[Patch(color=LABEL_COLORS[c], label=LABEL_NAMES[c]) for c in (3, 1, 2)], loc="lower center",
               ncol=3, frameon=False, labelcolor="white")


def show_plotly(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                                              # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------------------------------------------------
# Rotating 3D video (orbit the camera around the tumor in the current 3D figure)
# ----------------------------------------------------------------------------------------------------------------------
VID_W, VID_H = 800, 600                    # even sizes (H.264 needs this)
VID_FRAMES, VID_FPS = 36, 12               # 36 frames / 12 fps = 3 s, one full 360-degree turn
VID_START_DEG = 45


def _orbit_camera(angle_deg, radius=1.5, height=0.8):
    """Plotly camera on a circle around the vertical (z) axis; the scene centre is the tumor (see _prep_video_fig)."""
    a = math.radians(angle_deg)
    return dict(eye=dict(x=radius * math.cos(a), y=radius * math.sin(a), z=height),
                up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0))


def _xyz(tr):
    return np.column_stack([tr.x, tr.y, tr.z]).astype(float)


def _prep_video_fig(fig3d):
    """Copy of the on-screen figure tuned for video, so the on-screen one is never touched:
    - tumor centred in the scene (an invisible cube of points centred on the tumor makes the camera orbit it),
    - glossier lighting, a more visible brain/edema shell."""
    fig = go.Figure(fig3d)
    meshes = [t for t in fig.data if t.type == "mesh3d"]
    if not meshes:
        return fig
    core = [_xyz(t) for t in meshes if t.name in ("Necrotic / non-enh. core", "Enhancing tumor")] \
        or [_xyz(t) for t in meshes if t.name != "Brain"] or [_xyz(t) for t in meshes]
    ctr = np.vstack(core).mean(0)
    allv = np.vstack([_xyz(t) for t in fig.data if t.type in ("mesh3d", "scatter3d")])
    r = np.abs(allv - ctr).max()                                    # cube big enough to contain everything
    cube = np.array([ctr - r, ctr + r])
    fig.add_trace(go.Scatter3d(x=cube[:, 0], y=cube[:, 1], z=cube[:, 2], mode="markers", marker=dict(size=1, opacity=0),
                               name="_bounds", showlegend=False, hoverinfo="skip"))
    for t in meshes:
        t.update(lighting=dict(ambient=0.35, diffuse=0.95, specular=0.6, roughness=0.35, fresnel=0.4),
                 lightposition=dict(x=3000, y=2000, z=4000))
        if t.name == "Brain":
            t.opacity = 0.12
        elif t.name == "Edema":
            t.opacity = 0.28
    return fig


def _frames_plotly(fig, n_frames):
    """Preferred path: rotate the camera on the prepared Plotly figure, rasterise each frame with kaleido."""
    fig.update_layout(width=VID_W, height=VID_H)
    frames = []
    for i in range(n_frames):
        fig.update_layout(scene_camera=_orbit_camera(VID_START_DEG + 360.0 * i / n_frames))
        png = fig.to_image(format="png", width=VID_W, height=VID_H, scale=1)   # raises if kaleido/Chrome is missing
        frames.append(np.array(Image.open(io.BytesIO(png)).convert("RGB")))
    return frames


def _frames_matplotlib(fig3d, n_frames, max_faces=15000):
    """Fallback (no kaleido needed): redraw the traces with Matplotlib, CPU only, with per-face Lambert shading and a
    light that follows the camera. Meshes are thinned to `max_faces` triangles each, so it is a bit coarser than Plotly."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = Figure(figsize=(VID_W / 100, VID_H / 100), dpi=100, facecolor="#0e1117")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d", facecolor="#0e1117")
    shaded, pts, handles = [], [], []
    for tr in fig3d.data:
        if tr.type == "mesh3d":
            v, f = _xyz(tr), np.column_stack([tr.i, tr.j, tr.k]).astype(int)
            if len(f) > max_faces:
                f = f[::int(np.ceil(len(f) / max_faces))]
            tri = v[f]                                              # (F, 3, 3)
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            col = Poly3DCollection(tri, edgecolors="none")
            ax.add_collection3d(col)
            shaded.append((col, nrm, np.array(to_rgb(tr.color)), float(tr.opacity)))
            pts.append(v); handles.append(Patch(color=tr.color, label=tr.name))
        elif tr.type == "scatter3d":
            pts.append(_xyz(tr))
            if tr.name != "_bounds":                                # planned trajectory
                ax.plot(tr.x, tr.y, tr.z, color=tr.line.color, lw=3)
                ax.scatter(tr.x, tr.y, tr.z, color=tr.line.color, s=18)
                handles.append(Patch(color=tr.line.color, label=tr.name))
    allp = np.vstack(pts)
    lo, hi = allp.min(0), allp.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.3)                            # equal scaling, like aspectmode="data"
    ax.set_axis_off()
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, labelcolor="white", fontsize=9)
    title = getattr(fig3d.layout.title, "text", None)
    if title:
        fig.text(0.5, 0.96, title, color="white", ha="center", fontsize=11)
    el = 25
    frames = []
    for i in range(n_frames):
        az = VID_START_DEG + 360.0 * i / n_frames
        ax.view_init(elev=el, azim=az)
        a, e = math.radians(az), math.radians(el)
        view = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        light = view + np.array([-0.4 * math.sin(a), 0.4 * math.cos(a), 0.5])   # headlight, nudged up/sideways
        light /= np.linalg.norm(light)
        for col, nrm, rgb, alpha in shaded:
            lam = np.abs(nrm @ light)                               # two-sided diffuse term
            rgba = np.empty((len(lam), 4))
            rgba[:, :3] = np.clip(rgb[None] * (0.30 + 0.70 * lam)[:, None] + 0.18 * (lam ** 20)[:, None], 0, 1)  # + highlight
            rgba[:, 3] = alpha
            col.set_facecolor(rgba)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    return frames


def _encode_video(frames, fps):
    """MP4 (H.264) through imageio-ffmpeg; if that is unavailable, fall back to an animated GIF via Pillow."""
    try:
        import imageio.v2 as iio
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rotation.mp4")
            iio.mimsave(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=1)
            with open(path, "rb") as fh:
                return fh.read(), "mp4"
    except Exception:
        ims = [Image.fromarray(f) for f in frames]
        buf = io.BytesIO()
        ims[0].save(buf, format="GIF", save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
        return buf.getvalue(), "gif"


def render_rotation_video(fig3d, n_frames=VID_FRAMES, fps=VID_FPS):
    """Return (bytes, "mp4"|"gif", backend_name) for a full turn around the tumor in `fig3d`."""
    vfig = _prep_video_fig(fig3d)
    try:
        frames, backend = _frames_plotly(vfig, n_frames), "Plotly + kaleido"
    except Exception:                                               # kaleido / Chrome missing or failing
        frames, backend = _frames_matplotlib(vfig, n_frames), "Matplotlib fallback"
    data, kind = _encode_video(frames, fps)
    return data, kind, backend


# ----------------------------------------------------------------------------------------------------------------------
# Sidebar: inputs (everything is uploaded in the browser - no folder paths to type)
# ----------------------------------------------------------------------------------------------------------------------
SRC_UPLOAD, SRC_CACHED, SRC_PHANTOM = "Upload MRI scan", "Cached held-out case", "Demo phantom (no AI)"


def detect_uploads(files, read=False):
    """Map sequence key (flair/t1/t1ce/t2/seg) -> (file name, bytes or None) from loose .nii/.nii.gz uploads
    and/or a zipped BraTS case folder. `read=False` only looks at names, so it is cheap enough to run on every rerun."""
    found = {}
    for f in files or []:
        if f.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                    for n in z.namelist():
                        base = Path(n).name
                        k = find_modality(base)
                        if k and not base.startswith(".") and base.lower().endswith((".nii", ".nii.gz")):
                            found[k] = (base, z.read(n) if read else None)
            except zipfile.BadZipFile:
                pass
        elif (k := find_modality(f.name)):
            found[k] = (f.name, f.getvalue() if read else None)
    return found


def save_weights(files):
    """Store uploaded weight files where load_models() looks for them (app folder, else a temp folder)."""
    keep = {f.name: f.getvalue() for f in files if f.name in WEIGHT_FILES + ["config.json", "results_table.csv"]}
    for d in (Path(DEFAULT_BUNDLE) / "checkpoints", Path(tempfile.gettempdir()) / "neuroassist"):
        try:
            d.mkdir(parents=True, exist_ok=True)
            for name, data in keep.items():
                (d / name).write_bytes(data)
            st.session_state["bundle_dir"] = str(d.parent if d.name == "checkpoints" else d)
            return
        except OSError:                                             # read-only app folder -> try the temp folder
            continue


st.sidebar.header("Model")
bundle = st.session_state.get("bundle_dir", DEFAULT_BUNDLE)
try:
    models = load_models(bundle)
except Exception as e:                                              # e.g. wrong or corrupt weight file
    models = None
    st.sidebar.error(f"Could not load the model weights: {e}")
if models is None:
    if lfs_pointer_found(bundle):
        st.sidebar.error("Found a Git LFS pointer instead of the real weights. Run:  git lfs install && git lfs pull")
    st.sidebar.warning("Model weights not found. Upload them once to enable the AI models "
                       "(until then only the demo phantom works).")
    w_up = st.sidebar.file_uploader("teacher_best.pt + student_kd_best.pt (config.json / results_table.csv optional)",
                                    type=["pt", "json", "csv"], accept_multiple_files=True, key="weights_up")
    have_both = {f.name for f in w_up or []} >= set(WEIGHT_FILES)
    if st.sidebar.button("Load models", disabled=not have_both):
        with st.spinner("Loading models…"):
            save_weights(w_up); load_models.clear()
        st.rerun()
else:
    st.sidebar.success(f"Models ready · {models['device'].upper()}")
    with st.sidebar.expander("Model details"):
        for label, k in (("Baseline", "teacher"), ("Lightweight", "student")):
            v = models["info"][k].get("val")
            st.caption(f"**{label}:** " + (f"val mean Dice {v['mean']:.3f}" if v else "loaded"))

st.sidebar.header("Case")
cached = sorted(Path(bundle).glob("cases/*_img.npy"))
sources = [SRC_UPLOAD] + ([SRC_CACHED] if cached and models is not None else []) + [SRC_PHANTOM]
default = SRC_CACHED if SRC_CACHED in sources else (SRC_UPLOAD if models is not None else SRC_PHANTOM)
src = st.sidebar.radio("Source", sources, index=sources.index(default))
uploads, sel_case, ready, hint = None, None, False, ""
if src == SRC_UPLOAD:
    uploads = st.sidebar.file_uploader("4 sequences (.nii / .nii.gz), or one .zip of a BraTS case folder",
                                       type=["nii", "gz", "zip"], accept_multiple_files=True, key="case_up")
    found = detect_uploads(uploads)
    missing = [MOD_TITLES[m] for m in MODALITIES if m not in found]
    if uploads:
        st.sidebar.caption("  ".join(f"{'✅' if m in found else '⬜'} {MOD_TITLES[m]}" for m in MODALITIES)
                           + ("  ·  ✅ ground truth" if "seg" in found else ""))
    ready = models is not None and not missing
    hint = ("Upload the model weights first." if models is None else
            "Upload the 4 MRI sequences to begin." if not uploads else f"Still needed: {', '.join(missing)}")
elif src == SRC_CACHED:
    sel_case = st.sidebar.selectbox("Patient", [p.name.replace("_img.npy", "") for p in cached])
    ready = True
else:
    ready = True
with st.sidebar.expander("Advanced"):
    tta = st.checkbox("Flip test-time augmentation (8x slower, smoother)", value=False)
run = st.sidebar.button("▶ Run segmentation", type="primary", disabled=not ready)
if hint and not ready:
    st.sidebar.caption(hint)


def load_case():
    """-> dict(name, img, gt, phantom) or None (error already shown)."""
    if src == SRC_CACHED and sel_case:
        return dict(name=sel_case, img=np.load(Path(bundle) / "cases" / f"{sel_case}_img.npy").astype(np.float32),
                    gt=np.load(Path(bundle) / "cases" / f"{sel_case}_lbl.npy"), phantom=False)
    if src == SRC_UPLOAD:
        try:
            files, tmp, paths = detect_uploads(uploads, read=True), Path(tempfile.mkdtemp()), {}
            for k, (fname, data) in files.items():
                paths[k] = tmp / fname
                paths[k].write_bytes(data)
            vols = {m: load_nii(str(paths[m])) for m in MODALITIES}
            if any(v.shape != (240, 240, 155) for v in vols.values()):
                st.error("Volumes must be in BraTS space (240x240x155). Register/resample first.")
                return None
            gt = preprocess_label(load_nii(str(paths["seg"]))) if "seg" in paths else None
            name = re.sub(r"_(flair|t1ce|t1gd|t1|t2|seg)\.nii(\.gz)?$", "", files["flair"][0], flags=re.I) or "Uploaded case"
            return dict(name=name, img=preprocess_modalities(vols), gt=gt, phantom=False)
        except Exception as e:
            st.error(f"Could not read the uploaded scan: {e}")
            return None
    if src == SRC_PHANTOM:
        img, gt = make_phantom()
        return dict(name="Synthetic phantom", img=img, gt=gt, phantom=True)
    st.error("Choose or provide a case first.")
    return None


def analyze(case):
    img, gt = case["img"], case["gt"]
    if case["phantom"]:
        p_t, p_s = phantom_probs(gt); t_ms = s_ms = None
    else:
        if models is None:
            st.error("Models are not loaded."); return None
        p_t, t_ms = run_model(models, "teacher", img, tta)
        p_s, s_ms = run_model(models, "student", img, tta)
    seg = {"pt": p_t.argmax(0).astype(np.uint8), "ps": p_s.argmax(0).astype(np.uint8), "gt": gt}
    brain = np.abs(img).sum(0) > 0
    ref = seg["ps"] if (seg["ps"] > 0).any() else (gt if gt is not None else seg["pt"])
    return dict(id=time.time(), name=case["name"], img=img, seg=seg, brain=brain, phantom=case["phantom"], t_ms=t_ms, s_ms=s_ms,
                dist={k: tc_distance(v) for k, v in seg.items() if v is not None},
                js=js_divergence_map(p_t, p_s), ent=entropy_map(p_s), best=best_slice(ref, 2), tta=tta)


if run:
    case = load_case()
    if case is not None:
        with st.spinner("Running both networks (CPU can take a few seconds per volume)…"):
            st.session_state["res"] = analyze(case)

res = st.session_state.get("res")
if res is None:
    st.markdown("#### Get started")
    g = st.columns(3)
    g[0].markdown("**1 · Add a case**\n\nUpload the 4 MRI sequences (.nii / .nii.gz) or one .zip of a BraTS case in the sidebar.")
    g[1].markdown("**2 · Run**\n\nPress **▶ Run segmentation**. The baseline and lightweight networks both run on your machine.")
    g[2].markdown("**3 · Explore**\n\nCompare segmentations, set a safety margin, plan a trajectory, export a rotating 3D video.")
    st.stop()
if res["phantom"]:
    st.markdown('<div class="pre">SYNTHETIC PHANTOM: UI preview only. These "predictions" are smoothed copies of a made-up mask; no AI model ran.</div>',
                unsafe_allow_html=True)

st.sidebar.header("3 · View")
plane = st.sidebar.radio("Plane", list(AXES), horizontal=True)
idx = st.sidebar.slider("Slice", 0, 127, int(res["best"]), key=f"slice_{res['id']}")
modality = st.sidebar.selectbox("Background sequence", MODALITIES, index=2, format_func=lambda m: MOD_TITLES[m])
modes = ["Side-by-side (baseline | lightweight | ground truth)", "Lightweight only", "Baseline only", "Ground truth only"]
mode = st.sidebar.radio("Segmentation shown", modes)
alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
margin = st.sidebar.slider("Safety margin around tumor core (mm)", 0, 20, 10)

seg, img, brain, dist = res["seg"], res["img"], res["brain"], res["dist"]
ax_i, mi = AXES[plane], MODALITIES.index(modality)
tabs = st.tabs(["Segmentation viewer", "4 MRI modalities", "3D tumor", "Surgical report", "Uncertainty", "Models & speed"])

# ---------------------------------------------------------------- viewer
with tabs[0]:
    if mode == modes[0]:
        panels = [("pt", "Baseline (teacher)"), ("ps", "Lightweight (student)")] + ([("gt", "Ground truth")] if seg["gt"] is not None else [])
    else:
        key = {modes[1]: "ps", modes[2]: "pt", modes[3]: "gt"}[mode]
        panels = [(key, dict(ps="Lightweight (student)", pt="Baseline (teacher)", gt="Ground truth")[key])] if seg[key] is not None else []
    if not panels:
        st.warning("No ground truth available for this case.")
    else:
        fig = Figure(figsize=(4.8 * (len(panels) + 1), 5.4), facecolor="#0e1117")
        axs = fig.subplots(1, len(panels) + 1)
        draw(axs[0], take(img[mi], ax_i, idx), None, f"{MOD_TITLES[modality]} (raw)")
        for a, (k, t) in zip(axs[1:], panels):
            d = dist.get(k)
            draw(a, take(img[mi], ax_i, idx), take(seg[k], ax_i, idx), t, alpha, None if d is None else take(d, ax_i, idx), margin)
        legend(fig)
        if margin:
            fig.text(0.99, 0.01, f"cyan = {margin} mm margin around tumor core", color="cyan", ha="right", fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        st.pyplot(fig)

# ---------------------------------------------------------------- modalities
with tabs[1]:
    fig = Figure(figsize=(19, 4.8), facecolor="#0e1117")
    for a, (m, i) in zip(fig.subplots(1, 4), zip(MODALITIES, range(4))):
        draw(a, take(img[i], ax_i, idx), take(seg["ps"], ax_i, idx), MOD_TITLES[m], alpha=0.35)
    legend(fig); fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.pyplot(fig)

# ---------------------------------------------------------------- 3D
with tabs[2]:
    c1, c2 = st.columns([1, 3])
    show = c1.radio("Model", ["Lightweight", "Baseline", "Ground truth"][:3 if seg["gt"] is not None else 2])
    key = {"Lightweight": "ps", "Baseline": "pt", "Ground truth": "gt"}[show]
    entry_key = f"entry_{res['id']}"
    use_traj = c1.checkbox("Show planned trajectory", value=True)
    traj = None
    if use_traj and dist.get(key) is not None:
        de = default_entry(seg[key], brain)
        e = [c1.number_input(ax, 0, 127, de[i], key=f"{entry_key}_{ax}_{key}") for i, ax in enumerate(("Entry x", "Entry y", "Entry z"))]
        tgt = np.argwhere((seg[key] == 1) | (seg[key] == 3)).mean(0)
        traj = (e, tgt.tolist())
        t = trajectory_report(e, seg[key], brain, dist[key])
        if t:
            for k_, v_ in t.items():
                c1.caption(f"**{k_}:** {v_}")
    with c2:
        with st.spinner("Building 3D meshes…"):
            fig3d = make_3d_figure(seg[key], brain, dist.get(key), margin, traj, f"{res['name']} · {show}")
            show_plotly(fig3d)

    # ---- rotating video of the current view (selected model, margin and trajectory) ----
    st.divider()
    # signature of everything that shapes the 3D view, used to flag a video that no longer matches
    vid_sig = repr((res["id"], key, margin, None if traj is None else (tuple(traj[0]), tuple(round(float(x), 1) for x in traj[1]))))
    v1, v2 = st.columns([1, 3])
    if v1.button("🎬 Generate Rotating 3D Video", key="gen_rot_video"):
        st.session_state.pop("rot_video", None)                    # clear any previous video first
        with v2:
            with st.spinner("Rendering rotating video…"):
                try:
                    data, kind, backend = render_rotation_video(fig3d)
                    st.session_state["rot_video"] = dict(data=data, kind=kind, backend=backend, sig=vid_sig)
                except Exception as ex:
                    st.error(f"Video generation failed: {ex}")
    vid = st.session_state.get("rot_video")
    if vid:
        with v2:
            if vid["kind"] == "mp4":
                st.video(vid["data"], format="video/mp4")
            else:
                st.image(vid["data"])                               # st.video cannot play GIFs
            st.caption(f"{VID_FRAMES} frames · {VID_FPS} fps · rendered with {vid['backend']}")
            if vid["sig"] != vid_sig:
                st.warning("The 3D view has changed since this video was generated. Click the button to refresh it.")
        v1.download_button(f"⬇️ Download video ({vid['kind'].upper()})", vid["data"],
                           file_name=f"{res['name']}_rotation.{vid['kind']}",
                           mime="video/mp4" if vid["kind"] == "mp4" else "image/gif", key="dl_rot_video")

# ---------------------------------------------------------------- report
with tabs[3]:
    rows = [("Lightweight", "ps"), ("Baseline", "pt")] + ([("Ground truth", "gt")] if seg["gt"] is not None else [])
    reps = {n: tumor_report(seg[k]) for n, k in rows}
    r = reps["Lightweight"]
    vol_keys = ["Enhancing tumor ET (mL)", "Necrotic core (mL)", "Edema (mL)", "Tumor core TC (mL)", "Whole tumor WT (mL)"]
    cols = st.columns(5)
    for c, k in zip(cols, vol_keys):
        c.metric(k.replace(" (mL)", ""), f"{r[k]:.1f} mL")
    st.subheader("Volumes by source (mL)")
    st.dataframe(pd.DataFrame({n: [reps[n][k] for k in vol_keys] for n, _ in rows}, index=[k.replace(" (mL)", "") for k in vol_keys]).round(1))
    if "centroid" in r:
        st.write(f"**Extent (lightweight):** bounding box {r['bbox_mm']} mm · max axial diameter {r['max_axial_diameter_mm']} mm · "
                 f"necrosis fraction of core {r['necrosis_fraction_of_TC']:.0%} · centroid voxel {r['centroid']}")
        st.subheader("Resection safety margin around tumor core (lightweight)")
        mt = margin_table(seg["ps"], brain, dist["ps"])
        st.dataframe(mt, hide_index=True)
    else:
        st.info("No tumor detected by the lightweight model.")
    if res["t_ms"]:
        st.caption(f"Inference (128³ volume, {models['device'].upper()}{', flip-TTA 8x' if res['tta'] else ''}): lightweight "
                   f"**{res['s_ms']:.0f} ms** vs baseline **{res['t_ms']:.0f} ms** ({res['t_ms']/max(res['s_ms'],1e-6):.1f}x faster)")
    if seg["gt"] is not None:
        d = lambda a, b: 2.0 * np.logical_and(a, b).sum() / max(a.sum() + b.sum(), 1)
        st.caption(f"Whole-tumor Dice vs ground truth: lightweight {d(seg['ps'] > 0, seg['gt'] > 0):.3f} · baseline {d(seg['pt'] > 0, seg['gt'] > 0):.3f}")
    csv = pd.DataFrame({n: [reps[n][k] for k in vol_keys] for n, _ in rows}, index=vol_keys).round(2).to_csv()
    st.download_button("Download volumes (CSV)", csv, file_name=f"{res['name']}_volumes.csv", mime="text/csv")

# ---------------------------------------------------------------- uncertainty
with tabs[4]:
    st.write("**Where should a surgeon double-check?** Teacher-student disagreement (Jensen-Shannon divergence) and predictive entropy of the "
             "lightweight model. Cyan marks the 5% most disagreeing voxels in a band around the tumor.")
    band = ndi.binary_dilation((seg["ps"] > 0) | (seg["pt"] > 0), iterations=8)
    if band.any():
        thr = np.percentile(res["js"][band], 95)
        flag = band & (res["js"] >= thr)
        fig = Figure(figsize=(15, 4.8), facecolor="#0e1117")
        a0, a1, a2 = fig.subplots(1, 3)
        base = take(img[mi], ax_i, idx)
        for a, (u, t) in zip((a0, a1), ((res["js"], "Teacher-student JS divergence"), (res["ent"], "Lightweight predictive entropy"))):
            uu = np.where(band, u, 0)
            a.imshow(np.transpose(np.stack([gray_norm(base)] * 3, -1), (1, 0, 2)), origin="lower", alpha=0.4)
            a.imshow(np.ma.masked_less_equal(take(uu, ax_i, idx), 0).T, origin="lower", cmap="magma")
            a.set_title(t, color="white", fontsize=10); a.axis("off")
        rgb = overlay_rgb(base, take(seg["ps"], ax_i, idx), 0.35).copy()
        rgb[take(flag, ax_i, idx).T] = [0, 0.9, 1]
        a2.imshow(rgb, origin="lower"); a2.set_title("Double-check voxels (cyan)", color="white", fontsize=10); a2.axis("off")
        fig.tight_layout(); st.pyplot(fig)
        st.caption(f"{flag.sum() * SPACING**3 / 1000:.2f} mL flagged ({flag.sum()} voxels) · models disagree on {int(((seg['pt'] != seg['ps']) & band).sum())} band voxels")
    else:
        st.info("No tumor predicted.")

# ---------------------------------------------------------------- models & speed
with tabs[5]:
    if models is not None:
        tp, sp = models["teacher_params"], models["student_params"]
        c = st.columns(4)
        c[0].metric("Baseline params", f"{tp/1e6:.1f} M"); c[1].metric("Lightweight params", f"{sp/1e6:.2f} M")
        c[2].metric("Parameter reduction", f"{tp/sp:.1f}x")
        c[3].metric("This case speed-up", f"{res['t_ms']/max(res['s_ms'],1e-6):.1f}x" if res["t_ms"] else "n/a")
    if models is not None:
        st.subheader("Loaded checkpoints (validation Dice recorded at the best epoch)")
        rows = {n: {"file": Path(models["info"][k]["file"]).name, "best epoch": (models["info"][k]["epoch"] or 0) + 1,
                    **{r: round(models["info"][k]["val"][r], 3) for r in ("ET", "TC", "WT", "mean")}}
                for n, k in (("Baseline", "teacher"), ("Lightweight", "student")) if models["info"][k].get("val")}
        if rows:
            st.dataframe(pd.DataFrame(rows).T)
    rp = next((p for p in (Path(bundle) / "results_table.csv", Path(bundle) / "checkpoints" / "results_table.csv",
                           BASE / "results_table.csv", BASE / "checkpoints" / "results_table.csv") if p.exists()), None)
    st.subheader("Held-out validation results (74 patients)")
    if rp is not None:
        st.dataframe(pd.read_csv(rp, index_col=0).T)
    else:
        st.dataframe(KNOWN_RESULTS)
        st.caption("Measured in our main training run (Run 1). Place results_table.csv next to the app to show your own.")