"""
NeuroAssist - lightweight AI surgical planning assistant for brain tumor resection (Streamlit demo)

Run:   streamlit run app.py
Needs: teacher_best.pt + student_kd_best.pt (from the training notebook) next to this file, or in ./checkpoints,
       or point the sidebar field "Model bundle folder" to the folder that holds them.
Without checkpoints the app still starts and offers a clearly labelled SYNTHETIC PHANTOM (UI preview only, no AI).

Research prototype. Not a medical device. Not for clinical use.
"""
import itertools, os, re, json, time, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import ndimage as ndi
from skimage import measure
from matplotlib.figure import Figure
from matplotlib.patches import Patch

st.set_page_config(page_title="NeuroAssist", page_icon="🧠", layout="wide")

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
WEIGHTS_URL = os.environ.get("NEUROASSIST_WEIGHTS_URL", "")        # e.g. https://github.com/<org>/neuroassist/releases/download/v1.0
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
.hdr{background:linear-gradient(90deg,#0b3d4a,#0e1117);padding:14px 22px;border-radius:10px;margin-bottom:8px}
.hdr h1{margin:0;font-size:1.9rem;color:#e8f6f8}.hdr p{margin:2px 0 0 0;color:#9fc5cc}
.warn{background:#3b2a05;color:#ffd98a;padding:8px 14px;border-radius:8px;font-size:.85rem;margin-bottom:10px}
.pre{background:#4a1d1d;color:#ffb4b4;padding:8px 14px;border-radius:8px;font-size:.9rem;margin-bottom:10px}
</style>
<div class="hdr"><h1>🧠 NeuroAssist</h1><p>Pre-operative brain tumor planning · multi-sequence MRI · lightweight distilled 3D network</p></div>
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


def download_weights(base_url, dest=None):
    """Download the weight files from a GitHub Release (or any folder URL) into <app>/checkpoints/."""
    import urllib.request
    dest = Path(dest or BASE / "checkpoints"); dest.mkdir(parents=True, exist_ok=True)
    for f in WEIGHT_FILES + ["config.json", "results_table.csv"]:
        try:
            urllib.request.urlretrieve(f"{base_url.rstrip('/')}/{f}", str(dest / f))
        except Exception as e:
            if f in WEIGHT_FILES:
                raise RuntimeError(f"could not download {f}: {e}")
    return dest


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
# Sidebar: inputs
# ----------------------------------------------------------------------------------------------------------------------
st.sidebar.header("1 · Case")
bundle = st.sidebar.text_input("Model bundle folder", DEFAULT_BUNDLE)
models = load_models(bundle)
if models is None:
    if lfs_pointer_found(bundle):
        st.sidebar.error("Found a Git LFS pointer instead of the real weights. Run:  git lfs install && git lfs pull")
    st.sidebar.warning("Weights not found. Put teacher_best.pt and student_kd_best.pt in ./checkpoints/ next to app.py "
                       "(or download them below). Until then only the synthetic phantom preview works.")
    with st.sidebar.expander("Download weights from a GitHub Release"):
        url = st.text_input("Release folder URL", WEIGHTS_URL, placeholder="https://github.com/<org>/<repo>/releases/download/v1.0")
        if st.button("Download weights") and url:
            try:
                with st.spinner("Downloading (about 70 MB)…"):
                    download_weights(url)
                load_models.clear(); st.rerun()
            except Exception as e:
                st.error(str(e))
else:
    inf = models["info"]
    fmt = lambda k: (f"val mean Dice {inf[k]['val']['mean']:.3f}" if inf[k].get("val") else "loaded")
    st.sidebar.success(f"Models loaded on {models['device'].upper()}\n\nBaseline: {fmt('teacher')}\n\nLightweight: {fmt('student')}")

sources = ["Cached held-out case (.npy)", "Upload 4 NIfTI sequences", "BraTS case folder on disk", "Synthetic phantom (UI preview, no AI)"]
has_cases = bool(list(Path(bundle).glob("cases/*_img.npy")))
src = st.sidebar.radio("Case source", sources, index=0 if (models is not None and has_cases) else (1 if models is not None else 3))
tta = st.sidebar.checkbox("Flip test-time augmentation (8x slower, smoother)", value=False)

case_files = sorted(Path(bundle).glob("cases/*_img.npy")) if src == sources[0] else []
sel_case = uploads = folder = None
if src == sources[0]:
    if case_files:
        sel_case = st.sidebar.selectbox("Patient", [p.name.replace("_img.npy", "") for p in case_files])
    else:
        st.sidebar.info("No cached cases found in <bundle>/cases/. Use 'Upload 4 NIfTI sequences' or 'BraTS case folder on disk'.")
elif src == sources[1]:
    uploads = st.sidebar.file_uploader("flair / t1 / t1ce / t2 (.nii, .nii.gz), BraTS space 240x240x155", type=["nii", "gz"],
                                       accept_multiple_files=True)
elif src == sources[2]:
    folder = st.sidebar.text_input("Path to e.g. .../BraTS20_Training_001")
run = st.sidebar.button("▶ Run segmentation", type="primary")


def load_case():
    """-> dict(name, img, gt, phantom) or None (error already shown)."""
    if src == sources[0] and sel_case:
        return dict(name=sel_case, img=np.load(Path(bundle) / "cases" / f"{sel_case}_img.npy").astype(np.float32),
                    gt=np.load(Path(bundle) / "cases" / f"{sel_case}_lbl.npy"), phantom=False)
    if src == sources[1]:
        found = {}
        for f in uploads or []:
            k = find_modality(f.name)
            if k in MODALITIES or k == "seg":
                tmp = Path(tempfile.mkdtemp()) / f.name
                tmp.write_bytes(f.getvalue()); found[k] = str(tmp)
        missing = [m for m in MODALITIES if m not in found]
        if missing:
            st.error(f"Missing sequences: {missing}. File names must end in _flair/_t1/_t1ce/_t2 (.nii or .nii.gz).")
            return None
        vols = {m: load_nii(found[m]) for m in MODALITIES}
        if any(v.shape != (240, 240, 155) for v in vols.values()):
            st.error("Volumes must be in BraTS space (240x240x155). Register/resample first.")
            return None
        gt = preprocess_label(load_nii(found["seg"])) if "seg" in found else None
        return dict(name="Uploaded case", img=preprocess_modalities(vols), gt=gt, phantom=False)
    if src == sources[2] and folder:
        d = Path(folder)
        files = {find_modality(f.name): f for f in d.glob("*.nii*") if find_modality(f.name)} if d.is_dir() else {}
        if not all(m in files for m in MODALITIES):
            st.error("Folder not found or it lacks the four sequences.")
            return None
        gt = preprocess_label(load_nii(str(files["seg"]))) if "seg" in files else None
        return dict(name=d.name, img=preprocess_modalities({m: load_nii(str(files[m])) for m in MODALITIES}), gt=gt, phantom=False)
    if src == sources[3]:
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
    st.info("👈 Select a case in the sidebar and press **Run segmentation**. "
            "The app compares the heavy baseline with the lightweight distilled model, then offers margin, 3D, trajectory and uncertainty tools.")
    st.stop()
if res["phantom"]:
    st.markdown('<div class="pre">SYNTHETIC PHANTOM: UI preview only. These "predictions" are smoothed copies of a made-up mask; no AI model ran.</div>',
                unsafe_allow_html=True)

st.sidebar.header("2 · View")
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
            show_plotly(make_3d_figure(seg[key], brain, dist.get(key), margin, traj, f"{res['name']} · {show}"))

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
