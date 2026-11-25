import os
import glob
import tempfile
import datetime as dt
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import viz

# ---------------- page config ----------------
st.set_page_config(
    page_title="Early Alzheimer's Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "final_oasis1_q8_cdr1_clin_mmse.pkl"

# ---------------- Clean Light Indigo SaaS Theme (inspired by preview.png) ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #FFFFFF;
    --bg-subtle: #F9FAFB;
    --surface: #FFFFFF;
    --border: #E5E7EB;
    --border-strong: #D1D5DB;
    --text: #111827;
    --muted: #6B7280;
    --indigo: #4F46E5;
    --indigo-bright: #6366F1;
    --indigo-soft: #EEF2FF;
    --indigo-border: #C7D2FE;
    --green-bg: #ECFDF5;
    --green-text: #15803D;
    --red-bg: #FEF2F2;
    --red-text: #DC2626;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(17, 24, 39, 0.04), 0 1px 2px rgba(17, 24, 39, 0.02);
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer { visibility: hidden; }

header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.9) !important;
}

.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-subtle) !important;
    border-right: 1px solid var(--border);
    width: 21.5rem !important;
    min-width: 21.5rem !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.1rem;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: var(--text);
}

.sb-brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding-bottom: 0.9rem;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.sb-logo {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    color: #FFFFFF !important;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    box-shadow: 0 2px 6px rgba(79, 70, 229, 0.28);
}
.sb-brand-name {
    font-weight: 700;
    font-size: 0.98rem;
    color: var(--text);
    line-height: 1.15;
}
.sb-brand-sub {
    font-size: 0.74rem;
    color: var(--muted) !important;
}

.sb-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted) !important;
    margin: 1rem 0 0.35rem 0;
}

/* Top Pill Navigation (st.tabs styled like Overview | Accounts | Pipeline) */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    background: transparent;
    padding: 0 0 0.85rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.42rem 0.9rem;
    font-weight: 500;
    font-size: 0.86rem;
    border-radius: 8px;
    color: var(--muted) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-subtle) !important;
    color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--indigo-soft) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    border-color: var(--indigo-border) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* Page Header */
.page-head {
    margin-bottom: 1.15rem;
}
.page-title {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1.2;
    margin: 0;
}
.page-sub {
    font-size: 0.86rem;
    color: var(--muted);
    margin-top: 0.25rem;
}

/* Segmented meta strip */
.seg-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.35rem;
}
.seg-group {
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: var(--surface);
}
.seg-item {
    padding: 0.34rem 0.78rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--muted);
    border-right: 1px solid var(--border);
    background: var(--surface);
}
.seg-item:last-child {
    border-right: none;
}
.seg-item.active {
    background: var(--indigo-soft);
    color: var(--indigo);
    font-weight: 600;
}
.pill-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem 0.72rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text);
}
.pill-chip.warn {
    border-color: #FECACA;
    background: var(--red-bg);
    color: var(--red-text);
}
.pill-chip.ok {
    border-color: #BBF7D0;
    background: var(--green-bg);
    color: var(--green-text);
}

/* Section Title */
.sec-title {
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--text);
    margin: 0.2rem 0 0.75rem 0;
}

/* KPI Cards (matching preview.png) */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.2rem;
    box-shadow: var(--shadow);
    height: 100%;
}
.kpi-top {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.45rem;
}
.kpi-check {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: var(--green-bg);
    color: var(--green-text);
    font-size: 0.65rem;
    font-weight: 700;
}
.kpi-val {
    font-size: 1.95rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums;
    color: var(--text);
    line-height: 1.15;
    margin-bottom: 0.55rem;
}
.kpi-foot {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex-wrap: wrap;
}
.delta-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.16rem 0.5rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
}
.delta-pill.up {
    background: var(--green-bg);
    color: var(--green-text);
}
.delta-pill.down {
    background: var(--red-bg);
    color: var(--red-text);
}
.delta-pill.neutral {
    background: var(--indigo-soft);
    color: var(--indigo);
}
.kpi-sub {
    font-size: 0.76rem;
    color: var(--muted);
}

/* Verdict Banner Card */
.verdict-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--v-accent);
    border-radius: var(--radius);
    padding: 1.2rem 1.35rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.1rem;
}
.verdict-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
}
.verdict-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.verdict-desc {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.55;
    margin: 0;
}

/* Onboarding Step Cards */
.onboard-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.3rem;
    box-shadow: var(--shadow);
    height: 100%;
}
.onboard-num {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem 0.52rem;
    border-radius: 999px;
    background: var(--indigo-soft);
    color: var(--indigo);
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 0.55rem;
}
.onboard-card h4 {
    font-size: 0.96rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.3rem 0;
}
.onboard-card p {
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.5;
    margin: 0;
}

/* Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow);
}

/* Buttons */
.stButton > button[kind="primary"] {
    background-color: var(--indigo) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.52rem 1rem !important;
    box-shadow: 0 1px 2px rgba(79, 70, 229, 0.25);
}
.stButton > button[kind="primary"]:hover {
    background-color: #4338CA !important;
}
.stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
}
.stDownloadButton > button:hover {
    border-color: var(--indigo-bright) !important;
    color: var(--indigo) !important;
    background: var(--indigo-soft) !important;
}

[data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    color: var(--text) !important;
}
[data-testid="stMetricLabel"] p {
    white-space: normal;
    color: var(--muted) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- cached helpers ----------------
@st.cache_data
def model_list(stamp):
    return viz.list_models(".")


@st.cache_resource
def get_model(path, stamp):
    with open(path, "rb") as fh:
        return pickle.load(fh)


@st.cache_resource
def get_cnn(weights, device):
    import torch
    from CNN_features import build_model
    return build_model(weights, torch.device(device))


def extract_features(path, weights, device):
    import torch
    from CNN_features import load_volume
    net = get_cnn(weights, device)
    with torch.no_grad():
        return net(load_volume(path).to(device)).cpu().numpy().ravel()


@st.cache_data
def read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def panel(fig, title=None, subtitle=None):
    with st.container(border=True):
        if title:
            st.markdown(
                f'<div style="font-weight:600; font-size:0.95rem; color:#111827; margin-bottom:0.1rem;">'
                f'{title} <span class="kpi-check">✓</span></div>',
                unsafe_allow_html=True,
            )
        if subtitle:
            st.markdown(
                f'<div style="font-size:0.8rem; color:#6B7280; margin-bottom:0.6rem;">{subtitle}</div>',
                unsafe_allow_html=True,
            )
        st.pyplot(fig, use_container_width=True)


# ---------------- state & discovery ----------------
stamp = tuple(sorted((p, os.path.getmtime(p)) for p in glob.glob("*.pkl")))
models = model_list(stamp)
st.session_state.setdefault("hist", [])

# ---------------- sidebar ----------------
sb = st.sidebar
sb.markdown(
    """
<div class="sb-brand">
    <div class="sb-logo">🧠</div>
    <div>
        <div class="sb-brand-name">Alzheimer's Screening</div>
        <div class="sb-brand-sub">Quantum-classical MRI scoring</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

sb.markdown('<div class="sb-label">1 · Patient</div>', unsafe_allow_html=True)
initials = sb.text_input(
    "Initials",
    max_chars=4,
    help="Letters only. Used to label the report. No full names are stored.",
).strip().upper()
c_sex, c_date = sb.columns(2)
sex = c_sex.selectbox("Sex", ["Not stated", "Female", "Male"])
scan_date = c_date.date_input("Scan date", value=dt.date.today())

if not models:
    st.markdown('<div class="page-head"><h1 class="page-title">Overview</h1></div>', unsafe_allow_html=True)
    st.error("No trained model files (.pkl) were found in this folder. Run a training script first.")
    st.stop()

sb.markdown('<div class="sb-label">2 · Scan</div>', unsafe_allow_html=True)
mode = sb.radio(
    "Source",
    ["Upload a registered scan", "Use a dataset subject (demo)"],
    label_visibility="collapsed",
)
demo = mode.startswith("Use")
up, confirm, sel_id, ds_name, hdf = None, False, None, None, None

paths = [m["path"] for m in models]
meta = {m["path"]: m for m in models}
idx = next((i for i, p in enumerate(paths) if os.path.basename(p) == DEFAULT_MODEL), 0)

sb.markdown('<div class="sb-label">3 · Model</div>', unsafe_allow_html=True)
model_path = sb.selectbox(
    "Model",
    paths,
    index=idx,
    format_func=lambda p: meta[p]["label"],
    label_visibility="collapsed",
)
mi = meta[model_path]
sb.caption(f"Trained on {mi['dataset']} · Inputs: {mi['inputs']}")

if demo:
    ds_name = sb.selectbox(
        "Dataset",
        ["OASIS-1", "OASIS-2"],
        index=1 if mi["dataset"] == "OASIS-2" else 0,
        label_visibility="collapsed",
    )
    n = "1" if ds_name == "OASIS-1" else "2"
    fdf, mdf = read_csv(f"oasis{n}_features.csv"), read_csv(f"oasis{n}_cleaned_metadata.csv")
    if fdf is None or mdf is None:
        sb.error(f"oasis{n}_features.csv or oasis{n}_cleaned_metadata.csv is missing from this folder.")
        st.stop()
    if mi["hippo"]:
        hdf = read_csv("hippo_features.csv")
        if hdf is None:
            sb.error("hippo_features.csv is missing from this folder.")
            st.stop()
    mids, hids = set(mdf["ID"]), (set(hdf["ID"]) if hdf is not None else None)
    ids = [i for i in fdf["ID"] if i in mids and (hids is None or i in hids)]
    sel_id = sb.selectbox("Subject ID", ids)
    mrow = mdf[mdf["ID"] == sel_id].iloc[0]
    age, nwbv = int(round(mrow["Age"])), float(mrow["nWBV"])
    mmse = float(mrow["MMSE"]) if "MMSE" in mrow and not pd.isna(mrow["MMSE"]) else None
    sb.caption(
        f"Age {age}, nWBV {nwbv:.3f}, MMSE {mmse} from dataset metadata. "
        "Dataset subjects may inflate apparent performance."
    )
else:
    up = sb.file_uploader("Registered scan (.nii.gz)", type=["gz", "nii"])
    confirm = sb.checkbox("Skull-stripped, normalised and registered like the training scans")
    c_age, c_vol = sb.columns(2)
    age = c_age.number_input("Age (years)", min_value=18, max_value=100, value=73)
    nwbv = c_vol.number_input(
        "nWBV",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.001,
        format="%.3f",
        help="Normalised whole-brain volume from the scan report or a segmentation tool. OASIS values run from about 0.64 to 0.89.",
    )
    mmse = (
        sb.number_input("MMSE score", min_value=0, max_value=30, value=27)
        if mi.get("clin_names") == ["nWBV", "Age", "MMSE"]
        else None
    )

with sb.expander("Advanced"):
    weights = st.text_input("MedicalNet weights file", value="resnet_18_23dataset.pth")
    device = st.selectbox("Device", ["cpu", "cuda"])
    layout = st.selectbox(
        "Scan axis layout (viewer only)",
        list(viz.LAYOUTS),
        help="Which view each array axis of the file gives. The prediction does not use this.",
    )
    flip_ud = st.checkbox("Flip viewer images up/down")
    flip_lr = st.checkbox("Flip viewer images left/right")

run = sb.button("Run prediction", type="primary", use_container_width=True)
sb.caption("Nothing is saved to disk. Scans stay in memory for this session.")

# ---------------- run ----------------
if run:
    problems = []
    if not initials.isalpha():
        problems.append("Enter the patient’s initials (letters only).")
    if not demo and up is None:
        problems.append("Upload a registered scan.")
    if not demo and up is not None and not confirm:
        problems.append("Confirm that the scan went through the same preprocessing as the training scans.")
    if not demo and mi["hippo"]:
        problems.append(
            "This model uses hippocampus features from hippo_features.csv, and the app cannot compute them from an uploaded scan. "
            "Use the dataset demo or a model without hippocampus features."
        )
    if mi.get("clin_names") == ["nWBV", "Age", "MMSE"] and mmse is None:
        problems.append("This model needs an MMSE score.")
    if demo and ds_name != mi["dataset"] and mi["dataset"] != "OASIS-1 + OASIS-2":
        problems.append(f"This model was trained on {mi['dataset']}, but the subject is from {ds_name}. Pick the matching model.")

    for p in problems:
        st.error(p)

    if not problems:
        tmp = None
        try:
            with st.spinner("Reading the scan and running the quantum circuit..."):
                m = get_model(model_path, stamp)
                if demo:
                    cnn = fdf[fdf["ID"] == sel_id].iloc[0][[c for c in fdf if c[0] == "f" and c[1:].isdigit()]].values.astype(float)
                    hip = hdf[hdf["ID"] == sel_id].iloc[0][m["hippo"]].values.astype(float) if m.get("hippo") else None
                    feats = viz.assemble(m, cnn, hip)
                    spath = viz.find_scan(sel_id)
                    truth = f"CDR {mrow['CDR']}" if "CDR" in mrow and not pd.isna(mrow["CDR"]) else "unknown"
                else:
                    suffix = ".nii" if up.name.lower().endswith(".nii") else ".nii.gz"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                        f.write(up.getbuffer())
                    tmp = spath = f.name
                    truth = None
                    feats = extract_features(spath, weights, device)

                vol = viz.load_volume_nib(spath) if spath else None
                e = viz.explain(m, feats, age, nwbv, mmse)

            warn = []
            if e["ood"] > 0.10:
                warn.append("This scan’s features differ a lot from the training scans. Check that it was preprocessed and registered the same way.")
            if np.any(np.abs(e["clin_z"]) > 3):
                warn.append("Age or brain volume is far outside the range the model was trained on.")
            if vol is not None and (vol > 0).mean() < 0.05:
                warn.append("Very little of the scan contains signal. It may be empty or badly registered.")
            ref = "mni152_1mm.nii.gz"
            if vol is not None and os.path.exists(ref):
                import nibabel as nib
                rs = tuple(nib.load(ref).shape[:3])
                if rs != tuple(vol.shape[:3]):
                    warn.append(f"The scan size {tuple(vol.shape[:3])} differs from the registration template {rs}.")

            positive = e["p"] >= e["tau"]
            info = {
                "Patient": initials,
                "Sex": sex,
                "Scan date": scan_date,
                "Age": age,
                "Brain volume (nWBV)": f"{nwbv:.3f}",
                "Model": mi["label"],
                "Result": "Impairment signal" if positive else "No impairment signal",
                "Model score": f"{e['p']:.3f}",
                "Decision threshold": f"{e['tau']:.3f}",
                "Run at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

            with viz.theme("light"):
                lfigs = [viz.fig_ruler(e, m), viz.fig_angles(e), viz.fig_zscores(e)]
                pdf = viz.make_pdf(
                    info,
                    lfigs,
                    extra=dict(e=e, positive=positive, side=mi["target"].split(" vs ")[1], warn=warn, vol=vol),
                )
            for g in lfigs:
                viz.plt.close(g)

            st.session_state["res"] = dict(
                e=e,
                info=info,
                vol=vol,
                warn=warn,
                positive=positive,
                truth=truth,
                mi=mi,
                pdf=pdf,
                figs=[viz.fig_ruler(e, m), viz.fig_angles(e), viz.fig_zscores(e)],
                source=("dataset subject " + sel_id) if demo else up.name,
            )
            st.session_state["hist"].append(
                {
                    "Run at": info["Run at"],
                    "Patient": initials,
                    "Model": mi["dataset"],
                    "Score": round(e["p"], 3),
                    "Threshold": round(e["tau"], 3),
                    "Result": info["Result"],
                }
            )
        except ImportError as ex:
            st.error(f"A required package is missing: {ex}. Install torch and monai to read uploaded scans, or use the dataset demo.")
        except Exception as ex:
            st.error(f"The prediction failed: {ex}")
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

res = st.session_state.get("res")

# ---------------- top pill navigation ----------------
tab_names = ["Overview", "Scan viewer", "Why this score", "Model evidence", "About & limits"]
if st.session_state["hist"]:
    tab_names.append(f"Session log ({len(st.session_state['hist'])})")

tabs = st.tabs(tab_names)
t_over, t_scan, t_why, t_perf, t_about = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]

# ---------------- OVERVIEW TAB ----------------
with t_over:
    if res is None:
        st.markdown(
            f"""
<div class="page-head">
    <h1 class="page-title">Overview</h1>
    <div class="page-sub">{mi['dataset']} model · {mi['inputs']} · ready for screening</div>
</div>
<div class="seg-strip">
    <div class="seg-group">
        <span class="seg-item active">{mi['dataset']}</span>
        <span class="seg-item">8-Qubit VQC</span>
        <span class="seg-item">ResNet-18</span>
    </div>
    <span class="pill-chip warn">Research prototype</span>
    <span class="pill-chip">Not a diagnosis</span>
</div>
<div class="sec-title">Getting started</div>
""",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        cards = [
            (c1, "Step 1", "Add patient details", "Enter initials, sex, and scan date in the left sidebar. Initials only label the report."),
            (c2, "Step 2", "Select MRI source", "Upload a registered .nii.gz scan or choose an OASIS benchmark subject."),
            (c3, "Step 3", "Run and review", "Click Run prediction to inspect the score, decision ruler, slices, and model evidence."),
        ]
        for col, badge, title, desc in cards:
            col.markdown(
                f"""
<div class="onboard-card">
    <div class="onboard-num">{badge}</div>
    <h4>{title}</h4>
    <p>{desc}</p>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        e, info = res["e"], res["info"]
        verdict = "Impairment signal" if res["positive"] else "No impairment signal"
        v_accent = "#DC2626" if res["positive"] else "#16A34A"
        pill_cls = "warn" if res["positive"] else "ok"
        side = res["mi"]["target"].split(" vs ")[1]
        meaning = (
            f"The score sits above the decision threshold, closer to subjects with {side}. This calls for clinical follow-up. It is not a diagnosis."
            if res["positive"]
            else "The score sits below the decision threshold. This does not rule out early disease, because the model misses some mild cases."
        )
        margin = e["p"] - e["tau"]
        margin_cls = "down" if res["positive"] else "up"
        margin_arrow = "↑" if margin >= 0 else "↓"
        truth_seg = f'<span class="seg-item active">{res["truth"]}</span>' if res["truth"] else ""

        st.markdown(
            f"""
<div class="page-head">
    <h1 class="page-title">Overview</h1>
    <div class="page-sub">Patient {info['Patient']} · Age {info['Age']} · {info['Scan date']} · {res['mi']['dataset']} model</div>
</div>
<div class="seg-strip">
    <div class="seg-group">
        <span class="seg-item active">{res['mi']['dataset']}</span>
        <span class="seg-item">nWBV {info['Brain volume (nWBV)']}</span>
        <span class="seg-item">{res['source']}</span>
        {truth_seg}
    </div>
    <span class="pill-chip {pill_cls}">{verdict}</span>
    <span class="pill-chip">Research prototype · Not a diagnosis</span>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="verdict-box" style="--v-accent:{v_accent};">
    <div class="verdict-head">
        <h3 class="verdict-title">{verdict}</h3>
        <span class="pill-chip {pill_cls}">Score {e['p']:.3f} vs. τ {e['tau']:.3f}</span>
    </div>
    <p class="verdict-desc">{meaning}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        for w in res["warn"]:
            st.warning(w)

        st.markdown('<div class="sec-title">Screening metrics</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        svm_val = f"{e['ps']:.3f}" if e["ps"] is not None else "N/A"

        k1.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-top">Model score <span class="kpi-check">✓</span></div>
    <div class="kpi-val">{e['p']:.3f}</div>
    <div class="kpi-foot"><span class="delta-pill {margin_cls}">{margin_arrow} {margin:+.3f}</span><span class="kpi-sub">vs. threshold</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        k2.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-top">Decision threshold <span class="kpi-check">✓</span></div>
    <div class="kpi-val">{e['tau']:.3f}</div>
    <div class="kpi-foot"><span class="delta-pill neutral">τ cutoff</span><span class="kpi-sub">sensitivity-tuned</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        k3.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-top">Brain volume (nWBV) <span class="kpi-check">✓</span></div>
    <div class="kpi-val">{info['Brain volume (nWBV)']}</div>
    <div class="kpi-foot"><span class="delta-pill neutral">Age {info['Age']}</span><span class="kpi-sub">clinical input</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        k4.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-top">Classical SVM check <span class="kpi-check">✓</span></div>
    <div class="kpi-val">{svm_val}</div>
    <div class="kpi-foot"><span class="delta-pill neutral">Baseline</span><span class="kpi-sub">same input features</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        c_left, c_right = st.columns([1.45, 1])
        with c_left:
            panel(
                res["figs"][0],
                "Where this scan falls",
                viz.closeness(e["p"], e["tau"])
                + " Score is bounded between 0.27 and 0.73 (not a probability of disease).",
            )
        with c_right:
            if res["vol"] is not None:
                panel(
                    viz.fig_thumb(res["vol"], viz.LAYOUTS[layout], flip_ud, flip_lr),
                    "Scan preview",
                    f"Source: {res['source']}",
                )
            else:
                with st.container(border=True):
                    st.markdown(
                        '<div style="font-weight:600; font-size:0.95rem; color:#111827;">Scan preview</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("No registered scan file was found for this subject. The prediction used the saved features.")
                    st.caption(f"Source: {res['source']}")

        if res["truth"]:
            st.info(f"Known label in the dataset: {res['truth']}. This subject was part of the model’s training data.")

        d1, d2, _ = st.columns([1, 1, 2.5])
        d1.download_button(
            "Download report (PDF)",
            res["pdf"],
            file_name=f"report_{info['Patient']}_{dt.date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        d2.download_button(
            "Download result (CSV)",
            pd.DataFrame([info]).to_csv(index=False),
            file_name=f"result_{info['Patient']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------------- SCAN VIEWER TAB ----------------
with t_scan:
    st.markdown(
        '<div class="page-head"><h1 class="page-title">Scan viewer</h1>'
        '<div class="page-sub">Orthogonal MRI slice inspection &amp; voxel intensity distribution</div></div>',
        unsafe_allow_html=True,
    )
    if res is None:
        st.info("Run a prediction to open the scan viewer.")
    elif res["vol"] is None:
        st.info("No registered scan file was found for this subject, so there is nothing to display.")
    else:
        v = res["vol"]
        order = viz.plane_order(viz.LAYOUTS[layout])
        scols = st.columns(3)
        fracs = {p: scols[i].slider(p, 0.0, 1.0, 0.5, key=f"slider_{p}") for i, p in enumerate(order)}
        f = viz.fig_slices(v, fracs, viz.LAYOUTS[layout], flip_ud, flip_lr)
        panel(f, "Orthogonal slices", f"Axis layout: {layout}")
        viz.plt.close(f)

        a, b = st.columns([3, 2])
        with a:
            f = viz.fig_hist(v)
            panel(f, "Intensity distribution", "Voxel intensity histogram across the 3D volume")
            viz.plt.close(f)
        with b.container(border=True):
            st.markdown("**Volume summary**")
            st.write(f"**Dimensions:** {v.shape[0]} × {v.shape[1]} × {v.shape[2]} voxels")
            st.write(f"**Signal coverage:** {(v > 0).mean():.0%} of voxels contain signal")
            st.caption("This view is a quick visual check. The model uses 512 numbers extracted from the full 3D volume, not these slices.")

# ---------------- WHY THIS SCORE TAB ----------------
with t_why:
    st.markdown(
        '<div class="page-head"><h1 class="page-title">Why this score</h1>'
        '<div class="page-sub">Qubit rotation angles, training z-scores, and variational circuit</div></div>',
        unsafe_allow_html=True,
    )
    if res is None:
        st.info("Run a prediction to see how the score was reached.")
    else:
        e = res["e"]
        st.write(
            f"The circuit receives {len(e['angles'])} numbers. {len(e['angles']) - e['NC']} come from the scan "
            f"and {e['NC']} are age and brain volume. Each number rotates one qubit, entangling layers mix them, "
            "and measuring the first qubit gives the score."
        )
        a, b = st.columns(2)
        with a:
            panel(res["figs"][1], "Qubit rotation angles", "Encoded feature rotations across 8 qubits")
        with b:
            panel(res["figs"][2], "Compared with training subjects", "Standardised z-scores (orange = >3 SD)")

        r1, r2 = st.columns([2, 3])
        with r1:
            f = viz.fig_radar(e)
            panel(f, "Qubit profile")
            viz.plt.close(f)
        with r2:
            try:
                f = viz.fig_circuit(e)
                panel(f, "Variational circuit")
                viz.plt.close(f)
            except Exception:
                with st.container(border=True):
                    st.markdown("**Variational circuit**")
                    st.caption("The circuit diagram could not be drawn here.")

# ---------------- MODEL EVIDENCE TAB ----------------
with t_perf:
    st.markdown(
        f'<div class="page-head"><h1 class="page-title">Model evidence</h1>'
        f'<div class="page-sub">{mi["label"]} · subject-wise cross-validation benchmarks</div></div>',
        unsafe_allow_html=True,
    )
    m = get_model(model_path, stamp)
    perf = viz.performance(m, model_path)
    if perf is None:
        st.info("No saved validation results for this model. Train it again with the current training script to store them.")
    else:
        r = perf["rows"]["VQC"]
        st.caption(f"{perf['n']} samples, subject-wise cross-validation. Metrics {perf['source']}.")
        q1, q2, q3, q4 = st.columns(4)
        with q1.container(border=True):
            st.metric("Accuracy", f"{r['acc']:.1%}", f"{(r['acc'] - perf['majority']) * 100:+.1f} pts vs baseline")
        with q2.container(border=True):
            st.metric("AUC", f"{r['auc']:.2f}")
        with q3.container(border=True):
            st.metric("Sensitivity", f"{r['sens']:.1%}")
        with q4.container(border=True):
            st.metric("Specificity", f"{r['spec']:.1%}")

        a, b = st.columns([3, 2])
        with a:
            f = viz.fig_metrics(perf)
            panel(f, "Quantum model and classical baseline")
            viz.plt.close(f)
        if "oof" in m and "y" in m:
            with b:
                f = viz.fig_roc(m)
                panel(f, "ROC curve")
                viz.plt.close(f)
        if "SVM" in perf["rows"] and perf["rows"]["SVM"]["auc"] >= perf["rows"]["VQC"]["auc"]:
            st.info("The classical SVM scores at least as well as the quantum model on the same inputs, so the circuit shows no clear advantage.")

    csvs = sorted(glob.glob("*_results.csv"))
    if csvs:
        st.markdown('<div class="sec-title">Fold-by-fold results</div>', unsafe_allow_html=True)
        p1_, p2_ = st.columns(2)
        pick = p1_.selectbox("Results file", csvs)
        metric = p2_.selectbox(
            "Metric",
            [a for a, _ in viz.METRIC_NAMES if a != "bal"] + ["f1"],
            format_func=lambda a: dict(viz.METRIC_NAMES + [("f1", "F1")])[a],
        )
        rdf = read_csv(pick)
        if rdf is not None and {"fold", "model", metric} <= set(rdf.columns):
            f = viz.fig_folds(rdf, metric)
            panel(f, "Per-fold performance")
            viz.plt.close(f)

    ab = read_csv("ablation_summary.csv")
    if ab is not None:
        f = viz.fig_ablation(ab)
        panel(f, "Design experiments", "Accuracy when one setting changes (differences are small vs. fold variation)")
        viz.plt.close(f)

# ---------------- ABOUT & LIMITS TAB ----------------
with t_about:
    st.markdown(
        '<div class="page-head"><h1 class="page-title">About &amp; limits</h1>'
        '<div class="page-sub">Architecture overview and clinical caveats</div></div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**How it works**")
        st.markdown(
            """
1. The MRI is skull-stripped, normalised and registered to a standard brain.
2. A 3D ResNet-18 pretrained on medical images turns the volume into 512 numbers.
3. A scaler and selector keep the most informative numbers (CNN features, plus hippocampus features for some models). Age and brain volume are added.
4. Eight values rotate eight qubits. A 4-layer variational circuit is simulated and the first qubit is measured.
5. A threshold chosen on validation data separates “impairment signal” from “no impairment signal”. It favours sensitivity.
"""
        )
    with b.container(border=True):
        st.markdown("**Limits to keep in mind**")
        st.markdown(
            """
- It is a research prototype, not a medical device. It gives a score, not a diagnosis.
- It was trained on a few hundred subjects from OASIS-1 and OASIS-2. Other scanners and hospitals are untested.
- Early-stage detection is hard here. Accuracy is modest, and a classical SVM did about as well as the circuit.
- OASIS-1 and OASIS-2 scans differ a lot, so use the model that matches the scan’s source.
- Age and brain volume carry much of the signal. Scan features alone perform worse.
- The quantum circuit is simulated on a normal computer.
"""
        )

# ---------------- SESSION LOG TAB ----------------
if st.session_state["hist"]:
    with tabs[5]:
        st.markdown(
            '<div class="page-head"><h1 class="page-title">Session log</h1>'
            '<div class="page-sub">Predictions run during this browser session</div></div>',
            unsafe_allow_html=True,
        )
        h = pd.DataFrame(st.session_state["hist"])
        st.dataframe(h, use_container_width=True)
        st.download_button(
            "Download session log (CSV)",
            h.to_csv(index=False),
            file_name="session_log.csv",
            mime="text/csv",
        )