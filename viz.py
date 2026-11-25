import io, os, glob, pickle, contextlib, textwrap, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score
from predict import qprob, ru_full_prob

DARK = dict(ink="#E6EDF3", muted="#8CA0B3", grid="#2A3F55", wash="#1D2E40", teal="#4DB6AC", signal="#F2A33A", panel="#142130", scan="#080F16")
LIGHT = dict(ink="#15272D", muted="#5E7279", grid="#D3DEE0", wash="#E4EEEF", teal="#1F6F78", signal="#C2571A", panel="#FFFFFF", scan="#0E1A1E")
PAL = dict(LIGHT)

REPORTED = {
    "final_oasis1_q8_clin.pkl": dict(n=228, majority=0.579, VQC=dict(acc=.693, bal=.701, sens=.750, spec=.652, auc=.750)),
    "final_oasis1_q8_cdr1_clin.pkl": dict(n=162, majority=0.815, VQC=dict(acc=.753, bal=.797, sens=.867, spec=.727, auc=.796)),
    "oasis2_visits_q8.pkl": dict(n=357, majority=0.563, VQC=dict(acc=.622, bal=.623, sens=.628, spec=.617, auc=.648), SVM=dict(acc=.667, bal=.660, sens=.609, spec=.711, auc=.690)),
}

def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": PAL["grid"], "axes.labelcolor": PAL["ink"], "xtick.color": PAL["muted"],
                         "ytick.color": PAL["muted"], "text.color": PAL["ink"], "figure.facecolor": PAL["panel"],
                         "axes.facecolor": PAL["panel"], "savefig.facecolor": PAL["panel"], "axes.titlecolor": PAL["ink"]})
style()

@contextlib.contextmanager
def theme(mode):
    old = dict(PAL)
    PAL.update(LIGHT if mode == "light" else DARK)
    style()
    try:
        yield
    finally:
        PAL.clear(); PAL.update(old); style()

def describe_model(path, m=None):
    n = os.path.basename(path)
    ds = "OASIS-2" if n.startswith("oasis2") else ("OASIS-1 + OASIS-2" if "both" in n else "OASIS-1")
    if "cdr1" in n: target = "normal vs clear dementia (CDR 1 or higher)"
    elif "early" in n: target = "normal vs very mild impairment (CDR 0.5)"
    else: target = "normal vs impaired (CDR 0.5 or higher)"
    hc = (m or {}).get("hippo")
    if hc:
        scan = "hippocampus features" if m["NF"] == len(hc) else "CNN and hippocampus features"
        inputs = scan + (", age and brain volume" if m.get("NC") == 2 else "")
    else:
        if n.endswith("_plain.pkl") or "clin" not in n and not n.startswith("oasis2"):
            inputs = "CNN features only"
        elif m and m.get("NC") == 3:
            names = m.get("clin_names") or []
            inputs = "CNN features, age, brain volume and MMSE" if names == ["nWBV", "Age", "MMSE"] else "CNN features, age, brain volume and volume slope"
        else:
            inputs = "CNN features, age and brain volume"
    kind = "all visits" if "visits" in n else ""
    label = f"{ds}: {target}" + (f", {kind}" if kind else "") + f" ({inputs})"
    return dict(label=label, dataset=ds, target=target, inputs=inputs)

def list_models(folder="."):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.pkl"))):
        try:
            m = pickle.load(open(p, "rb"))
        except Exception:
            continue
        if not isinstance(m, dict) or "w" not in m or "tau" not in m or "m1" in m:
            continue
        out.append(dict(path=p, nc=int(m["NC"]), q=int(np.shape(m["w"])[1]), has_perf="oof" in m,
                        hippo=bool(m.get("hippo")), clin_names=m.get("clin_names") or m.get("clin"),
                        **describe_model(p, m)))
    return out

def assemble(m, cnn, hip=None):
    """Build the feature vector in the same order as training: CNN features, then hippocampus features."""
    hc = m.get("hippo") or []
    if hc and m["NF"] == len(hc): return np.asarray(hip, float)      # "hippoonly" model
    return np.concatenate([np.asarray(cnn, float), np.asarray(hip, float)]) if hc else np.asarray(cnn, float)

def explain(m, feats, age, nwbv, mmse=None):
    NF, NC = m["NF"], m["NC"]
    clinical = {2: [nwbv, age], 3: [nwbv, age, mmse]}
    x = np.concatenate([np.asarray(feats, float).ravel()] + ([clinical[NC]] if NC in clinical and all(v is not None for v in clinical[NC]) else []))
    zs = m["sc"].transform(x[None, :])
    idx = m["sel"].get_support(indices=True)
    z = np.clip(m["mm"].transform(np.hstack([m["sel"].transform(zs[:, :NF]), zs[:, NF:]])), 0, np.pi)[0]
    p = ru_full_prob(m, z) if m.get("ansatz") == "ru_full" else qprob(m["w"], z)
    ps = float(m["svm"].predict_proba(z[None, :])[0, 1]) if "svm" in m else None
    hc = m.get("hippo") or []
    n_cnn = NF - len(hc)
    clinical_names = {2: ["Brain volume (nWBV)", "Age"], 3: ["Brain volume (nWBV)", "Age", "MMSE"]}
    names = [f"CNN feature {i}" if i < n_cnn else f"Hippocampus: {hc[i - n_cnn]}" for i in idx] + clinical_names.get(NC, [])
    return dict(p=p, ps=ps, tau=float(m["tau"]), angles=z, names=names, zsel=np.concatenate([zs[0, idx], zs[0, NF:]]),
                ood=float((np.abs(zs[0, :NF]) > 4).mean()), clin_z=zs[0, NF:], w=np.asarray(m["w"]), NC=NC)

def closeness(p, tau):
    d = abs(p - tau)
    if d < 0.03: return "Very close to the threshold. Treat this result as uncertain."
    if d < 0.08: return "Moderately close to the threshold."
    return "Clearly on this side of the threshold."

def fig_ruler(e, m):
    lo, hi = 0.25, 0.75
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axvspan(lo, e["tau"], color=PAL["teal"], alpha=.12)
    ax.axvspan(e["tau"], hi, color=PAL["signal"], alpha=.12)
    if "oof" in m and "y" in m:
        bins = np.linspace(lo, hi, 17)
        for c, col, lab in [(0, PAL["teal"], "Normal subjects (validation)"), (1, PAL["signal"], "Impaired subjects (validation)")]:
            ax.hist(np.clip(m["oof"][m["y"] == c], lo, hi), bins=bins, density=True, histtype="stepfilled", alpha=.35, color=col, label=lab)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.3)
    ax.axvline(e["tau"], color=PAL["ink"], lw=1.1, ls="--")
    right = e["tau"] > (lo + hi) / 2
    ax.text(e["tau"] + (-.004 if right else .004), top * 1.22, "threshold", ha="right" if right else "left", va="top", fontsize=8, color=PAL["muted"])
    ax.axvline(e["p"], color=PAL["ink"], lw=3)
    ax.scatter([e["p"]], [top * 1.3], marker="v", s=90, color=PAL["ink"], zorder=5, clip_on=False)
    ax.text(e["p"], top * 1.36, "this scan", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(lo, hi); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Model score (this circuit can only output values between 0.27 and 0.73)")
    ax.text(lo + .005, top * .04, "no impairment signal", color=PAL["teal"], fontsize=8, va="bottom")
    ax.text(hi - .005, top * .04, "impairment signal", color=PAL["signal"], fontsize=8, ha="right", va="bottom")
    fig.tight_layout()
    return fig

def fig_angles(e):
    n = len(e["angles"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * n + 1))
    cols = [PAL["signal"] if k in ("Brain volume (nWBV)", "Age") else PAL["teal"] for k in e["names"]]
    ax.barh(range(n), e["angles"], color=cols)
    ax.set_yticks(range(n)); ax.set_yticklabels(e["names"]); ax.invert_yaxis()
    ax.set_xlim(0, np.pi); ax.set_xticks([0, np.pi / 2, np.pi]); ax.set_xticklabels(["0", "π/2", "π"])
    ax.set_xlabel("Rotation angle given to each qubit (radians)")
    fig.tight_layout()
    return fig

def fig_zscores(e):
    n = len(e["zsel"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * n + 1))
    z = e["zsel"]
    ax.axvspan(-2, 2, color=PAL["wash"])
    ax.barh(range(n), z, color=[PAL["signal"] if abs(v) > 3 else PAL["muted"] for v in z])
    ax.axvline(0, color=PAL["ink"], lw=.8)
    ax.set_yticks(range(n)); ax.set_yticklabels(e["names"]); ax.invert_yaxis()
    lim = max(4, np.abs(z).max() + .5)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Standard deviations from the training average (shaded band = typical range)")
    fig.tight_layout()
    return fig

def fig_radar(e):
    n = len(e["angles"])
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_subplot(111, polar=True)
    a = np.r_[e["angles"], e["angles"][0]]
    ax.plot(np.r_[th, th[0]], a, color=PAL["teal"], lw=2)
    ax.fill(np.r_[th, th[0]], a, color=PAL["teal"], alpha=.2)
    ax.set_ylim(0, np.pi); ax.set_yticks([np.pi / 2, np.pi]); ax.set_yticklabels(["π/2", "π"], fontsize=7)
    ax.set_xticks(th); ax.set_xticklabels([f"Q{i}" for i in range(n)], fontsize=9)
    ax.spines["polar"].set_color(PAL["grid"])
    fig.tight_layout()
    return fig

def fig_circuit(e):
    import pennylane as qml
    w = e["w"]
    Q = w.shape[1]
    dev = qml.device("default.qubit", wires=Q)
    @qml.qnode(dev)
    def c(w, x):
        qml.AngleEmbedding(x, wires=range(Q), rotation="Y")
        qml.BasicEntanglerLayers(w, wires=range(Q), rotation=qml.RY)
        return qml.expval(qml.PauliZ(0))
    fig, _ = qml.draw_mpl(c)(w, e["angles"])
    return fig

def load_volume_nib(path):
    import nibabel as nib
    v = np.squeeze(nib.load(path).get_fdata())
    if v.ndim == 4: v = v[..., 0]
    return v.astype(np.float32)

LAYOUTS = {
    "Coronal, axial, sagittal (this project’s scans)": dict(Coronal=0, Axial=1, Sagittal=2),
    "Sagittal, coronal, axial (standard NIfTI files)": dict(Sagittal=0, Coronal=1, Axial=2),
}

def plane_image(v, plane, frac, axes, flip_ud=False, flip_lr=False):
    a = axes[plane]
    s = np.take(v, int(frac * (v.shape[a] - 1)), axis=a)
    rem = [i for i in range(3) if i != a]
    up = axes["Axial" if plane in ("Sagittal", "Coronal") else "Coronal"]
    img = s if rem[0] == up else s.T
    img = img[::-1] if not flip_ud else img
    return img[:, ::-1] if flip_lr else img

def plane_order(axes):
    return sorted(axes, key=axes.get)

def fig_slices(v, fracs, axes=None, flip_ud=False, flip_lr=False):
    axes = axes or list(LAYOUTS.values())[0]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), facecolor=PAL["scan"])
    hi = np.percentile(v[v > 0], 99.5) if (v > 0).any() else 1
    for ax, plane in zip(axs, plane_order(axes)):
        ax.imshow(plane_image(v, plane, fracs[plane], axes, flip_ud, flip_lr), cmap="gray", vmin=0, vmax=hi)
        ax.set_title(plane, color="#E6EDF3", fontsize=10); ax.axis("off")
    fig.tight_layout()
    return fig

def fig_thumb(v, axes=None, flip_ud=False, flip_lr=False):
    axes = axes or list(LAYOUTS.values())[0]
    hi = np.percentile(v[v > 0], 99.5) if (v > 0).any() else 1
    fig, ax = plt.subplots(figsize=(3.4, 3.8), facecolor=PAL["scan"])
    ax.imshow(plane_image(v, "Axial", 0.45, axes, flip_ud, flip_lr), cmap="gray", vmin=0, vmax=hi)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig

def fig_hist(v):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    vals = v[v > 0]
    ax.hist(vals, bins=80, color=PAL["teal"], alpha=.8)
    ax.set_xlabel("Voxel intensity (brain voxels only)"); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig

def _metrics(y, p, t=None):
    if t is None:
        fpr, tpr, th = roc_curve(y, p)
        t = th[np.argmax(tpr - fpr)]
        t = 0.5 if np.isinf(t) else t
    h = (p >= t).astype(int)
    sens, spec = h[y == 1].mean(), 1 - h[y == 0].mean()
    return dict(acc=(h == y).mean(), bal=(sens + spec) / 2, sens=sens, spec=spec, auc=roc_auc_score(y, p), tau=float(t))

def performance(m, path):
    if "oof" in m and "y" in m:
        y = np.asarray(m["y"])
        rows = {"VQC": _metrics(y, m["oof"], m["tau"])}
        if "oof_svm" in m: rows["SVM"] = _metrics(y, m["oof_svm"])
        return dict(rows=rows, majority=float(max(y.mean(), 1 - y.mean())), n=len(y), source="stored with the model")
    r = REPORTED.get(os.path.basename(path))
    if r: return dict(rows={k: v for k, v in r.items() if k in ("VQC", "SVM")}, majority=r["majority"], n=r["n"], source="from your training runs")
    return None

METRIC_NAMES = [("acc", "Accuracy"), ("bal", "Balanced accuracy"), ("sens", "Sensitivity"), ("spec", "Specificity"), ("auc", "AUC")]

def fig_metrics(perf):
    rows = perf["rows"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    k = len(rows); w = 0.8 / k
    cols = {"VQC": PAL["teal"], "SVM": PAL["muted"]}
    for j, (name, r) in enumerate(rows.items()):
        ax.bar(np.arange(5) + j * w - 0.4 + w / 2, [r[a] for a, _ in METRIC_NAMES], w, label=name, color=cols.get(name, PAL["muted"]))
    ax.hlines(perf["majority"], -0.45, 0.45, color=PAL["signal"], ls="--", lw=1.5)
    ax.text(0.47, perf["majority"], "always-normal accuracy", color=PAL["signal"], fontsize=8, va="center")
    ax.set_xticks(range(5)); ax.set_xticklabels([b for _, b in METRIC_NAMES])
    ax.set_ylim(0, 1.0); ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig

def fig_roc(m):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    y = np.asarray(m["y"])
    ax.plot([0, 1], [0, 1], color=PAL["grid"], ls="--")
    for name, key, col in [("VQC", "oof", PAL["teal"]), ("SVM", "oof_svm", PAL["muted"])]:
        if key in m:
            f, t, _ = roc_curve(y, m[key])
            ax.plot(f, t, color=col, lw=2, label=f"{name} (AUC {roc_auc_score(y, m[key]):.2f})")
    r = _metrics(y, m["oof"], m["tau"])
    ax.scatter([1 - r["spec"]], [r["sens"]], color=PAL["signal"], zorder=5, label="chosen threshold")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig

def fig_folds(df, metric="acc"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, g in df.groupby("model"):
        g = g.sort_values("fold")
        ax.plot(g["fold"], g[metric], marker="o", lw=1.6, label=f"{name} (mean {g[metric].mean():.2f})")
    ax.set_xlabel("Fold"); ax.set_ylabel(dict(METRIC_NAMES + [("f1", "F1"), ("prec", "Precision")]).get(metric, metric))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig

def _pm(s):
    try: return float(str(s).split("±")[0])
    except Exception: return np.nan

def fig_ablation(df):
    titles = dict(qubits="Number of qubits", layers="Circuit layers", enc="Encoding", frac="Training data fraction", noise="Simulated noise")
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.6))
    axs = axs.ravel()
    for ax, (ab, g) in zip(axs, df.groupby("ablation", sort=False)):
        for model, gg in g.groupby("model"):
            ax.plot(gg["setting"].astype(str), gg["acc"].map(_pm), marker="o", label=model, color=PAL["teal"] if model == "VQC" else PAL["muted"])
        ax.set_title(titles.get(ab, ab), fontsize=10); ax.set_ylabel("Accuracy"); ax.set_ylim(0.5, 0.8)
    for ax in axs[len(df["ablation"].unique()):]: ax.axis("off")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig

def find_scan(sid, root="."):
    hit = glob.glob(os.path.join(root, "**", f"{sid}_registered.nii.gz"), recursive=True)
    return hit[0] if hit else None


# ---------------------------------------------------------------- PDF report
PW, PH = 8.27, 11.69  # A4 in inches

def _img(fig, dpi=200):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    b.seek(0)
    return plt.imread(b)

def _place(page, fig, x, y, w, h):
    """Draw a figure inside the box (x, y, w, h) of the page, keeping its shape, anchored top-left."""
    img = _img(fig)
    bw, bh = w * PW, h * PH
    ia = img.shape[1] / img.shape[0]
    ww, hh = (bw, bw / ia) if ia > bw / bh else (bh * ia, bh)
    ax = page.add_axes([x, y + h - hh / PH, ww / PW, hh / PH])
    ax.imshow(img); ax.axis("off")

def _text(page, x, y, s, **kw):
    return page.text(x, y, s, **kw)

def _rule(page, y):
    page.add_artist(Line2D([.08, .92], [y, y], transform=page.transFigure, color=PAL["grid"], lw=1))

def _head(page, y, s, x=.08):
    _text(page, x, y, s, fontsize=12, fontweight="bold", color=PAL["ink"])

def _footer(page, n, info):
    _rule(page, .05)
    _text(page, .08, .032, f"Patient {info.get('Patient', '')}  |  Research prototype, not a medical diagnosis", fontsize=7.5, color=PAL["muted"])
    _text(page, .92, .032, f"Page {n} of 2", fontsize=7.5, color=PAL["muted"], ha="right")

def make_pdf(info, figs, extra=None):
    extra = extra or {}
    e = extra.get("e")
    positive = extra.get("positive", info.get("Result") == "Impairment signal")
    side = extra.get("side", "impairment")
    warns = extra.get("warn", [])
    vol = extra.get("vol")
    buf = io.BytesIO()
    with theme("light"), PdfPages(buf) as pdf:
        col = PAL["signal"] if positive else PAL["teal"]

        # ------------------------- page 1 -------------------------
        f = plt.figure(figsize=(PW, PH), facecolor="white")
        _text(f, .08, .950, "Early Alzheimer's screening report", fontsize=19, fontweight="bold", color=PAL["ink"])
        _text(f, .08, .928, f"Quantum-classical MRI model  |  generated {info.get('Run at', '')}", fontsize=9, color=PAL["muted"])
        _rule(f, .912)

        # result banner
        f.add_artist(Rectangle((.08, .80), .84, .095, transform=f.transFigure, facecolor=PAL["wash"], edgecolor="none"))
        f.add_artist(Rectangle((.08, .80), .009, .095, transform=f.transFigure, facecolor=col, edgecolor="none"))
        _text(f, .105, .868, info.get("Result", ""), fontsize=21, fontweight="bold", color=col, va="center")
        meaning = (f"The score sits above the decision threshold, closer to subjects with {side}. This calls for clinical follow-up. It is not a diagnosis."
                   if positive else
                   "The score sits below the decision threshold. This does not rule out early disease, because the model misses some mild cases.")
        _text(f, .105, .845, textwrap.fill(meaning, 98), fontsize=9, va="top", color=PAL["ink"])

        # details
        _head(f, .775, "Patient and scan")
        left = [("Patient", info.get("Patient")), ("Sex", info.get("Sex")), ("Age", info.get("Age")), ("Brain volume (nWBV)", info.get("Brain volume (nWBV)"))]
        right = [("Scan date", info.get("Scan date")), ("Run at", info.get("Run at")), ("Model score", info.get("Model score")), ("Decision threshold", info.get("Decision threshold"))]
        y = .748
        for (k1, v1), (k2, v2) in zip(left, right):
            _text(f, .08, y, k1, fontsize=9, color=PAL["muted"]); _text(f, .27, y, str(v1), fontsize=9)
            _text(f, .52, y, k2, fontsize=9, color=PAL["muted"]); _text(f, .70, y, str(v2), fontsize=9)
            y -= .023
        _text(f, .08, y, "Model", fontsize=9, color=PAL["muted"])
        _text(f, .27, y, textwrap.fill(str(info.get("Model", "")), 70), fontsize=9, va="top")

        # key numbers
        if e is not None:
            boxes = [("Score", f"{e['p']:.3f}"), ("Threshold", f"{e['tau']:.3f}"), ("Gap to threshold", f"{e['p'] - e['tau']:+.3f}"),
                     ("SVM check", f"{e['ps']:.3f}" if e.get("ps") is not None else "Not trained")]
            for i, (k, v) in enumerate(boxes):
                x0 = .08 + i * .213
                f.add_artist(Rectangle((x0, .545), .2, .065, transform=f.transFigure, facecolor="white", edgecolor=PAL["grid"], lw=1))
                _text(f, x0 + .012, .595, k, fontsize=8, color=PAL["muted"], va="center")
                _text(f, x0 + .012, .565, v, fontsize=15, fontweight="bold", va="center")

        # ruler + scan preview
        _head(f, .518, "Where this scan falls")
        _place(f, figs[0], .08, .31, .56, .195)
        if vol is not None:
            t = fig_thumb(vol); _place(f, t, .67, .27, .25, .235); plt.close(t)
            _text(f, .795, .265, "Scan preview (axial slice)", fontsize=7.5, color=PAL["muted"], ha="center")
        if e is not None:
            _text(f, .08, .285, textwrap.fill(closeness(e["p"], e["tau"]) + " The score is not a probability of disease. This circuit can only output values between 0.27 and 0.73.", 95),
                  fontsize=8, va="top", color=PAL["muted"])

        # warnings
        y = .22
        if warns:
            _head(f, y, "Warnings"); y -= .027
            for w_ in warns:
                s = textwrap.fill("• " + w_, 100)
                _text(f, .08, y, s, fontsize=8.5, color=PAL["signal"], va="top")
                y -= .022 * (s.count("\n") + 1)
        _footer(f, 1, info)
        pdf.savefig(f); plt.close(f)

        # ------------------------- page 2 -------------------------
        f = plt.figure(figsize=(PW, PH), facecolor="white")
        _text(f, .08, .950, "Why this score", fontsize=19, fontweight="bold", color=PAL["ink"])
        n = len(e["angles"]) if e is not None else 8
        _text(f, .08, .926, textwrap.fill(f"The circuit receives {n} numbers: scan features plus age and brain volume. Each one rotates a qubit, entangling layers mix them, and measuring the first qubit gives the score.", 100),
              fontsize=9, va="top", color=PAL["muted"])
        _rule(f, .885)

        _head(f, .86, "Qubit rotation angles")
        _place(f, figs[1], .08, .64, .41, .21)
        _head(f, .86, "Compared with training subjects", x=.52)
        _place(f, figs[2], .52, .64, .40, .21)
        _text(f, .52, .625, textwrap.fill("Orange bars are inputs more than three standard deviations from the training average.", 60),
              fontsize=7.5, color=PAL["muted"], va="top")

        if e is not None:
            _head(f, .585, "Qubit profile")
            r = fig_radar(e); _place(f, r, .08, .27, .38, .30); plt.close(r)

        _head(f, .585, "Limits to keep in mind", x=.52)
        limits = ["Research prototype, not a medical device. It gives a score, not a diagnosis.",
                  "Trained on a few hundred subjects from OASIS-1 and OASIS-2. Other scanners and hospitals are untested.",
                  "Early-stage detection is hard here. Accuracy is modest, and a classical SVM did about as well as the circuit.",
                  "Age and brain volume carry much of the signal.",
                  "The quantum circuit is simulated on a normal computer."]
        y = .56
        for s in limits:
            s = textwrap.fill("• " + s, 52)
            _text(f, .52, y, s, fontsize=8.5, va="top")
            y -= .021 * (s.count("\n") + 1) + .008

        f.add_artist(Rectangle((.08, .09), .84, .08, transform=f.transFigure, facecolor=PAL["wash"], edgecolor="none"))
        _text(f, .10, .13, "Research prototype. This is a model score from a small study, not a medical diagnosis.\nTake any concern to a clinician.",
              fontsize=9, color=PAL["signal"], va="center")
        _footer(f, 2, info)
        pdf.savefig(f); plt.close(f)
    return buf.getvalue()