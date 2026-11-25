import argparse, json, pickle, numpy as np, pandas as pd, pennylane as qml
from pennylane import numpy as pnp

NAMES = ["Normal", "Early (mild)", "Alzheimer's"]

def scan_to_features(scan, weights="resnet_18_23dataset.pth", device="cpu"):
    import torch
    from CNN_features import load_volume, build_model
    model = build_model(weights, torch.device(device))
    with torch.no_grad():
        return model(load_volume(scan).to(device)).cpu().numpy().ravel()

def qprob(w, z):
    Q = w.shape[1]
    dev = qml.device("default.qubit", wires=Q)
    @qml.qnode(dev)
    def circuit(w, x):
        qml.AngleEmbedding(x, wires=range(Q), rotation="Y")
        qml.BasicEntanglerLayers(w, wires=range(Q), rotation=qml.RY)
        return qml.expval(qml.PauliZ(0))
    return float(1 / (1 + np.exp(-circuit(pnp.array(w), pnp.array(z)))))

def ru_full_prob(m, z):
    if "ensemble" not in m:
        return qprob(m["w"], z)
    Q = m["w"].shape[1]
    dev = qml.device("default.qubit", wires=Q)
    @qml.qnode(dev)
    def circuit(w, s, x):
        for l in range(w.shape[0]):
            qml.AngleEmbedding(x * s[l], wires=range(Q), rotation="Y")
            qml.BasicEntanglerLayers(w[l:l + 1], wires=range(Q), rotation=qml.RY)
        return [qml.expval(qml.PauliZ(i)) for i in range(Q)]
    values = []
    for member in m["ensemble"]:
        mean_z = float(np.mean(circuit(pnp.array(member["w"]), pnp.array(member["s"]), pnp.array(z))))
        values.append(1 / (1 + np.exp(-(member["scale"] * mean_z + member["bias"]))))
    return float(np.mean(values))

def score(m, x):
    z = m["sc"].transform(x[None, :])
    z = np.clip(m["mm"].transform(np.hstack([m["sel"].transform(z[:, :m["NF"]]), z[:, m["NF"]:]])), 0, np.pi)[0]
    p = ru_full_prob(m, z) if m.get("ansatz") == "ru_full" else qprob(m["w"], z)
    ps = float(m["svm"].predict_proba(z[None, :])[0, 1]) if "svm" in m else None
    return p, ps

def predict_one(features, age=None, nwbv=None, mmse=None, model_path="final_oasis1_q8_cdr1_clin_mmse.pkl"):
    m = pickle.load(open(model_path, "rb"))
    features = np.asarray(features, dtype=float).ravel()
    if "m1" in m:
        x = np.concatenate([features] + ([[nwbv, age]] if m["m1"]["NC"] else []))
        p1, _ = score(m["m1"], x)
        p2, _ = score(m["m2"], x)
        lab = 0 if p1 < m["t1"] else (1 if p2 < m["t2"] else 2)
        return dict(result=NAMES[lab], p_impaired=round(p1, 3), p_alzheimers_given_impaired=round(p2, 3), thresholds=[round(m["t1"], 3), round(m["t2"], 3)])
    names = m.get("clin_names") or m.get("clin")
    if names == ["nWBV", "Age", "MMSE"]:
        clinical_values = [nwbv, age, mmse]
    elif names and "nWBV slope/year" in names:
        raise ValueError("This longitudinal model needs baseline and follow-up scans, which one scan cannot provide.")
    else:
        clinical_values = [nwbv, age] if m["NC"] == 2 else []
    clinical = {m["NC"]: clinical_values} if clinical_values else {}
    if m["NC"] in clinical and any(v is None for v in clinical[m["NC"]]):
        required = "--age and --nwbv" + (" and --mmse" if names == ["nWBV", "Age", "MMSE"] else "")
        raise ValueError(f"This model needs {required}.")
    x = np.concatenate([features] + ([clinical[m["NC"]]] if m["NC"] in clinical else []))
    p, ps = score(m, x)
    out = dict(result="Possible early impairment / Alzheimer's (CDR >= 0.5)" if p >= m["tau"] else "No impairment signal", probability=round(p, 3), threshold=round(float(m["tau"]), 3))
    if ps is not None: out["svm_probability"] = round(ps, 3)
    out["note"] = "Research prototype, not a medical diagnosis."
    return out

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--scan", help="registered scan, <ID>_registered.nii.gz")
    a.add_argument("--features_csv", help="csv with an ID column and f0..f511")
    a.add_argument("--id", help="row ID inside --features_csv (default: first row)")
    a.add_argument("--age", type=float)
    a.add_argument("--nwbv", type=float)
    a.add_argument("--mmse", type=float)
    a.add_argument("--model", default="final_oasis1_q8_cdr1_clin_mmse.pkl")
    a.add_argument("--weights", default="resnet_18_23dataset.pth")
    a.add_argument("--device", default="cpu")
    a = a.parse_args()
    if a.scan:
        f = scan_to_features(a.scan, a.weights, a.device)
    else:
        df = pd.read_csv(a.features_csv)
        row = df[df["ID"] == a.id].iloc[0] if a.id else df.iloc[0]
        f = row[[c for c in df if c[0] == "f" and c[1:].isdigit()]].values
        with open(a.model, "rb") as mf:
            selected_model = pickle.load(mf)
        if selected_model.get("hippo"):
            hippo = pd.read_csv("hippo_features.csv")
            hippo_row = hippo[hippo["ID"] == row["ID"]]
            if hippo_row.empty:
                raise ValueError(f"No hippocampus features found for {row['ID']}.")
            f = np.concatenate([f, hippo_row.iloc[0][selected_model["hippo"]].values.astype(float)])
        if a.mmse is None and selected_model["NC"] == 3:
            metadata = pd.read_csv("oasis1_cleaned_metadata.csv")
            meta_row = metadata[metadata["ID"] == row["ID"]]
            if meta_row.empty:
                raise ValueError(f"No metadata found for {row['ID']}.")
            a.mmse = float(meta_row.iloc[0]["MMSE"])
    print(json.dumps(predict_one(f, a.age, a.nwbv, a.mmse, a.model), indent=2))