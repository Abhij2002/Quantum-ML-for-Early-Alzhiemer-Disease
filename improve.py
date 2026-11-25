import sys, warnings, os
import numpy as np, pandas as pd, pennylane as qml
from pennylane import numpy as pnp
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings("ignore")
VAR, EPOCHS, K, LR = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
Q, L, SEED = 8, 4, 42
d = pd.read_csv("oasis1_features.csv").merge(pd.read_csv("oasis1_cleaned_metadata.csv"), on="ID").dropna(subset=["CDR", "nWBV", "Age"])
cnn = [c for c in d if c[0] == "f" and c[1:].isdigit()]; NF = len(cnn)
X = np.hstack([d[cnn].values, d[["nWBV", "Age"]].values]); y = (d["CDR"] > 0).astype(int).values; g = d["ID"].values
dev = qml.device("default.qubit", wires=Q)

def unpack(p):
    return p[:L * Q].reshape(L, Q), p[L * Q:2 * L * Q].reshape(L, Q), p[-2], p[-1]

@qml.qnode(dev, diff_method="backprop")
def circ(p, x):
    w, s, a, b = unpack(p)
    if VAR in ("base", "ob"):
        qml.AngleEmbedding(x, wires=range(Q), rotation="Y")
        for l in range(L):
            qml.BasicEntanglerLayers(w[l:l + 1], wires=range(Q), rotation=qml.RY)
    else:
        for l in range(L):
            qml.AngleEmbedding(x * s[l], wires=range(Q), rotation="Y")
            qml.BasicEntanglerLayers(w[l:l + 1], wires=range(Q), rotation=qml.RY)
    if VAR == "ru_full":
        return [qml.expval(qml.PauliZ(i)) for i in range(Q)]
    return qml.expval(qml.PauliZ(0))

def out(p, x):
    o = circ(p, x)
    if VAR == "ru_full":
        o = pnp.mean(pnp.stack(o), axis=0)
    a, b = p[-2], p[-1]
    return 1 / (1 + pnp.exp(-(a * o + b))) if VAR != "base" else 1 / (1 + pnp.exp(-o))

def init(seed):
    r = np.random.default_rng(seed)
    return pnp.array(np.concatenate([r.uniform(-np.pi, np.pi, L * Q), np.ones(L * Q), [1.0 if VAR == "base" else 3.0, 0.0]]), requires_grad=True)

def fit(x, t, seed):
    p, opt = init(seed), qml.AdamOptimizer(LR)
    x, t = pnp.array(x, requires_grad=False), pnp.array(t, requires_grad=False)
    def loss(p):
        h = out(p, x)
        return -pnp.mean(t * pnp.log(h + 1e-7) + (1 - t) * pnp.log(1 - h + 1e-7))
    for _ in range(EPOCHS):
        p = opt.step(loss, p)
    return p

def prob(p, x): return np.array(out(p, pnp.array(x, requires_grad=False)))
rows = []
for k, (tr, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=SEED).split(X, y, g), 1):
    a, v = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X[tr], y[tr], g[tr])); a, v = tr[a], tr[v]
    sc = StandardScaler().fit(X[a]); pc = SelectKBest(f_classif, k=Q - 2).fit(sc.transform(X[a])[:, :NF], y[a])
    red = lambda z: np.hstack([pc.transform(z[:, :NF]), z[:, NF:]])
    mm = MinMaxScaler((0, np.pi)).fit(red(sc.transform(X[a])))
    f = lambda i: np.clip(mm.transform(red(sc.transform(X[i]))), 0, np.pi)
    xa, xv, xe = f(a), f(v), f(te)
    rg = np.random.default_rng(SEED + k); n = np.bincount(y[a]).max()
    ix = np.concatenate([rg.choice(np.where(y[a] == c)[0], n) for c in (0, 1)])
    p = fit(xa[ix], y[a][ix], SEED + k)
    fpr, tpr, th = roc_curve(y[v], prob(p, xv)); t = th[np.argmax(tpr - fpr)]
    pe = prob(p, xe); h = (pe >= t).astype(int)
    rows.append(dict(fold=k, acc=accuracy_score(y[te], h), sens=recall_score(y[te], h), spec=recall_score(y[te], h, pos_label=0), auc=roc_auc_score(y[te], pe)))
R = pd.DataFrame(rows)
out_dir = os.path.join("artifacts", "vqc_benchmarks")
os.makedirs(out_dir, exist_ok=True)
R.to_csv(os.path.join(out_dir, f"{VAR}.csv"), index=False)
print(VAR, EPOCHS, K, LR, R[["acc", "sens", "spec", "auc"]].mean().round(3).to_dict())