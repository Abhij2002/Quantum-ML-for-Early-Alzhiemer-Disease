import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

features = pd.read_csv("oasis2_features.csv")
metadata = pd.read_csv("oasis2_cleaned_metadata.csv")
data = features.merge(metadata, on="ID").sort_values(["Subject ID", "Visit"])
first = data.groupby("Subject ID").first()
last = data.groupby("Subject ID").last()
years = (last["MR Delay"] - first["MR Delay"]) / 365.25
valid = years > 0
first, last, years = first[valid], last[valid], years[valid]

y = (first["Group"] != "Nondemented").astype(int).to_numpy()
# "Impaired" is stronger in the first row: Demented or Converted.
y = ((first["Group"] == "Demented") | (first["Group"] == "Converted")).astype(int).to_numpy()
cnn = [c for c in features if c.startswith("f") and c[1:].isdigit()]
slope = ((last["nWBV"] - first["nWBV"]) / years).to_numpy()

sets = {
    "scan_change": np.column_stack([first[cnn].to_numpy(float), first["nWBV"].to_numpy(float), first["Age"].to_numpy(float), slope]),
    "with_mmse": np.column_stack([first[cnn].to_numpy(float), first["nWBV"].to_numpy(float), first["Age"].to_numpy(float), first["MMSE"].to_numpy(float), slope]),
    "clinical_only": np.column_stack([first["nWBV"].to_numpy(float), first["Age"].to_numpy(float), first["MMSE"].to_numpy(float), slope]),
}

def lr(k):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(f_classif, k=min(k, 30)),
                         LogisticRegression(C=.1, max_iter=3000, class_weight="balanced", random_state=100))

def svm(k):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(f_classif, k=min(k, 30)),
                         SVC(C=.1, kernel="linear", class_weight="balanced", probability=True, random_state=100))

cv = StratifiedKFold(10, shuffle=True, random_state=100)
print("n", len(y), "impaired", y.sum(), "normal", (1-y).sum())
for name, X in sets.items():
    print("\n" + name)
    for model_name, factory in {
        "lr": lambda: lr(X.shape[1]),
        "svm": lambda: svm(X.shape[1]),
        "hist": lambda: make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, l2_regularization=1, random_state=100)),
        "extra": lambda: ExtraTreesClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=100, n_jobs=-1),
    }.items():
        prob = np.zeros(len(y))
        pred = np.zeros(len(y), dtype=int)
        for train, test in cv.split(X, y):
            model = factory()
            model.fit(X[train], y[train])
            prob[test] = model.predict_proba(X[test])[:, 1]
            pred[test] = model.predict(X[test])
        print(model_name, "acc", round(accuracy_score(y, pred), 3), "bal", round(balanced_accuracy_score(y, pred), 3), "auc", round(roc_auc_score(y, prob), 3))
