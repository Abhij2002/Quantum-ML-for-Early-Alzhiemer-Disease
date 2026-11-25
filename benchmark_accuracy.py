import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

features = pd.read_csv("oasis1_features.csv")
metadata = pd.read_csv("oasis1_cleaned_metadata.csv")
hippo = pd.read_csv("hippo_features.csv")

metadata = metadata.copy()
metadata["Subject ID"] = metadata["ID"].str.replace(r"_MR\d+$", "", regex=True)
data = features.merge(
    metadata[["ID", "Subject ID", "CDR", "nWBV", "Age", "MMSE", "Educ", "SES"]],
    on="ID",
)
data = data.merge(hippo, on="ID", how="inner").dropna(
    subset=["CDR", "nWBV", "Age", "MMSE"]
)
data = data[data["CDR"] != 0.5]

y = (data["CDR"] >= 1).astype(int).to_numpy()
groups = data["Subject ID"].fillna(data["ID"]).astype(str).to_numpy()
cnn = [c for c in data if c.startswith("f") and c[1:].isdigit()]
hippo_cols = [c for c in hippo if c != "ID"]

feature_sets = {
    "cnn_clin": cnn + ["nWBV", "Age"],
    "cnn_hippo_clin": cnn + hippo_cols + ["nWBV", "Age"],
    "all_clin": cnn + hippo_cols + ["nWBV", "Age", "MMSE", "Educ", "SES"],
    "clin_mmse": ["nWBV", "Age", "MMSE", "Educ", "SES"],
    "hippo_mmse": hippo_cols + ["nWBV", "Age", "MMSE", "Educ", "SES"],
}

def make_lr():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SelectKBest(f_classif, k=30),
        LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=42),
    )

def make_svm_linear():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SelectKBest(f_classif, k=30),
        SVC(C=0.1, kernel="linear", class_weight="balanced", probability=True, random_state=42),
    )

def make_svm_rbf():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SelectKBest(f_classif, k=30),
        SVC(C=2, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    )

models = {
    "lr": make_lr,
    "svm_linear": make_svm_linear,
    "svm_rbf": make_svm_rbf,
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "hist_gradient": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, l2_regularization=1, random_state=42
    ),
}

cv = StratifiedGroupKFold(10, shuffle=True, random_state=42)
print("n", len(y), "positive", y.sum(), "negative", (1 - y).sum())

for feature_name, columns in feature_sets.items():
    X = data[columns].to_numpy(float)
    print("\n" + feature_name)
    for model_name, factory in models.items():
        probabilities = np.zeros(len(y))
        predictions = np.zeros(len(y), dtype=int)
        for train, test in cv.split(X, y, groups):
            model = factory()
            model.fit(X[train], y[train])
            probabilities[test] = model.predict_proba(X[test])[:, 1]
            predictions[test] = model.predict(X[test])
        print(
            model_name,
            "acc", round(accuracy_score(y, predictions), 3),
            "bal", round(balanced_accuracy_score(y, predictions), 3),
            "auc", round(roc_auc_score(y, probabilities), 3),
        )
