import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

features = pd.read_csv("oasis2_features.csv")
metadata = pd.read_csv("oasis2_cleaned_metadata.csv")
data = features.merge(metadata, on="ID").dropna(subset=["CDR", "nWBV", "Age", "MMSE"])
cnn = [c for c in data if c.startswith("f") and c[1:].isdigit()]

sets = {
    "cnn_clin": cnn + ["nWBV", "Age"],
    "clinical_mmse": cnn + ["nWBV", "Age", "MMSE"],
    "clinical_extended": cnn + ["nWBV", "Age", "MMSE", "Educ", "SES"],
}

models = {
    "logistic": lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(f_classif, k=30),
        LogisticRegression(C=.1, max_iter=3000, class_weight="balanced", random_state=42),
    ),
    "svm_linear": lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(f_classif, k=30),
        SVC(C=.1, kernel="linear", class_weight="balanced", probability=True, random_state=42),
    ),
    "svm_rbf": lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(f_classif, k=30),
        SVC(C=2, gamma="scale", class_weight="balanced", probability=True, random_state=42),
    ),
    "hist_gradient": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, l2_regularization=1, random_state=42),
    ),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

y = (data["CDR"] >= .5).astype(int).to_numpy()
groups = data["Subject ID"].astype(str).to_numpy()
cv = StratifiedGroupKFold(10, shuffle=True, random_state=100)
print("n", len(y), "impaired", y.sum(), "normal", (1-y).sum())

for name, columns in sets.items():
    X = data[columns].to_numpy(float)
    print("\n" + name)
    for model_name, factory in models.items():
        prob = np.zeros(len(y))
        pred = np.zeros(len(y), dtype=int)
        for train, test in cv.split(X, y, groups):
            model = factory()
            model.fit(X[train], y[train])
            prob[test] = model.predict_proba(X[test])[:, 1]
            pred[test] = model.predict(X[test])
        print(
            model_name,
            "acc", round(accuracy_score(y, pred), 3),
            "bal", round(balanced_accuracy_score(y, pred), 3),
            "auc", round(roc_auc_score(y, prob), 3),
        )
