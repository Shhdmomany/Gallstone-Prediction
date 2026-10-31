"""Run once:  python train.py dataset-uci.csv
Trains several models with cross-validation, keeps the best, saves it to ./artifacts"""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TARGET = "Gallstone Status"
# Names as used in your notebook (0 = No gallstone, 1 = Gallstone).
# CHECK this against the UCI dataset page: I believe UCI documents the coding the other way round.
LABEL_NAMES = {0: "No gallstone", 1: "Gallstone"}

csv = sys.argv[1] if len(sys.argv) > 1 else "dataset-uci.csv"
out = Path("artifacts"); out.mkdir(exist_ok=True)

def read_any_encoding(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):   # Excel often saves CSVs as cp1252
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the CSV with any common encoding")

df = read_any_encoding(csv).drop_duplicates().reset_index(drop=True)
X = df.drop(columns=TARGET).apply(pd.to_numeric, errors="coerce")
medians = X.median()
X = X.fillna(medians)
y = df[TARGET]

# stratify keeps the class balance equal in train and test (your original split did not)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cands = {
    "Logistic Regression": (Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
                            {"m__C": [0.1, 1, 10]}),
    "Random Forest": (Pipeline([("m", RandomForestClassifier(random_state=42))]),
                      {"m__n_estimators": [100, 300], "m__max_depth": [None, 10], "m__min_samples_leaf": [1, 2]}),
    "SVM": (Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True, random_state=42))]),
            {"m__C": [0.1, 1, 10], "m__kernel": ["linear", "rbf"]}),
}
try:
    from xgboost import XGBClassifier
    cands["XGBoost"] = (Pipeline([("m", XGBClassifier(eval_metric="logloss", random_state=42))]),
                        {"m__n_estimators": [100, 200], "m__max_depth": [3, 5], "m__learning_rate": [0.05, 0.1]})
except ImportError:
    print("xgboost not installed, skipping it")

rows, best = [], None
for name, (pipe, grid) in cands.items():
    gs = GridSearchCV(pipe, grid, cv=5, scoring="f1_macro").fit(Xtr, ytr)
    p = gs.predict(Xte)
    row = {"Model": name, "CV F1 (macro)": round(gs.best_score_, 3),
           "Test accuracy": round(accuracy_score(yte, p), 3),
           "Test precision": round(precision_score(yte, p), 3),
           "Test recall": round(recall_score(yte, p), 3),
           "Test F1": round(f1_score(yte, p), 3)}
    rows.append(row)
    print(row, "\n", confusion_matrix(yte, p))
    if best is None or gs.best_score_ > best[0]:
        best = (gs.best_score_, name, gs.best_estimator_)

print("\nBest model (by cross-validation):", best[1])
final = clone(best[2]).fit(X, y)          # refit on all data for the app
joblib.dump(final, out / "model.joblib")
Xtr.to_csv(out / "background.csv", index=False)
json.dump({"features": X.columns.tolist(),
           "medians": medians.to_dict(),
           "mins": X.min().to_dict(), "maxs": X.max().to_dict(),
           "class_names": [LABEL_NAMES[int(c)] for c in final.classes_],
           "best_model": best[1], "results": rows},
          open(out / "meta.json", "w"), indent=2)
print("Saved to", out.resolve())
