"""
train_model.py
Trains the Random Forest false-positive classifier (and, optionally, two comparison models).

Target convention:  y = 1 for a FALSE POSITIVE, y = 0 for a GENUINE attack,
so predict_proba(...)[:, 1] is the "false-positive probability" shown on slide 13.
"""
from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eve_parser import load_eve, NUMERIC, CATEGORICAL

ROOT = Path(__file__).resolve().parent.parent


def build_dataset():
    X = load_eve(ROOT / "data" / "eve.json")
    lab = pd.read_csv(ROOT / "data" / "labels.csv")
    df = X.merge(lab, on="flow_id")
    y_fp = (df["label"] == 0).astype(int)                 # 1 = false positive
    return df.drop(columns=["flow_id", "label"]), y_fp, df["label"]


def make_pipeline(model, scale=False):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("num", num, NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", model)])


def get_models():
    return {
        "Random Forest": make_pipeline(RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42)),
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"), scale=True),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=42)),
    }


def split():
    X, y_fp, label = build_dataset()
    return train_test_split(X, y_fp, test_size=0.30, stratify=y_fp, random_state=42)


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = split()
    print(f"train={len(Xtr)}  test={len(Xte)}  FP share (train)={ytr.mean():.2%}")
    rf = get_models()["Random Forest"].fit(Xtr, ytr)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(rf, ROOT / "models" / "rf_fp_model.joblib")
    print("Saved models/rf_fp_model.joblib")
