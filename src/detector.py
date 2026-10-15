"""Part 1: a bias detector trained on our own audit outputs.

Trains a small classifier to predict a name's race and gender group from the generated TEXT
ALONE, with the name masked out. If it can guess the group above chance, the text itself
carries demographic signal that a human reader (or a hiring manager) could pick up on too,
even without ever seeing the name.

Usage:
  python src/detector.py train   results/outputs.csv
  python src/detector.py score   results/outputs.csv --out results/detector_scores.csv

Split by NAME, not by row: all 10 samples for a given name land in the same split, so the
detector can't just memorize "this exact letter -> this name" and call it generalization.
"""
import argparse, json, re, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def mask_name(text, name):
    """Replace every occurrence of the candidate's name with a neutral placeholder."""
    return re.sub(re.escape(name), "[CANDIDATE]", str(text), flags=re.IGNORECASE)


def load(path):
    df = pd.read_csv(path)
    df = df[df.task == "letter"].copy()  # letters carry more free text than the hire task
    df["masked"] = [mask_name(t, n) for t, n in zip(df.output, df.name)]
    return df


def train_one(df, label_col, out_path):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, test_idx = next(gss.split(df, groups=df.name))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(train.masked)
    Xte = vec.transform(test.masked)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, train[label_col])
    pred = clf.predict(Xte)

    report = classification_report(test[label_col], pred, zero_division=0)
    acc = accuracy_score(test[label_col], pred)
    n_classes = test[label_col].nunique()
    print(f"\n== {label_col} ==")
    print(f"held-out names: {test.name.nunique()}, rows: {len(test)}, classes: {n_classes}")
    print(f"accuracy: {acc:.3f}  (chance ~= {1/n_classes:.3f})")
    print(report)

    with open(out_path, "wb") as f:
        pickle.dump({"vectorizer": vec, "model": clf, "label_col": label_col}, f)
    return {"label_col": label_col, "accuracy": acc, "chance": 1 / n_classes, "report": report}


def cmd_train(args):
    df = load(args.csv)
    MODEL_DIR.mkdir(exist_ok=True)
    results = []
    results.append(train_one(df, "race", MODEL_DIR / "detector_race.pkl"))
    results.append(train_one(df, "gender", MODEL_DIR / "detector_gender.pkl"))
    (MODEL_DIR / "detector_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved models to {MODEL_DIR}/ and results to {MODEL_DIR}/detector_results.json")


def cmd_score(args):
    df = load(args.csv)
    out = df[["name", "race", "gender", "task", "condition", "sample"]].copy()
    for label_col in ["race", "gender"]:
        with open(MODEL_DIR / f"detector_{label_col}.pkl", "rb") as f:
            saved = pickle.load(f)
        X = saved["vectorizer"].transform(df.masked)
        proba = saved["model"].predict_proba(X)
        classes = saved["model"].classes_
        # "tilt" = max predicted probability; high tilt = text leans hard toward one group
        out[f"{label_col}_pred"] = saved["model"].predict(X)
        out[f"{label_col}_tilt"] = proba.max(axis=1)
        for i, c in enumerate(classes):
            out[f"{label_col}_p_{c}"] = proba[:, i]
    out["bias_score"] = out[["race_tilt", "gender_tilt"]].mean(axis=1)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} scored rows to {args.out}")
    print("Lower bias_score = text leans less toward any one demographic group (closer to chance).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("csv"); t.set_defaults(func=cmd_train)
    s = sub.add_parser("score"); s.add_argument("csv"); s.add_argument("--out", default=str(ROOT / "results" / "detector_scores.csv")); s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)
