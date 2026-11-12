"""Live demo of the trained bias detector (models/detector_gender.pkl, detector_race.pkl).

Usage:
  python src/demo_detector.py                     # scores two real letters from results/outputs.csv
  python src/demo_detector.py "your own text"     # scores any text you type
The candidate's name is hidden before scoring, exactly as in the experiment.
"""
import pickle, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from detector import mask_name


def load(label):
    with open(ROOT / "models" / f"detector_{label}.pkl", "rb") as f:
        return pickle.load(f)


def score(text, name=None):
    masked = mask_name(text, name) if name else text
    for label in ["gender", "race"]:
        m = load(label)
        p = m["model"].predict_proba(m["vectorizer"].transform([masked]))[0]
        parts = "   ".join(f"{c}: {v:.0%}" for c, v in zip(m["model"].classes_, p))
        print(f"  {label:<7} -> {parts}")


if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
    print(f'\nText: "{text}"')
    score(text)
else:
    df = pd.read_csv(ROOT / "results" / "outputs.csv")
    L = df[(df.task == "letter") & (df.output.fillna("").str.len() > 50)]
    gendered = L.output.str.contains(r"\b(?:she|her|hers|he|him|his)\b", case=False, regex=True)
    picks = [("default prompt", L[(L.name == "Emily") & (L.condition == "default") & gendered].iloc[0]),
             ("with bias-reduction instruction", L[(L.name == "Emily") & (L.condition == "mitigation") & ~gendered].iloc[0])]
    for label, row in picks:
        name = row["name"]
        masked = mask_name(row.output, name)
        start = masked.find("I am")
        print(f"\n=== Real letter for {name} ({label}), name hidden ===")
        print("  " + masked[start:start + 260].replace("\n", " ") + "...")
        print("  Detector's guess:")
        score(row.output, name)
