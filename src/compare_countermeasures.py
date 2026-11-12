"""Compare the untuned model, countermeasure v1 (12 templates) and v2 (60 templates) on one
shared footing. Writes results/countermeasure_comparison.json.
Usage: python src/compare_countermeasures.py
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "results"
GEN = r"\b(?:she|her|hers|herself|he|him|his|himself)\b"
RUNS = {"base": (R/"outputs_mlx_base.csv", R/"detector_scores_base.csv"),
        "v1_12_templates": (R/"v1_12templates"/"outputs_mlx_tuned.csv", R/"v1_12templates"/"detector_scores_tuned.csv"),
        "v2_60_templates": (R/"outputs_mlx_tuned.csv", R/"detector_scores_tuned.csv")}

def mask(d): return [re.sub(re.escape(n), "[CANDIDATE]", str(t), flags=re.I) for t, n in zip(d.output, d.name)]

def main():
    data = {k: pd.read_csv(p) for k, (p, _) in RUNS.items()}
    letters = {k: d[d.task == "letter"].reset_index(drop=True) for k, d in data.items()}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(sum((mask(l) for l in letters.values()), []))
    out = {}
    for k, L in letters.items():
        o = L.output.fillna("").astype(str)
        s = pd.read_csv(RUNS[k][1])
        dflt = L[L.condition == "default"].reset_index(drop=True)
        S = cosine_similarity(vec.transform(mask(dflt)))
        names = np.asarray(dflt["name"].astype(str).tolist(), dtype=object)
        out[k] = {"letters": int(len(L)),
                  "gendered_pronoun_rate": round(float(o.str.contains(GEN, case=False, regex=True).mean()), 3),
                  "mean_words": round(float(o.str.split().str.len().mean()), 1),
                  "unique_letter_fraction": round(o.nunique() / len(o), 3),
                  "cross_name_cosine_default": round(float(S[names[:, None] != names[None, :]].mean()), 3),
                  "detector_gender_acc": round(float((s.gender_pred == s.gender).mean()), 3),
                  "detector_race_acc": round(float((s.race_pred == s.race).mean()), 3),
                  "mean_bias_score": round(float(s.bias_score.mean()), 3)}
    (R / "countermeasure_comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
