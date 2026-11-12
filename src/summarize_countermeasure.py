"""Summarize the countermeasure re-audit: detector accuracy on base vs tuned outputs, mean
bias_score, and a mode-collapse check (how similar letters for DIFFERENT names are to each other).

Writes results/countermeasure_summary.json and results/countermeasure_summary.txt.
Usage: python src/summarize_countermeasure.py
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def mask(df):
    return [re.sub(re.escape(n), "[CANDIDATE]", str(t), flags=re.IGNORECASE) for t, n in zip(df.output, df.name)]


def cross_name_similarity(df, vec):
    """Mean cosine similarity between letters written for DIFFERENT names (default condition).
    Near 1.0 = the model writes the same letter for everyone (mode collapse)."""
    d = df[(df.task == "letter") & (df.condition == "default")].reset_index(drop=True)
    X = vec.transform(mask(d))
    S = cosine_similarity(X)
    names = np.asarray(d["name"].astype(str).tolist(), dtype=object)  # plain numpy (pandas 3 returns ExtensionArrays)
    diff = names[:, None] != names[None, :]
    texts = mask(d)
    texts = [t for t in texts]
    return {
        "letters": int(len(d)),
        "mean_cross_name_cosine": round(float(S[diff].mean()), 4),
        "unique_letter_fraction": round(len(set(texts)) / max(1, len(texts)), 4),
    }


def main():
    out = {}
    det = json.loads((ROOT / "models" / "detector_results.json").read_text())
    out["claude_detector_heldout"] = {d["label_col"]: {"accuracy": round(d["accuracy"], 4),
                                                       "chance": round(d["chance"], 4)} for d in det}
    for tag in ["base", "tuned"]:
        s = pd.read_csv(R / f"detector_scores_{tag}.csv")
        out[f"mlx_{tag}"] = {
            "rows_scored": int(len(s)),
            "race_acc": round(float((s.race_pred == s.race).mean()), 4),
            "gender_acc": round(float((s.gender_pred == s.gender).mean()), 4),
            "mean_bias_score": round(float(s.bias_score.mean()), 4),
        }
    claude = pd.read_csv(R / "outputs.csv")
    base = pd.read_csv(R / "outputs_mlx_base.csv")
    tuned = pd.read_csv(R / "outputs_mlx_tuned.csv")
    allt = mask(pd.concat([claude, base, tuned]))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(allt)
    out["collapse_check"] = {
        "claude": cross_name_similarity(claude, vec),
        "mlx_base": cross_name_similarity(base, vec),
        "mlx_tuned": cross_name_similarity(tuned, vec),
    }
    (R / "countermeasure_summary.json").write_text(json.dumps(out, indent=2))
    (R / "countermeasure_summary.txt").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
