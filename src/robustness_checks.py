"""Robustness checks on the Claude audit (results/outputs.csv), letters only.

1. Data quality: empty / truncated outputs (excluded from everything below).
2. Pronoun use by prompt condition and name gender, and whether pronouns match the name's gender.
3. Detector accuracy with 4-fold cross-validation grouped by name, under two maskings:
   (a) name masked, (b) name + gendered pronouns/titles masked. If gender accuracy collapses
   under (b), the gender signal is carried by pronouns, not by other wording.

Writes results/robustness_checks.json and .txt.  Usage: python src/robustness_checks.py
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
FEM = r"\b(?:she|her|hers|herself)\b"
MAL = r"\b(?:he|him|his|himself)\b"
THEY = r"\b(?:they|them|their|theirs|themselves)\b"
GENDERED = r"\b(?:he|she|him|her|his|hers|himself|herself|mr|mrs|ms|miss|man|woman|gentleman|lady)\b"


def cv_acc(df, col, label, k=4):
    accs = []
    for tr, te in GroupKFold(n_splits=k).split(df, groups=df.name):
        v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        X = v.fit_transform(df[col].iloc[tr])
        c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, df[label].iloc[tr])
        accs.append(float((c.predict(v.transform(df[col].iloc[te])) == df[label].iloc[te]).mean()))
    return {"mean": round(float(np.mean(accs)), 4), "sd": round(float(np.std(accs)), 4), "folds": k}


def main():
    out = {}
    df = pd.read_csv(R / "outputs.csv")
    text = df.output.fillna("").astype(str)
    empty = text.str.strip().str.len() < 50
    out["data_quality"] = {"total_outputs": int(len(df)), "empty_or_truncated": int(empty.sum()),
                           "empty_by_task": df[empty].task.value_counts().to_dict()}
    L = df[(df.task == "letter") & ~empty].copy()
    L["output"] = L.output.astype(str)
    L["fem"] = L.output.str.contains(FEM, case=False, regex=True)
    L["mal"] = L.output.str.contains(MAL, case=False, regex=True)
    L["they"] = L.output.str.contains(THEY, case=False, regex=True)
    L["any_gendered"] = L.fem | L.mal
    tab = L.groupby(["condition", "gender"])[["fem", "mal", "they", "any_gendered"]].mean().round(3)
    out["pronoun_use"] = {f"{c}|{g}": r.to_dict() for (c, g), r in tab.iterrows()}
    out["gendered_pronoun_rate_by_condition"] = L.groupby("condition").any_gendered.mean().round(3).to_dict()
    d = L[L.condition == "default"]
    match = ((d.gender == "female") & d.fem & ~d.mal) | ((d.gender == "male") & d.mal & ~d.fem)
    opp = ((d.gender == "female") & d.mal & ~d.fem) | ((d.gender == "male") & d.fem & ~d.mal)
    out["default_pronoun_vs_name_gender"] = {"letters": int(len(d)), "gendered": int(d.any_gendered.sum()),
                                             "matches_name_gender": int(match.sum()), "opposite": int(opp.sum())}
    L["m_name"] = [re.sub(re.escape(n), "[CANDIDATE]", t, flags=re.I) for t, n in zip(L.output, L.name)]
    L["m_name_pron"] = [re.sub(GENDERED, "[PRON]", t, flags=re.I) for t in L.m_name]
    L = L.reset_index(drop=True)
    out["detector_cv"] = {}
    for col, desc in [("m_name", "name_masked"), ("m_name_pron", "name_and_pronouns_masked")]:
        for lab in ["gender", "race"]:
            out["detector_cv"][f"{desc}|{lab}"] = cv_acc(L, col, lab)
            for cond in ["default", "mitigation"]:
                sub = L[L.condition == cond].reset_index(drop=True)
                out["detector_cv"][f"{desc}|{lab}|{cond}"] = cv_acc(sub, col, lab)
    # Word-rate regressions: OLS with name-clustered SEs (the mixed model's name variance is ~0,
    # and MixedLM failed to converge for the rate outcomes).
    import statsmodels.formula.api as smf
    sys_path_src = Path(__file__).resolve().parent
    import sys; sys.path.insert(0, str(sys_path_src))
    from analyze import score
    S = score(L[["name", "race", "gender", "condition", "output"]].copy())
    out["ols_clustered"] = {}
    for y in ["communal_rate", "agentic_rate", "n_words"]:
        m = smf.ols(f"{y} ~ C(race) * C(gender) * C(condition)", S).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(S["name"])[0]})
        out["ols_clustered"][y] = {k: {"coef": round(float(m.params[k]), 4), "p": round(float(m.pvalues[k]), 4)}
                                   for k in m.params.index}
    out["word_rate_means"] = S.groupby(["condition", "race", "gender"])[["communal_rate", "agentic_rate", "n_words"]].mean().round(3).reset_index().to_dict("records")
    s = json.dumps(out, indent=2)
    (R / "robustness_checks.json").write_text(s)
    (R / "robustness_checks.txt").write_text(s)
    print(s)


if __name__ == "__main__":
    main()
