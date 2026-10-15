"""Score outputs and test for differences. Writes results/scores.csv and results/summary.txt.

Word lists come from Gaucher, Friesen & Kay (2011), Appendix A (see below).
"""
import re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
# Word lists: Appendix A of Gaucher, Friesen & Kay (2011), "Evidence that gendered wording in job
# advertisements exists and sustains gender inequality", JPSP. The paper calls them masculine- and
# feminine-coded; they were built from published agentic/communal trait lists, and we use them as
# the agentic/communal proxy. A trailing * means "any ending" (stem match). The spelling
# "Implusive" is as printed in the source appendix.
AGENTIC = ['Active', 'Adventurous', 'Aggress*', 'Ambitio*', 'Analy*', 'Assert*', 'Athlet*', 'Autonom*', 'Boast*', 'Challeng*', 'Compet*', 'Confident', 'Courag*', 'Decide', 'Decisive', 'Decision*', 'Determin*', 'Dominant', 'Domina*', 'Force*', 'Greedy', 'Headstrong', 'Hierarch*', 'Hostil*', 'Implusive', 'Independen*', 'Individual*', 'Intellect*', 'Lead*', 'Logic', 'Masculine', 'Objective', 'Opinion', 'Outspoken', 'Persist', 'Principle*', 'Reckless', 'Stubborn', 'Superior', 'Self-confiden*', 'Self-sufficien*', 'Self-relian*']
COMMUNAL = ['Affectionate', 'Child*', 'Cheer*', 'Commit*', 'Communal', 'Compassion*', 'Connect*', 'Considerate', 'Cooperat*', 'Depend*', 'Emotiona*', 'Empath*', 'Feminine', 'Flatterable', 'Gentle', 'Honest', 'Interpersonal', 'Interdependen*', 'Interpersona*', 'Kind', 'Kinship', 'Loyal*', 'Modesty', 'Nag', 'Nurtur*', 'Pleasant*', 'Polite', 'Quiet*', 'Respon*', 'Sensitiv*', 'Submissive', 'Support*', 'Sympath*', 'Tender*', 'Together', 'Trust*', 'Understand*', 'Warm*', 'Whin*', 'Yield*']


def _matcher(words):
    exact = {w.lower() for w in words if not w.endswith("*")}
    stems = tuple(w.lower().rstrip("*") for w in words if w.endswith("*"))
    return lambda tok: tok in exact or tok.startswith(stems)


is_agentic, is_communal = _matcher(AGENTIC), _matcher(COMMUNAL)


def tokens(text):
    return re.findall(r"[a-z']+", str(text).lower())


def score(df):
    toks = df.output.map(tokens)
    df["n_words"] = toks.map(len)
    df["agentic"] = toks.map(lambda t: sum(map(is_agentic, t)))
    df["communal"] = toks.map(lambda t: sum(map(is_communal, t)))
    df["agentic_rate"] = df.agentic / df.n_words.clip(lower=1) * 100
    df["communal_rate"] = df.communal / df.n_words.clip(lower=1) * 100
    df["hire"] = df.output.str.upper().str.contains(r"DECISION:\s*HIRE") & ~df.output.str.upper().str.contains(r"DECISION:\s*NO HIRE")
    return df


def main():
    df = pd.read_csv(ROOT / "results" / "outputs.csv")
    df = score(df)
    df.to_csv(ROOT / "results" / "scores.csv", index=False)
    lines = [f"Outputs analysed: {len(df)}  (model: {', '.join(df.model.unique())})", ""]
    if (df.model == "dry-run").any():
        lines += ["*** DRY-RUN DATA: these numbers are fake and mean nothing. ***", ""]

    letters = df[df.task == "letter"]
    for outcome in ["communal_rate", "agentic_rate", "n_words"]:
        lines.append(f"== Letters: {outcome} ~ race * gender * condition (name as random effect) ==")
        try:
            m = smf.mixedlm(f"{outcome} ~ C(race) * C(gender) * C(condition)", letters, groups=letters["name"]).fit()
            lines.append(str(m.summary()))
        except Exception as e:
            lines.append(f"model failed: {e}")
        lines.append("")
        lines.append(letters.groupby(["condition", "race", "gender"])[outcome].agg(["mean", "std", "count"]).round(3).to_string())
        lines.append("")

    hire = df[df.task == "hire"]
    lines.append("== Hire rate by group and condition ==")
    lines.append(hire.groupby(["condition", "race", "gender"]).hire.agg(["mean", "count"]).round(3).to_string())
    for cond, g in hire.groupby("condition"):
        for var in ["race", "gender"]:
            table = pd.crosstab(g[var], g.hire)
            if table.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(table)
                lines.append(f"[{cond}] hire ~ {var}: chi2={chi2:.2f}, p={p:.3f}")
    text = "\n".join(lines)
    (ROOT / "results" / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
