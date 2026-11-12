"""Class demo of the whole pipeline, one step at a time. Reads the real project files.

  python src/demo_pipeline.py prompt Emily Jamal        # Step 1: the CLEAR prompt; only the name changes
  python src/demo_pipeline.py write Emily               # Step 2: what Claude wrote (saved from the full run)
  python src/demo_pipeline.py write Emily --task hire   #          ... the hire decision for the same name
  python src/demo_pipeline.py write Emily --live        #          ... ask Claude NOW (needs ANTHROPIC_API_KEY)
  python src/demo_pipeline.py results                   # Step 4: key numbers after robustness_checks.py

--live saves to results/demo_live.csv and never touches results/outputs.csv.
"""
import argparse, csv, json, os, re, sys, textwrap
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config

B, DIM, YEL, GRN, RED, END = "\033[1m", "\033[2m", "\033[93m", "\033[92m", "\033[91m", "\033[0m"
GEN = r"\b(she|her|hers|herself|he|him|his|himself)\b"


def wrap(t, width=100):
    return "\n".join(textwrap.fill(p, width) if p.strip() else "" for p in str(t).split("\n"))


def title(t):
    print(f"\n{B}{'=' * 8} {t} {'=' * 8}{END}")


def cmd_prompt(a):
    for name in a.names:
        title(f"CLEAR prompt for {name}  (task: {a.task}, prompt: {a.condition})")
        p = config.build_prompt(a.task, a.condition, name)
        p = re.sub(r"^(Context|Logic|Expectations|Action|Restrictions):", lambda m: f"{GRN}{B}{m.group(1)}:{END}", p, flags=re.M)
        print(wrap(p).replace(name, f"{YEL}{B}{name}{END}"))
    print(f"\n{DIM}Only the highlighted name differs. Model: {config.MODEL}, temperature {config.TEMPERATURE}.{END}")


def show(text, name):
    t = wrap(text).replace(name, f"{YEL}{B}{name}{END}")
    print(re.sub(GEN, lambda m: f"{RED}{B}{m.group(0)}{END}", t, flags=re.I))


def cmd_write(a):
    if a.live:
        try:
            import anthropic
            from run_experiment import call_model
            client = anthropic.Anthropic()
            title(f"Asking Claude now: {a.task} for {a.name} ({a.condition})")
            text = call_model(client, a.task, a.condition, a.name)
        except Exception as e:
            print(f"{RED}Live call failed ({type(e).__name__}). Check ANTHROPIC_API_KEY. Showing the saved output instead.{END}")
            a.live = False
        else:
            out = ROOT / "results" / "demo_live.csv"
            new = not out.exists()
            with open(out, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new: w.writerow(["name", "task", "condition", "model", "output"])
                w.writerow([a.name, a.task, a.condition, config.MODEL, text])
            show(text, a.name)
            print(f"\n{DIM}Saved to results/demo_live.csv (outputs.csv untouched).{END}")
            return
    df = pd.read_csv(ROOT / "results" / "outputs.csv")
    rows = df[(df.name == a.name) & (df.task == a.task) & (df.condition == a.condition)]
    rows = rows[rows.output.fillna("").str.len() > 50]
    row = rows.iloc[min(a.sample, len(rows) - 1)]
    title(f"Claude's {a.task} for {a.name}  ({a.condition} prompt, sample {int(row['sample'])} of {len(rows)})")
    show(row.output, a.name)
    print(f"\n{DIM}Saved output from the full run: results/outputs.csv ({len(df)} rows, model {row.model}). "
          f"Name in yellow, gendered pronouns in red.{END}")


def cmd_results(a):
    rc = json.loads((ROOT / "results" / "robustness_checks.json").read_text())
    df = pd.read_csv(ROOT / "results" / "outputs.csv")
    hire = df[df.task == "hire"]
    hr = hire.output.str.contains("DECISION: HIRE").groupby([hire.race, hire.gender]).mean()
    cv = rc["detector_cv"]
    m = rc["default_pronoun_vs_name_gender"]
    pr = rc["gendered_pronoun_rate_by_condition"]
    title("AUDIT RESULTS (800 Claude outputs)")
    print(f"{B}1. Decisions{END}  hire rate by group:")
    for (r, g), v in hr.items():
        print(f"     {r:>5} {g:<6}  {v:.0%}")
    print(f"{B}2. Pronouns{END}   letters using he/she:  default {RED}{pr['default']:.1%}{END}   with instruction {GRN}{pr['mitigation']:.1%}{END}")
    print(f"               matched the name's gender: {m['matches_name_gender']} of {m['gendered']}")
    print(f"{B}3. Detector{END}   gender guessed with the NAME hidden (chance 50%):")
    print(f"               default {RED}{cv['name_masked|gender|default']['mean']:.0%}{END}   with instruction {GRN}{cv['name_masked|gender|mitigation']['mean']:.0%}{END}")
    k = "name_and_pronouns_masked|gender|default"
    if k in cv:
        print(f"               with name AND pronouns hidden (default): {cv[k]['mean']:.0%}")
    print(f"\n{DIM}Computed by src/robustness_checks.py -> results/robustness_checks.json{END}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prompt"); p.add_argument("names", nargs="+")
p.add_argument("--task", default="letter", choices=["letter", "hire"]); p.add_argument("--condition", default="default", choices=["default", "mitigation"])
p.set_defaults(func=cmd_prompt)
w = sub.add_parser("write"); w.add_argument("name")
w.add_argument("--task", default="letter", choices=["letter", "hire"]); w.add_argument("--condition", default="default", choices=["default", "mitigation"])
w.add_argument("--sample", type=int, default=0); w.add_argument("--live", action="store_true")
w.set_defaults(func=cmd_write)
r = sub.add_parser("results"); r.set_defaults(func=cmd_results)
a = ap.parse_args(); a.func(a)
