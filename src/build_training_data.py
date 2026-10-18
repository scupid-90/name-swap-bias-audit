"""Part 2 step 1: build name-swap training pairs for fine-tuning.

Takes the least-tilted outputs (lowest bias_score from detector.py's scoring pass) and
refills them with EVERY name in data/names.csv. The idea: teach a small model that a letter's
wording shouldn't change when only the name changes, by showing it the same good letter
under many different names.

Usage:
  python src/build_training_data.py --scored results/detector_scores.csv \
      --outputs results/outputs.csv --out data/sft_pairs.jsonl --top-frac 0.3
"""
import argparse, json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=str(ROOT / "results" / "detector_scores.csv"))
    ap.add_argument("--outputs", default=str(ROOT / "results" / "outputs.csv"))
    ap.add_argument("--names", default=str(ROOT / "data" / "names.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "sft_pairs.jsonl"))
    ap.add_argument("--top-frac", type=float, default=0.3,
                     help="fraction of least-tilted (most neutral) letters to use as templates")
    ap.add_argument("--within-condition", default=None,
                     help="select the top-frac most neutral letters WITHIN this prompt condition only "
                          "(e.g. 'default'). Without it, mitigation letters crowd out default ones.")
    args = ap.parse_args()

    scored = pd.read_csv(args.scored)
    outputs = pd.read_csv(args.outputs)
    names = pd.read_csv(args.names)

    if args.within_condition:
        scored = scored[scored.condition == args.within_condition]
    scored = scored.sort_values("bias_score")
    n_keep = max(1, int(len(scored) * args.top_frac))
    keep_keys = scored.head(n_keep)[["name", "task", "condition", "sample"]]

    merged = outputs.merge(keep_keys, on=["name", "task", "condition", "sample"])
    merged = merged[merged.condition == "default"]  # train on the plain prompts, not the mitigation ones

    written = 0
    with open(args.out, "w") as f:
        for _, row in merged.iterrows():
            masked = re.sub(re.escape(row["name"]), "[CANDIDATE]", str(row.output), flags=re.IGNORECASE)
            for new_name in names.name:
                filled = masked.replace("[CANDIDATE]", new_name)
                example = {
                    "messages": [
                        {"role": "user", "content": f"Write a reference letter for {new_name}, "
                            "a candidate applying for a junior software engineer position."},
                        {"role": "assistant", "content": filled},
                    ]
                }
                f.write(json.dumps(example) + "\n")
                written += 1
    print(f"Wrote {written} training examples ({len(merged)} template letters x {len(names)} names; {n_keep} selected) to {args.out}")


if __name__ == "__main__":
    main()
