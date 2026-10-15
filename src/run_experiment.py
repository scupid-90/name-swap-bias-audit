"""Run the name-swap experiment and append every output to results/outputs.csv.

Usage:
  python src/run_experiment.py --dry-run --samples 2      # no API key, fake outputs, tests the pipeline
  python src/run_experiment.py --samples 1                # small pilot with the real API
  python src/run_experiment.py                             # full run (SAMPLES_PER_NAME from config)

Safe to stop and restart: rows already in the CSV are skipped.
"""
import argparse, csv, os, random, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "outputs.csv"
FIELDS = ["name", "race", "gender", "task", "condition", "sample", "model", "temperature", "output"]


def fake_output(task, name):
    """Placeholder text for --dry-run only. Contains NO real model behaviour."""
    if task == "hire":
        return f"DECISION: {random.choice(['HIRE', 'NO HIRE'])}\n{name} has solid skills and experience."
    words = random.sample(["confident", "helpful", "warm", "ambitious", "supportive", "analytical"], 3)
    return f"{name} is {words[0]}, {words[1]}, and {words[2]}. I recommend them."


def call_model(client, task, condition, name):
    prompt = config.build_prompt(task, condition, name)
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=config.MODEL, max_tokens=config.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                # anthropic>=1.14 dropped `temperature` as a typed kwarg (replaced by
                # output_config.effort for reasoning depth). extra_body still reaches the
                # server as a raw field, which is the only way left to request sampling
                # temperature for SAMPLES_PER_NAME > 1 to produce varied outputs.
                extra_body={"temperature": config.TEMPERATURE},
            )
            return "".join(b.text for b in msg.content if b.type == "text")
        except Exception as e:  # rate limits, transient errors
            wait = 2 ** attempt
            print(f"  retry in {wait}s ({type(e).__name__})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("API failed after 5 attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=config.SAMPLES_PER_NAME)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = pd.read_csv(ROOT / "data" / "names.csv")
    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        done = set(zip(old.name, old.task, old.condition, old["sample"]))
    new_file = not OUT.exists()

    todo = [(r, t, c, s) for r in names.itertuples() for t in config.TASKS
            for c in config.CONDITIONS for s in range(args.samples)
            if (r.name, t, c, s) not in done]
    print(f"{len(todo)} calls to make ({len(done)} already saved)")

    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, (r, t, c, s) in enumerate(todo, 1):
            text = fake_output(t, r.name) if args.dry_run else call_model(client, t, c, r.name)
            w.writerow(dict(name=r.name, race=r.race, gender=r.gender, task=t, condition=c, sample=s,
                            model="dry-run" if args.dry_run else config.MODEL,
                            temperature=config.TEMPERATURE, output=text))
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
