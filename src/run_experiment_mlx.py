"""Re-run the SAME audit protocol against the local base model and the LoRA-tuned model,
so you can compare before/after. Mirrors run_experiment.py's structure but calls mlx_lm
locally instead of the Claude API -- no API key needed for this part.

Usage:
  python src/run_experiment_mlx.py --tag base   --samples 3
  python src/run_experiment_mlx.py --tag tuned  --samples 3 --adapter models/lora_adapters
"""
import argparse, csv, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["name", "race", "gender", "task", "condition", "sample", "model", "temperature", "output"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapters; omit for the base model")
    ap.add_argument("--tag", required=True, help="'base' or 'tuned' -- goes in the output filename")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    mx.random.seed(0)  # reproducible sampling
    # Sample at the same temperature as the Claude audit (config.TEMPERATURE). mlx_lm's default is
    # greedy decoding, which made every 'sample' for a given prompt identical.
    sampler = make_sampler(temp=config.TEMPERATURE)

    print(f"Loading {args.model}" + (f" + adapter {args.adapter}" if args.adapter else "") + " ...")
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    names = pd.read_csv(ROOT / "data" / "names.csv")
    out_path = ROOT / "results" / f"outputs_mlx_{args.tag}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in names.itertuples():
            for task in config.TASKS:
                for condition in config.CONDITIONS:
                    prompt = config.build_prompt(task, condition, r.name)
                    messages = [{"role": "user", "content": prompt}]
                    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    for s in range(args.samples):
                        text = generate(model, tokenizer, prompt=formatted, max_tokens=config.MAX_TOKENS, sampler=sampler, verbose=False)
                        w.writerow(dict(name=r.name, race=r.race, gender=r.gender, task=task,
                                        condition=condition, sample=s, model=f"mlx:{args.tag}",
                                        temperature=config.TEMPERATURE, output=text))
                        f.flush()
            print(f"  done: {r.name}")
    print("done ->", out_path)


if __name__ == "__main__":
    main()
