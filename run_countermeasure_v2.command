#!/bin/bash
# Countermeasure v2: same pipeline, but training templates are the most neutral 30% of letters
# WITHIN the default prompt condition (60 templates x 20 names = 1,200 examples), instead of the
# overall most-neutral 30% (which was 108/120 mitigation letters, leaving only 12 default templates).
# v1 artifacts are kept under results/v1_12templates and models/v1_12templates.
# The base model is unchanged, so its re-audit (results/outputs_mlx_base.csv) is reused.
cd "$(dirname "$0")" || exit 1
STATUS="results/pipeline_status.txt"; LOG="results/pipeline_full.log"
stamp(){ echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$STATUS"; }
run(){ local name="$1"; shift; stamp "START $name"
  if "$@" >> "$LOG" 2>&1; then stamp "DONE  $name"; else stamp "FAILED $name (exit $?) -- see results/pipeline_full.log"; exit 1; fi; }
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1

mkdir -p results/v1_12templates models/v1_12templates
cp results/outputs_mlx_tuned.csv results/detector_scores_tuned.csv results/countermeasure_summary.json data/sft_pairs.jsonl results/v1_12templates/ 2>/dev/null
cp -R models/lora_adapters models/v1_12templates/lora_adapters
stamp "v1 (12-template) countermeasure saved to results/v1_12templates"

run "v2 1/5 Build training pairs (within default, 60 templates)" python src/build_training_data.py --top-frac 0.3 --within-condition default
rm -rf models/lora_adapters 2>/dev/null; mkdir -p models/lora_adapters
run "v2 2/5 LoRA fine-tune (MLX)" bash src/train_mlx.sh
run "v2 3/5 Re-audit tuned model" python src/run_experiment_mlx.py --tag tuned --samples 3 --adapter models/lora_adapters
run "v2 4/5 Score tuned outputs" python src/detector.py score results/outputs_mlx_tuned.csv --out results/detector_scores_tuned.csv
run "v2 5/5 Countermeasure summary" python src/summarize_countermeasure.py
stamp "ALL DONE V2"
