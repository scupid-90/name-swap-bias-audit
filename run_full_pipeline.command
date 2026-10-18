#!/bin/bash
# Full-scale pipeline runner for name-swap-bias-audit.
# Double-click in Finder (or `bash run_full_pipeline.command`) to start.
# Progress: results/pipeline_status.txt (one line per stage) and results/pipeline_full.log (everything).
# Safe to re-run: the Claude API step resumes where it stopped; pilot files are backed up once.
cd "$(dirname "$0")" || exit 1
mkdir -p results
STATUS="results/pipeline_status.txt"
LOG="results/pipeline_full.log"
stamp(){ echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$STATUS"; }

# Keep the Mac awake for 12 hours, independent of whether this script finishes or fails.
# (-d display, -i idle, -m disk, -s system on AC power. Lid must stay OPEN.)
nohup caffeinate -dims -t 43200 >/dev/null 2>&1 &
stamp "CAFFEINATE on for 12h (pid $!)"

run(){  # hard stage: stop the pipeline if it fails
  local name="$1"; shift
  stamp "START $name"
  if "$@" >> "$LOG" 2>&1; then stamp "DONE  $name"
  else stamp "FAILED $name (exit $?) -- see results/pipeline_full.log"; exit 1; fi
}
soft(){ # soft stage: log failure but keep going
  local name="$1"; shift
  stamp "START $name"
  if "$@" >> "$LOG" 2>&1; then stamp "DONE  $name"
  else stamp "WARN  $name failed (exit $?), continuing"; fi
}

if [ ! -f .env ]; then stamp "FAILED preflight: .env with ANTHROPIC_API_KEY is missing"; exit 1; fi
set -a; source .env; set +a
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
stamp "Environment ready ($(python --version 2>&1))"

if [ ! -d results/pilot ]; then
  mkdir -p results/pilot models/pilot
  cp results/outputs.csv results/outputs_mlx_base.csv results/outputs_mlx_tuned.csv results/detector_scores*.csv results/pilot/
  cp data/sft_pairs.jsonl results/pilot/
  cp models/detector_*.pkl models/detector_results.json models/pilot/
  cp -R models/lora_adapters models/pilot/lora_adapters
  stamp "Pilot artifacts backed up to results/pilot and models/pilot"
fi

run  "1/8 Claude API full run (resumes; 800 outputs total)" python src/run_experiment.py
soft "2/8 Statistical analysis (analyze.py)" python src/analyze.py
run  "3/8 Train bias detector on full Claude data" python src/detector.py train results/outputs.csv
run  "3/8 Score Claude letters" python src/detector.py score results/outputs.csv --out results/detector_scores.csv
run  "4/8 Build training pairs" python src/build_training_data.py --top-frac 0.3
run  "5/8 LoRA fine-tune (MLX)" bash src/train_mlx.sh
run  "6/8 Re-audit base model" python src/run_experiment_mlx.py --tag base --samples 3
run  "7/8 Re-audit tuned model" python src/run_experiment_mlx.py --tag tuned --samples 3 --adapter models/lora_adapters
run  "8/8 Score base outputs" python src/detector.py score results/outputs_mlx_base.csv --out results/detector_scores_base.csv
run  "8/8 Score tuned outputs" python src/detector.py score results/outputs_mlx_tuned.csv --out results/detector_scores_tuned.csv
run  "8/8 Countermeasure summary" python src/summarize_countermeasure.py
stamp "ALL DONE"
# --- appended while running: start the corrected countermeasure (v2) automatically ---
bash ./run_countermeasure_v2.command
