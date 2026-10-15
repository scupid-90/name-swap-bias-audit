#!/bin/bash
# Part 2 step 2: fine-tune a small model with MLX + LoRA on your Mac.
# Must be run ON a Mac with Apple Silicon. Will not run in a cloud container.
#
# Usage: bash src/train_mlx.sh
#
# Requires: pip install mlx-lm   (Apple's MLX framework + LoRA trainer)
# Model: defaults to a small instruct model. Check Hugging Face for the current best small
# option before running -- model names change. Llama models need you to accept their license
# on huggingface.co first and run `huggingface-cli login`.
set -e
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_DIR="data/mlx_sft"
ADAPTER_DIR="models/lora_adapters"

mkdir -p "$DATA_DIR" "$ADAPTER_DIR"

echo "Splitting data/sft_pairs.jsonl into train/valid/test for mlx_lm..."
python3 - <<PYEOF
import json, random
random.seed(0)
rows = [json.loads(l) for l in open("data/sft_pairs.jsonl")]
random.shuffle(rows)
n = len(rows)
splits = {"train": rows[:int(n*0.8)], "valid": rows[int(n*0.8):int(n*0.9)], "test": rows[int(n*0.9):]}
for name, part in splits.items():
    with open(f"$DATA_DIR/{name}.jsonl", "w") as f:
        for r in part:
            f.write(json.dumps(r) + "\n")
    print(name, len(part))
PYEOF

echo "Fine-tuning $MODEL with LoRA (this is the slow, CPU/GPU-heavy step)..."
python3 -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --iters 300 \
    --batch-size 2 \
    --num-layers 8

echo "Done. Adapters saved to $ADAPTER_DIR"
echo "Try it:  python3 -m mlx_lm generate --model $MODEL --adapter-path $ADAPTER_DIR --prompt \"Write a reference letter for Jamal, a candidate applying for a junior software engineer position.\""
