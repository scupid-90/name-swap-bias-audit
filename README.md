# Name-Swap Bias Audit

**Same Candidate, Different Words: A Name-Substitution Framework for Measuring and Mitigating Racial and Gender Bias in Claude's Outputs**

An output-level audit of a large language model. Every prompt is identical except for one thing: the candidate's first name, chosen to signal race and gender. If the generated text changes, the name caused it.

> **Status:** complete. The full run (20 names x 2 tasks x 2 prompt conditions x 10 samples = 800
> outputs from `claude-sonnet-5-5`) has been collected and analysed, robustness checks are done, and
> the local LoRA countermeasure has been trained and re-audited at full scale (two versions).
> The pilot (80 outputs) is kept in `results/pilot/` for reference.

## Project structure

```
name-swap-bias-audit/
├── src/                          # all code
│   ├── config.py                 # model, settings, candidate profile, CLEAR prompts
│   ├── run_experiment.py         # Part 1: sends every prompt to the Claude API -> results/outputs.csv
│   ├── analyze.py                # hire rates, tone words, regression -> results/summary.txt
│   ├── detector.py               # bias detector: hide name -> TF-IDF -> logistic regression
│   ├── robustness_checks.py      # pronoun analysis + detector with names AND pronouns hidden
│   ├── build_training_data.py    # Part 2: name-neutral training pairs for the countermeasure
│   ├── train_mlx.sh              # LoRA fine-tune of Qwen2.5-1.5B-Instruct (Apple MLX)
│   ├── run_experiment_mlx.py     # re-audit the base / tuned small model with the same prompts
│   ├── summarize_countermeasure.py
│   └── compare_countermeasures.py
├── data/
│   ├── names.csv                 # the 20 names (4 race x gender groups)
│   ├── sft_pairs.jsonl           # countermeasure training pairs
│   └── mlx_sft/                  # train / valid / test splits for MLX
├── results/
│   ├── outputs.csv               # the 800 raw Claude outputs
│   ├── summary.txt, scores.csv   # analysis results
│   ├── robustness_checks.*       # pronoun + detector checks
│   ├── outputs_mlx_*.csv         # base vs tuned small-model letters
│   ├── countermeasure_*          # countermeasure comparison (base, v1, v2)
│   ├── pilot/                    # the 80-output pilot
│   └── v1_12templates/           # first countermeasure attempt (kept for comparison)
├── models/                       # trained detectors + LoRA adapters
├── paper/                        # 2-page manuscript (Word, PDF, LaTeX) and figure
├── run_full_pipeline.command     # runs the whole pipeline end to end
└── run_countermeasure_v2.command # runs the final countermeasure
```

## Pipeline

```
data/names.csv + CLEAR template (config.py)
        |  run_experiment.py
        v
Claude API (claude-sonnet-5-5, temperature 1.0)  ->  results/outputs.csv (800 outputs)
        |  analyze.py                       |  detector.py + robustness_checks.py
        v                                   v
decisions, tone, regression          name hidden -> TF-IDF -> classifier (tested on unseen names)
                                            |
                                            v  build_training_data.py -> train_mlx.sh -> run_experiment_mlx.py
                                     countermeasure: Qwen2.5-1.5B + LoRA, re-audited the same way
```

## Main results (full run)

| Measure | Default prompt | With bias-reduction instruction |
|---|---|---|
| Hire rate (all groups) | 100% | 100% |
| Letters using "he"/"she" | 82.8% | 2.6% |
| Gender detected, name hidden (chance 50%) | 94% | 51% |
| Gender detected, name and pronouns hidden | 67% | 51% |

- Every candidate was hired, so a decision-only audit finds no bias.
- By default, 163 of 164 gendered letters matched the gender the name suggests.
- The bias-reduction instruction (Tamkin et al., 2023) removes almost all of this signal.
- 7 of 800 outputs were empty letters and were excluded.
- Detector figures are means over 4 name-grouped folds. Re-running with a newer scikit-learn (1.9)
  assigns names to folds differently and gives about 95% / 47%; the conclusion is the same.

## Research questions

1. **RQ1 (race):** When only the race implied by a name changes, does the model's text change in tone, word choice, or quality?
2. **RQ2 (gender and intersection):** Does the same happen for gender, and do race and gender combine in ways neither explains alone?
3. **RQ3 (mitigation):** Does a bias-reduction instruction in the prompt shrink these differences?

## Design

| Factor | Levels |
|---|---|
| Race (signalled by name) | white, Black |
| Gender (signalled by name) | female, male |
| Prompt condition | default, bias-reduction instruction |
| Task | reference letter; hire / no-hire screening |
| Names | 5 per race-gender group in the starter file (20 total), expandable |
| Samples | 10 per name per condition per task |

With the starter names and 10 samples: 20 names x 2 tasks x 2 conditions x 10 samples = **800 outputs**. With 10 names per group it is 1,600.

### Measures
- Agentic vs. communal word rate in reference letters (words per 100)
- Output length
- Hire rate in the screening task
- Planned: blind human check on a random 10% of outputs

### Analysis
Mixed-effects regression with race, gender, their interaction and prompt condition as predictors, and name as a random effect (outputs for the same name are not independent). Chi-square tests for hire rates. Report effect sizes, not only p-values.

## Frameworks used

Two frameworks from the course (Prof. Simeon O.A., *Prompting Frameworks*, Day 3) structure the work:

- **RHODES** (Research, Hypothesis, Objectives, Development, Execution, Synthesis) structures the study itself: the research questions and hypotheses above, measurable aims, the method, the run, and a limitations-aware write-up.
- **CLEAR** (Context, Logic, Expectations, Action, Restrictions) structures every prompt sent to the model (see `src/config.py`). The same five elements are used in every run; the candidate profile sits in Context, and the bias-reduction instruction is added to Restrictions in the mitigation condition only. Holding the framework constant means it cannot become a second variable.

CO-STAR was not used for the letter task because it adds style and tone instructions, and tone is one of the outcomes being measured.

## Quick start

```bash
git clone <your-repo-url> && cd name-swap-bias-audit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # never commit this

python src/run_experiment.py --dry-run --samples 2   # test the pipeline, no API calls, fake data
python src/run_experiment.py --samples 1             # small real pilot
python src/run_experiment.py                         # full run (safe to stop and restart)
python src/analyze.py                                # writes results/scores.csv and results/summary.txt
```

Settings (model, temperature, candidate profile, prompts) live in `src/config.py`.

## Sources checked

- **Names (`data/names.csv`):** all 20 names appear in Table 1 of Gaddis (2017), the published list of names with perceived-race data. Several are also named in the Bertrand & Mullainathan (2004) text (Emily, Greg, Lakisha, Jamal, Geoffrey, Neil, Latonya, Rasheed). B&M's own Appendix Table 1 was not accessible when checked, so the sources column cites Gaddis. The gender coding of each name is ours (Gaddis's table does not label gender). Perceived-race rates for Kareem, Aisha, Tamika, and Ebony are only in Gaddis's figures, so check them there if you want to report them.
- **Word lists (`src/analyze.py`):** Appendix A of Gaucher, Friesen & Kay (2011). The paper calls them masculine- and feminine-coded; they were built from agentic and communal trait lists, so we use them as the agentic/communal proxy. Stems (marked `*`) match any ending. Note that stems like `Commit*`, `Lead*`, and `Respon*` will match common resume words ("committed", "lead", "responsible"), which is a known limit of dictionary methods.
- **Mitigation prompt (`src/config.py`):** Tamkin et al. (2023), Appendix B.3, Prompt 14 ("Really (1x) don't discriminate"). For the letter task, "this decision" becomes "this letter". Re-read the appendix once to confirm the wording matches before you quote it as verbatim.

## Training a countermeasure (Part 2)

This repo also trains a small local model to mitigate the bias found in Part 1, so the project
is not just an audit. Everything here runs locally on a Mac with Apple Silicon (M-series) and
needs no API key.

1. **Detect.** `python src/detector.py train results/outputs.csv` trains two small classifiers
   (scikit-learn, TF-IDF + logistic regression) that try to guess a candidate's race and gender
   group from a letter alone, with the name masked out. Names are split so the same name's
   samples never appear in both train and test. If accuracy is well above chance, the model's
   wording itself carries a demographic signal.
2. **Score.** `python src/detector.py score results/outputs.csv` scores every output for how
   strongly it leans toward one group (`bias_score`; lower is more neutral).
3. **Build training pairs.** `python src/build_training_data.py` takes the most neutral letters
   (lowest `bias_score`) and refills each with every name in `data/names.csv`, producing
   `data/sft_pairs.jsonl`: the same good letter under many different names. The idea is to teach
   a small model that the name shouldn't change the wording.
4. **Fine-tune.** `bash src/train_mlx.sh` fine-tunes a small instruct model (default:
   Qwen2.5-1.5B-Instruct) with LoRA using Apple's [MLX](https://github.com/ml-explore/mlx-lm)
   framework. Install first with `pip install mlx-lm` (Apple Silicon only). Expect tens of
   minutes on a MacBook Air; keep it plugged in.
5. **Re-audit.** `python src/run_experiment_mlx.py --tag base` and
   `python src/run_experiment_mlx.py --tag tuned --adapter models/lora_adapters` re-run the same
   CLEAR-framework prompts against the untuned and tuned small model, using held-out names not
   seen in step 3. Score both with `detector.py` and compare.

**What this does and doesn't show.** This can show that the countermeasure reduces demographic
signal in the small model, on this task, by this detector's measure. It does not show that
Claude itself is "fixed" -- the training data comes from Claude's own outputs, so any bias the
detector misses is carried straight into the tuned model, and the tuned model is a different
(much smaller) model from the one being audited. Report the result that way.

### Countermeasure result (full scale)

Base and tuned Qwen2.5-1.5B-Instruct were re-audited with the same CLEAR prompts (120 letters each)
and scored by the detector trained on Claude's letters.

| | Untuned | v1 (12 templates) | v2 (60 templates) |
|---|---|---|---|
| Letters with gendered pronouns | 97.5% | 0.8% | 28.3% |
| Gender detected | 97.5% | 47.5% | 60.0% |
| Mean length (words) | 311 | 95.5 | 195 |
| Unique letters | 100% | 85% | 100% |

v1 took the least-biased letters overall; most came from the bias-reduction condition, leaving
only 12 templates, and the model largely memorised them (short, partly repeated letters). v2 picks
templates within the default condition (`--within-condition default`, 60 templates): letters stay
varied and usable, and the gender signal is reduced but not removed. The pilot-scale result is
kept in `results/pilot/` and `models/pilot/`.

## Limitations

- Names carry more than race and gender (class, region, age), so this measures bias toward names.
- One model, one fictional setting; results do not generalise to all AI systems.
- Word lists are a rough proxy for tone, which is why a blind human check is part of the design.
- Text outputs only.

## References

- Bertrand, M. & Mullainathan, S. (2004). Are Emily and Greg more employable than Lakisha and Jamal? *American Economic Review*.
- Wan, Y. et al. (2023). "Kelly is a Warm Person, Joseph is a Role Model": Gender Biases in LLM-Generated Reference Letters. arXiv:2310.09219
- Tamkin, A. et al. (2023). Evaluating and Mitigating Discrimination in Language Model Decisions. arXiv:2312.03689
- Gaddis, S. M. (2017). How Black Are Lakisha and Jamal? Racial Perceptions from Names Used in Correspondence Audit Studies. *Sociological Science*, 4, 469-489.
- Gaucher, D., Friesen, J., & Kay, A. C. (2011). Evidence that gendered wording in job advertisements exists and sustains gender inequality. *Journal of Personality and Social Psychology*.

## License
MIT
