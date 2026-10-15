# Name-Swap Bias Audit

**Same Candidate, Different Words: A Name-Substitution Framework for Measuring and Mitigating Racial and Gender Bias in Claude's Outputs**

An output-level audit of a large language model. Every prompt is identical except for one thing: the candidate's first name, chosen to signal race and gender. If the generated text changes, the name caused it.

> **Status:** work in progress. A small real pilot has been run against the live Claude API
> (20 names x 2 tasks x 2 conditions x 1 sample = 80 outputs; see `results/outputs.csv`), and the
> full local countermeasure pipeline (detector, training-pair construction, LoRA fine-tune,
> re-audit) has been run end to end on that pilot data -- see **Pilot countermeasure result**
> below. The full-scale run (800 outputs, `SAMPLES_PER_NAME = 10`) has not been run yet; the
> numbers below are from the pilot only and are not yet statistically meaningful on their own.

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

### Pilot countermeasure result

Ran once, end to end, on pilot-scale data: 16 training examples (12 low-`bias_score` letters x
every name, held-out split), 300 LoRA iterations on Qwen2.5-1.5B-Instruct, then re-audited both
the base and the tuned model on 20 names x 2 tasks x 2 conditions x 3 samples (120 letters each),
scored by the same race/gender detector trained on the real Claude pilot data:

| | race accuracy (chance 0.500) | gender accuracy (chance 0.500) | mean bias_score |
|---|---|---|---|
| base (untuned) | 0.500 | 0.975 | 0.571 |
| LoRA-tuned | 0.500 | 0.500 | 0.555 |

Gender was strongly detectable from the base model's letters alone (97.5% accuracy, name
masked) and fell to chance after fine-tuning. Race was already at chance in both, so there was no
race signal for the detector to remove at this scale.

**Caveat, stated plainly:** the LoRA run trained on only 16 examples and the training loss
dropped to ~0.01, i.e. the model is close to memorizing those examples rather than learning a
general pattern. A gender accuracy of chance is consistent with real debiasing, but is also
consistent with the tuned model producing more homogeneous, less name-responsive letters overall
(a "mode collapse" explanation that is not a countermeasure success). A manual read of several
`results/outputs_mlx_tuned.csv` letters across different names is needed to tell these apart
before reporting this as a positive result, and the full-scale run (more templates, more names)
is needed before drawing a real conclusion either way.

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
