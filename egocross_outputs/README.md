# EgoCross Outputs

Each subdirectory is one inference experiment. Keep `predictions.json`, `submission.zip`, `metrics_summary.txt`, and `raw_outputs.json` together so every submission remains reproducible.

## Current Candidate

```text
why_grpo_direct_tail_dense_max12_vid006_4f
Overall: 0.536050
Surgery: 0.533569
Industry: 0.538776
XSports: 0.459350
Animal: 0.639344
```

Use this as the current submission candidate unless a future fixed strategy is selected from support/validation.

## Historical Groups

```text
baseline_*                         Original full-SFT baseline outputs
weighted_answer_only_*             Weighted answer-only model outputs
router_*                           Domain/model routing or blend outputs
full_sft_*_cot_*                   CoT/short-CoT experiments
fewshot_*                          Few-shot and enhanced-reasoning probes
why_grpo_*                         External GRPO model inference variants
support_eval/                      Public support-set strategy checks; not submissions
scratch_tests/                     Small debugging subsets; not final submissions
```

## Safety

- Do not overwrite existing output directories.
- Do not tune future router/fallback/prompt choices from hidden leaderboard feedback.
- If `raw_outputs.json` contains many errors or `metrics_summary.txt` shows error fallback, do not submit that run.
