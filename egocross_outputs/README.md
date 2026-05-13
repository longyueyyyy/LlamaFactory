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

Use this as the final kept submission candidate. Later fixed challengers did
not exceed it:

```text
dpo_lora_B_full_direct_tail_dense_max12_vid006_4f
  fixed hidden Overall: 0.526646

kto_lora_from_grpo_answer_reward_perm4_fixed_direct_tail_dense_max12_vid006_4f_20260512_140203
  fixed hidden Overall: 0.527691

kto_perm24_v2_no_merge_fixed_direct_tail_dense_max12_vid006_4f_20260513_011159
  fixed hidden Overall: 0.533960
```

Final report record:

```text
final_report/experiment_record_20260513.md
```

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
final_report/                      Compact final score/cleanup record for report writing
```

## Safety

- Do not overwrite existing output directories.
- Do not tune future router/fallback/prompt choices from hidden leaderboard feedback.
- If `raw_outputs.json` contains many errors or `metrics_summary.txt` shows error fallback, do not submit that run.
- Hidden challenger scores are fixed-evaluation records only; they are not development signals.
