# EgoCross File Map

This file explains where EgoCross-specific files live. Existing validated entrypoints are kept in place to avoid breaking commands.

## Root

```text
README_EGOCROSS_lsg.md       Current EgoCross handoff and experiment record
agents.md                    Long-term agent rules and safety constraints
submission_template.json     Official submission template; do not overwrite
```

Legacy root scripts kept for reproducibility:

```text
generate_egocross_submission.py
generate_egocross_submission_cot_test.py
generate_egocross_submission_shortcot_test.py
```

Prefer `scripts/egocross_router_infer.py` for new inference.

## Scripts

```text
scripts/egocross_router_infer.py
  Main inference router. Supports prompt modes, frame routes, retry, voting,
  and frame sampling strategies: uniform / endpoint / tail_dense.

scripts/egocross_support_eval.py
  Runs a fixed inference strategy on the labeled support set and writes
  support_predictions.json, support_metrics.json, and support_metrics.txt.

scripts/egocross_blend_submit.py
  Historical submission blending helper.

scripts/prepare_egocross_weighted_answer_only.py
  Builds weighted answer-only SFT data.

scripts/prepare_egocross_preference_answer_only.py
  Builds answer-only DPO preference data and optional 4-fold support splits.
```

## Configs

```text
configs/egocross_full_sft.yaml
configs/egocross_lora.yaml
configs/egocross_weighted_answer_only_full_ep1.yaml
configs/egocross_dpo_answer_only_full_all_equal_lr1e6_ep1.yaml
configs/egocross_dpo_answer_only_lora_all_equal_lr1e6_ep1.yaml
configs/egocross_answer_only_full_all_equal_lr2e6_ep1.yaml
```

## Data Registry

```text
data/dataset_info.json
```

Registers EgoCross raw, weighted answer-only, all-equal answer-only, and preference datasets. Server paths are used for training; local `../data_local/egocross` is only a mirror for inspection/debugging.

## Outputs

```text
egocross_outputs/
```

Every experiment should write to a new subdirectory containing:

```text
predictions.json
submission.zip
metrics_summary.txt
raw_outputs.json
```

Current best output:

```text
egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
```

## Organization Policy

- Do not move validated scripts or output directories without updating commands and README.
- Add new experiment outputs under `egocross_outputs/<descriptive_name>/`.
- Keep scratch subsets under `egocross_outputs/scratch_tests/`.
- Keep final or candidate submissions with their raw outputs and metrics together.
