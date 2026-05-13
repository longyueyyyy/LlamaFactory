# EgoCross File Map

This map separates report notes, reproducible entrypoints, experiment outputs,
and protected assets. Existing scripts/configs are kept in their original
locations so old commands remain valid.

## Report And Handoff

```text
docs/egocross/README.md
  Compact final report index.

docs/egocross/file_map.md
  This file.

README_EGOCROSS_lsg.md
  Long historical appendix. Do not read end to end; use rg for specific
  fragments.

agents.md
  Current project rules, protected assets, compliance notes, and final status.

egocross_outputs/final_report/experiment_record_20260513.md
  Final score table and cleanup note.
```

## Protected Assets

Do not overwrite or delete:

```text
submission_template.json
saves/egocross/qwen3vl4b/full_sft_32k_200k
saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1
/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
```

## Primary Scripts

```text
scripts/egocross_router_infer.py
  Main fixed-test inference router. Produces predictions.json and submission.zip.

scripts/egocross_support_eval.py
  Public support-set evaluation with the same inference strategy.

scripts/run_egocross_support_eval_grid.py
  Support-set strategy grid runner used during diagnostics.

scripts/prepare_egocross_grpo_train_safe.py
  Builds a train-safe symlink/config view of the external GRPO model.

scripts/prepare_egocross_preference_answer_only.py
scripts/check_egocross_pref_data.py
  DPO preference-data preparation and validation.

scripts/prepare_egocross_kto_answer_only.py
scripts/check_egocross_kto_data.py
  KTO answer-only augmentation preparation and validation.
```

Historical root scripts (`generate_egocross_submission*.py`) are kept for
reproducibility, but new inference should use `scripts/egocross_router_infer.py`.

## Configs

```text
configs/egocross_full_sft.yaml
configs/egocross_weighted_answer_only_full_ep1.yaml
configs/egocross_dpo_*                     Historical DPO configs
configs/egocross_kto_*                     Historical KTO configs
configs/egocross_export_*                  Export configs, kept for audit only
```

The KTO/DPO challenger weights were removed on the server after final
evaluation, but configs and logs are retained for report writing.

## Outputs

```text
egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
  Final kept candidate.

egocross_outputs/final_report
  Final report-facing record.

egocross_outputs/support_eval
  Public support/fold diagnostics. Not submissions.

egocross_outputs/kto_* and egocross_outputs/dpo_*
  Fixed challenger outputs kept as evidence, not promoted.
```

All submission-capable output directories should keep their
`predictions.json`, `submission.zip`, `raw_outputs.json`, and
`metrics_summary.txt` together.

## Logs

```text
logs/egocross_grpo_baseline_fold_eval_20260510_111639
logs/egocross_dpo_lora_folds_abc_memsafe_20260510_030247
logs/egocross_kto_lora_from_grpo_20260512_113232
logs/egocross_kto_perm24_v2_20260512_174523
```

These are useful for reporting training or fold-eval provenance. Python
`__pycache__` folders and ad-hoc smoke artifacts can be removed safely.
