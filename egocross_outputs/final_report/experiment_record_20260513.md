# EgoCross Final Experiment Record, 2026-05-13

## Best Kept Candidate

- Model: `/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212`
- Inference: `direct + tail_dense + max12 + VID006=4 + temperature=0`
- Fixed hidden score: Overall `0.536050`, coverage `1.0`
- Output kept: `egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f`

## Fixed Challengers That Did Not Promote

- DPO LoRA B from external GRPO, fixed hidden Overall `0.526646`; not promoted.
- KTO v1 perm4/rank16/lr3e-6/ep2:
  - support: `0.887500` (71/80), format `1.0`, coverage `1.0`, parse/error `0`
  - fixed hidden: Overall `0.527691`, coverage `1.0`; not promoted.
- KTO v2 perm24/rank8/lr1e-6/ep1:
  - support: `0.875000` (70/80), format `1.0`, coverage `1.0`, parse/error `0`
  - fixed hidden: Overall `0.533960`, Surgery `0.526502`, Industry `0.538776`, XSports `0.459350`, Animal `0.639344`, coverage `1.0`; not promoted.

## Compliance Note

Hidden results above are recorded as fixed evaluations only. The final kept strategy remains the pre-existing external GRPO best. Do not use the challenger hidden scores to tune prompt, router, frame sampling, checkpoint selection, reward, or training hyperparameters.

## Cleanup Note

Large non-promoted model weights and failed/obsolete merged copies were removed after this record was written. Small inference outputs, support metrics, configs, and logs are retained for report writing.
