# EgoCross Report Notes

This folder is the compact report-facing index for the finished EgoCross work.
The long handoff file `README_EGOCROSS_lsg.md` is kept as historical appendix
only; use `rg` on it when a specific old experiment needs to be recovered.

## Final Kept Candidate

- Model: `/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212`
- Inference: `direct + tail_dense + max12 + VID006=4 + temperature=0`
- Hidden fixed score: Overall `0.536050`, coverage `1.0`
- Kept output: `egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f`

## Challengers Not Promoted

| Candidate | Support / fold signal | Hidden fixed score | Decision |
| --- | --- | --- | --- |
| DPO LoRA B from external GRPO | Fold mean `0.8750`, above GRPO fold mean `0.8625` | Overall `0.526646` | Not promoted |
| KTO v1 perm4/rank16/lr3e-6/ep2 | Support `0.887500`, format/coverage clean | Overall `0.527691` | Not promoted |
| KTO v2 perm24/rank8/lr1e-6/ep1 | Support `0.875000`, format/coverage clean | Overall `0.533960` | Not promoted |

Hidden scores are recorded as fixed evaluations only. They must not be used to
tune prompts, routing, frame sampling, rewards, checkpoints, or training
hyperparameters.

## Report Source Files

- Final experiment record: `egocross_outputs/final_report/experiment_record_20260513.md`
- Output directory guide: `egocross_outputs/README.md`
- File map: `docs/egocross/file_map.md`
- Agent/safety rules: `agents.md`
- Historical appendix: `README_EGOCROSS_lsg.md`

## Server Cleanup State

Large non-promoted model weights were removed from
`/share/home/group9/lsg/LlamaFactory/saves/egocross/qwen3vl4b`.
The remaining local server saves are expected to be:

```text
grpo_v3_train_rope_default
full_sft_32k_200k
weighted_answer_only_full_i4_x4_lr5e6_ep1
```

The two full SFT directories are protected historical assets. The train-safe
GRPO view is a tiny symlink/config view used only for reproducibility.
