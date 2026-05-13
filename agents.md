# EgoCross RL Training Agent Rules

## Mission

- Task: EgoCross egocentric video multiple-choice QA.
- Project status: finished and cleaned for report writing.
- Final kept candidate is the external GRPO model with `direct + tail_dense + max12 + VID006=4 + temperature=0`.
- Do not continue hidden-test-driven inference tuning. Treat previous inference work as history only.
- Output format remains one letter `A/B/C/D`; final submissions use `predictions.json` only.

## Paths

- Server repo: `/share/home/group9/lsg/LlamaFactory`.
- Support data: `/share/home/group9/data/egocross`.
- Test data: `/share/home/group9/data/egocross_full/egocross_testbed`.
- Test JSON: `/share/home/group9/data/egocross_full/egocross_testbed/egocross_testbed_imgs.json`.
- Submission template: `/share/home/group9/lsg/LlamaFactory/submission_template.json`.
- Local mirror for debugging only: repo sibling `../data_local/egocross`.

## Baselines And Protected Assets

- Current best model: `/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212`.
- Current best inference: `direct + tail_dense + max12 + VID006=4 + temperature=0`.
- Current best test score: Overall `0.536050`, Surgery `0.533569`, Industry `0.538776`, XSports `0.459350`, Animal `0.639344`, coverage `1.0`.
- Final report record: `egocross_outputs/final_report/experiment_record_20260513.md`.
- Compact report index: `docs/egocross/README.md`.
- Do not overwrite:
  - `submission_template.json`
  - `saves/egocross/qwen3vl4b/full_sft_32k_200k`
  - `saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1`
  - `/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212`
  - `egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f`

## Compliance

- Do not infer hidden labels by repeated submissions, leaderboard probing, domain score feedback, or manual hidden-test inspection.
- Do not tune training, router, prompt, frame sampling, checkpoint selection, or reward design from hidden-test feedback.
- Hidden results may be recorded as fixed evaluations only.
- Use support set, fold splits, training logs, reward/format stability, coverage, parse failures, and error rates for model selection.
- Reports should say strategies/models were selected using support/fold diagnostics; hidden test is final fixed evaluation only.

## Training Rules

- Start by reading this file and `docs/egocross/README.md`, then selectively search `README_EGOCROSS_lsg.md`; do not read the whole historical README into context.
- Prefer reproducible fold-first protocols before full training.
- Keep one variable isolated per experiment: base model, data, reward, RL hyperparameters, or inference strategy.
- For RL/DPO/GRPO experiments, preserve answer-only behavior and evaluate with the fixed current-best inference strategy unless a support/fold protocol explicitly tests otherwise.
- Do not promote a model unless support/fold metrics improve over the external GRPO baseline and format/error metrics remain stable.
- Record every experiment with config path, base model, data, reward/loss, hyperparameters, output dir, support/fold result, and conclusion.

## Known Training Pitfalls

- Prior LoRA DPO from the external GRPO looked good on folds (`B` mean `0.8750`) but fixed test dropped to Overall `0.526646`; do not repeat the same small DPO sweep as the main plan.
- KTO v1 from GRPO improved support to `0.887500` but fixed test dropped to Overall `0.527691`; not promoted.
- KTO v2 perm24/rank8/lr1e-6 support was `0.875000` and fixed test was Overall `0.533960`; still below current best and not promoted.
- Full DPO can OOM; prior memsafe profile used `cutoff_len=24576`, image/video pixels `131072`, and `LLAMAFACTORY_LOGPS_CHUNK_SIZE=128`.
- Transformers training/export and vLLM inference may need different Qwen3-VL rope config:
  - train/export-safe view may use `text_config.rope_scaling={"rope_type":"default"}`
  - vLLM merged inference copies historically prefer `text_config.rope_scaling=null`
- vLLM multimodal Qwen/Qwen-derived inference is more stable with `--enforce-eager`, `--mm-processor-cache-gb 0`, and careful context length.

## Historical Index

Use these report/historical files:

- `docs/egocross/README.md`
- `docs/egocross/file_map.md`
- `egocross_outputs/final_report/experiment_record_20260513.md`

Use `rg` on `README_EGOCROSS_lsg.md` for older history:

- `rg -n "0\\.536050|DPO LoRA B|Fixed challenger|Current best|GRPO direct tail_dense max12" README_EGOCROSS_lsg.md`
- `rg -n "Conservative LoRA DPO|Current DPO memory profile|Fold protocol|Result record" README_EGOCROSS_lsg.md`
- `rg -n "vLLM|rope_scaling|max-model-len|context length|mm_hash|deepstack" README_EGOCROSS_lsg.md`
- `rg -n "hidden|leaderboard|Do not|submission_template|Do not promote|Do not overwrite" README_EGOCROSS_lsg.md`
