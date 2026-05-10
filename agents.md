# EgoCross Agent Rules

## Project

- Task: EgoCross egocentric video multiple-choice QA.
- Input/output: video frames + question/options -> one letter `A/B/C/D`.
- Current repo on server: `/share/home/group9/lsg/LlamaFactory`.
- Local mirror workspace may not have server model/data paths. Do not assume server paths exist locally.

## Data And Models

- Server support data: `/share/home/group9/data/egocross`.
- Server test data: `/share/home/group9/data/egocross_full/egocross_testbed`.
- Test query: `/share/home/group9/data/egocross_full/egocross_testbed/egocross_testbed_imgs.json`.
- Submission template: `/share/home/group9/lsg/LlamaFactory/submission_template.json`.
- Original baseline model: `saves/egocross/qwen3vl4b/full_sft_32k_200k`.
- Weighted SFT model: `saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1`.
- External GRPO candidate: `/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212`.

## Current Best Candidate

- Model: external GRPO candidate.
- Inference: `direct` prompt, `--default-max-frames 12`, `--frame-sampling tail_dense`, `--frame-route VID006=4`.
- Output: `egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f`.
- Score record: Overall `0.536050`, Surgery `0.533569`, Industry `0.538776`, XSports `0.459350`, Animal `0.639344`, coverage `1.0`.
- Recent challenger: LoRA DPO B from the external GRPO model (`lr5e-6`, `beta=0.03`, `pref_ftx=0.05`, memsafe `cutoff_len=24576`, image/video pixels `131072`) passed fold/support format checks but fixed test evaluation reached Overall `0.526646`, coverage `1.0`; it should not replace the GRPO best.
- Recent support-only inference grid (`egocross_outputs/support_eval/grpo_infer_grid_20260510_161738`): `ENIGMA=8 + VID006=4` tied the baseline support score at `0.8625` with identical error samples, improved `ok_after_frame_retry` from `12` to `0`, and reduced runtime from `201.942s` to `161.924s`; treat it as robustness/runtime-only and do not run hidden test for it.
- Recent fixed inference challenger: `direct + tail_dense + max12 + VID006=4 + ExtrameSportFPV=8 + temperature=0` was selected from support/fold (`70/80`, fold mean `0.8750`) and run once on hidden test; it tied the current best exactly at Overall `0.536050` with identical domain scores, so it should not replace the simpler current best and must not be used for further hidden-driven router tuning.
- Post-test support/fold diagnostic: `direct + tail_dense + max12 + VID006=4 + VID059=8 + temperature=0` reached full support `70/80` and fold acc `0.95, 0.90, 0.85, 0.80` (mean `0.8750`). It is narrower than `ExtrameSportFPV=8`, but because it was explored after a related fixed hidden test, record it as support/fold-positive only and do not run another hidden test for this rule in the same feedback cycle.

## Compliance

- Do not infer hidden labels by leaderboard probing, reverse engineering, or exploiting evaluation behavior.
- Do not manually label hidden test samples.
- Do not tune router/fallback/prompt/frame strategy from hidden leaderboard or hidden domain feedback.
- Do not use the negative LoRA DPO B hidden result to launch hidden-driven sweeps. Treat it as a recorded fixed challenger evaluation only.
- Final reports should say strategies were fixed using support/validation data and robustness diagnostics, with hidden test used only for final evaluation.
- For new inference strategy selection, run fixed candidates on public support set with `scripts/egocross_support_eval.py`, then apply the chosen strategy unchanged to hidden test.

## File Safety

- Do not overwrite protected models, old best outputs, or official template files.
- Use new output directories under `egocross_outputs/` for every experiment.
- Keep `predictions.json`, `raw_outputs.json`, `metrics_summary.txt`, and `submission.zip` together in each output directory.
- Keep active scripts in place; avoid moving validated entrypoints unless all commands and docs are updated.
- Preserve answer-only behavior. Avoid CoT/few-shot/enhanced reasoning for final submissions unless validated on public support data.

## Stable Commands

Start vLLM for multimodal Qwen/Qwen-derived models:

```bash
cd /share/home/group9/lsg/LlamaFactory
conda activate lsg
export PATH="/share/home/group9/miniconda3/envs/lsg/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
  --model /share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212 \
  --port 8000 \
  --served-model-name egocross \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --mm-processor-cache-gb 0
```

Run current best candidate:

```bash
python scripts/egocross_router_infer.py \
  --template submission_template.json \
  --default-prompt-mode direct \
  --default-max-frames 12 \
  --frame-sampling tail_dense \
  --frame-route VID006=4 \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
```

Evaluate a fixed strategy on support set:

```bash
python scripts/egocross_support_eval.py \
  --support-dir /share/home/group9/data/egocross \
  --prompt-mode direct \
  --max-frames 12 \
  --frame-sampling tail_dense \
  --frame-route VID006=4 \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir egocross_outputs/support_eval/grpo_direct_tail_dense_max12_vid006_4f
```

Run the default support strategy grid:

```bash
python scripts/run_egocross_support_eval_grid.py
```

The default grid is inference-only and conservative: baseline `direct_tail_dense_max12_vid006_4f`, support candidate `direct_tail_dense_max12_enigma_8f_vid006_4f`, and diagnostic-only `max8`, `endpoint`, and `uniform` variants. Historical negative candidates such as `max16` or `strict_direct` are explicit-only via `--candidates`.

The grid writes separate directories under `egocross_outputs/support_eval/` and a summary TSV at `egocross_outputs/support_eval/_grid_summary.tsv`. It must not overwrite existing experiment outputs.

Dynamic frame candidates are explicit-only:

```bash
python scripts/run_egocross_support_eval_grid.py \
  --support-dir /share/home/group9/data/egocross \
  --base-url http://127.0.0.1:8000/v1 \
  --out-root egocross_outputs/support_eval/grpo_dynamic_frame_${STAMP} \
  --log-dir logs/support_eval_dynamic_frame_${STAMP} \
  --candidates direct_query_diverse_max12_vid006_4f,direct_query_diverse_tail_max12_vid006_4f
```

Use `scripts/egocross_preview_frame_sampling.py` to inspect `tail_dense`, `query_diverse`, and `query_diverse_tail` selections without calling vLLM. These samplers are deterministic and use public input metadata only: question type, question/options text, option time ranges, timeline coverage, and diversity.

Analyze support or fold predictions:

```bash
python scripts/egocross_support_analyze.py \
  --predictions egocross_outputs/support_eval/grpo_direct_tail_dense_max12_vid006_4f/support_predictions.json \
  --metrics egocross_outputs/support_eval/grpo_direct_tail_dense_max12_vid006_4f/support_metrics.json \
  --support-dir /share/home/group9/data/egocross \
  --output-dir egocross_outputs/support_eval/grpo_direct_tail_dense_max12_vid006_4f_analysis
```

Promotion gate for any new inference router: full support beats baseline by at least one sample; four-fold mean beats `0.8625` with at least `3/4` folds not below baseline; `coverage=1.0`, `parse_fail=0`, `error_fallback=0`; no obvious per-domain or answer distribution collapse. If the gate fails, record a support negative result and do not run hidden test.

Latest support grid result: `direct_tail_dense_max12_enigma_8f_vid006_4f` did not meet the promotion gate because it tied full support accuracy. It may be useful as a runtime/robustness reference, but it is not a new accuracy candidate.

## Known Pitfalls

- VID006 is long and expensive; use `--frame-route VID006=4` for the GRPO tail-dense setup.
- Router retry only handles explicit context-length errors. vLLM 500/deepstack/mm_hash errors usually require restarting vLLM.
- For training with transformers, `rope_scaling` may need train-safe config; for vLLM, null `rope_scaling` has been safer historically.
