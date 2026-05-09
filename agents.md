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

## Compliance

- Do not infer hidden labels by leaderboard probing, reverse engineering, or exploiting evaluation behavior.
- Do not manually label hidden test samples.
- Do not tune router/fallback/prompt/frame strategy from hidden leaderboard or hidden domain feedback.
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

## Known Pitfalls

- VID006 is long and expensive; use `--frame-route VID006=4` for the GRPO tail-dense setup.
- Router retry only handles explicit context-length errors. vLLM 500/deepstack/mm_hash errors usually require restarting vLLM.
- For training with transformers, `rope_scaling` may need train-safe config; for vLLM, null `rope_scaling` has been safer historically.
