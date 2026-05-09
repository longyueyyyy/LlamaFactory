# EgoCross Experiment README

本文档是 EgoCross 项目的当前交接记录。目标是让后续接手者快速知道：当前最强方案、服务器路径、合规边界、关键脚本、已经尝试过的方向，以及哪些文件不能覆盖。

## 1. 项目概况

任务形式：

```text
多帧第一视角视频帧 + 多选题文本 -> 输出 A/B/C/D
```

最终提交只需要：

```text
submission.zip
└── predictions.json
```

`predictions.json` 每题只填一个答案字母，不提交解释、CoT 或中间证据。

## 2. 比赛合规规则

必须遵守：

```text
Do not attempt to infer hidden labels by probing, reverse engineering, or exploiting evaluation behavior.
Manual per-example labeling of hidden test data is prohibited.
```

本项目执行规则：

```text
1. 不通过反复提交 hidden test、观察榜单/domain 分数来反推标签、定位错误样本或调整 router/fallback/prompt。
2. 不手工标注 hidden test 样本。
3. 推理策略应在提交前固定，依据公开 support/validation、训练日志、格式稳定性、coverage、运行错误率等非 hidden-label 信号决定。
4. 报告中不要写“根据 hidden 榜单反馈调整策略”。可写“基于 support set 与鲁棒性诊断选择固定推理策略，hidden test 仅用于最终评估”。
```

历史 Codabench 分数只作为实验记录。后续不要把 hidden 榜单当开发集。

## 3. 服务器路径

```bash
CODE=/share/home/group9/lsg/LlamaFactory
SUPPORT=/share/home/group9/data/egocross
TESTBED=/share/home/group9/data/egocross_full/egocross_testbed
TEST_JSON=/share/home/group9/data/egocross_full/egocross_testbed/egocross_testbed_imgs.json
TEMPLATE=/share/home/group9/lsg/LlamaFactory/submission_template.json
```

重要模型：

```text
baseline:
saves/egocross/qwen3vl4b/full_sft_32k_200k

weighted answer-only SFT:
saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1

external GRPO candidate:
/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
```

本地数据镜像在仓库同级：

```text
../data_local/egocross
```

本地镜像只用于非破坏性检查和脚本调试。训练/推理命令里的服务器路径不要假设本地可用。

## 4. 当前最强候选

当前最强方案：

```text
model: /share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
prompt: direct
frame_sampling: tail_dense
max_frames: 12
frame_route: VID006=4
router/fallback: none
output: egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
```

分数记录：

```text
Overall: 0.536050
Surgery: 0.533569
Industry: 0.538776
XSports: 0.459350
Animal: 0.639344
coverage: 1.0
```

当前建议：先保留这版作为提交候选，不继续基于 hidden 榜单反馈做细调。`ctx65536 + max12 + no VID006 route + temp0` 已复现同分 0.536050，但工程复杂度更高；`max_frames=16`、`ctx65536 + max24`、`temperature=0.2` 和 `temperature=0.7 + vote` 均低于当前最佳。

## 5. 当前最佳运行命令

启动 vLLM：

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

检查服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

跑当前最强候选：

```bash
cd /share/home/group9/lsg/LlamaFactory
mkdir -p logs

python scripts/egocross_router_infer.py \
  --template submission_template.json \
  --default-prompt-mode direct \
  --default-max-frames 12 \
  --frame-sampling tail_dense \
  --frame-route VID006=4 \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f \
  2>&1 | tee logs/why_grpo_direct_tail_dense_max12_vid006_4f.log
```

输出检查：

```bash
cat egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f/metrics_summary.txt
```

如果 `status_dist` 里有大量 `error_fallback`，不要提交该结果。

## 5.1 Support 固定策略验证

后续如果要比较 prompt / frame_sampling / max_frames，优先在公开 support set 上固定策略后验证，不要根据 hidden 榜单反馈继续调。

支持脚本：

```bash
scripts/egocross_support_eval.py
```

示例：验证当前最强策略在 support set 上的 acc：

```bash
cd /share/home/group9/lsg/LlamaFactory

python scripts/egocross_support_eval.py \
  --support-dir /share/home/group9/data/egocross \
  --prompt-mode direct \
  --max-frames 12 \
  --frame-sampling tail_dense \
  --frame-route VID006=4 \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir egocross_outputs/support_eval/grpo_direct_tail_dense_max12_vid006_4f
```

输出文件：

```text
support_predictions.json
support_metrics.json
support_metrics.txt
```

一次性跑多个预设候选策略：

```bash
cd /share/home/group9/lsg/LlamaFactory

python scripts/run_egocross_support_eval_grid.py
```

默认候选包括：

```text
direct + tail_dense + max8/12/16 + VID006=4
direct + endpoint + max12 + VID006=4
direct + uniform + max12 + VID006=4
strict_direct + tail_dense + max12 + VID006=4
```

汇总表：

```text
egocross_outputs/support_eval/_grid_summary.tsv
```

可选环境变量：

```bash
python scripts/run_egocross_support_eval_grid.py \
  --base-url http://127.0.0.1:8000/v1 \
  --support-dir /share/home/group9/data/egocross \
  --out-root egocross_outputs/support_eval
```

默认不覆盖已有非空输出目录；如要只打印命令不运行，使用 `--dry-run`。

服务器 support 选策略建议流程：

```bash
cd /share/home/group9/lsg/LlamaFactory
conda activate lsg
STAMP=$(date +%Y%m%d_%H%M%S)

python scripts/run_egocross_support_eval_grid.py \
  --support-dir /share/home/group9/data/egocross \
  --base-url http://127.0.0.1:8000/v1 \
  --out-root egocross_outputs/support_eval/grpo_grid_${STAMP} \
  --log-dir logs/support_eval_grid_${STAMP}
```

跑完查看汇总：

```bash
cat egocross_outputs/support_eval/grpo_grid_${STAMP}/_grid_summary.tsv
```

选择规则：

```text
1. 先看 overall support acc。
2. 若 overall 接近，优先 answer_only_format_rate=1.0、status_dist 无 error、used_max_frames_dist 不频繁降帧的策略。
3. 再看 per-domain，避免某个 domain 明显崩掉。
4. 选定后用同一组 prompt/frame_sampling/max_frames/VID006 配置跑 test；不要根据 hidden 榜单再改策略。
```

建议策略：预先列出少量候选，例如 `uniform/endpoint/tail_dense`、`max8/max12`、`direct/strict_direct`，一次性在 support 上比较 overall、per-domain、per-question-type、answer-only 格式率和 error rate。选定后直接应用到 test；不要再用 hidden domain 分数反推修改。

## 6. 文件结构

关键文件分布：

```text
README_EGOCROSS_lsg.md                 当前实验交接 README
agents.md                              长期 agent 规则和当前最佳命令
submission_template.json               官方提交模板，禁止覆盖

scripts/egocross_router_infer.py        当前主推理脚本
scripts/egocross_support_eval.py        support set 固定策略验证脚本
scripts/run_egocross_support_eval_grid.py
scripts/egocross_blend_submit.py        历史 blend 工具
scripts/prepare_egocross_weighted_answer_only.py
scripts/prepare_egocross_preference_answer_only.py

configs/egocross_*.yaml                 EgoCross 训练配置
data/dataset_info.json                  LLaMA-Factory 数据集注册
egocross_outputs/                       历史输出目录，每个实验单独建目录
```

根目录旧脚本：

```text
generate_egocross_submission.py
generate_egocross_submission_cot_test.py
generate_egocross_submission_shortcot_test.py
```

这些是历史入口，保留用于复现；新实验优先使用 `scripts/egocross_router_infer.py`。

## 7. 推理脚本能力

主脚本：

```bash
scripts/egocross_router_infer.py
```

支持：

```text
1. 按 dataset/domain 路由 prompt mode 和 max_frames。
2. 只跑部分 domain，其他样本用 fallback submission 填满完整输出。
3. 保存 predictions.json、raw_outputs.json、metrics_summary.txt、submission.zip。
4. 对明确 context length 400 错误自动降帧 retry。
5. frame sampling: uniform / endpoint / tail_dense。
6. prompt mode: direct / strict_direct / domain_direct / type_direct / domain_type_direct。
7. option-order voting，但当前 GRPO 模型上 voting 实测不佳。
8. `--temperature` 控制采样温度；默认是 0.0，历史 direct/vote 实验均为 deterministic 推理。
```

降帧序列：

```text
start -> 12 -> 8 -> 6 -> 4 -> 2 -> 1 中不超过 start 的序列
```

例如：

```text
max_frames=12: 12 -> 8 -> 6 -> 4 -> 2 -> 1
VID006=4: 4 -> 2 -> 1
```

采帧策略：

```text
uniform: 默认旧行为，均匀取 max_frames 帧，但不保证包含最后一帧。
endpoint: 包含首尾帧，中间均匀取样。
tail_dense: 包含首尾帧，并把更多采样点分配到视频后半段。
```

当前 GRPO 模型上，`direct + tail_dense + max12 + VID006=4` 是当前最强。

Prompt 经验：

```text
direct: 当前最稳。
strict_direct: 可做最小 prompt 对照。
domain_type_direct/type_direct: 已发现可能让 answer-only GRPO 模型偏离分布。
temperature=0.7: 可以作为 self-consistency/随机采样对照，但可能引入噪声；优先在 support 上验证。
CoT/few-shot/enhanced reasoning: 历史上均未超过 direct/router，不建议用于最终提交。
```

## 8. 历史关键结果

| 方案 | Overall | Surgery | Industry | XSports | Animal | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline full SFT direct max8 | 0.480669 | 0.515901 | 0.400000 | 0.414634 | 0.622951 | 旧 baseline |
| weighted answer-only direct | 0.481714 | 0.501767 | 0.416327 | 0.430894 | 0.606557 | 弱域提升，强域下降 |
| old router baseline strong + weighted weak | 0.489028 | 0.515901 | 0.416327 | 0.430894 | 0.622951 | 旧最佳 |
| CoT 49k 256 | 0.444096 | 0.508834 | 0.342857 | 0.418699 | 0.513661 | 明显下降 |
| short CoT 49k 512 | 0.459770 | 0.515901 | 0.375510 | 0.414634 | 0.546448 | 仍低于 direct |
| few-shot weak domains | 0.479624 | 0.515901 | 0.412245 | 0.398374 | 0.622951 | few-shot 伤 XSports |
| enhanced short reasoning few-shot | 0.481714 | 0.515901 | 0.387755 | 0.430894 | 0.622951 | enhanced 伤 Industry |
| external GRPO direct max8 VID006=4 | 0.528736 | 0.530035 | 0.534694 | 0.459350 | 0.612022 | 新主力模型 |
| GRPO direct endpoint max8 | 0.526646 | 0.526502 | 0.526531 | 0.459350 | 0.617486 | 低于 tail_dense |
| GRPO direct tail_dense max8 | 0.529781 | 0.533569 | 0.530612 | 0.459350 | 0.617486 | 采帧改进有效 |
| GRPO direct tail_dense max12 | 0.536050 | 0.533569 | 0.538776 | 0.459350 | 0.639344 | 当前最强候选 |
| GRPO direct tail_dense max16 | 0.532915 | 0.522968 | 0.538776 | 0.459350 | 0.639344 | 低于 max12，Surgery 回落 |
| GRPO ctx65536 direct tail_dense max12 no VID006 route temp0 | 0.536050 | 0.533569 | 0.538776 | 0.459350 | 0.639344 | 与当前最佳同分；长上下文可避免特殊 VID006 route，但无提分 |
| GRPO ctx65536 direct tail_dense max24 | 0.531870 | 0.522968 | 0.542857 | 0.459350 | 0.628415 | 长上下文更多帧提升 Industry，但损伤 Surgery/Animal |
| GRPO ctx65536 direct tail_dense max12 no VID006 route temp0.2 | 0.529781 | 0.522968 | 0.526531 | 0.459350 | 0.639344 | 轻微温度采样仍伤分 |
| GRPO domain_type_direct + vote max12 | 0.472309 | 0.487633 | 0.383673 | 0.447154 | 0.601093 | 复杂 prompt 明显伤分 |
| GRPO direct + vote max8 | 0.491118 | 0.505300 | 0.412245 | 0.451220 | 0.628415 | voting 不适合当前 GRPO |
| GRPO direct tail_dense max12 vote temp0.7 | 0.486938 | 0.505300 | 0.420408 | 0.426829 | 0.628415 | 高温投票明显伤分，不建议继续 |

说明：表中分数是实验记录，不应用于继续对 hidden test 做反推调参。

## 9. 训练相关记录

已完成训练：

```text
Full SFT baseline:
base: Qwen/Qwen3-VL-4B-Instruct
data: /share/home/group9/data/egocross/train.json
epochs: 2
cutoff_len: 32768
image_max_pixels: 200000
output: saves/egocross/qwen3vl4b/full_sft_32k_200k

Weighted answer-only SFT:
base: saves/egocross/qwen3vl4b/full_sft_32k_200k
data: train_weighted_answer_only_i4_x4.json
weights: animal=1,surgery=1,industry=4,xsports=4
lr: 5e-6
epochs: 1
output: saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1
```

已新增但不一定已跑通的配置：

```text
configs/egocross_dpo_answer_only_full_all_equal_lr1e6_ep1.yaml
configs/egocross_dpo_answer_only_lora_all_equal_lr1e6_ep1.yaml
configs/egocross_answer_only_full_all_equal_lr2e6_ep1.yaml
```

显存限制下，full DPO 容易 OOM；基础 SFT 更容易跑。当前最佳来自外部 GRPO 模型推理优化，不是本地 DPO。

生成全域 answer-only SFT 数据：

```bash
python scripts/prepare_egocross_weighted_answer_only.py \
  --data-dir /share/home/group9/data/egocross \
  --output /share/home/group9/data/egocross/train_answer_only_all_equal.json \
  --weights animal=1,surgery=1,industry=1,xsports=1 \
  --seed 2026
```

生成全域 DPO preference 数据：

```bash
python scripts/prepare_egocross_preference_answer_only.py \
  --data-dir /share/home/group9/data/egocross \
  --output /share/home/group9/data/egocross/train_pref_answer_only_all_equal_wrong3_fmt1.json \
  --fold-output-dir /share/home/group9/data/egocross/pref_answer_only_all_equal_folds
```

## 10. vLLM 和环境坑点

推荐启动参数：

```text
VLLM_USE_V1=0
--enforce-eager
--mm-processor-cache-gb 0
```

原因：

```text
1. Qwen3-VL 多图输入曾遇到 deepstack token 对齐问题。
2. vLLM v1 多模态路径和 mm processor cache 曾出现 mm_hash/cache 问题。
3. 关闭 cache 和 old engine 更慢，但更稳。
```

VID006：

```text
ExtrameSportFPV_VID006 是长样本，视觉 token 多。
当前 GRPO tail_dense 最强候选使用 --frame-route VID006=4。
如果出现 context length 或 vLLM 错误，可尝试更低 VID006=2，但不要用 hidden 反馈做细调。
```

训练/推理 config：

```text
transformers 训练阶段可能需要 text_config.rope_scaling={"rope_type":"default"}。
vLLM 推理阶段历史上 null rope_scaling 更稳。
切换前检查模型 config，避免训练版/推理版混用。
```

## 11. 文件保护规则

不要覆盖：

```text
submission_template.json
saves/egocross/qwen3vl4b/full_sft_32k_200k
saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1
egocross_outputs/baseline_max8
egocross_outputs/router_baseline_strong_weighted_weak_direct_max8_vid006_2f
egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
```

每个新实验必须新建 output dir，命名包含模型、prompt、sampling、max_frames、VID006 设置。

## 12. 建议的后续工作

当前建议先停在：

```text
egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f/submission.zip
```

如果后续继续探索，优先在 support/validation 上预先验证固定策略，再提交 hidden test。建议顺序：

```text
1. 支持集验证不同 frame_sampling，而不是用 hidden 分数调。
2. 保持 direct prompt，不优先加 CoT/few-shot/type prompt。
3. 若要改 VID006 帧数，先基于 context length / error rate 等鲁棒性信号，不基于 hidden 分数。
4. ctx65536 + max12 no VID006 route 已与当前最佳同分，可作为工程对照，但无提分；最终提交优先保留更简单稳定的原 max12 + VID006=4。
5. 不建议继续温度采样或高温投票；temperature=0.2 与 temperature=0.7 + vote 均低于 deterministic direct。
```

## 13. Conservative LoRA DPO From External GRPO

Purpose:
```text
Run the first new training experiment as LoRA DPO from the external GRPO model.
Do not start with full DPO. Do not change inference strategy in the same round.
Do not use hidden leaderboard/domain feedback for tuning or model promotion.
```

Original external GRPO model:
```text
/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
```

Train/export-safe model view used by the new DPO configs:
```text
saves/egocross/qwen3vl4b/grpo_v3_train_rope_default
```

Create the train-safe model view before training:
```bash
python scripts/prepare_egocross_grpo_train_safe.py --overwrite-config
```

This creates symlinks to the external GRPO files and writes only a local `config.json` with `text_config.rope_scaling={"rope_type":"default"}`. Do not edit the external GRPO directory in place.

The ctx65536/65538 variant is treated as an inference/config comparison, not the first training source. It had no score gain and adds engineering variables.

Data check before DPO:
```bash
python scripts/check_egocross_pref_data.py \
  --data-dir /share/home/group9/data/egocross \
  --pref-file /share/home/group9/data/egocross/train_pref_answer_only_all_equal_wrong3_fmt1.json \
  --fold-dir /share/home/group9/data/egocross/pref_answer_only_all_equal_folds
```

Local mirror/debug only:
```bash
python scripts/check_egocross_pref_data.py \
  --data-dir ../data_local/egocross \
  --pref-file ../data_local/egocross/train_pref_answer_only_all_equal_wrong3_fmt1.json \
  --fold-dir ../data_local/egocross/pref_answer_only_all_equal_folds \
  --skip-image-check
```

Candidate configs:
```text
A: configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_lr1e5_beta003_ftx005_ep1.yaml
B: configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_lr5e6_beta003_ftx005_ep1.yaml
C: configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_lr1e5_beta005_ftx005_ep1.yaml
```

Fold protocol:
```text
foldN training must use only egocross_pref_answer_only_all_equal_wrong3_fmt1_foldN.
foldN accuracy must be measured only on pref_answer_only_all_equal_folds/eval_answer_only_all_equal_foldN.json.
Preference train rows are only for loss/format stability checks, not promotion accuracy.
Run GRPO baseline 4-fold heldout first, then A/B/C fold heldout. Run full-support training only after fold metrics are stable.
```

Fold0 smoke command:
```bash
LLAMAFACTORY_LOGPS_CHUNK_SIZE=512 CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 llamafactory-cli train \
  configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_fold0_lr1e5_beta003_ftx005_ep1.yaml
```

DPO OOM note:
```text
Long multimodal DPO may OOM inside get_batch_logps at logits.log_softmax(-1). DDP does not shard this per-rank tensor. This repo now avoids full-logits fp32 upcast in DPO trainer and computes label log-probs as label_logit - logsumexp(logits), chunked by LLAMAFACTORY_LOGPS_CHUNK_SIZE. If fold0 still OOMs, retry with LLAMAFACTORY_LOGPS_CHUNK_SIZE=512, 256, or 128 before lowering cutoff/image pixels.
If it still OOMs after the logsumexp patch, use the memsafe fold0 config:
configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_fold0_lr1e5_beta003_ftx005_ep1_memsafe_ctx24576_px131k.yaml
```

Heldout fold eval example:
```bash
python scripts/egocross_support_eval.py \
  --support-dir /share/home/group9/data/egocross \
  --eval-json /share/home/group9/data/egocross/pref_answer_only_all_equal_folds/eval_answer_only_all_equal_fold0.json \
  --prompt-mode direct \
  --max-frames 12 \
  --frame-sampling tail_dense \
  --frame-route VID006=4 \
  --temperature 0 \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir egocross_outputs/support_eval/dpo_lora_from_grpo_A_fold0_direct_tail_dense_max12_vid006_4f
```

Full training and export use only the winning candidate:
```bash
LLAMAFACTORY_LOGPS_CHUNK_SIZE=512 CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 llamafactory-cli train \
  configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_lr<W>_beta<B>_ftx005_ep1.yaml

llamafactory-cli export \
  configs/egocross_export_dpo_lora_from_grpo_all_equal_wrong3_fmt1_lr<W>_beta<B>_ftx005_ep1.yaml
```

Reference-model sanity:
```text
LLaMA-Factory stage=dpo + finetuning_type=lora + pref_loss=sigmoid creates DPO reference behavior by disabling the new LoRA adapter on the same base model when ref_model is unset. For this experiment, the reference should therefore be the frozen train-safe view of the external GRPO base, not original SFT.
```

After export, vLLM may still prefer null rope_scaling. If vLLM fails to load a merged model, inspect the exported `config.json` and set `text_config.rope_scaling` back to null for inference only. Keep the train-safe base config at `{"rope_type":"default"}` for transformers training/export.

Promotion criteria:
```text
At least 3/4 folds have heldout accuracy >= GRPO baseline.
Mean heldout accuracy is above baseline, or tied with better format/error metrics.
answer_only_format_rate >= 0.995.
coverage = 1.0.
parse_fail is not above baseline.
support eval has error/error_fallback = 0.
```

Hard stop:
```text
coverage < 1.0
answer_only_format_rate < 0.995
support overall drops by more than 0.5-1.0 point vs GRPO baseline
any domain drops by more than 2 points without stable fold/overall compensation
answer distribution collapses heavily to one letter
merged model appears not to have loaded/merged LoRA
```

First-round inference must stay fixed:
```text
direct + tail_dense + max12 + VID006=4 + temperature=0
No prompt/frame/router/vote changes in the same round as DPO.
```

Final comparison table template:
```text
model	strategy	support_overall	Surgery	Industry	XSports	Animal	answer_only_format_rate	coverage	parse_fail	error	avg_used_frames	runtime_seconds
GRPO baseline	direct_tail_dense_max12_vid006_4f									
DPO LoRA A	direct_tail_dense_max12_vid006_4f									
DPO LoRA B	direct_tail_dense_max12_vid006_4f									
DPO LoRA C	direct_tail_dense_max12_vid006_4f									
```
