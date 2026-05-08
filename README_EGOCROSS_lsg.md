# EgoCross Progress README

本文档用于交接当前 EgoCross 实验进度。阅读后应能理解目前做了什么、哪些配置已经验证、哪些问题踩过坑、下一步该怎么继续。

## 1. 当前任务目标

我们在参加 EgoCross egocentric video QA 比赛。任务形式是：

```text
多帧第一视角视频帧 + 多选题文本 -> 输出 A/B/C/D
```

目前目标不是继续搭环境，而是：

```text
使用已经 Full SFT 训练好的 Qwen3-VL-4B 模型
对 test set 批量推理
生成 Codabench 可提交的 predictions.json / submission.zip
并继续尝试 CoT / 思维链 prompt 是否能提高分数
```

## 2. 服务器关键路径

代码目录：

```bash
/share/home/group9/lsg/LlamaFactory
```

训练 support set：

```bash
/share/home/group9/data/egocross
```

test set：

```bash
/share/home/group9/data/egocross_full/egocross_testbed
```

test query 文件：

```bash
/share/home/group9/data/egocross_full/egocross_testbed/egocross_testbed_imgs.json
```

官方提交模板：

```bash
/share/home/group9/lsg/LlamaFactory/submission_template.json
```

当前最重要的 baseline 模型：

```bash
/share/home/group9/lsg/LlamaFactory/saves/egocross/qwen3vl4b/full_sft_32k_200k
```

当前最重要的新模型：

```bash
/share/home/group9/lsg/LlamaFactory/saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1
```

当前最佳提交策略：

```text
Surgery / Animal 使用 baseline 模型结果。
Industry / XSports 使用 weighted_answer_only_full_i4_x4_lr5e6_ep1 模型结果。
Codabench Overall = 0.489028。
```

## 3. Conda 环境

当前使用 conda 环境：

```bash
conda activate lsg
```

多卡/torchrun 曾经遇到过环境串到系统 Python 的问题，所以每次建议执行：

```bash
export PATH="/share/home/group9/miniconda3/envs/lsg/bin:$PATH"
```

这个命令的作用是保证 `python`、`torchrun`、`llamafactory-cli` 优先使用 `lsg` 环境里的版本。

## 4. 已完成的训练

已经完成一次 Full SFT：

```text
base model: Qwen/Qwen3-VL-4B-Instruct
training method: Full SFT
training data: /share/home/group9/data/egocross/train.json
epochs: 2
cutoff_len: 32768
image_max_pixels: 200000
output model: saves/egocross/qwen3vl4b/full_sft_32k_200k
```

训练完成日志中关键指标：

```text
train_loss = 4.7132
epoch = 2.0
train_runtime = 0:04:43.48
```

注意：`train_loss` 不是比赛分数，只说明训练完成。比赛分数来自 Codabench test set 评测。

### 4.1 Weighted answer-only Full SFT

已完成一次继续 Full SFT，用于增强弱 domain：

```text
base model: saves/egocross/qwen3vl4b/full_sft_32k_200k
training method: Full SFT
training data: /share/home/group9/data/egocross/train_weighted_answer_only_i4_x4.json
data construction: original support set, assistant 强制为单字母 A/B/C/D
domain weights: animal=1, surgery=1, industry=4, xsports=4
epochs: 1
learning_rate: 5e-6
cutoff_len: 32768
image_max_pixels: 200000
output model: saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1
```

直接用新模型全量 direct 推理的 Codabench 得分：

```text
Overall: 0.481714
Surgery: 0.501767
Industry: 0.416327
XSports: 0.430894
Animal: 0.606557
coverage: 1.0
```

结论：

```text
weighted Full SFT 确实提升了 Industry / XSports。
但它损伤了 Surgery / Animal，因此不应直接全 domain 替代 baseline。
最佳用法是 domain router：弱 domain 用新模型，强 domain 保留 baseline。
```

## 5. 已跑通的 baseline 推理与分数

使用 vLLM 加载模型，`max-frames 8`，直接答案 prompt，已经完整跑通 test set 并提交 Codabench。

baseline 设置：

```text
model: saves/egocross/qwen3vl4b/full_sft_32k_200k
vLLM max_model_len: 32768
vLLM enforce_eager: true
max_frames: 8
prompt: direct answer, only A/B/C/D
script: generate_egocross_submission.py
output: submission_full_sft_32k_200k_max8.json
```

Codabench 得分：

```text
Overall: 0.480669
acc: 0.480669
Surgery: 0.515901
Industry: 0.400000
XSports: 0.414634
Animal: 0.622951
coverage: 1.0
```

这个 baseline 要保留，不要覆盖。后续所有实验都应新建输出文件。

## 5.1 已完成推理实验汇总

| 实验名 | 脚本 | Prompt | vLLM max_model_len | max_frames | max_tokens | Overall | Surgery | Industry | XSports | Animal | coverage | 输出目录 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_max8 | `generate_egocross_submission.py` | 直接输出答案 | 32768 | 8 | 8 | 0.480669 | 0.515901 | 0.400000 | 0.414634 | 0.622951 | 1.0 | `egocross_outputs/baseline_max8/` |
| cot_49k_256 | `generate_egocross_submission_cot_test.py` | 普通 CoT | 49152 | 8 | 256 | 0.444096 | 0.508834 | 0.342857 | 0.418699 | 0.513661 | 1.0 | `egocross_outputs/full_sft_32k_200k_max8_cot_49k_256/` |
| shortcot_49k_512 | `generate_egocross_submission_shortcot_test.py` | 短 CoT | 49152 | 8 | 512 | 0.459770 | 0.515901 | 0.375510 | 0.414634 | 0.546448 | 1.0 | `egocross_outputs/full_sft_32k_200k_max8_shortcot_49k_512/` |
| router_domain_direct_lowdomains | `scripts/egocross_router_infer.py` | Industry/XSports domain_direct，其他 baseline fallback | 32768 | 8 | 8 | 0.480669 | 0.515901 | 0.391837 | 0.422764 | 0.622951 | 1.0 | `egocross_outputs/router_domain_direct_lowdomains/` |
| weighted_full_direct_max8 | `scripts/egocross_router_infer.py` | 新模型 direct，全 domain | 32768/49152 | 8，VID006=2 | 8 | 0.481714 | 0.501767 | 0.416327 | 0.430894 | 0.606557 | 1.0 | `egocross_outputs/weighted_answer_only_full_i4_x4_lr5e6_ep1_direct_max8_vid006_2f/` |
| router_baseline_strong_weighted_weak | `scripts/egocross_router_infer.py` | Surgery/Animal baseline，Industry/XSports 新模型 | 32768/49152 | 8，VID006=2 | 8 | 0.489028 | 0.515901 | 0.416327 | 0.430894 | 0.622951 | 1.0 | `egocross_outputs/router_baseline_strong_weighted_weak_direct_max8_vid006_2f/` |
| fewshot_k1_2f_weakdomains | `scripts/egocross_router_infer.py` | Industry/XSports 使用同 domain + question type 的视觉 few-shot，Surgery/Animal fallback | 32768/49152 | 当前题 8，few-shot 每例 2，VID006=2 | 8 | 0.479624 | 0.515901 | 0.412245 | 0.398374 | 0.622951 | 1.0 | few-shot 弱域实验输出目录 |
| fewshot_shortreason_enhanced_industry | `scripts/egocross_router_infer.py` | Industry 使用 enhanced 蒸馏短推理 + `Answer: X` few-shot，XSports/Surgery/Animal fallback | 32768/49152 | 当前题 8，few-shot 每例 2，VID006=2 | 64 | 0.481714 | 0.515901 | 0.387755 | 0.430894 | 0.622951 | 1.0 | `egocross_outputs/fewshot_shortreason_enhanced_k1_2f_industry_probe/` |

当前结论：

```text
直接答案 baseline 仍然最高。
普通 CoT 256 明显下降，主要问题是思维链输出偏长，部分样本在 max_tokens=256 内没有输出 Final answer。
缺少最终答案时，脚本会从输出中抽取答案或 fallback 到 A，容易污染预测。
Short CoT 512 比普通 CoT 256 更好，但仍低于直接答案 baseline。
router_domain_direct_lowdomains 的 Overall 与 baseline 持平：XSports 提升 2 题，但 Industry 下降 2 题，互相抵消。
weighted answer-only Full SFT 让 Industry / XSports 变强，但 Surgery / Animal 变弱。
当前最佳结果是模型级 domain router：Surgery/Animal 用 baseline，Industry/XSports 用 weighted Full SFT，Overall = 0.489028。
few-shot prompt 当前没有带来收益：视觉 few-shot 明显损伤 XSports；enhanced 短推理 few-shot 明显损伤 Industry。
```

因此，后续如果继续做 CoT，建议优先：

```text
1. 继续缩短 prompt，让模型更快输出 Final answer。
2. 检查 *_raw_cot.json 中 no_final_answer 的比例。
3. 不只增加 max_tokens，因为更长输出会显著变慢，也可能增加无关解释。
4. 目前不建议直接用 enhanced 长解释做主训练集，也不建议直接把 enhanced 短推理 few-shot 用于提交；优先保留 answer-only / direct prompt 稳定性。
```

## 5.2 Few-shot 实验结论（2026-05-03）

已尝试在当前 test 推理 pipeline 上加入 support/train set few-shot 示例：

```text
方案 A：Industry / XSports 使用同 domain + question type 的视觉 few-shot 示例，每个示例 2 帧，目标题 8 帧。
结果：Overall = 0.479624，Industry = 0.412245，XSports = 0.398374。
结论：Industry 略低于 weighted direct，XSports 明显下降，不适合作为提交策略。

方案 B：Industry 使用 train_*_enhanced 蒸馏一句短 Reasoning，并要求目标输出同样格式：
Reasoning: ...
Answer: X
结果：Overall = 0.481714，Industry = 0.387755，XSports = 0.430894。
结论：短推理格式没有帮助，反而显著损伤 Industry。即使最终只抽取 Answer 字母，模型仍可能被推理格式带偏。
```

当前判断：

```text
1. few-shot 示例和目标视频放在同一个多图 user message 中，可能带来视觉上下文污染。
2. support set 每个 domain 只有 20 条，同 question type 容易反复选到同一个示例，造成答案字母或题型先验偏置。
3. weighted answer-only 模型对 direct answer prompt 更稳定，few-shot / short reasoning 都属于分布外 prompt。
4. XSports 的方向、动作序列、时间定位更依赖目标视频动态，参考视频示例泛化弱，容易误导。
5. Industry counting/localization 也未从 enhanced 短推理中受益，说明 enhanced 描述即使压缩后仍可能是噪声。
```

后续建议：

```text
不要把 few-shot 结果用于正式提交。
当前最佳仍是 router_baseline_strong_weighted_weak_direct_max8_vid006_2f，Overall = 0.489028。
如果继续研究 few-shot，建议只做小样本诊断，例如 text-only few-shot、答案字母平衡、或只针对极少数错误类型做手工示例；不要直接全量提交。
```

## 5.3 当前最佳训练方向（2026-05-02）

当前最佳已验证路线是训练 + domain router：

```text
从当前最强模型 saves/egocross/qwen3vl4b/full_sft_32k_200k 继续 Full SFT
训练数据使用原始 support set，强制 assistant answer-only
Industry / XSports 过采样，提高弱 domain 权重
训练后只把新模型用于 Industry / XSports
Surgery / Animal 保留 baseline 结果
```

新增文件：

```text
scripts/prepare_egocross_weighted_answer_only.py
configs/egocross_weighted_answer_only_full_ep1.yaml
data/dataset_info.json 中新增 egocross_weighted_answer_only_i4_x4
```

推荐训练数据权重：

```text
animal=1
surgery=1
industry=4
xsports=4
```

注意：`train_*_enhanced.json` 里有视觉描述和分析，但不保证正确，且输出格式偏长。不要直接作为主训练集，否则模型可能学会长解释，影响最终 A/B/C/D 输出稳定性。若后续使用 enhanced，建议只做短证据蒸馏并过滤最终答案不一致的样本。

当前最佳提交：

```text
egocross_outputs/router_baseline_strong_weighted_weak_direct_max8_vid006_2f/submission.zip
Overall: 0.489028
```

## 6. 当前已有脚本

### 6.1 baseline 脚本

```bash
generate_egocross_submission.py
```

作用：

```text
读取 egocross_testbed_imgs.json
读取每道题的视频帧
调用本地 vLLM OpenAI-compatible API
要求模型直接输出 A/B/C/D
从输出中抽取答案
填入 submission_template.json
生成 submission JSON
```

已验证可用。

### 6.2 CoT 测试脚本

```bash
generate_egocross_submission_cot_test.py
```

作用：

```text
读取 test set
使用 CoT / 思维链 prompt
要求模型先简短分析视觉证据和选项
最后输出 Final answer: X
生成提交 JSON
同时生成 *_raw_cot.json 保存完整模型输出，方便检查是否真的输出思维链
```

### 6.3 Short CoT 512 测试脚本

```bash
generate_egocross_submission_shortcot_test.py
```

作用：

```text
使用更短的 CoT prompt。
要求模型最多用 2 句短推理，然后输出 Final answer: X。
默认 max_tokens=512，用于降低推理文本被截断、缺少最终答案的概率。
默认 base_url 为 http://127.0.0.1:8001/v1。
```

这个脚本用于测试：

```text
short CoT prompt + max_frames 8 + vLLM max_model_len 49152 + max_tokens 512
```

### 6.4 Router 推理脚本

```bash
scripts/egocross_router_infer.py
```

作用：

```text
按 dataset/domain 选择 prompt mode 和 max_frames。
支持只跑某些 domain，其他样本用 baseline fallback 填满完整 submission。
支持 raw_outputs.json 和 metrics_summary.txt。
对明确的 context length 400 错误支持自动降帧 retry：start -> 12 -> 8 -> 6 -> 4 -> 2 -> 1 中不超过 start 的序列。
```

已知结果：

```text
Industry + XSports 同时使用 domain_direct 时，XSports 上升但 Industry 下降，Overall 持平。
如果继续做 router，优先只让 XSports 使用 domain_direct，Industry 回到 baseline/direct。
```

### 6.5 Weighted answer-only 数据构建脚本

```bash
scripts/prepare_egocross_weighted_answer_only.py
```

作用：

```text
读取 train_animal.json / train_surgery.json / train_industry.json / train_xsports.json。
把 assistant 内容强制转换为单字母 A/B/C/D。
按指定权重复制并 shuffle，生成 weighted answer-only 训练集。
默认建议输出：/share/home/group9/data/egocross/train_weighted_answer_only_i4_x4.json
```

## 7. vLLM 启动方式

baseline 稳定启动命令：

```bash
cd /share/home/group9/lsg/LlamaFactory
conda activate lsg
export PATH="/share/home/group9/miniconda3/envs/lsg/bin:$PATH"

CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
  --model saves/egocross/qwen3vl4b/full_sft_32k_200k \
  --port 8000 \
  --served-model-name egocross \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

检查服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

如果返回模型列表且 `id` 是 `egocross`，说明服务可用。

`--enforce-eager` 很重要。之前 vLLM 在 Qwen3-VL 多图输入上遇到过 deepstack token 对齐问题，使用 eager 模式更稳。

当前更推荐的稳定启动方式：

```bash
cd /share/home/group9/lsg/LlamaFactory
conda activate lsg
export PATH="/share/home/group9/miniconda3/envs/lsg/bin:$PATH"

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
  --model saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1 \
  --port 8000 \
  --served-model-name egocross \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --mm-processor-cache-gb 0
```

说明：

```text
VLLM_USE_V1=0：切回 vLLM old engine，绕开部分 v1 多模态路径问题。
--mm-processor-cache-gb 0：关闭 multimodal processor cache，避免 mm_hash/cache 不一致。
--enforce-eager：继续保留，Qwen3-VL 多图输入更稳。
代价是速度可能稍慢，但比赛提交更看重稳定性。
```

注意：

```text
这个启动方式不能解决 context length 超限。
ExtrameSportFPV_VID006 仍然需要推理命令里加 --frame-route VID006=2。
```

## 8. 已遇到的重要问题与解决方案

### 8.1 环境串到系统 Python

现象：

```text
No module named 'llamafactory'
binary: /usr/bin/python3
```

原因：`torchrun` 或命令优先用了系统 Python。

解决：

```bash
export PATH="/share/home/group9/miniconda3/envs/lsg/bin:$PATH"
```

### 8.2 tokenizer / HuggingFace 网络问题

现象：

```text
Network is unreachable
OSError: Failed to load tokenizer
```

解决：使用离线模式：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### 8.3 Full SFT OOM

尝试过 `cutoff_len=65536`，Full SFT OOM。

最终跑通配置：

```text
cutoff_len: 32768
image_max_pixels: 200000
```

### 8.4 test set 路径解析问题

test set 中 `video_path` 形如：

```text
/egocross_testbed/CholecTrack20/generated/VID01/frames/oc_q1/frame_00000.jpg
```

这不是服务器真实绝对路径。脚本中已经处理为：

```text
/share/home/group9/data/egocross_full/egocross_testbed/CholecTrack20/generated/...
```

### 8.5 submission_template 问题

`egocross_testbed_imgs.json` 只有：

```text
id, dataset, question_text, options, video_path
```

没有 `question_id`。

所以生成提交时必须使用：

```bash
submission_template.json
```

脚本只填入 `answer`，保留 template 中的 `id/question_id/dataset`。

### 8.6 vLLM 上下文长度超限

现象：

```text
Input length (34244) exceeds model's maximum context length (32768)
```

原因：图片帧过多或 prompt 变长后，视觉 token + 文本 token 超过 `--max-model-len 32768`。

已知：

```text
direct prompt + max_frames 8 已经跑通
CoT prompt + max_frames 12 容易超长
```

### 8.7 vLLM deepstack token 错误

现象：

```text
Requested more deepstack tokens than available in buffer:
num_tokens=480 > self.deepstack_input_embeds_num_tokens=479
```

这是 vLLM + Qwen3-VL + 多图输入的视觉 token 对齐问题，不是训练模型坏了。

缓解：

```text
使用 --enforce-eager
减少 max_frames
避免自动 fallback 连续发送超长请求
```

### 8.8 自动降帧 fallback 尝试不理想

曾尝试在 CoT 脚本中做：

```text
max_frames 12 -> 超长后 fallback 8 -> fallback 6 -> fallback 4 -> fallback 2 -> fallback 1
```

但实际效果不好。原因是 vLLM 遇到前一个超长请求后，有时服务进入不稳定状态，后续较少帧请求也会返回 500。

更新结论：

```text
对 vLLM 500 / deepstack token 错误，仍然不建议自动 fallback，优先重启服务。
对明确的 400 context length 错误，可以安全做同一样本降帧 retry。
当前 router 脚本已实现 context length retry：start -> 12 -> 8 -> 6 -> 4 -> 2 -> 1 中不超过 start 的序列。
ExtrameSportFPV_VID006 已验证需要显式降帧，推荐 --frame-route VID006=2。
```

### 8.8.1 vLLM 多模态 cache / mm_hash 问题

现象：

```text
vLLM 在预处理多模态输入时失败，multimodal cache 中找不到对应 mm_hash。
请求返回 400 Bad Request，脚本标记为 error_fallback。
```

推荐启动 vLLM 时关闭多模态 processor cache，并切回旧 engine：

```bash
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
  --model saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1 \
  --port 8000 \
  --served-model-name egocross \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --mm-processor-cache-gb 0
```

说明：

```text
VLLM_USE_V1=0 可绕开部分 vLLM v1 多模态路径问题。
--mm-processor-cache-gb 0 可避免 mm_hash cache 缺失/不一致。
```

### 8.8.2 ExtrameSportFPV_VID006 特殊问题

`ExtrameSportFPV_VID006` 是当前 test set 中最麻烦的一组样本，涉及：

```text
question_id/id 范围：ExtrameSportFPV_VID006_q5 到 q12
路径位置：/share/home/group9/data/egocross_full/egocross_testbed/ExtrameSportFPV/generated/VID006/frames
```

遇到过两类问题：

```text
1. context length 超限：
   Input length (653xx) exceeds model's maximum context length (49152)

2. vLLM multimodal cache / mm_hash 错误：
   vLLM 预处理多模态输入时，multimodal cache 里找不到对应 mm_hash，请求返回 400 Bad Request。
```

原因：

```text
VID006 的帧视觉 token 异常多。
即使使用 max_frames=8，也可能达到 65k 左右输入长度，超过 49k 上下文。
同时，多图请求在 vLLM 的 multimodal cache 路径上容易触发 mm_hash/cache 不一致。
```

最终稳定解决方案：

```text
启动 vLLM 时使用：
VLLM_USE_V1=0
--enforce-eager
--mm-processor-cache-gb 0

推理时对 VID006 显式降帧：
--frame-route VID006=2
```

注意：

```text
普通的自动 retry 只能解决明确的 context length 400。
当前 retry 顺序为 start -> 12 -> 8 -> 6 -> 4 -> 2 -> 1 中不超过 start 的序列。
如果已经触发 mm_hash/cache 错误，最好重启 vLLM，并关闭 mm processor cache。
对完整 test set，推荐直接显式指定 --frame-route VID006=2，避免先发送 8 帧污染服务状态。
```

只测试 VID006 的方法：

```bash
cd /share/home/group9/lsg/LlamaFactory

python - <<'PY'
import json
from pathlib import Path

src = Path("/share/home/group9/data/egocross_full/egocross_testbed/egocross_testbed_imgs.json")
dst = Path("egocross_outputs/scratch_tests/egocross_testbed_vid006.json")
dst.parent.mkdir(parents=True, exist_ok=True)

data = json.load(open(src))

def hit(x):
    text = json.dumps(x, ensure_ascii=False)
    return "VID006" in text and "ExtrameSportFPV" in text

sub = [x for x in data if hit(x)]
print("total:", len(data))
print("matched:", len(sub))
if sub:
    print("first sample:", json.dumps(sub[0], ensure_ascii=False)[:800])

json.dump(sub, open(dst, "w"), indent=2, ensure_ascii=False)
print("saved:", dst)
PY
```

只跑 VID006 小测试：

```bash
python scripts/egocross_router_infer.py \
  --input-json egocross_outputs/scratch_tests/egocross_testbed_vid006.json \
  --default-prompt-mode direct \
  --default-max-frames 8 \
  --frame-route VID006=2 \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir egocross_outputs/scratch_tests/vid006_2f_test
```

如果 2 帧仍失败，再降到 1 帧：

```bash
python scripts/egocross_router_infer.py \
  --input-json egocross_outputs/scratch_tests/egocross_testbed_vid006.json \
  --default-prompt-mode direct \
  --default-max-frames 8 \
  --frame-route VID006=1 \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir egocross_outputs/scratch_tests/vid006_1f_test
```

已经验证：

```text
VID006 使用 --frame-route VID006=2 可以跑通。
当前最佳完整 test set 提交也使用该设置。
```

### 8.9 transformers / Qwen3-VL rope_scaling 兼容问题

现象 1：

```text
transformers==5.6.2 不兼容 LLaMA-Factory。
LLaMA-Factory 要求 transformers>=4.55.0,<=5.2.0,!=4.57.0。
```

处理：

```bash
pip install -U "transformers==4.57.3"
```

现象 2：

```text
训练加载 Qwen3-VL 时失败：
AttributeError: 'NoneType' object has no attribute 'get'
原因是模型 config 里的 text_config.rope_scaling 为 null，而 transformers 4.57.x 某处代码按 dict 处理。
```

处理方式：只 patch 继续训练的基座模型 config，不改 HuggingFace cache 或源码。

```bash
cd /share/home/group9/lsg/LlamaFactory

cp saves/egocross/qwen3vl4b/full_sft_32k_200k/config.json \
   saves/egocross/qwen3vl4b/full_sft_32k_200k/config.json.bak_rope_null

python - <<'PY'
import json
from pathlib import Path

p = Path("saves/egocross/qwen3vl4b/full_sft_32k_200k/config.json")
cfg = json.loads(p.read_text())
text_config = cfg.setdefault("text_config", {})
if text_config.get("rope_scaling") is None:
    text_config["rope_scaling"] = {"rope_type": "default"}
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
print("patched text_config.rope_scaling =", cfg["text_config"]["rope_scaling"])
PY
```

验证：

```bash
python - <<'PY'
from transformers import AutoConfig
p = "saves/egocross/qwen3vl4b/full_sft_32k_200k"
cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
print("loaded ok")
print("rope_scaling:", cfg.text_config.rope_scaling)
PY
```

重要区别：

```text
训练阶段（transformers 4.57.x）：text_config.rope_scaling 需要 {"rope_type": "default"}，否则可能 None.get 报错。
vLLM 推理阶段：text_config.rope_scaling 更稳的形式是 null，否则可能触发 rotary position / q/k shape 不匹配。
```

因此，如果为了训练改过 `full_sft_32k_200k/config.json`，在用 vLLM 启动 baseline 模型前也需要恢复为 null。

推荐把每个模型目录都保留两份 config 备份：

```text
config.json.bak_rope_default_for_train  # 训练版：text_config.rope_scaling = {"rope_type": "default"}
config.json.bak_rope_null               # 推理版：text_config.rope_scaling = null
```

训练/推理时只切换 `config.json`：

```bash
# 训练前
cp config.json.bak_rope_default_for_train config.json

# vLLM 推理前
cp config.json.bak_rope_null config.json
```

baseline 模型创建/恢复推理版 config：

```bash
cd /share/home/group9/lsg/LlamaFactory

cp saves/egocross/qwen3vl4b/full_sft_32k_200k/config.json \
   saves/egocross/qwen3vl4b/full_sft_32k_200k/config.json.bak_rope_default_for_train

python - <<'PY'
import json
from pathlib import Path

p = Path("saves/egocross/qwen3vl4b/full_sft_32k_200k/config.json")
cfg = json.loads(p.read_text())

if isinstance(cfg.get("text_config"), dict):
    cfg["text_config"]["rope_scaling"] = None

if "rope_scaling" in cfg:
    cfg["rope_scaling"] = None

p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
print("patched full_sft_32k_200k for vLLM:")
print("top rope_scaling =", cfg.get("rope_scaling"))
print("text rope_scaling =", cfg.get("text_config", {}).get("rope_scaling"))
PY

cp saves/egocross/qwen3vl4b/full_sft_32k_200k/config.json \
   saves/egocross/qwen3vl4b/full_sft_32k_200k/config.json.bak_rope_null
```

weighted 新模型创建/恢复推理版 config：

```bash
cd /share/home/group9/lsg/LlamaFactory

cp saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1/config.json \
   saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1/config.json.bak_rope_default_for_train

python - <<'PY'
import json
from pathlib import Path

p = Path("saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1/config.json")
cfg = json.loads(p.read_text())

if isinstance(cfg.get("text_config"), dict):
    cfg["text_config"]["rope_scaling"] = None

if "rope_scaling" in cfg:
    cfg["rope_scaling"] = None

p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
print("patched weighted model for vLLM:")
print("top rope_scaling =", cfg.get("rope_scaling"))
print("text rope_scaling =", cfg.get("text_config", {}).get("rope_scaling"))
PY

cp saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1/config.json \
   saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1/config.json.bak_rope_null
```

检查当前目录 config 属于训练版还是推理版：

```bash
python - <<'PY'
import json
cfg = json.load(open("config.json"))
print("top:", cfg.get("rope_scaling"))
print("text:", cfg.get("text_config", {}).get("rope_scaling"))
PY
```

## 9. 当前正在尝试的 CoT 实验

目标：

```text
在不改变模型的前提下，只改变推理 prompt
测试让模型输出简短思维链/分析过程是否能提高 Codabench 分数
```

推荐干净实验：

```text
baseline: direct prompt + max_frames 8 + max_model_len 32768
cot test: CoT prompt + max_frames 8 + max_model_len 32768 或 49152
```

CoT prompt 的核心要求：

```text
First briefly describe visual evidence.
Then compare options.
Finally output exactly one line: Final answer: X.
```

最终提交 JSON 中仍然只允许：

```json
"answer": "A/B/C/D"
```

思维链只保存在 `_raw_cot.json` 中，不提交。

## 10. CoT 推荐运行命令

### 10.1 启动 vLLM，32768 稳定版

```bash
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
  --model saves/egocross/qwen3vl4b/full_sft_32k_200k \
  --port 8000 \
  --served-model-name egocross \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

### 10.2 CoT max8 测试

```bash
python generate_egocross_submission_cot_test.py \
  --template submission_template.json \
  --max-frames 8 \
  --output submission_full_sft_32k_200k_max8_cot_test.json
```

### 10.3 如果想试更大上下文

启动 vLLM 时改为：

```bash
--max-model-len 49152
```

输出文件名建议写清楚：

```bash
python generate_egocross_submission_cot_test.py \
  --template submission_template.json \
  --max-frames 8 \
  --max-tokens 256 \
  --output submission_full_sft_32k_200k_max8_cot_49k_256.json
```

注意：`49152` 可能减少上下文超限，但也可能更吃显存或触发 vLLM 多图 bug。它是新实验，不应替代 baseline。

### 10.4 Short CoT 512 + 49k 上下文实验

如需并行运行实验，可在空闲 GPU 上启动独立 vLLM 服务。以下示例使用 GPU 5 和端口 8001：

```bash
cd /share/home/group9/lsg/LlamaFactory
conda activate lsg
export PATH="/share/home/group9/miniconda3/envs/lsg/bin:$PATH"

CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
  --model saves/egocross/qwen3vl4b/full_sft_32k_200k \
  --port 8001 \
  --served-model-name egocross \
  --trust-remote-code \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

检查服务：

```bash
curl http://127.0.0.1:8001/v1/models
```

运行 short CoT 512：

```bash
python generate_egocross_submission_shortcot_test.py \
  --template submission_template.json \
  --max-frames 8 \
  --max-tokens 512 \
  --base-url http://127.0.0.1:8001/v1 \
  --output submission_full_sft_32k_200k_max8_shortcot_49k_512.json
```

这个实验的目的：

```text
解决普通 CoT 输出过长、还没到 Final answer 就被截断的问题。
通过 short prompt 限制推理长度，同时用 max_tokens=512 保证能输出最终答案。
```

## 11. 检查输出

检查 submission：

```bash
python - <<'PY'
import json
p = "submission_full_sft_32k_200k_max8_cot_test.json"
with open(p) as f:
    data = json.load(f)
print("num:", len(data))
print("empty:", sum(1 for x in data if not x.get("answer")))
print("answers:", {k: sum(1 for x in data if x.get("answer") == k) for k in "ABCD"})
print(data[0])
print(data[-1])
PY
```

检查 raw CoT 是否有错误：

```bash
python - <<'PY'
import json
p = "submission_full_sft_32k_200k_max8_cot_test_raw_cot.json"
with open(p) as f:
    data = json.load(f)
errs = [x for x in data if str(x.get("raw_output", "")).startswith("ERROR:")]
print("errors:", len(errs))
print(errs[:3])
PY
```

理想结果：

```text
num: 957
empty: 0
errors: 0
```

检查 short CoT 是否真的输出了 `Final answer`：

```bash
python - <<'PY'
import json
p = "submission_full_sft_32k_200k_max8_shortcot_49k_512_raw_cot.json"
with open(p) as f:
    data = json.load(f)
errs = [x for x in data if str(x.get("raw_output", "")).startswith("ERROR:")]
no_final = [x for x in data if "FINAL ANSWER" not in str(x.get("raw_output", "")).upper()]
print("num:", len(data))
print("errors:", len(errs))
print("no_final_answer:", len(no_final))
print("first no_final:", no_final[:1])
PY
```

如果 `no_final_answer` 很多，说明 prompt 仍然不够强，或者输出仍被截断。此时应继续缩短 prompt，而不是只增加 `max_tokens`。

普通 CoT 256 的检查文件：

```bash
python - <<'PY'
import json
p = "egocross_outputs/full_sft_32k_200k_max8_cot_49k_256/submission_full_sft_32k_200k_max8_cot_49k_256_raw_cot.json"
with open(p) as f:
    data = json.load(f)
errs = [x for x in data if str(x.get("raw_output", "")).startswith("ERROR:")]
no_final = [x for x in data if "FINAL ANSWER" not in str(x.get("raw_output", "")).upper()]
print("num:", len(data))
print("errors:", len(errs))
print("no_final_answer:", len(no_final))
print("first no_final:", no_final[:1])
PY
```

Short CoT 512 的检查文件：

```bash
python - <<'PY'
import json
p = "egocross_outputs/full_sft_32k_200k_max8_shortcot_49k_512/submission_full_sft_32k_200k_max8_shortcot_49k_512_raw_cot.json"
with open(p) as f:
    data = json.load(f)
errs = [x for x in data if str(x.get("raw_output", "")).startswith("ERROR:")]
no_final = [x for x in data if "FINAL ANSWER" not in str(x.get("raw_output", "")).upper()]
print("num:", len(data))
print("errors:", len(errs))
print("no_final_answer:", len(no_final))
print("first no_final:", no_final[:1])
PY
```

## 12. 打包提交

Codabench 需要：

```text
submission.zip
└── predictions.json
```

打包命令：

```bash
cp submission_full_sft_32k_200k_max8_cot_test.json predictions.json
zip submission.zip predictions.json
```

如果已经在实验输出目录中打包为 `submission.zip`，直接上传该文件即可。

## 13. 重要原则

1. 不覆盖 baseline 文件。
2. 新实验必须新文件名，例如带 `cot`、`49k`、`max8`。
3. 每次只改一个关键变量，方便比较分数。
4. 如果 vLLM 出现 500，优先重启服务，不要继续跑污染结果。
5. 如果 raw 输出里有 `ERROR`，对应提交不建议直接上传。
6. CoT 输出只能用于分析，最终 predictions.json 只填 A/B/C/D。

## 14. 后续可能优化方向

可继续尝试：

```text
1. 当前最佳是 router_baseline_strong_weighted_weak，Overall = 0.489028，先保留并作为新基线。
2. 如果继续训练，可尝试 industry=5,xsports=5 或只增强 Industry/XSports，但要继续用 domain router 防止损伤 Surgery/Animal。
3. 可尝试第二个 weighted Full SFT：learning_rate=3e-6 或 weights industry=3,xsports=5，比较弱 domain 提升和强 domain 损伤。
4. enhanced 数据已经验证为 few-shot 短推理不稳定，暂不建议用于正式提交 prompt；若继续使用，只做小样本诊断或离线错误分析。
5. 对 ExtrameSportFPV 特别长样本继续使用 --frame-route VID006=2。
6. few-shot 当前两轮均未超过 direct/router，后续不要全量盲跑；若继续尝试，优先做 text-only few-shot、答案字母平衡、或错误类型定向示例。
```

当前最可靠 baseline 仍然是：

```text
full_sft_32k_200k + direct prompt + max_frames 8
Codabench Overall = 0.480669
```

当前最佳提交是：

```text
router_baseline_strong_weighted_weak_direct_max8_vid006_2f
Codabench Overall = 0.489028
```

## 14.1 外部 GRPO 模型结果（2026-05-09）

同学训练的 GRPO answer 模型路径：

```text
/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
```

已完成一次 direct 推理提交，结果明显超过旧 router：

```text
Overall: 0.528736
Surgery: 0.530035
Industry: 0.534694
XSports: 0.459350
Animal: 0.612022
coverage: 1.0
```

当前判断：

```text
这个 GRPO 模型应作为新的主力候选。
Animal 略低于旧 baseline Animal=0.622951，因此优先尝试 Animal fallback router：
Surgery / Industry / XSports 使用 GRPO，Animal 使用 baseline。
下一步可尝试 direct max_frames=12，并对 VID006 显式使用 --frame-route VID006=4。
router 脚本已更新 context length retry 顺序：start -> 12 -> 8 -> 6 -> 4 -> 2 -> 1 中不超过 start 的序列。
例如 default-max-frames=12 时，会按 12 -> 8 -> 6 -> 4 -> 2 -> 1 尝试；
VID006=4 时，会按 4 -> 2 -> 1 尝试。
```

## 15. Weighted answer-only Full SFT 推荐命令

准备环境：

```bash
cd /share/home/group9/lsg/LlamaFactory
conda activate lsg
export PATH="/share/home/group9/miniconda3/envs/lsg/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p logs
```

构建 weighted answer-only 数据：

```bash
python scripts/prepare_egocross_weighted_answer_only.py \
  --data-dir /share/home/group9/data/egocross \
  --output /share/home/group9/data/egocross/train_weighted_answer_only_i4_x4.json \
  --weights animal=1,surgery=1,industry=4,xsports=4 \
  --seed 2026
```

4 卡训练示例（卡号可按实际空闲情况改）：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=4 \
llamafactory-cli train configs/egocross_weighted_answer_only_full_ep1.yaml \
  2>&1 | tee logs/weighted_answer_only_full_ep1_$(date +%Y%m%d_%H%M%S).log
```

输出模型：

```text
saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1
```

训练完成后建议先用 direct prompt + max_frames 8 跑完整 test，不要叠加 router/domain prompt。这样可以单独评估“模型本身是否变强”。

## 16. 当前最佳 router 提交流程

前提：vLLM 已启动新模型：

```text
saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1
```

只让新模型推理 Industry / XSports，Surgery / Animal 使用 baseline fallback：

```bash
python scripts/egocross_router_infer.py \
  --template submission_template.json \
  --fallback-submission egocross_outputs/baseline_max8/submission_full_sft_32k_200k_max8.json \
  --only-datasets ENIGMA,ExtrameSportFPV \
  --default-prompt-mode direct \
  --default-max-frames 8 \
  --frame-route VID006=2 \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir egocross_outputs/router_baseline_strong_weighted_weak_direct_max8_vid006_2f
```

提交文件：

```text
egocross_outputs/router_baseline_strong_weighted_weak_direct_max8_vid006_2f/submission.zip
```

Codabench 结果：

```text
Overall: 0.489028
Surgery: 0.515901
Industry: 0.416327
XSports: 0.430894
Animal: 0.622951
coverage: 1.0
```
