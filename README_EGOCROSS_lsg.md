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

当前最重要的模型：

```bash
/share/home/group9/lsg/LlamaFactory/saves/egocross/qwen3vl4b/full_sft_32k_200k
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

当前结论：

```text
直接答案 baseline 仍然最高。
普通 CoT 256 明显下降，主要问题是思维链输出偏长，部分样本在 max_tokens=256 内没有输出 Final answer。
缺少最终答案时，脚本会从输出中抽取答案或 fallback 到 A，容易污染预测。
Short CoT 512 比普通 CoT 256 更好，但仍低于直接答案 baseline。
router_domain_direct_lowdomains 的 Overall 与 baseline 持平：XSports 提升 2 题，但 Industry 下降 2 题，互相抵消。
```

因此，后续如果继续做 CoT，建议优先：

```text
1. 继续缩短 prompt，让模型更快输出 Final answer。
2. 检查 *_raw_cot.json 中 no_final_answer 的比例。
3. 不只增加 max_tokens，因为更长输出会显著变慢，也可能增加无关解释。
4. 目前不建议直接用 enhanced 长解释做主训练集。优先做 domain-weighted answer-only SFT，让 assistant 仍只输出 A/B/C/D。
```

## 5.2 当前最新方向（2026-05-02）

当前最值得继续的是训练而不是继续调长 CoT prompt：

```text
从当前最强模型 saves/egocross/qwen3vl4b/full_sft_32k_200k 继续 Full SFT
训练数据使用原始 support set，强制 assistant answer-only
Industry / XSports 过采样，提高弱 domain 权重
先跑 1 epoch、低学习率，避免破坏 Surgery / Animal
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
对明确的 context length 400 错误支持自动降帧 retry：8 -> 4 -> 2 -> 1。
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
max_frames 12 -> 超长后 fallback 8 -> fallback 4 -> fallback 1
```

但实际效果不好。原因是 vLLM 遇到前一个超长请求后，有时服务进入不稳定状态，后续较少帧请求也会返回 500。

更新结论：

```text
对 vLLM 500 / deepstack token 错误，仍然不建议自动 fallback，优先重启服务。
对明确的 400 context length 错误，可以安全做同一样本降帧 retry。
当前 router 脚本已实现 context length retry：8 -> 4 -> 2 -> 1。
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
1. 先跑 weighted answer-only Full SFT ep1，验证训练是否提升整体分。
2. 如果 weighted Full SFT 提升 Industry/XSports 但损伤 Surgery/Animal，降低权重到 industry=3,xsports=3 或降低 learning_rate。
3. 如果 Full SFT 不涨，再考虑只对 XSports 使用 domain_direct router，Industry 回到 baseline/direct。
4. enhanced 数据只作为后续短证据蒸馏候选，不直接用长解释训练主模型。
5. 对 ExtrameSportFPV 特别长样本可使用 context length retry 降帧，减少视觉 token。
```

当前最可靠基准仍然是：

```text
full_sft_32k_200k + direct prompt + max_frames 8
Codabench Overall = 0.480669
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
