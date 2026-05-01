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

结论：目前不建议自动降帧。更推荐固定一个较稳的 `max-frames`。

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
  --output submission_full_sft_32k_200k_max8_cot_49k.json
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

## 12. 打包提交

Codabench 需要：

```text
submission.zip
└── predictions.json
```

打包命令：

```bash
cp submission_full_sft_32k_200k_max8_cot_test.json predictions.json
zip submission_cot_max8.zip predictions.json
```

如果平台要求 zip 名必须是 `submission.zip`：

```bash
cp submission_cot_max8.zip submission.zip
```

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
1. CoT prompt + max_frames 8
2. CoT prompt + max_model_len 49152
3. 使用 train_*_enhanced.json 这类带思维链标注的数据重新训练
4. 分 domain 训练或分 domain 推理
5. 对帧做筛选，去掉无效帧，减少视觉 token
6. 对 prompt 做更细的 domain-specific 设计
```

当前最可靠基准仍然是：

```text
full_sft_32k_200k + direct prompt + max_frames 8
Codabench Overall = 0.480669
```
