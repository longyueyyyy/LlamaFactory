# EgoCross Experiment README

## Final Status, 2026-05-13

This README is now a historical appendix. The project is finished and cleaned for report writing.

Final kept candidate:

```text
model: /share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
inference: direct + tail_dense + max12 + VID006=4 + temperature=0
hidden fixed score: Overall 0.536050, coverage 1.0
kept output: egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
```

Report-facing docs:

```text
docs/egocross/README.md
docs/egocross/file_map.md
egocross_outputs/final_report/experiment_record_20260513.md
```

Final challenger outcomes:

```text
DPO LoRA B fixed hidden Overall 0.526646; not promoted.
KTO v1 perm4 support 0.887500, fixed hidden Overall 0.527691; not promoted.
KTO v2 perm24 support 0.875000, fixed hidden Overall 0.533960; not promoted.
```

Hidden results are recorded as fixed evaluations only. Do not use them to tune prompt, router, frame sampling, checkpoint selection, reward, or training hyperparameters.

## 0. RL Training Handoff For New Codex

This README is a historical appendix. It is intentionally long. For a new RL-training-focused Codex session:

1. Read `agents.md` completely.
2. Do **not** read this README end to end.
3. Use `rg` to pull only the needed historical fragments.

Historical mission during active experimentation:

```text
Focus was RL/model training from scratch or from a strong base.
Hidden-test-driven inference tuning was disallowed.
The current best remains the external GRPO model with direct + tail_dense + max12 + VID006=4.
Any future model must be selected by support/fold diagnostics before a single fixed hidden evaluation.
```

Recommended selective searches:

```bash
rg -n -C 4 "0\\.536050|Fixed challenger|DPO LoRA B|XSports=8|VID059=8|Dynamic frame|GRPO direct tail_dense max12" README_EGOCROSS_lsg.md
rg -n -C 4 "hidden|leaderboard|Do not|submission_template|Do not promote|Do not overwrite" README_EGOCROSS_lsg.md
rg -n -C 5 "Conservative LoRA DPO|Current DPO memory profile|Fold protocol|Promotion criteria|Result record|GRPO baseline" README_EGOCROSS_lsg.md
rg -n -C 4 "vLLM|Qwen3|rope_scaling|max-model-len|context length|mm_hash|deepstack" README_EGOCROSS_lsg.md
rg -n -C 3 "CODE=/share|SUPPORT=/share|support-dir|submission_template|egocross_support_eval|router_infer|fold" README_EGOCROSS_lsg.md
```

Short history summary:

```text
Current best fixed test:
/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
direct + tail_dense + max12 + VID006=4 + temperature=0
Overall 0.536050, coverage 1.0.

Prior DPO LoRA B:
fold/support looked stable, but fixed test Overall 0.526646; do not repeat that exact small DPO sweep as the main plan.

Recent inference diagnostics:
ENIGMA=8 improved runtime only; XSports=8/VID059=8 were support/fold positive but related fixed test tied current best; dynamic query_diverse did not improve support.
These are historical records, not hidden-feedback tuning signals.
```

鏈枃妗ｆ槸 EgoCross 椤圭洰鐨勫綋鍓嶄氦鎺ヨ褰曘€傜洰鏍囨槸璁╁悗缁帴鎵嬭€呭揩閫熺煡閬擄細褰撳墠鏈€寮烘柟妗堛€佹湇鍔″櫒璺緞銆佸悎瑙勮竟鐣屻€佸叧閿剼鏈€佸凡缁忓皾璇曡繃鐨勬柟鍚戯紝浠ュ強鍝簺鏂囦欢涓嶈兘瑕嗙洊銆?
## 1. 椤圭洰姒傚喌

浠诲姟褰㈠紡锛?
```text
澶氬抚绗竴瑙嗚瑙嗛甯?+ 澶氶€夐鏂囨湰 -> 杈撳嚭 A/B/C/D
```

鏈€缁堟彁浜ゅ彧闇€瑕侊細

```text
submission.zip
鈹斺攢鈹€ predictions.json
```

`predictions.json` 姣忛鍙～涓€涓瓟妗堝瓧姣嶏紝涓嶆彁浜よВ閲娿€丆oT 鎴栦腑闂磋瘉鎹€?
## 2. 姣旇禌鍚堣瑙勫垯

蹇呴』閬靛畧锛?
```text
Do not attempt to infer hidden labels by probing, reverse engineering, or exploiting evaluation behavior.
Manual per-example labeling of hidden test data is prohibited.
```

鏈」鐩墽琛岃鍒欙細

```text
1. 涓嶉€氳繃鍙嶅鎻愪氦 hidden test銆佽瀵熸鍗?domain 鍒嗘暟鏉ュ弽鎺ㄦ爣绛俱€佸畾浣嶉敊璇牱鏈垨璋冩暣 router/fallback/prompt銆?2. 涓嶆墜宸ユ爣娉?hidden test 鏍锋湰銆?3. 鎺ㄧ悊绛栫暐搴斿湪鎻愪氦鍓嶅浐瀹氾紝渚濇嵁鍏紑 support/validation銆佽缁冩棩蹇椼€佹牸寮忕ǔ瀹氭€с€乧overage銆佽繍琛岄敊璇巼绛夐潪 hidden-label 淇″彿鍐冲畾銆?4. 鎶ュ憡涓笉瑕佸啓鈥滄牴鎹?hidden 姒滃崟鍙嶉璋冩暣绛栫暐鈥濄€傚彲鍐欌€滃熀浜?support set 涓庨瞾妫掓€ц瘖鏂€夋嫨鍥哄畾鎺ㄧ悊绛栫暐锛宧idden test 浠呯敤浜庢渶缁堣瘎浼扳€濄€?```

鍘嗗彶 Codabench 鍒嗘暟鍙綔涓哄疄楠岃褰曘€傚悗缁笉瑕佹妸 hidden 姒滃崟褰撳紑鍙戦泦銆?
## 3. 鏈嶅姟鍣ㄨ矾寰?
```bash
CODE=/share/home/group9/lsg/LlamaFactory
SUPPORT=/share/home/group9/data/egocross
TESTBED=/share/home/group9/data/egocross_full/egocross_testbed
TEST_JSON=/share/home/group9/data/egocross_full/egocross_testbed/egocross_testbed_imgs.json
TEMPLATE=/share/home/group9/lsg/LlamaFactory/submission_template.json
```

閲嶈妯″瀷锛?
```text
baseline:
saves/egocross/qwen3vl4b/full_sft_32k_200k

weighted answer-only SFT:
saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1

external GRPO candidate:
/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
```

鏈湴鏁版嵁闀滃儚鍦ㄤ粨搴撳悓绾э細

```text
../data_local/egocross
```

鏈湴闀滃儚鍙敤浜庨潪鐮村潖鎬ф鏌ュ拰鑴氭湰璋冭瘯銆傝缁?鎺ㄧ悊鍛戒护閲岀殑鏈嶅姟鍣ㄨ矾寰勪笉瑕佸亣璁炬湰鍦板彲鐢ㄣ€?
## 4. 褰撳墠鏈€寮哄€欓€?
褰撳墠鏈€寮烘柟妗堬細

```text
model: /share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
prompt: direct
frame_sampling: tail_dense
max_frames: 12
frame_route: VID006=4
router/fallback: none
output: egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
```

鍒嗘暟璁板綍锛?
```text
Overall: 0.536050
Surgery: 0.533569
Industry: 0.538776
XSports: 0.459350
Animal: 0.639344
coverage: 1.0
```

褰撳墠寤鸿锛氬厛淇濈暀杩欑増浣滀负鎻愪氦鍊欓€夛紝涓嶇户缁熀浜?hidden 姒滃崟鍙嶉鍋氱粏璋冦€俙ctx65536 + max12 + no VID006 route + temp0` 宸插鐜板悓鍒?0.536050锛屼絾宸ョ▼澶嶆潅搴︽洿楂橈紱`max_frames=16`銆乣ctx65536 + max24`銆乣temperature=0.2` 鍜?`temperature=0.7 + vote` 鍧囦綆浜庡綋鍓嶆渶浣炽€?
## 5. 褰撳墠鏈€浣宠繍琛屽懡浠?
鍚姩 vLLM锛?
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

妫€鏌ユ湇鍔★細

```bash
curl http://127.0.0.1:8000/v1/models
```

璺戝綋鍓嶆渶寮哄€欓€夛細

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

杈撳嚭妫€鏌ワ細

```bash
cat egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f/metrics_summary.txt
```

濡傛灉 `status_dist` 閲屾湁澶ч噺 `error_fallback`锛屼笉瑕佹彁浜よ缁撴灉銆?
## 5.1 Support 鍥哄畾绛栫暐楠岃瘉

鍚庣画濡傛灉瑕佹瘮杈?prompt / frame_sampling / max_frames锛屼紭鍏堝湪鍏紑 support set 涓婂浐瀹氱瓥鐣ュ悗楠岃瘉锛屼笉瑕佹牴鎹?hidden 姒滃崟鍙嶉缁х画璋冦€?
鏀寔鑴氭湰锛?
```bash
scripts/egocross_support_eval.py
```

绀轰緥锛氶獙璇佸綋鍓嶆渶寮虹瓥鐣ュ湪 support set 涓婄殑 acc锛?
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

杈撳嚭鏂囦欢锛?
```text
support_predictions.json
support_metrics.json
support_metrics.txt
```

Support-only analysis for an existing eval output:

```bash
python scripts/egocross_support_analyze.py \
  --predictions egocross_outputs/support_eval/grpo_direct_tail_dense_max12_vid006_4f/support_predictions.json \
  --metrics egocross_outputs/support_eval/grpo_direct_tail_dense_max12_vid006_4f/support_metrics.json \
  --support-dir /share/home/group9/data/egocross \
  --output-dir egocross_outputs/support_eval/grpo_direct_tail_dense_max12_vid006_4f_analysis
```

For fold outputs, pass the fold eval JSON so per-video diagnostics can be reconstructed from public metadata:

```bash
python scripts/egocross_support_analyze.py \
  --predictions egocross_outputs/support_eval/grpo_baseline_folds_20260510_111639/fold0_direct_tail_dense_max12_vid006_4f/support_predictions.json \
  --metrics egocross_outputs/support_eval/grpo_baseline_folds_20260510_111639/fold0_direct_tail_dense_max12_vid006_4f/support_metrics.json \
  --support-dir /share/home/group9/data/egocross \
  --eval-json /share/home/group9/data/egocross/pref_answer_only_all_equal_folds/eval_answer_only_all_equal_fold0.json \
  --output-dir egocross_outputs/support_eval/analysis_grpo_baseline_fold0
```

涓€娆℃€ц窇澶氫釜棰勮鍊欓€夌瓥鐣ワ細

```bash
cd /share/home/group9/lsg/LlamaFactory

python scripts/run_egocross_support_eval_grid.py
```

榛樿鍊欓€夊寘鎷細

```text
baseline: direct + tail_dense + max12 + VID006=4
diagnostic: direct + tail_dense + max8 + VID006=4
direct + tail_dense + max12 + ENIGMA=8 + VID006=4
direct + endpoint + max12 + VID006=4
direct + uniform + max12 + VID006=4
```

Current default grid intentionally excludes historical negative or unstable branches such as `max16` and `strict_direct`; they remain available only through explicit `--candidates` names. Use `--list-candidates` to inspect the registry.

Dynamic frame selection candidates are explicit-only. They keep `direct + temperature=0` fixed and change only frame sampling:

```bash
python scripts/run_egocross_support_eval_grid.py \
  --support-dir /share/home/group9/data/egocross \
  --base-url http://127.0.0.1:8000/v1 \
  --out-root egocross_outputs/support_eval/grpo_dynamic_frame_${STAMP} \
  --log-dir logs/support_eval_dynamic_frame_${STAMP} \
  --candidates direct_query_diverse_max12_vid006_4f,direct_query_diverse_tail_max12_vid006_4f
```

Before running model inference, preview deterministic frame choices without calling vLLM:

```bash
python scripts/egocross_preview_frame_sampling.py \
  --support-dir /share/home/group9/data/egocross \
  --sampling tail_dense,query_diverse,query_diverse_tail \
  --max-frames 12 \
  --frame-route VID006=4 \
  --limit 20
```

`query_diverse` and `query_diverse_tail` are lightweight, deterministic samplers. They use public input metadata only: question type, question/options text, option time ranges when present, timeline coverage, and diversity. They do not use labels, hidden feedback, CoT, voting, or extra trained weights.

姹囨€昏〃锛?
```text
egocross_outputs/support_eval/_grid_summary.tsv
```

鍙€夌幆澧冨彉閲忥細

```bash
python scripts/run_egocross_support_eval_grid.py \
  --base-url http://127.0.0.1:8000/v1 \
  --support-dir /share/home/group9/data/egocross \
  --out-root egocross_outputs/support_eval
```

Fold validation example:

```bash
python scripts/run_egocross_support_eval_grid.py \
  --support-dir /share/home/group9/data/egocross \
  --eval-json /share/home/group9/data/egocross/pref_answer_only_all_equal_folds/eval_answer_only_all_equal_fold0.json \
  --base-url http://127.0.0.1:8000/v1 \
  --out-root egocross_outputs/support_eval/grpo_infer_grid_fold0_${STAMP} \
  --log-dir logs/support_eval_grid_fold0_${STAMP}
```

Promotion gate for any inference router:

```text
1. Full support must beat baseline by at least one sample.
2. Four-fold mean must beat 0.8625, with at least 3/4 folds not below baseline.
3. coverage=1.0, parse_fail=0, error_fallback=0.
4. No obvious per-domain collapse and no answer distribution collapse.
5. If these fail, record a support negative result and do not run hidden test.
```

Support-only inference grid result, 2026-05-10:

```text
run: egocross_outputs/support_eval/grpo_infer_grid_20260510_161738
model: external GRPO v3
fixed settings: direct + tail_dense + temperature=0 + answer-only

baseline direct_tail_dense_max12_vid006_4f:
overall_acc: 0.862500 (69/80)
Animal/Surgery/Industry/XSports: 0.800000 / 0.950000 / 0.900000 / 0.800000
answer_only_format_rate: 1.000000
coverage: 1.000000
parse_fail/error/error_fallback: 0/0/0
avg_used_frames: 11.4
runtime_seconds: 201.942
status_dist: {"ok": 68, "ok_after_frame_retry": 12}
used_max_frames_dist: {"12": 68, "8": 12}
answer_dist: {"A": 17, "B": 18, "C": 21, "D": 24}

ENIGMA=8 route direct_tail_dense_max12_enigma_8f_vid006_4f:
overall_acc: 0.862500 (69/80)
Animal/Surgery/Industry/XSports: 0.800000 / 0.950000 / 0.900000 / 0.800000
answer_only_format_rate: 1.000000
coverage: 1.000000
parse_fail/error/error_fallback: 0/0/0
avg_used_frames: 11.0
runtime_seconds: 161.924
status_dist: {"ok": 80}
used_max_frames_dist: {"12": 60, "8": 20}
answer_dist: {"A": 17, "B": 18, "C": 21, "D": 24}

Analysis:
ENIGMA=8 produced the same predictions and the same 11 support errors as baseline.
It removed support retry events and reduced runtime, so it is a robustness/runtime improvement only.
It does not satisfy the promotion gate because full support accuracy did not beat baseline by at least one sample.
Do not run hidden test for this candidate.

Other grid notes:
direct_tail_dense_max8_vid006_4f tied overall at 0.862500 but shifted domains (Animal 0.75, XSports 0.85); diagnostic only.
direct_endpoint_max12_vid006_4f tied baseline without robustness gain.
direct_uniform_max12_vid006_4f dropped to 0.850000.
```

Fixed challenger test result, 2026-05-10:

```text
strategy fixed from support/fold before test:
direct + tail_dense + max12 + VID006=4 + ExtrameSportFPV=8 + temperature=0

support/fold validation:
full support: 0.875000 (70/80), baseline was 0.862500 (69/80)
fold acc: 0.950000, 0.900000, 0.850000, 0.800000; mean 0.875000
baseline fold acc: 0.900000, 0.900000, 0.850000, 0.800000; mean 0.862500
coverage/parse/error_fallback: 1.0/0/0

fixed test result:
Overall: 0.536050
Surgery: 0.533569
Industry: 0.538776
XSports: 0.459350
Animal: 0.639344
coverage: 1.0

Conclusion:
XSports=8 was a valid support/fold-selected fixed challenger, but hidden fixed test tied the current GRPO best exactly.
It did not exceed 0.536050 and should not replace the simpler current best strategy.
Do not use this hidden tie to further tune XSports/video/frame routes.
Keep current best as direct + tail_dense + max12 + VID006=4 + temperature=0.
```

Post-test support/fold diagnostic, 2026-05-10:

```text
strategy:
direct + tail_dense + max12 + VID006=4 + VID059=8 + temperature=0

full support:
overall_acc: 0.875000 (70/80)
Animal: 0.800000 (16/20)
Industry: 0.900000 (18/20)
Surgery: 0.950000 (19/20)
XSports: 0.850000 (17/20)
answer_only_format_rate: 1.000000
coverage: 1.000000
parse_fail/error/error_fallback: 0/0/0
status_dist: {"ok": 68, "ok_after_frame_retry": 12}
answer_dist: {"A": 18, "B": 18, "C": 21, "D": 23}
used_max_frames_dist: {"12": 59, "8": 21}

fold validation:
fold acc: 0.950000, 0.900000, 0.850000, 0.800000; mean 0.875000
baseline fold acc: 0.900000, 0.900000, 0.850000, 0.800000; mean 0.862500
coverage/parse_fail/error_fallback: 1.0/0/0 on all folds

Conclusion:
VID059=8 is support/fold-positive and narrower than the earlier ExtrameSportFPV=8 rule.
However, it was explored after a related fixed hidden test for ExtrameSportFPV=8 tied the current best.
Record as a support/fold diagnostic candidate only; do not run an additional hidden test for this rule in the same feedback cycle.
```

Dynamic frame selection support result, 2026-05-10:

```text
run: egocross_outputs/support_eval/grpo_dynamic_frame_20260510_210129
fixed settings: external GRPO v3 + direct + max12 + VID006=4 + temperature=0 + answer-only

direct_query_diverse_max12_vid006_4f:
overall_acc: 0.862500 (69/80)
Animal/Surgery/Industry/XSports: 0.800000 / 0.950000 / 0.900000 / 0.800000
answer_only_format_rate: 1.000000
coverage: 1.000000
parse_fail/error/error_fallback: 0/0/0
avg_used_frames: 11.4
runtime_seconds: 212.377
status_dist: {"ok": 68, "ok_after_frame_retry": 12}
used_max_frames_dist: {"12": 68, "8": 12}
answer_dist: {"A": 17, "B": 18, "C": 21, "D": 24}
Conclusion: tied baseline on support; no fold/test promotion.

direct_query_diverse_tail_max12_vid006_4f:
overall_acc: 0.850000 (68/80)
Animal/Surgery/Industry/XSports: 0.800000 / 0.950000 / 0.850000 / 0.800000
answer_only_format_rate: 1.000000
coverage: 1.000000
parse_fail/error/error_fallback: 0/0/0
avg_used_frames: 11.4
runtime_seconds: 211.015
status_dist: {"ok": 68, "ok_after_frame_retry": 12}
used_max_frames_dist: {"12": 68, "8": 12}
answer_dist: {"A": 17, "B": 18, "C": 22, "D": 23}
Conclusion: support negative; do not retry on hidden test.

Prediction-diff analysis:
query_diverse changed 0/80 predictions vs baseline, so the current heuristic is effectively answer-equivalent to tail_dense on support.
query_diverse_tail changed only one prediction vs baseline:
Industry_8 / ENIGMA/111 / counting, gold=D, baseline=D correct, query_diverse_tail=C wrong.
Frame preview showed Industry_8 has exactly 12 frames and max_frames=12, so tail_dense/query_diverse/query_diverse_tail use the same frames for that sample.
This suggests the observed one-sample drop is not a useful frame-selection signal; treat query_diverse_tail as support-negative and do not tune it further without a new support-only design.
```

榛樿涓嶈鐩栧凡鏈夐潪绌鸿緭鍑虹洰褰曪紱濡傝鍙墦鍗板懡浠や笉杩愯锛屼娇鐢?`--dry-run`銆?
鏈嶅姟鍣?support 閫夌瓥鐣ュ缓璁祦绋嬶細

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

璺戝畬鏌ョ湅姹囨€伙細

```bash
cat egocross_outputs/support_eval/grpo_grid_${STAMP}/_grid_summary.tsv
```

閫夋嫨瑙勫垯锛?
```text
1. 鍏堢湅 overall support acc銆?2. 鑻?overall 鎺ヨ繎锛屼紭鍏?answer_only_format_rate=1.0銆乻tatus_dist 鏃?error銆乽sed_max_frames_dist 涓嶉绻侀檷甯х殑绛栫暐銆?3. 鍐嶇湅 per-domain锛岄伩鍏嶆煇涓?domain 鏄庢樉宕╂帀銆?4. 閫夊畾鍚庣敤鍚屼竴缁?prompt/frame_sampling/max_frames/VID006 閰嶇疆璺?test锛涗笉瑕佹牴鎹?hidden 姒滃崟鍐嶆敼绛栫暐銆?```

寤鸿绛栫暐锛氶鍏堝垪鍑哄皯閲忓€欓€夛紝渚嬪 `uniform/endpoint/tail_dense`銆乣max8/max12`銆乣direct/strict_direct`锛屼竴娆℃€у湪 support 涓婃瘮杈?overall銆乸er-domain銆乸er-question-type銆乤nswer-only 鏍煎紡鐜囧拰 error rate銆傞€夊畾鍚庣洿鎺ュ簲鐢ㄥ埌 test锛涗笉瑕佸啀鐢?hidden domain 鍒嗘暟鍙嶆帹淇敼銆?
## 6. 鏂囦欢缁撴瀯

鍏抽敭鏂囦欢鍒嗗竷锛?
```text
README_EGOCROSS_lsg.md                 褰撳墠瀹為獙浜ゆ帴 README
agents.md                              闀挎湡 agent 瑙勫垯鍜屽綋鍓嶆渶浣冲懡浠?submission_template.json               瀹樻柟鎻愪氦妯℃澘锛岀姝㈣鐩?
scripts/egocross_router_infer.py        褰撳墠涓绘帹鐞嗚剼鏈?scripts/egocross_support_eval.py        support set 鍥哄畾绛栫暐楠岃瘉鑴氭湰
scripts/run_egocross_support_eval_grid.py
scripts/egocross_support_analyze.py     support/fold prediction diagnostics
scripts/egocross_preview_frame_sampling.py
scripts/egocross_blend_submit.py        鍘嗗彶 blend 宸ュ叿
scripts/prepare_egocross_weighted_answer_only.py
scripts/prepare_egocross_preference_answer_only.py

configs/egocross_*.yaml                 EgoCross 璁粌閰嶇疆
data/dataset_info.json                  LLaMA-Factory 鏁版嵁闆嗘敞鍐?egocross_outputs/                       鍘嗗彶杈撳嚭鐩綍锛屾瘡涓疄楠屽崟鐙缓鐩綍
```

鏍圭洰褰曟棫鑴氭湰锛?
```text
generate_egocross_submission.py
generate_egocross_submission_cot_test.py
generate_egocross_submission_shortcot_test.py
```

杩欎簺鏄巻鍙插叆鍙ｏ紝淇濈暀鐢ㄤ簬澶嶇幇锛涙柊瀹為獙浼樺厛浣跨敤 `scripts/egocross_router_infer.py`銆?
## 7. 鎺ㄧ悊鑴氭湰鑳藉姏

涓昏剼鏈細

```bash
scripts/egocross_router_infer.py
```

鏀寔锛?
```text
1. 鎸?dataset/domain 璺敱 prompt mode 鍜?max_frames銆?2. 鍙窇閮ㄥ垎 domain锛屽叾浠栨牱鏈敤 fallback submission 濉弧瀹屾暣杈撳嚭銆?3. 淇濆瓨 predictions.json銆乺aw_outputs.json銆乵etrics_summary.txt銆乻ubmission.zip銆?4. 瀵规槑纭?context length 400 閿欒鑷姩闄嶅抚 retry銆?5. frame sampling: uniform / endpoint / tail_dense / query_diverse / query_diverse_tail銆?6. query_diverse modes use question type/options/time ranges plus deterministic diversity; no labels or hidden feedback.
7. prompt mode: direct / strict_direct / domain_direct / type_direct / domain_type_direct銆?8. option-order voting锛屼絾褰撳墠 GRPO 妯″瀷涓?voting 瀹炴祴涓嶄匠銆?9. `--temperature` 鎺у埗閲囨牱娓╁害锛涢粯璁ゆ槸 0.0锛屽巻鍙?direct/vote 瀹為獙鍧囦负 deterministic 鎺ㄧ悊銆?```

闄嶅抚搴忓垪锛?
```text
start -> 12 -> 8 -> 6 -> 4 -> 2 -> 1 涓笉瓒呰繃 start 鐨勫簭鍒?```

渚嬪锛?
```text
max_frames=12: 12 -> 8 -> 6 -> 4 -> 2 -> 1
VID006=4: 4 -> 2 -> 1
```

閲囧抚绛栫暐锛?
```text
uniform: 榛樿鏃ц涓猴紝鍧囧寑鍙?max_frames 甯э紝浣嗕笉淇濊瘉鍖呭惈鏈€鍚庝竴甯с€?endpoint: 鍖呭惈棣栧熬甯э紝涓棿鍧囧寑鍙栨牱銆?tail_dense: 鍖呭惈棣栧熬甯э紝骞舵妸鏇村閲囨牱鐐瑰垎閰嶅埌瑙嗛鍚庡崐娈点€?query_diverse: query-aware deterministic coverage/diversity sampler.
query_diverse_tail: query-aware deterministic sampler with extra tail bias for prediction/sequence questions.
```

褰撳墠 GRPO 妯″瀷涓婏紝`direct + tail_dense + max12 + VID006=4` 鏄綋鍓嶆渶寮恒€?
Prompt 缁忛獙锛?
```text
direct: 褰撳墠鏈€绋炽€?strict_direct: 鍙仛鏈€灏?prompt 瀵圭収銆?domain_type_direct/type_direct: 宸插彂鐜板彲鑳借 answer-only GRPO 妯″瀷鍋忕鍒嗗竷銆?temperature=0.7: 鍙互浣滀负 self-consistency/闅忔満閲囨牱瀵圭収锛屼絾鍙兘寮曞叆鍣０锛涗紭鍏堝湪 support 涓婇獙璇併€?CoT/few-shot/enhanced reasoning: 鍘嗗彶涓婂潎鏈秴杩?direct/router锛屼笉寤鸿鐢ㄤ簬鏈€缁堟彁浜ゃ€?```

## 8. 鍘嗗彶鍏抽敭缁撴灉

| 鏂规 | Overall | Surgery | Industry | XSports | Animal | 缁撹 |
|---|---:|---:|---:|---:|---:|---|
| baseline full SFT direct max8 | 0.480669 | 0.515901 | 0.400000 | 0.414634 | 0.622951 | 鏃?baseline |
| weighted answer-only direct | 0.481714 | 0.501767 | 0.416327 | 0.430894 | 0.606557 | 寮卞煙鎻愬崌锛屽己鍩熶笅闄?|
| old router baseline strong + weighted weak | 0.489028 | 0.515901 | 0.416327 | 0.430894 | 0.622951 | 鏃ф渶浣?|
| CoT 49k 256 | 0.444096 | 0.508834 | 0.342857 | 0.418699 | 0.513661 | 鏄庢樉涓嬮檷 |
| short CoT 49k 512 | 0.459770 | 0.515901 | 0.375510 | 0.414634 | 0.546448 | 浠嶄綆浜?direct |
| few-shot weak domains | 0.479624 | 0.515901 | 0.412245 | 0.398374 | 0.622951 | few-shot 浼?XSports |
| enhanced short reasoning few-shot | 0.481714 | 0.515901 | 0.387755 | 0.430894 | 0.622951 | enhanced 浼?Industry |
| external GRPO direct max8 VID006=4 | 0.528736 | 0.530035 | 0.534694 | 0.459350 | 0.612022 | 鏂颁富鍔涙ā鍨?|
| GRPO direct endpoint max8 | 0.526646 | 0.526502 | 0.526531 | 0.459350 | 0.617486 | 浣庝簬 tail_dense |
| GRPO direct tail_dense max8 | 0.529781 | 0.533569 | 0.530612 | 0.459350 | 0.617486 | 閲囧抚鏀硅繘鏈夋晥 |
| GRPO direct tail_dense max12 | 0.536050 | 0.533569 | 0.538776 | 0.459350 | 0.639344 | 褰撳墠鏈€寮哄€欓€?|
| DPO LoRA B from GRPO direct tail_dense max12 | 0.526646 | 0.526502 | 0.542857 | 0.434959 | 0.628415 | fixed challenger run; lower than GRPO best, do not promote |
| GRPO direct tail_dense max16 | 0.532915 | 0.522968 | 0.538776 | 0.459350 | 0.639344 | 浣庝簬 max12锛孲urgery 鍥炶惤 |
| GRPO ctx65536 direct tail_dense max12 no VID006 route temp0 | 0.536050 | 0.533569 | 0.538776 | 0.459350 | 0.639344 | 涓庡綋鍓嶆渶浣冲悓鍒嗭紱闀夸笂涓嬫枃鍙伩鍏嶇壒娈?VID006 route锛屼絾鏃犳彁鍒?|
| GRPO ctx65536 direct tail_dense max24 | 0.531870 | 0.522968 | 0.542857 | 0.459350 | 0.628415 | 闀夸笂涓嬫枃鏇村甯ф彁鍗?Industry锛屼絾鎹熶激 Surgery/Animal |
| GRPO ctx65536 direct tail_dense max12 no VID006 route temp0.2 | 0.529781 | 0.522968 | 0.526531 | 0.459350 | 0.639344 | 杞诲井娓╁害閲囨牱浠嶄激鍒?|
| GRPO domain_type_direct + vote max12 | 0.472309 | 0.487633 | 0.383673 | 0.447154 | 0.601093 | 澶嶆潅 prompt 鏄庢樉浼ゅ垎 |
| GRPO direct + vote max8 | 0.491118 | 0.505300 | 0.412245 | 0.451220 | 0.628415 | voting 涓嶉€傚悎褰撳墠 GRPO |
| GRPO direct tail_dense max12 vote temp0.7 | 0.486938 | 0.505300 | 0.420408 | 0.426829 | 0.628415 | 楂樻俯鎶曠エ鏄庢樉浼ゅ垎锛屼笉寤鸿缁х画 |

璇存槑锛氳〃涓垎鏁版槸瀹為獙璁板綍锛屼笉搴旂敤浜庣户缁 hidden test 鍋氬弽鎺ㄨ皟鍙傘€?
## 9. 璁粌鐩稿叧璁板綍

宸插畬鎴愯缁冿細

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

宸叉柊澧炰絾涓嶄竴瀹氬凡璺戦€氱殑閰嶇疆锛?
```text
configs/egocross_dpo_answer_only_full_all_equal_lr1e6_ep1.yaml
configs/egocross_dpo_answer_only_lora_all_equal_lr1e6_ep1.yaml
configs/egocross_answer_only_full_all_equal_lr2e6_ep1.yaml
```

鏄惧瓨闄愬埗涓嬶紝full DPO 瀹规槗 OOM锛涘熀纭€ SFT 鏇村鏄撹窇銆傚綋鍓嶆渶浣虫潵鑷閮?GRPO 妯″瀷鎺ㄧ悊浼樺寲锛屼笉鏄湰鍦?DPO銆?
鐢熸垚鍏ㄥ煙 answer-only SFT 鏁版嵁锛?
```bash
python scripts/prepare_egocross_weighted_answer_only.py \
  --data-dir /share/home/group9/data/egocross \
  --output /share/home/group9/data/egocross/train_answer_only_all_equal.json \
  --weights animal=1,surgery=1,industry=1,xsports=1 \
  --seed 2026
```

鐢熸垚鍏ㄥ煙 DPO preference 鏁版嵁锛?
```bash
python scripts/prepare_egocross_preference_answer_only.py \
  --data-dir /share/home/group9/data/egocross \
  --output /share/home/group9/data/egocross/train_pref_answer_only_all_equal_wrong3_fmt1.json \
  --fold-output-dir /share/home/group9/data/egocross/pref_answer_only_all_equal_folds
```

## 10. vLLM 鍜岀幆澧冨潙鐐?
鎺ㄨ崘鍚姩鍙傛暟锛?
```text
VLLM_USE_V1=0
--enforce-eager
--mm-processor-cache-gb 0
```

鍘熷洜锛?
```text
1. Qwen3-VL 澶氬浘杈撳叆鏇鹃亣鍒?deepstack token 瀵归綈闂銆?2. vLLM v1 澶氭ā鎬佽矾寰勫拰 mm processor cache 鏇惧嚭鐜?mm_hash/cache 闂銆?3. 鍏抽棴 cache 鍜?old engine 鏇存參锛屼絾鏇寸ǔ銆?```

VID006锛?
```text
ExtrameSportFPV_VID006 鏄暱鏍锋湰锛岃瑙?token 澶氥€?褰撳墠 GRPO tail_dense 鏈€寮哄€欓€変娇鐢?--frame-route VID006=4銆?濡傛灉鍑虹幇 context length 鎴?vLLM 閿欒锛屽彲灏濊瘯鏇翠綆 VID006=2锛屼絾涓嶈鐢?hidden 鍙嶉鍋氱粏璋冦€?```

璁粌/鎺ㄧ悊 config锛?
```text
transformers 璁粌闃舵鍙兘闇€瑕?text_config.rope_scaling={"rope_type":"default"}銆?vLLM 鎺ㄧ悊闃舵鍘嗗彶涓?null rope_scaling 鏇寸ǔ銆?鍒囨崲鍓嶆鏌ユā鍨?config锛岄伩鍏嶈缁冪増/鎺ㄧ悊鐗堟贩鐢ㄣ€?```

## 11. 鏂囦欢淇濇姢瑙勫垯

涓嶈瑕嗙洊锛?
```text
submission_template.json
saves/egocross/qwen3vl4b/full_sft_32k_200k
saves/egocross/qwen3vl4b/weighted_answer_only_full_i4_x4_lr5e6_ep1
egocross_outputs/baseline_max8
egocross_outputs/router_baseline_strong_weighted_weak_direct_max8_vid006_2f
egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f
```

姣忎釜鏂板疄楠屽繀椤绘柊寤?output dir锛屽懡鍚嶅寘鍚ā鍨嬨€乸rompt銆乻ampling銆乵ax_frames銆乂ID006 璁剧疆銆?
## 12. 寤鸿鐨勫悗缁伐浣?
褰撳墠寤鸿鍏堝仠鍦細

```text
egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f/submission.zip
```

濡傛灉鍚庣画缁х画鎺㈢储锛屼紭鍏堝湪 support/validation 涓婇鍏堥獙璇佸浐瀹氱瓥鐣ワ紝鍐嶆彁浜?hidden test銆傚缓璁『搴忥細

```text
1. 鏀寔闆嗛獙璇佷笉鍚?frame_sampling锛岃€屼笉鏄敤 hidden 鍒嗘暟璋冦€?2. 淇濇寔 direct prompt锛屼笉浼樺厛鍔?CoT/few-shot/type prompt銆?3. 鑻ヨ鏀?VID006 甯ф暟锛屽厛鍩轰簬 context length / error rate 绛夐瞾妫掓€т俊鍙凤紝涓嶅熀浜?hidden 鍒嗘暟銆?4. ctx65536 + max12 no VID006 route 宸蹭笌褰撳墠鏈€浣冲悓鍒嗭紝鍙綔涓哄伐绋嬪鐓э紝浣嗘棤鎻愬垎锛涙渶缁堟彁浜や紭鍏堜繚鐣欐洿绠€鍗曠ǔ瀹氱殑鍘?max12 + VID006=4銆?5. 涓嶅缓璁户缁俯搴﹂噰鏍锋垨楂樻俯鎶曠エ锛泃emperature=0.2 涓?temperature=0.7 + vote 鍧囦綆浜?deterministic direct銆?```

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

Current DPO memory profile:
```text
All fold/full LoRA DPO configs now use cutoff_len=24576 and image_max_pixels/video_max_pixels=131072.
Their output_dir paths include _memsafe_ctx24576_px131k.
Keep this setting identical for A/B/C and all folds so preference parameters are the only intended comparison variable.
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LLAMAFACTORY_LOGPS_CHUNK_SIZE=128 CUDA_VISIBLE_DEVICES=4,5 FORCE_TORCHRUN=1 llamafactory-cli train \
  configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_fold0_lr1e5_beta003_ftx005_ep1.yaml
```

Queued A/B/C fold run:
```bash
# Conservative: one 2-GPU lane, runs all A/B/C x fold0-3 sequentially.
bash scripts/run_egocross_dpo_lora_folds_abc_memsafe.sh

# Faster on four free A800s: two 2-GPU lanes, each lane runs a sequential subset.
GPU_LANES="4,5 6,7" bash scripts/run_egocross_dpo_lora_folds_abc_memsafe.sh

# Optional subsets:
CANDIDATES="A B" FOLDS="0 1 2 3" GPU_LANES="4,5 6,7" bash scripts/run_egocross_dpo_lora_folds_abc_memsafe.sh
```

The queue writes logs to `logs/egocross_dpo_lora_folds_abc_memsafe_<timestamp>/`, with one log per candidate/fold, plus `driver.log` and `summary.tsv`. If an output directory already exists and is non-empty, the script stops before training; set `SKIP_EXISTING=1` only when you intentionally want to skip existing runs.

Export a trained fold adapter for heldout evaluation:
```bash
bash scripts/export_egocross_dpo_lora_fold_memsafe.sh A 0
```

Automated heldout evaluation:
```bash
# GRPO baseline: one vLLM serves the baseline model and runs fold0-3 sequentially.
GPU=4 PORT=8000 bash scripts/run_egocross_grpo_baseline_fold_eval.sh

# DPO A/B/C folds: export missing merged fold models, then evaluate through vLLM lanes.
# If baseline is running on GPU4/port8000, use the other three GPUs here.
GPU_LANES="5:8001 6:8002 7:8003" bash scripts/run_egocross_dpo_lora_fold_eval_abc_memsafe.sh

# If baseline is already done, DPO eval can use all four GPUs.
GPU_LANES="4:8001 5:8002 6:8003 7:8004" bash scripts/run_egocross_dpo_lora_fold_eval_abc_memsafe.sh
```

The DPO eval script lazily exports a missing merged model immediately before that fold is evaluated, using a small lock so exports do not run concurrently. Existing eval outputs can be skipped with `SKIP_EXISTING_EVAL=1`.
If the user quota is tight, set `DELETE_MERGED_AFTER_EVAL=1` to remove each merged fold model after its heldout metrics are written. This keeps the LoRA adapter directories and eval outputs.

For vLLM inference, exported DPO fold models are patched so `text_config.rope_scaling` is `null`. Keep the train-safe base at `{"rope_type":"default"}` for Transformers training/export; only the merged inference copy should be patched for vLLM.

DPO OOM note:
```text
Long multimodal DPO may OOM inside get_batch_logps at logits.log_softmax(-1). DDP does not shard this per-rank tensor. This repo now avoids full-logits fp32 upcast in DPO trainer and computes label log-probs as label_logit - logsumexp(logits), chunked by LLAMAFACTORY_LOGPS_CHUNK_SIZE.
The first working profile is the current default: cutoff_len=24576, image/video max pixels=131072, LLAMAFACTORY_LOGPS_CHUNK_SIZE=128.
If it still OOMs, stop and inspect the batch/media length distribution before making another global reduction. Do not mix different cutoff/image settings across A/B/C fold comparisons.
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LLAMAFACTORY_LOGPS_CHUNK_SIZE=128 CUDA_VISIBLE_DEVICES=4,5 FORCE_TORCHRUN=1 llamafactory-cli train \
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

Result record, 2026-05-10:
```text
Fold heldout protocol:
GRPO baseline fold acc: 0.90, 0.90, 0.85, 0.80; mean 0.8625.
DPO LoRA A fold acc:    0.95, 0.90, 0.90, 0.70; mean 0.8625.
DPO LoRA B fold acc:    0.95, 0.90, 0.90, 0.75; mean 0.8750.
DPO LoRA C fold acc:    0.95, 0.85, 0.90, 0.70; mean 0.8500.

Winner selected from fold validation: B
config: configs/egocross_dpo_lora_from_grpo_all_equal_wrong3_fmt1_lr5e6_beta003_ftx005_ep1.yaml
merged model: saves/egocross/qwen3vl4b/dpo_lora_from_grpo_all_equal_wrong3_fmt1_lr5e6_beta003_ftx005_ep1_memsafe_ctx24576_px131k_merged

Full support eval for B:
overall_acc: 0.862500 (69/80)
Animal: 0.850000 (17/20)
Industry: 0.900000 (18/20)
Surgery: 0.900000 (18/20)
XSports: 0.800000 (16/20)
answer_only_format_rate: 1.000000
coverage: 1.000000
parse_fail/error/error_fallback: 0/0/0
answer_dist: A=17, B=19, C=20, D=24

Fixed test evaluation for B, same inference strategy:
direct + tail_dense + max12 + VID006=4 + temperature=0
Overall: 0.526646
Surgery: 0.526502
Industry: 0.542857
XSports: 0.434959
Animal: 0.628415
coverage: 1.0
```

Conclusion:
```text
LoRA DPO from GRPO was a valid conservative challenger but did not beat the current GRPO best test record.
Do not promote DPO LoRA B as the main submission candidate.
Keep current best as:
/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212
direct + tail_dense + max12 + VID006=4 + temperature=0
egocross_outputs/why_grpo_direct_tail_dense_max12_vid006_4f

This fixed challenger result is recorded for audit/history only. Do not use this hidden result to tune prompt, router, frame sampling, beta, or checkpoint selection.
If training is revisited, use support/fold diagnostics first; prefer new data or stronger public validation rather than another small hidden-driven DPO sweep.
```
