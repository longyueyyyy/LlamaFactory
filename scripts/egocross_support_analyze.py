#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import egocross_support_eval as support_eval


def safe_div(num, den):
    return num / den if den else 0.0


def video_id_from_images(images):
    if not images:
        return "unknown"

    raw = str(images[0]).replace("\\", "/").strip("/")
    parts = raw.split("/")
    if "frames" in parts:
        idx = parts.index("frames")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}/{parts[idx + 2]}"

    for dataset in support_eval.router.DOMAIN_BY_DATASET:
        if dataset in parts:
            idx = parts.index(dataset)
            if idx + 1 < len(parts):
                return f"{dataset}/{parts[idx + 1]}"

    return "/".join(parts[:2]) if len(parts) >= 2 else raw or "unknown"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_sample_index(support_dir, eval_json):
    samples = []
    warnings = []
    if eval_json:
        path = Path(eval_json)
        if path.exists():
            samples = support_eval.load_eval_samples(path)
        else:
            warnings.append(f"eval_json not found: {path}")
    else:
        path = Path(support_dir)
        if path.exists():
            samples = support_eval.load_support_samples(path)
        else:
            warnings.append(f"support_dir not found, video metadata disabled: {path}")

    index = {}
    for sample in samples:
        meta = {
            "video_id": video_id_from_images(sample.get("video_path", [])),
            "frame_count": len(sample.get("video_path", [])),
            "question_text": sample.get("question_text", ""),
        }
        for key in (sample.get("id"), sample.get("question_id")):
            if key:
                index[str(key)] = meta
    return index, warnings


def enrich_rows(rows, sample_index):
    enriched = []
    for row in rows:
        item = dict(row)
        meta = sample_index.get(str(item.get("id"))) or sample_index.get(str(item.get("question_id"))) or {}
        if "video_id" not in item:
            if meta.get("video_id"):
                item["video_id"] = meta["video_id"]
            elif item.get("video_path"):
                item["video_id"] = video_id_from_images(item.get("video_path", []))
            elif item.get("images"):
                item["video_id"] = video_id_from_images(item.get("images", []))
            else:
                item["video_id"] = "unknown"
        if "frame_count" not in item:
            if "frame_count" in meta:
                item["frame_count"] = meta["frame_count"]
            elif item.get("video_path"):
                item["frame_count"] = len(item.get("video_path", []))
            elif item.get("images"):
                item["frame_count"] = len(item.get("images", []))
            else:
                item["frame_count"] = None
        if "question_text" not in item and meta.get("question_text"):
            item["question_text"] = meta["question_text"]
        enriched.append(item)
    return enriched


def add_bucket(bucket, key, correct):
    item = bucket[str(key or "unknown")]
    item["total"] += 1
    item["correct"] += int(bool(correct))


def finalize_bucket(bucket):
    return {
        key: {
            "correct": value["correct"],
            "total": value["total"],
            "acc": round(safe_div(value["correct"], value["total"]), 6),
        }
        for key, value in sorted(bucket.items())
    }


def build_analysis(rows, metrics, warnings):
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("correct")))
    valid_answers = sum(1 for row in rows if row.get("answer") in support_eval.router.VALID_ANSWERS)
    answer_only = sum(1 for row in rows if row.get("answer_only_format"))

    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    by_question_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_video = defaultdict(lambda: {"correct": 0, "total": 0})
    by_video_type = defaultdict(lambda: {"correct": 0, "total": 0})
    error_by_domain_type = Counter()
    error_by_video = Counter()
    status_dist = Counter()
    used_frames = Counter()
    answer_dist = Counter()
    gold_dist = Counter()

    errors = []
    for row in rows:
        is_correct = bool(row.get("correct"))
        domain = row.get("domain") or "unknown"
        question_type = row.get("question_type") or "unknown"
        video_id = row.get("video_id") or "unknown"
        add_bucket(by_domain, domain, is_correct)
        add_bucket(by_question_type, question_type, is_correct)
        add_bucket(by_video, video_id, is_correct)
        add_bucket(by_video_type, f"{video_id}\t{question_type}", is_correct)
        status_dist[str(row.get("status") or "unknown")] += 1
        used_frames[str(row.get("used_max_frames") or "")] += 1
        answer_dist[str(row.get("answer") or "")] += 1
        gold_dist[str(row.get("gold_answer") or "")] += 1
        if not is_correct:
            error_by_domain_type[f"{domain}\t{question_type}"] += 1
            error_by_video[video_id] += 1
            errors.append(
                {
                    "id": row.get("id"),
                    "question_id": row.get("question_id"),
                    "domain": domain,
                    "dataset": row.get("dataset"),
                    "video_id": video_id,
                    "question_type": question_type,
                    "gold_answer": row.get("gold_answer"),
                    "answer": row.get("answer"),
                    "status": row.get("status"),
                    "used_max_frames": row.get("used_max_frames"),
                    "frame_count": row.get("frame_count"),
                }
            )

    metric_overall = (metrics or {}).get("overall", {})
    overall = {
        "correct": metric_overall.get("correct", correct),
        "total": metric_overall.get("total", total),
        "acc": metric_overall.get("acc", round(safe_div(correct, total), 6)),
        "answer_only_format_rate": metric_overall.get(
            "answer_only_format_rate", round(safe_div(answer_only, total), 6)
        ),
        "coverage": metric_overall.get("coverage", round(safe_div(valid_answers, total), 6)),
        "parse_fail": metric_overall.get("parse_fail", total - valid_answers),
        "error": metric_overall.get("error", sum(1 for row in rows if "error" in str(row.get("status", "")))),
        "error_fallback": metric_overall.get(
            "error_fallback", sum(1 for row in rows if "error_fallback" in str(row.get("status", "")))
        ),
        "avg_used_frames": metric_overall.get("avg_used_frames"),
        "runtime_seconds": metric_overall.get("runtime_seconds"),
    }

    return {
        "warnings": warnings,
        "strategy": (metrics or {}).get("strategy", {}),
        "overall": overall,
        "by_domain": finalize_bucket(by_domain),
        "by_question_type": finalize_bucket(by_question_type),
        "by_video": finalize_bucket(by_video),
        "by_video_question_type": finalize_bucket(by_video_type),
        "status_dist": dict(sorted(status_dist.items())),
        "used_max_frames_dist": dict(sorted(used_frames.items())),
        "answer_dist": dict(sorted(answer_dist.items())),
        "gold_dist": dict(sorted(gold_dist.items())),
        "error_by_domain_question_type": dict(sorted(error_by_domain_type.items())),
        "error_by_video": dict(sorted(error_by_video.items())),
        "errors": sorted(errors, key=lambda row: (row["domain"], row["video_id"], row["question_type"], str(row["id"]))),
    }


def bucket_lines(title, bucket):
    lines = [f"{title}:"]
    for key, value in bucket.items():
        lines.append(f"  {key}: {value['acc']:.6f} ({value['correct']}/{value['total']})")
    return lines


def format_text(analysis, max_errors):
    overall = analysis["overall"]
    lines = []
    for warning in analysis.get("warnings", []):
        lines.append(f"warning: {warning}")
    lines.append(f"overall_acc: {overall['acc']:.6f} ({overall['correct']}/{overall['total']})")
    lines.append(f"answer_only_format_rate: {overall['answer_only_format_rate']:.6f}")
    lines.append(f"coverage: {overall['coverage']:.6f}")
    lines.append(f"parse_fail: {overall['parse_fail']}")
    lines.append(f"error: {overall['error']}")
    lines.append(f"error_fallback: {overall['error_fallback']}")
    if overall.get("avg_used_frames") is not None:
        lines.append(f"avg_used_frames: {overall['avg_used_frames']}")
    if overall.get("runtime_seconds") is not None:
        lines.append(f"runtime_seconds: {overall['runtime_seconds']}")
    strategy = analysis.get("strategy") or {}
    if strategy:
        lines.append("strategy: " + " ".join(f"{key}={value}" for key, value in strategy.items() if value not in [None, [], ""]))
    lines.extend(bucket_lines("by_domain", analysis["by_domain"]))
    lines.extend(bucket_lines("by_question_type", analysis["by_question_type"]))
    lines.extend(bucket_lines("by_video", analysis["by_video"]))
    lines.append("status_dist: " + json.dumps(analysis["status_dist"], ensure_ascii=False, sort_keys=True))
    lines.append("used_max_frames_dist: " + json.dumps(analysis["used_max_frames_dist"], ensure_ascii=False, sort_keys=True))
    lines.append("answer_dist: " + json.dumps(analysis["answer_dist"], ensure_ascii=False, sort_keys=True))
    lines.append("gold_dist: " + json.dumps(analysis["gold_dist"], ensure_ascii=False, sort_keys=True))
    lines.append("error_by_domain_question_type: " + json.dumps(analysis["error_by_domain_question_type"], ensure_ascii=False, sort_keys=True))
    lines.append("error_by_video: " + json.dumps(analysis["error_by_video"], ensure_ascii=False, sort_keys=True))
    lines.append("errors:")
    for row in analysis["errors"][:max_errors]:
        lines.append(
            "  {id} {domain} {video_id} {question_type} gold={gold_answer} pred={answer} "
            "status={status} used={used_max_frames}".format(**row)
        )
    if len(analysis["errors"]) > max_errors:
        lines.append(f"  ... {len(analysis['errors']) - max_errors} more")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze EgoCross support/fold predictions without hidden-label feedback.")
    parser.add_argument("--predictions", required=True, help="support_predictions.json from scripts/egocross_support_eval.py")
    parser.add_argument("--metrics", default=None, help="support_metrics.json. Defaults to sibling of --predictions.")
    parser.add_argument("--support-dir", default="/share/home/group9/data/egocross")
    parser.add_argument("--eval-json", default=None, help="Fold eval JSON used for the predictions, if any.")
    parser.add_argument("--output-dir", default=None, help="Directory for support_analysis.json/txt. Defaults to predictions parent.")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--text-output", default=None)
    parser.add_argument("--max-errors", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_path = Path(args.predictions)
    metrics_path = Path(args.metrics) if args.metrics else predictions_path.with_name("support_metrics.json")
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_json(predictions_path)
    metrics = load_json(metrics_path) if metrics_path.exists() else None
    sample_index, warnings = build_sample_index(args.support_dir, args.eval_json)
    analysis = build_analysis(enrich_rows(rows, sample_index), metrics, warnings)
    text = format_text(analysis, args.max_errors)

    json_output = Path(args.json_output) if args.json_output else output_dir / "support_analysis.json"
    text_output = Path(args.text_output) if args.text_output else output_dir / "support_analysis.txt"
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    with open(text_output, "w", encoding="utf-8") as f:
        f.write(text)

    print(text, end="")
    print(f"Saved support analysis to {output_dir}")


if __name__ == "__main__":
    main()
