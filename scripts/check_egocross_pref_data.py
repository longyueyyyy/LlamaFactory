#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LETTERS = {"A", "B", "C", "D"}
OPTION_RE = re.compile(r"(?<![A-Za-z0-9])([ABCD])\s*[:\).]\s*", re.I)
DOMAIN_FILES = {
    "animal": "train_animal.json",
    "surgery": "train_surgery.json",
    "industry": "train_industry.json",
    "xsports": "train_xsports.json",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_message_content(message):
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(message or "")


def answer_content(value):
    if isinstance(value, dict):
        return str(value.get("content", "")).strip()
    return str(value or "").strip()


def is_letter(text):
    return text.strip().upper() in LETTERS and re.fullmatch(r"\s*[ABCD]\s*", text.strip(), re.I) is not None


def is_format_negative(text, chosen):
    clean = str(text or "").strip()
    upper = clean.upper()
    if is_letter(clean):
        return False
    if chosen.upper() not in upper:
        return False
    return bool(re.search(r"\b(ANSWER|FINAL\s+ANSWER|BECAUSE|EVIDENCE|SUPPORTS)\b", upper)) or len(clean.split()) > 2


def prompt_text(row):
    messages = row.get("messages") or []
    user_messages = [extract_message_content(msg) for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"]
    return "\n".join(user_messages).strip()


def option_letters(text):
    return [match.group(1).upper() for match in OPTION_RE.finditer(text or "")]


def infer_question_type(text):
    clean = str(text or "").lower()
    if "how many" in clean or "number of" in clean or "distinct types" in clean:
        return "counting"
    if "not visible" in clean or "not shown" in clean or "not seen" in clean:
        return "not_visible"
    if "which region" in clean or "located" in clean or re.search(r"\bwhere\b", clean):
        return "region_localization"
    if "approximate time" in clean or "timestamp" in clean or "first interact" in clean:
        return "temporal_localization"
    if "next direction" in clean or "direction of movement" in clean:
        return "next_direction"
    if "predicted next" in clean or "next type of interaction" in clean:
        return "next_interaction"
    if "immediately follows" in clean or "next phase" in clean or "key action that will begin" in clean:
        return "phase_sequence"
    if "sequence of actions" in clean:
        return "action_sequence"
    if "what action is being performed" in clean:
        return "action_identification"
    if "extreme sport" in clean:
        return "sport_identification"
    if "type of animal" in clean:
        return "animal_identification"
    if "object is the cat interacting" in clean:
        return "object_interaction"
    if "which tool" in clean or "surgical instrument" in clean:
        return "tool_interaction"
    return "unknown"


def infer_dataset_video(images):
    if not images:
        return "unknown", "unknown"

    parts = str(images[0]).replace("\\", "/").split("/")
    if "frames" in parts:
        idx = parts.index("frames")
        dataset = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        video = parts[idx + 2] if idx + 2 < len(parts) else "unknown"
        return dataset, video

    for idx, part in enumerate(parts):
        if part.startswith("VID") or part.startswith("video"):
            dataset = parts[idx - 1] if idx > 0 else "unknown"
            return dataset, part

    return "unknown", "unknown"


def source_key(row, fallback_idx):
    domain = str(row.get("domain") or "").strip().lower()
    source_index = row.get("source_index")
    if source_index is None:
        source_index = row.get("question_id") or row.get("id") or fallback_idx
    return domain, str(source_index)


def resolve_image_path(data_dir, image_path):
    raw = str(image_path).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return path
    return data_dir / raw.lstrip("/")


def classify_pair(row):
    chosen = answer_content(row.get("chosen")).upper()
    rejected = answer_content(row.get("rejected")).strip()
    rejected_upper = rejected.upper()
    if is_letter(rejected):
        return "wrong_letter" if rejected_upper != chosen else "same_letter"
    if is_format_negative(rejected, chosen):
        return "format_negative"
    return "invalid_rejected"


def check_pref_rows(rows, data_dir, skip_image_check):
    errors = []
    warnings = []
    domain_counts = Counter()
    dataset_counts = Counter()
    video_counts = Counter()
    chosen_counts = Counter()
    rejection_counts = Counter()
    per_source = defaultdict(Counter)
    seen_pairs = Counter()
    question_type_counts = Counter()

    for idx, row in enumerate(rows, start=1):
        domain, source_index = source_key(row, idx)
        if not domain:
            errors.append(f"row {idx}: missing domain")
        domain_counts[domain] += 1
        source = (domain, source_index)

        prompt = prompt_text(row)
        if not prompt:
            errors.append(f"row {idx}: empty prompt")
        options = option_letters(prompt)
        if len(options) < 4:
            errors.append(f"row {idx}: prompt has fewer than four A/B/C/D options")

        chosen = answer_content(row.get("chosen")).upper()
        rejected = answer_content(row.get("rejected")).strip()
        if not is_letter(chosen):
            errors.append(f"row {idx}: chosen is not a single A/B/C/D letter: {chosen!r}")
        if chosen == rejected.upper():
            errors.append(f"row {idx}: chosen equals rejected for {source}")

        kind = classify_pair(row)
        if kind == "same_letter":
            errors.append(f"row {idx}: rejected is the same letter as chosen")
        elif kind == "invalid_rejected":
            errors.append(f"row {idx}: rejected is neither wrong A/B/C/D nor format negative: {rejected[:80]!r}")

        chosen_counts[chosen] += 1
        rejection_counts[kind] += 1
        per_source[source][kind] += 1
        question_type_counts[str(row.get("question_type") or infer_question_type(prompt))] += 1

        images = row.get("images") or []
        dataset, video = infer_dataset_video(images)
        dataset_counts[dataset] += 1
        video_counts[f"{dataset}/{video}"] += 1
        if not images:
            errors.append(f"row {idx}: no images")
        elif not skip_image_check:
            missing = [str(path) for path in images if not resolve_image_path(data_dir, path).exists()]
            if missing:
                errors.append(f"row {idx}: missing image path, first={missing[0]}")

        pair_key = (
            tuple(extract_message_content(msg) for msg in row.get("messages") or []),
            tuple(str(item) for item in images),
            chosen,
            rejected,
        )
        seen_pairs[pair_key] += 1

    for source, counts in sorted(per_source.items()):
        if counts["wrong_letter"] != 3 or counts["format_negative"] != 1:
            errors.append(
                f"source {source}: expected wrong3+fmt1, got wrong={counts['wrong_letter']} "
                f"fmt={counts['format_negative']} invalid={counts['invalid_rejected']}"
            )

    duplicate_pairs = sum(count - 1 for count in seen_pairs.values() if count > 1)
    if duplicate_pairs:
        warnings.append(f"duplicate exact preference pairs: {duplicate_pairs}")

    return {
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "pairs": len(rows),
            "unique_sources": len(per_source),
            "domain_counts": dict(sorted(domain_counts.items())),
            "dataset_counts": dict(sorted(dataset_counts.items())),
            "video_counts": dict(sorted(video_counts.items())),
            "chosen_counts": dict(sorted(chosen_counts.items())),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "question_type_counts": dict(sorted(question_type_counts.items())),
            "duplicate_exact_pairs": duplicate_pairs,
        },
        "source_keys": set(per_source),
    }


def load_support_source_keys(data_dir):
    keys_by_domain = {}
    for domain, file_name in DOMAIN_FILES.items():
        path = data_dir / file_name
        if not path.exists():
            continue
        rows = load_json(path)
        keys_by_domain[domain] = {(domain, str(idx)) for idx in range(1, len(rows) + 1)}
    return keys_by_domain


def eval_source_key(row, fallback_idx):
    domain = str(row.get("domain") or "").strip().lower()
    source_index = row.get("source_index")
    if source_index is None:
        source_index = row.get("question_id") or row.get("id") or fallback_idx
    return domain, str(source_index)


def check_folds(fold_dir, data_dir, skip_image_check):
    errors = []
    warnings = []
    summaries = {}
    support_keys = load_support_source_keys(data_dir)

    for fold_id in range(4):
        train_path = fold_dir / f"train_pref_answer_only_all_equal_wrong3_fmt1_fold{fold_id}.json"
        eval_path = fold_dir / f"eval_answer_only_all_equal_fold{fold_id}.json"
        if not train_path.exists():
            errors.append(f"missing fold train file: {train_path}")
            continue
        if not eval_path.exists():
            errors.append(f"missing fold eval file: {eval_path}")
            continue

        train_rows = load_json(train_path)
        eval_rows = load_json(eval_path)
        train_report = check_pref_rows(train_rows, data_dir, skip_image_check)
        errors.extend(f"fold{fold_id} train: {msg}" for msg in train_report["errors"])
        warnings.extend(f"fold{fold_id} train: {msg}" for msg in train_report["warnings"])

        eval_keys = set()
        eval_domains = Counter()
        eval_answer_counts = Counter()
        for idx, row in enumerate(eval_rows, start=1):
            key = eval_source_key(row, idx)
            eval_keys.add(key)
            eval_domains[key[0]] += 1
            answer = str(row.get("answer") or "").strip().upper()
            if not is_letter(answer):
                errors.append(f"fold{fold_id} eval row {idx}: answer is not A/B/C/D: {answer!r}")
            eval_answer_counts[answer] += 1
            if not prompt_text(row):
                errors.append(f"fold{fold_id} eval row {idx}: empty prompt")
            images = row.get("images") or []
            if not images:
                errors.append(f"fold{fold_id} eval row {idx}: no images")
            elif not skip_image_check:
                missing = [str(path) for path in images if not resolve_image_path(data_dir, path).exists()]
                if missing:
                    errors.append(f"fold{fold_id} eval row {idx}: missing image path, first={missing[0]}")

        leakage = train_report["source_keys"] & eval_keys
        if leakage:
            errors.append(f"fold{fold_id}: train/eval source leakage: {sorted(leakage)[:5]}")

        known_support = set().union(*support_keys.values()) if support_keys else set()
        if known_support:
            unknown_eval = eval_keys - known_support
            if unknown_eval:
                warnings.append(f"fold{fold_id}: eval keys not found in support files: {sorted(unknown_eval)[:5]}")

        summaries[f"fold{fold_id}"] = {
            "train": train_report["summary"],
            "eval_sources": len(eval_rows),
            "eval_domain_counts": dict(sorted(eval_domains.items())),
            "eval_answer_counts": dict(sorted(eval_answer_counts.items())),
            "leakage": len(leakage),
        }

    return errors, warnings, summaries


def main():
    parser = argparse.ArgumentParser(description="Validate EgoCross answer-only DPO preference data before training.")
    parser.add_argument("--data-dir", default="/share/home/group9/data/egocross")
    parser.add_argument(
        "--pref-file",
        default="/share/home/group9/data/egocross/train_pref_answer_only_all_equal_wrong3_fmt1.json",
    )
    parser.add_argument("--fold-dir", default="/share/home/group9/data/egocross/pref_answer_only_all_equal_folds")
    parser.add_argument("--skip-image-check", action="store_true", help="Skip image existence checks for local mirrors.")
    parser.add_argument("--output-json", default=None, help="Optional path for a machine-readable report.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pref_path = Path(args.pref_file)
    fold_dir = Path(args.fold_dir)

    rows = load_json(pref_path)
    report = check_pref_rows(rows, data_dir, args.skip_image_check)
    fold_errors, fold_warnings, fold_summary = check_folds(fold_dir, data_dir, args.skip_image_check)
    report["errors"].extend(fold_errors)
    report["warnings"].extend(fold_warnings)
    report["folds"] = fold_summary
    report["ok"] = not report["errors"]
    report.pop("source_keys", None)

    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)

    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
