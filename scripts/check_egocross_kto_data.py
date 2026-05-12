#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LETTERS = {"A", "B", "C", "D"}
DOMAIN_FILES = {
    "animal": "train_animal.json",
    "surgery": "train_surgery.json",
    "industry": "train_industry.json",
    "xsports": "train_xsports.json",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_letter(text):
    return re.fullmatch(r"\s*[ABCD]\s*", str(text or ""), re.I) is not None


def assistant_text(row):
    messages = row.get("messages") or []
    if not messages:
        return ""
    return str(messages[-1].get("content", "")).strip()


def prompt_text(row):
    messages = row.get("messages") or []
    return "\n".join(str(msg.get("content", "")) for msg in messages if msg.get("role") == "user")


def source_key(row, fallback_idx):
    return str(row.get("domain") or "").lower(), str(row.get("source_index") or fallback_idx)


def source_aug_key(row, fallback_idx):
    domain, source = source_key(row, fallback_idx)
    return domain, source, str(row.get("augmentation") or 0)


def resolve_image_path(data_dir, image_path):
    raw = str(image_path).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return path
    return data_dir / raw.lstrip("/")


def classify(row):
    text = assistant_text(row)
    gold = str(row.get("gold_answer") or "").strip().upper()
    if row.get("kto_tag") is True:
        if is_letter(text) and text.upper() == gold:
            return "positive_exact"
        return "bad_positive"
    if is_letter(text):
        return "negative_wrong_letter" if text.upper() != gold else "negative_same_letter"
    if gold in text.upper():
        return "negative_format"
    return "negative_invalid"


def check_rows(rows, data_dir, skip_image_check, expected_permutations, expected_format_negatives):
    errors = []
    warnings = []
    domain_counts = Counter()
    feedback_counts = Counter()
    gold_counts = Counter()
    text_counts = Counter()
    kto_counts = Counter()
    per_source_aug = defaultdict(Counter)
    source_aug_to_rows = defaultdict(list)

    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages") or []
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            errors.append(f"row {idx}: messages must end with assistant response")
        if row.get("kto_tag") not in (True, False):
            errors.append(f"row {idx}: kto_tag must be boolean")

        key = source_aug_key(row, idx)
        source_aug_to_rows[key].append(idx)
        domain_counts[key[0]] += 1
        kind = classify(row)
        feedback_counts[kind] += 1
        per_source_aug[key][kind] += 1

        gold = str(row.get("gold_answer") or "").strip().upper()
        if gold not in LETTERS:
            errors.append(f"row {idx}: gold_answer is not A/B/C/D: {gold!r}")
        gold_counts[gold] += 1
        text_counts[assistant_text(row).upper()] += 1
        kto_counts[str(row.get("kto_tag"))] += 1

        prompt = prompt_text(row)
        if not prompt:
            errors.append(f"row {idx}: empty prompt")
        for letter in sorted(LETTERS):
            if not re.search(rf"(?m)^\s*{letter}[.)：:]\s+", prompt):
                errors.append(f"row {idx}: missing visible option {letter}")
                break

        images = row.get("images") or []
        if not images:
            errors.append(f"row {idx}: no images")
        elif not skip_image_check:
            missing = [str(path) for path in images if not resolve_image_path(data_dir, path).exists()]
            if missing:
                errors.append(f"row {idx}: missing image path, first={missing[0]}")

    expected = {
        "positive_exact": 1,
        "negative_wrong_letter": 3,
        "negative_format": expected_format_negatives,
    }
    for key, counts in sorted(per_source_aug.items()):
        for label, value in expected.items():
            if counts[label] != value:
                errors.append(f"source_aug {key}: expected {label}={value}, got {counts[label]}")
        for bad in ("bad_positive", "negative_same_letter", "negative_invalid"):
            if counts[bad]:
                errors.append(f"source_aug {key}: invalid {bad} count={counts[bad]}")

    per_source = defaultdict(set)
    for domain, source, aug in source_aug_to_rows:
        per_source[(domain, source)].add(aug)
    for source, aug_set in sorted(per_source.items()):
        if len(aug_set) != expected_permutations:
            errors.append(f"source {source}: expected {expected_permutations} augmentations, got {len(aug_set)}")

    max_gold = max(gold_counts.values()) if gold_counts else 0
    min_gold = min(gold_counts.values()) if gold_counts else 0
    if min_gold > 0 and max_gold / min_gold > 1.35:
        warnings.append(f"gold answer distribution is imbalanced: {dict(sorted(gold_counts.items()))}")

    return {
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "rows": len(rows),
            "unique_sources": len(per_source),
            "source_augmentations": len(per_source_aug),
            "domain_counts": dict(sorted(domain_counts.items())),
            "feedback_counts": dict(sorted(feedback_counts.items())),
            "gold_answer_counts": dict(sorted(gold_counts.items())),
            "assistant_text_counts": dict(sorted(text_counts.items())),
            "kto_tag_counts": dict(sorted(kto_counts.items())),
        },
        "source_keys": set(per_source),
    }


def load_support_source_keys(data_dir):
    keys = set()
    for domain, file_name in DOMAIN_FILES.items():
        path = data_dir / file_name
        if not path.exists():
            continue
        rows = load_json(path)
        keys.update((domain, str(idx)) for idx in range(1, len(rows) + 1))
    return keys


def check_folds(fold_dir, data_dir, skip_image_check, expected_permutations, expected_format_negatives):
    errors = []
    warnings = []
    summaries = {}
    support_keys = load_support_source_keys(data_dir)
    for fold_id in range(4):
        train_path = fold_dir / f"train_kto_answer_reward_perm4_wrong3_fmt2_fold{fold_id}.json"
        eval_path = fold_dir / f"eval_answer_only_fold{fold_id}.json"
        if not train_path.exists():
            errors.append(f"missing fold train file: {train_path}")
            continue
        if not eval_path.exists():
            errors.append(f"missing fold eval file: {eval_path}")
            continue

        train_report = check_rows(
            load_json(train_path),
            data_dir,
            skip_image_check,
            expected_permutations,
            expected_format_negatives,
        )
        errors.extend(f"fold{fold_id} train: {msg}" for msg in train_report["errors"])
        warnings.extend(f"fold{fold_id} train: {msg}" for msg in train_report["warnings"])

        eval_rows = load_json(eval_path)
        eval_keys = set()
        eval_domain_counts = Counter()
        eval_answer_counts = Counter()
        for idx, row in enumerate(eval_rows, start=1):
            key = source_key(row, idx)
            eval_keys.add(key)
            eval_domain_counts[key[0]] += 1
            answer = str(row.get("answer") or "").strip().upper()
            if answer not in LETTERS:
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
        if support_keys:
            unknown_eval = eval_keys - support_keys
            if unknown_eval:
                warnings.append(f"fold{fold_id}: eval keys not found in support files: {sorted(unknown_eval)[:5]}")

        summaries[f"fold{fold_id}"] = {
            "train": train_report["summary"],
            "eval_sources": len(eval_rows),
            "eval_domain_counts": dict(sorted(eval_domain_counts.items())),
            "eval_answer_counts": dict(sorted(eval_answer_counts.items())),
            "leakage": len(leakage),
        }
    return errors, warnings, summaries


def main():
    parser = argparse.ArgumentParser(description="Validate EgoCross answer-only KTO data before training.")
    parser.add_argument("--data-dir", default="/share/home/group9/data/egocross")
    parser.add_argument(
        "--kto-file",
        default="/share/home/group9/data/egocross/train_kto_answer_reward_perm4_wrong3_fmt2_seed2026.json",
    )
    parser.add_argument(
        "--fold-dir",
        default="/share/home/group9/data/egocross/kto_answer_reward_perm4_wrong3_fmt2_folds",
    )
    parser.add_argument("--expected-permutations", type=int, default=4)
    parser.add_argument("--expected-format-negatives", type=int, default=2)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows = load_json(Path(args.kto_file))
    report = check_rows(
        rows,
        data_dir,
        args.skip_image_check,
        args.expected_permutations,
        args.expected_format_negatives,
    )
    fold_errors, fold_warnings, fold_summary = check_folds(
        Path(args.fold_dir),
        data_dir,
        args.skip_image_check,
        args.expected_permutations,
        args.expected_format_negatives,
    )
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
