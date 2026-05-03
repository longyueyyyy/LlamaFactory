#!/usr/bin/env python3
import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DOMAIN_FILES = {
    "animal": "train_animal.json",
    "surgery": "train_surgery.json",
    "industry": "train_industry.json",
    "xsports": "train_xsports.json",
}
LETTERS = ["A", "B", "C", "D"]
VALID_ANSWERS = set(LETTERS)


def extract_answer(text):
    upper = str(text or "").strip().upper()
    patterns = [
        r"(?:FINAL\s+)?ANSWER\s*[:：]\s*([ABCD])\b",
        r"^\s*([ABCD])\s*$",
        r"\b([ABCD])\b",
        r"([ABCD])",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract A/B/C/D answer from assistant content: {text!r}")


def split_prompt_and_answer(sample):
    messages = sample.get("messages") or []
    assistant_positions = [idx for idx, message in enumerate(messages) if message.get("role") == "assistant"]
    if not assistant_positions:
        raise ValueError(f"Sample has no assistant message: {sample!r}")

    assistant_idx = assistant_positions[-1]
    answer = extract_answer(messages[assistant_idx].get("content", ""))
    prompt_messages = copy.deepcopy(messages[:assistant_idx])
    if len(prompt_messages) % 2 != 1:
        raise ValueError(f"Expected prompt messages ending with user turn, got {len(prompt_messages)}: {sample!r}")

    return prompt_messages, answer


def make_assistant(content):
    return {"role": "assistant", "content": content}


def format_rejected(answer, style):
    if style == "because":
        return f"Answer: {answer} because the visual evidence supports this option."
    if style == "final_answer":
        return f"Final answer: {answer}"
    raise ValueError(f"Unknown format rejection style: {style}")


def indexed_row(row_entry, fallback_idx):
    if isinstance(row_entry, tuple):
        return row_entry
    return fallback_idx, row_entry


def build_pairs(rows, domain, wrong_per_sample, add_format_negative):
    output = []
    for fallback_idx, row_entry in enumerate(rows, start=1):
        source_idx, sample = indexed_row(row_entry, fallback_idx)
        prompt_messages, answer = split_prompt_and_answer(sample)
        wrong_answers = [letter for letter in LETTERS if letter != answer][:wrong_per_sample]

        for rejected in wrong_answers:
            item = {
                "messages": copy.deepcopy(prompt_messages),
                "chosen": make_assistant(answer),
                "rejected": make_assistant(rejected),
                "images": copy.deepcopy(sample.get("images") or []),
                "domain": domain,
                "source_index": source_idx,
                "rejection_type": "wrong_letter",
            }
            output.append(item)

        if add_format_negative:
            item = {
                "messages": copy.deepcopy(prompt_messages),
                "chosen": make_assistant(answer),
                "rejected": make_assistant(format_rejected(answer, "because")),
                "images": copy.deepcopy(sample.get("images") or []),
                "domain": domain,
                "source_index": source_idx,
                "rejection_type": "format_long_explanation",
            }
            output.append(item)

    return output


def load_domain_rows(data_dir):
    by_domain = {}
    for domain, file_name in DOMAIN_FILES.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing EgoCross support file: {path}")
        with open(path, encoding="utf-8") as f:
            by_domain[domain] = json.load(f)
    return by_domain


def summarize(rows):
    domain_counts = Counter(row["domain"] for row in rows)
    rejection_counts = Counter(row["rejection_type"] for row in rows)
    chosen_counts = Counter(row["chosen"]["content"] for row in rows)
    source_keys = {(row["domain"], row["source_index"]) for row in rows}
    return {
        "pairs": len(rows),
        "unique_sources": len(source_keys),
        "domain_counts": dict(sorted(domain_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "chosen_counts": dict(sorted(chosen_counts.items())),
    }


def write_json(path, rows, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def build_fold_rows(by_domain, fold_id, num_folds, wrong_per_sample, add_format_negative):
    train_rows = []
    eval_rows = []
    fold_summary = {}

    for domain, rows in by_domain.items():
        held_out = []
        train = []
        for idx, row in enumerate(rows, start=1):
            if idx % num_folds == fold_id:
                held_out.append((idx, row))
            else:
                train.append((idx, row))

        train_rows.extend(build_pairs(train, domain, wrong_per_sample, add_format_negative))
        for source_idx, sample in held_out:
            prompt_messages, answer = split_prompt_and_answer(sample)
            eval_rows.append({
                "messages": prompt_messages,
                "images": copy.deepcopy(sample.get("images") or []),
                "domain": domain,
                "answer": answer,
                "fold": fold_id,
                "source_index": source_idx,
            })

        fold_summary[domain] = {"train_sources": len(train), "eval_sources": len(held_out)}

    return train_rows, eval_rows, fold_summary


def main():
    parser = argparse.ArgumentParser(description="Build all-domain answer-only EgoCross DPO preference data.")
    parser.add_argument("--data-dir", default="/share/home/group9/data/egocross")
    parser.add_argument(
        "--output",
        default="/share/home/group9/data/egocross/train_pref_answer_only_all_equal_wrong3_fmt1.json",
    )
    parser.add_argument("--wrong-per-sample", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--format-negatives", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fold-output-dir",
        default=None,
        help="Optional directory for 4-fold support validation train/eval JSON files.",
    )
    parser.add_argument("--num-folds", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    by_domain = load_domain_rows(data_dir)

    rows = []
    source_counts = {}
    for domain, domain_rows in by_domain.items():
        source_counts[domain] = len(domain_rows)
        rows.extend(build_pairs(domain_rows, domain, args.wrong_per_sample, bool(args.format_negatives)))

    random.Random(args.seed).shuffle(rows)
    write_json(output_path, rows, args.overwrite)

    print(f"Saved {len(rows)} preference pairs to {output_path}")
    print("source_counts:", dict(sorted(source_counts.items())))
    print("summary:", json.dumps(summarize(rows), ensure_ascii=False, sort_keys=True))

    if args.fold_output_dir:
        fold_dir = Path(args.fold_output_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        all_fold_summaries = {}
        for fold_id in range(args.num_folds):
            train_rows, eval_rows, fold_summary = build_fold_rows(
                by_domain,
                fold_id,
                args.num_folds,
                args.wrong_per_sample,
                bool(args.format_negatives),
            )
            random.Random(args.seed + fold_id).shuffle(train_rows)
            train_path = fold_dir / f"train_pref_answer_only_all_equal_wrong3_fmt1_fold{fold_id}.json"
            eval_path = fold_dir / f"eval_answer_only_all_equal_fold{fold_id}.json"
            write_json(train_path, train_rows, args.overwrite)
            write_json(eval_path, eval_rows, args.overwrite)
            all_fold_summaries[f"fold{fold_id}"] = {
                "train": summarize(train_rows),
                "eval_sources": len(eval_rows),
                "by_domain": fold_summary,
            }

        summary_path = fold_dir / "fold_summary.json"
        write_json(summary_path, all_fold_summaries, args.overwrite)
        print(f"Saved {args.num_folds}-fold validation data to {fold_dir}")


if __name__ == "__main__":
    main()
