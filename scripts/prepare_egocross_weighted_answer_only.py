#!/usr/bin/env python3
import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_DOMAIN_FILES = {
    "animal": "train_animal.json",
    "surgery": "train_surgery.json",
    "industry": "train_industry.json",
    "xsports": "train_xsports.json",
}

DEFAULT_WEIGHTS = {
    "animal": 1,
    "surgery": 1,
    "industry": 4,
    "xsports": 4,
}


def extract_answer(text):
    upper = str(text).upper()
    patterns = [
        r"FINAL\s+ANSWER\s*:\s*([ABCD])",
        r"ANSWER\s*:\s*([ABCD])",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    match = re.search(r"([ABCD])", upper)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract answer from assistant content: {text!r}")


def make_answer_only(sample, domain):
    item = copy.deepcopy(sample)
    assistant_seen = False
    for message in item.get("messages", []):
        if message.get("role") == "assistant":
            message["content"] = extract_answer(message.get("content", ""))
            assistant_seen = True

    if not assistant_seen:
        raise ValueError(f"Sample has no assistant message: {sample!r}")

    item["domain"] = domain
    return item


def parse_weights(value):
    weights = dict(DEFAULT_WEIGHTS)
    if not value:
        return weights

    for part in value.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Weight must use domain=value format: {part}")
        domain, raw_weight = part.split("=", 1)
        domain = domain.strip().lower()
        if domain not in weights:
            raise ValueError(f"Unknown domain: {domain}")
        weights[domain] = int(raw_weight)

    return weights


def main():
    parser = argparse.ArgumentParser(description="Build weighted answer-only EgoCross SFT data.")
    parser.add_argument("--data-dir", default="/share/home/group9/data/egocross")
    parser.add_argument("--output", default="/share/home/group9/data/egocross/train_weighted_answer_only_i4_x4.json")
    parser.add_argument("--weights", default=None, help="Comma-separated weights, e.g. animal=1,surgery=1,industry=4,xsports=4")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    weights = parse_weights(args.weights)

    weighted = []
    source_counts = Counter()
    output_counts = Counter()

    for domain, file_name in DEFAULT_DOMAIN_FILES.items():
        path = data_dir / file_name
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)

        source_counts[domain] = len(rows)
        answer_only_rows = [make_answer_only(row, domain) for row in rows]
        for _ in range(weights[domain]):
            weighted.extend(copy.deepcopy(answer_only_rows))
            output_counts[domain] += len(answer_only_rows)

    random.Random(args.seed).shuffle(weighted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weighted, f, indent=2, ensure_ascii=False)

    answer_counts = Counter()
    for row in weighted:
        for message in row.get("messages", []):
            if message.get("role") == "assistant":
                answer_counts[message.get("content", "")] += 1

    print(f"Saved {len(weighted)} samples to {output_path}")
    print("source_counts:", dict(source_counts))
    print("weights:", weights)
    print("output_counts:", dict(output_counts))
    print("answer_counts:", dict(sorted(answer_counts.items())))


if __name__ == "__main__":
    main()
