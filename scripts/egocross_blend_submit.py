#!/usr/bin/env python3
import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


DOMAIN_BY_DATASET = {
    "CholecTrack20": "Surgery",
    "EgoSurgery": "Surgery",
    "ENIGMA": "Industry",
    "ExtrameSportFPV": "XSports",
    "EgoPet": "Animal",
}

VALID_ANSWERS = {"A", "B", "C", "D"}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def row_key(row, idx):
    if row.get("id") is not None:
        return ("id", str(row["id"]))
    if row.get("question_id"):
        return ("question_id", str(row["question_id"]))
    return ("idx", str(idx))


def parse_selectors(value):
    selectors = [item.strip() for item in value.split(",") if item.strip()]
    if not selectors:
        raise ValueError(f"Empty selector in override: {value}")
    return selectors


def selector_matches(dataset, selectors):
    domain = DOMAIN_BY_DATASET.get(dataset, dataset)
    names = {dataset.lower(), domain.lower()}
    return any(selector.lower() in names for selector in selectors)


def parse_override(value):
    if "=" not in value:
        raise ValueError(f"Override must use SELECTOR=FILE format: {value}")

    selector_text, file_text = value.split("=", 1)
    selectors = parse_selectors(selector_text)
    path = Path(file_text)
    if not path.exists():
        raise FileNotFoundError(path)
    return selectors, path


def index_rows(rows):
    return {row_key(row, idx): row for idx, row in enumerate(rows, start=1)}


def domain_for(row):
    return DOMAIN_BY_DATASET.get(row.get("dataset", ""), row.get("dataset", "UNKNOWN"))


def summarize(rows, reference_rows):
    answer_dist = Counter(row.get("answer", "") for row in rows)
    dataset_dist = Counter(row.get("dataset", "UNKNOWN") for row in rows)
    domain_dist = Counter(domain_for(row) for row in rows)
    invalid = [row for row in rows if row.get("answer") not in VALID_ANSWERS]
    empty = [row for row in rows if not row.get("answer")]

    changed_by_domain = Counter()
    changed_by_dataset = Counter()
    changed_total = 0
    for row, ref in zip(rows, reference_rows):
        if row.get("answer") != ref.get("answer"):
            changed_total += 1
            changed_by_domain[domain_for(row)] += 1
            changed_by_dataset[row.get("dataset", "UNKNOWN")] += 1

    lines = [
        f"num: {len(rows)}",
        f"empty: {len(empty)}",
        f"invalid: {len(invalid)}",
        "answer_dist: " + " ".join(f"{key}={answer_dist.get(key, 0)}" for key in ["A", "B", "C", "D"]),
        "dataset_dist: " + " ".join(f"{key}={value}" for key, value in sorted(dataset_dist.items())),
        "domain_dist: " + " ".join(f"{key}={value}" for key, value in sorted(domain_dist.items())),
        f"changed_vs_base_total: {changed_total}",
        "changed_vs_base_by_domain: "
        + " ".join(f"{key}={changed_by_domain.get(key, 0)}" for key in ["Surgery", "Industry", "XSports", "Animal"]),
        "changed_vs_base_by_dataset: "
        + " ".join(f"{key}={value}" for key, value in sorted(changed_by_dataset.items())),
    ]

    if invalid[:5]:
        lines.append("first_invalid: " + json.dumps(invalid[:5], ensure_ascii=False))

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Blend EgoCross submissions and build Codabench zip.")
    parser.add_argument("--base", required=True, help="Baseline submission used as the starting point.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override rows for a dataset or domain, e.g. XSports=path/to/submission.json.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zip-name", default="submission.zip")
    parser.add_argument("--expected-num", type=int, default=957)
    args = parser.parse_args()

    base_path = Path(args.base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_json(base_path)
    blended_rows = [dict(row) for row in base_rows]
    override_summaries = []

    for override_arg in args.override:
        selectors, override_path = parse_override(override_arg)
        override_rows = load_json(override_path)
        override_index = index_rows(override_rows)
        replaced = 0
        changed = 0
        by_domain = Counter()

        for idx, base_row in enumerate(base_rows, start=1):
            if not selector_matches(base_row.get("dataset", ""), selectors):
                continue

            key = row_key(base_row, idx)
            if key not in override_index:
                raise KeyError(f"Missing override row for {key} in {override_path}")

            new_row = dict(override_index[key])
            if new_row.get("answer") not in VALID_ANSWERS:
                raise ValueError(f"Invalid answer in override {override_path} for {key}: {new_row.get('answer')!r}")

            if new_row.get("answer") != blended_rows[idx - 1].get("answer"):
                changed += 1
                by_domain[domain_for(base_row)] += 1

            blended_rows[idx - 1] = new_row
            replaced += 1

        override_summaries.append(
            "override "
            + ",".join(selectors)
            + f" from {override_path}: replaced={replaced} changed={changed} "
            + " ".join(f"{key}={by_domain.get(key, 0)}" for key in ["Surgery", "Industry", "XSports", "Animal"])
        )

    if args.expected_num and len(blended_rows) != args.expected_num:
        raise SystemExit(f"Expected {args.expected_num} rows, got {len(blended_rows)}")

    invalid = [row for row in blended_rows if row.get("answer") not in VALID_ANSWERS]
    if invalid:
        raise SystemExit(f"Invalid answers found: {invalid[:5]}")

    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(blended_rows, f, indent=2, ensure_ascii=False)

    summary = "\n".join(override_summaries)
    if summary:
        summary += "\n"
    summary += summarize(blended_rows, base_rows)

    summary_path = output_dir / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    zip_path = output_dir / args.zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, arcname="predictions.json")

    print(summary, end="")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved zip to {zip_path}")


if __name__ == "__main__":
    main()
