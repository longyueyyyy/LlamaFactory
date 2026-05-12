#!/usr/bin/env python3
import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path


DOMAIN_FILES = {
    "animal": "train_animal.json",
    "surgery": "train_surgery.json",
    "industry": "train_industry.json",
    "xsports": "train_xsports.json",
}
LETTERS = ["A", "B", "C", "D"]
VALID_ANSWERS = set(LETTERS)
OPTION_LINE_RE = re.compile(r"^(?P<lead>\s*)(?P<label>[ABCD])(?P<sep>[.)：:])\s*(?P<text>.*)$", re.I)


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


def parse_options(prompt):
    lines = prompt.splitlines()
    option_by_label = {}
    line_indices = {}
    for idx, line in enumerate(lines):
        match = OPTION_LINE_RE.match(line)
        if not match:
            continue
        label = match.group("label").upper()
        option_by_label[label] = match.group("text")
        line_indices[label] = idx

    missing = [letter for letter in LETTERS if letter not in option_by_label]
    if missing:
        raise ValueError(f"Prompt is missing option lines for {missing}: {prompt[:240]!r}")
    return lines, option_by_label, line_indices


def permute_prompt(prompt, answer, permutation):
    if permutation == LETTERS:
        return prompt, answer

    lines, option_by_label, line_indices = parse_options(prompt)
    for new_idx, old_label in enumerate(permutation):
        new_label = LETTERS[new_idx]
        line_idx = line_indices[new_label]
        lines[line_idx] = f"{new_label}. {option_by_label[old_label]}"

    new_answer = LETTERS[permutation.index(answer)]
    return "\n".join(lines), new_answer


def make_permutations(rng, count):
    if count < 1 or count > 24:
        raise ValueError("--permutations must be in [1, 24]")

    output = [LETTERS[:]]
    seen = {tuple(LETTERS)}
    while len(output) < count:
        candidate = LETTERS[:]
        rng.shuffle(candidate)
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def make_messages(prompt_messages, assistant_text):
    messages = copy.deepcopy(prompt_messages)
    messages.append({"role": "assistant", "content": assistant_text})
    return messages


def make_row(prompt_messages, assistant_text, kto_tag, sample, domain, source_index, augmentation, permutation, gold, feedback_type):
    return {
        "messages": make_messages(prompt_messages, assistant_text),
        "kto_tag": bool(kto_tag),
        "images": copy.deepcopy(sample.get("images") or []),
        "domain": domain,
        "source_index": source_index,
        "augmentation": augmentation,
        "permutation": "".join(permutation),
        "gold_answer": gold,
        "assistant_text": assistant_text,
        "feedback_type": feedback_type,
    }


def build_rows(indexed_samples, domain, permutations, format_negatives):
    rows = []
    for fallback_idx, sample in indexed_samples:
        prompt_messages, answer = split_prompt_and_answer(sample)
        user_message = prompt_messages[-1]
        original_prompt = user_message.get("content", "")

        for aug_idx, permutation in enumerate(permutations):
            aug_messages = copy.deepcopy(prompt_messages)
            permuted_prompt, gold = permute_prompt(original_prompt, answer, permutation)
            aug_messages[-1]["content"] = permuted_prompt

            rows.append(
                make_row(
                    aug_messages,
                    gold,
                    True,
                    sample,
                    domain,
                    fallback_idx,
                    aug_idx,
                    permutation,
                    gold,
                    "exact_correct_letter",
                )
            )

            for wrong in [letter for letter in LETTERS if letter != gold]:
                rows.append(
                    make_row(
                        aug_messages,
                        wrong,
                        False,
                        sample,
                        domain,
                        fallback_idx,
                        aug_idx,
                        permutation,
                        gold,
                        "wrong_letter",
                    )
                )

            negative_templates = [
                "Final answer: {answer}",
                "Answer: {answer} because the visual evidence supports this option.",
            ]
            for template in negative_templates[:format_negatives]:
                rows.append(
                    make_row(
                        aug_messages,
                        template.format(answer=gold),
                        False,
                        sample,
                        domain,
                        fallback_idx,
                        aug_idx,
                        permutation,
                        gold,
                        "format_extra_text",
                    )
                )

    return rows


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
    return {
        "rows": len(rows),
        "unique_sources": len({(row["domain"], row["source_index"]) for row in rows}),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "feedback_counts": dict(sorted(Counter(row["feedback_type"] for row in rows).items())),
        "gold_answer_counts": dict(sorted(Counter(row["gold_answer"] for row in rows).items())),
        "kto_tag_counts": dict(sorted(Counter(str(row["kto_tag"]) for row in rows).items())),
    }


def write_json(path, rows, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def indexed(rows):
    return list(enumerate(rows, start=1))


def build_fold_rows(by_domain, fold_id, num_folds, permutations, format_negatives):
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

        train_rows.extend(build_rows(train, domain, permutations, format_negatives))
        for source_idx, sample in held_out:
            prompt_messages, answer = split_prompt_and_answer(sample)
            eval_rows.append(
                {
                    "messages": prompt_messages,
                    "images": copy.deepcopy(sample.get("images") or []),
                    "domain": domain,
                    "answer": answer,
                    "fold": fold_id,
                    "source_index": source_idx,
                }
            )

        fold_summary[domain] = {"train_sources": len(train), "eval_sources": len(held_out)}
    return train_rows, eval_rows, fold_summary


def main():
    parser = argparse.ArgumentParser(description="Build EgoCross answer-only KTO data from support labels.")
    parser.add_argument("--data-dir", default="/share/home/group9/data/egocross")
    parser.add_argument(
        "--output",
        default="/share/home/group9/data/egocross/train_kto_answer_reward_perm4_wrong3_fmt2_seed2026.json",
    )
    parser.add_argument("--permutations", type=int, default=4)
    parser.add_argument("--format-negatives", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fold-output-dir",
        default="/share/home/group9/data/egocross/kto_answer_reward_perm4_wrong3_fmt2_folds",
    )
    parser.add_argument("--num-folds", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    by_domain = load_domain_rows(data_dir)
    rng = random.Random(args.seed)
    permutations = make_permutations(rng, args.permutations)

    rows = []
    source_counts = {}
    for domain, domain_rows in by_domain.items():
        source_counts[domain] = len(domain_rows)
        rows.extend(build_rows(indexed(domain_rows), domain, permutations, args.format_negatives))

    random.Random(args.seed + 1000).shuffle(rows)
    write_json(Path(args.output), rows, args.overwrite)
    print(f"Saved {len(rows)} KTO rows to {args.output}")
    print("source_counts:", dict(sorted(source_counts.items())))
    print("permutations:", ["".join(item) for item in permutations])
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
                permutations,
                args.format_negatives,
            )
            random.Random(args.seed + 2000 + fold_id).shuffle(train_rows)
            train_path = fold_dir / f"train_kto_answer_reward_perm4_wrong3_fmt2_fold{fold_id}.json"
            eval_path = fold_dir / f"eval_answer_only_fold{fold_id}.json"
            write_json(train_path, train_rows, args.overwrite)
            write_json(eval_path, eval_rows, args.overwrite)
            all_fold_summaries[f"fold{fold_id}"] = {
                "train": summarize(train_rows),
                "eval_sources": len(eval_rows),
                "by_domain": fold_summary,
            }

        summary_path = fold_dir / "fold_summary.json"
        write_json(summary_path, all_fold_summaries, args.overwrite)
        print(f"Saved {args.num_folds}-fold KTO validation data to {fold_dir}")


if __name__ == "__main__":
    main()
