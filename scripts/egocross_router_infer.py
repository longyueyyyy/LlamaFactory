#!/usr/bin/env python3
import argparse
import base64
import json
import re
import time
import zipfile
from collections import Counter
from pathlib import Path


DOMAIN_BY_DATASET = {
    "CholecTrack20": "Surgery",
    "EgoSurgery": "Surgery",
    "ENIGMA": "Industry",
    "ExtrameSportFPV": "XSports",
    "EgoPet": "Animal",
}

DOMAIN_HINT_BY_DATASET = {
    "CholecTrack20": "laparoscopic surgery instrument and anatomy",
    "EgoSurgery": "first-person surgery",
    "ENIGMA": "industrial assembly operation",
    "ExtrameSportFPV": "first-person extreme sports action",
    "EgoPet": "pet-mounted first-person animal behavior",
}

VALID_ANSWERS = {"A", "B", "C", "D"}
LETTERS = ["A", "B", "C", "D"]


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def row_key(row, fallback_idx=None):
    if row.get("id") is not None:
        return ("id", str(row["id"]))
    if row.get("question_id"):
        return ("question_id", str(row["question_id"]))
    return ("idx", str(fallback_idx))


def load_rows(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_rows(rows):
    return {row_key(row, idx): row for idx, row in enumerate(rows, start=1)}


def parse_selectors(value):
    if not value:
        return None
    selectors = {item.strip().lower() for item in value.split(",") if item.strip()}
    return selectors or None


def selector_matches(dataset, selectors):
    if not selectors:
        return True
    domain = DOMAIN_BY_DATASET.get(dataset, dataset)
    return dataset.lower() in selectors or domain.lower() in selectors


def parse_routes(route_args, default_prompt_mode, default_max_frames):
    routes = {}
    for item in route_args or []:
        if "=" not in item:
            raise ValueError(f"Route must use DOMAIN=prompt_mode[:max_frames]: {item}")
        selector, value = item.split("=", 1)
        parts = value.split(":")
        prompt_mode = parts[0]
        if prompt_mode not in {"direct", "strict_direct", "domain_direct"}:
            raise ValueError(f"Unsupported prompt mode in route {item}")
        max_frames = default_max_frames if len(parts) == 1 or not parts[1] else int(parts[1])
        routes[selector.strip().lower()] = (prompt_mode, max_frames)

    def choose(dataset):
        domain = DOMAIN_BY_DATASET.get(dataset, dataset)
        for name in (dataset.lower(), domain.lower()):
            if name in routes:
                return routes[name]
        return default_prompt_mode, default_max_frames

    return choose


def parse_options(options):
    parsed = []
    if isinstance(options, dict):
        for letter in LETTERS:
            value = options.get(letter) or options.get(letter.lower())
            if value is not None:
                parsed.append((letter, str(value)))
    elif isinstance(options, list):
        for letter, value in zip(LETTERS, options[:4]):
            if isinstance(value, dict):
                text = value.get("text") or value.get("answer") or value.get("option") or str(value)
            else:
                text = str(value)
            parsed.append((letter, text))
    else:
        parsed = [(letter, "") for letter in LETTERS]
        parsed[0] = ("A", str(options))

    if len(parsed) != 4:
        raise ValueError(f"Expected 4 options, got {len(parsed)}: {options!r}")
    return parsed


def format_options(parsed_options, original_order):
    by_letter = dict(parsed_options)
    display_to_original = {}
    lines = []
    for display_letter, original_letter in zip(LETTERS, original_order):
        display_to_original[display_letter] = original_letter
        lines.append(f"{display_letter}) {by_letter[original_letter]}")
    return "\n".join(lines), display_to_original


def extract_answer(text):
    upper = str(text).strip().upper()
    match = re.search(r"\b([ABCD])\b", upper)
    if match:
        return match.group(1)
    match = re.search(r"([ABCD])", upper)
    if match:
        return match.group(1)
    return None


def resolve_frame_path(testbed_dir, dataset, frame_path):
    raw = str(frame_path).replace("\\", "/")

    p = Path(raw)
    if p.is_absolute() and p.exists():
        return str(p)

    rel = raw.lstrip("/")
    prefix = testbed_dir.name + "/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]

    candidates = [
        testbed_dir / rel,
        testbed_dir.parent / raw.lstrip("/"),
        testbed_dir / dataset / rel,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Cannot resolve frame path: {frame_path}. Tried: "
        + "; ".join(str(c) for c in candidates)
    )


def sample_frames(frame_paths, max_frames):
    if max_frames and len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        return [frame_paths[int(i * step)] for i in range(max_frames)]
    return list(frame_paths)


def build_prompt(sample, prompt_mode, option_lines):
    question = sample.get("question_text", "")
    dataset = sample.get("dataset", "")

    if prompt_mode == "strict_direct":
        return (
            "Return exactly one character: A, B, C, or D.\n"
            "No explanation. No punctuation.\n\n"
            f"Question: {question}\n"
            f"{option_lines}\n"
            "Answer:"
        )

    if prompt_mode == "domain_direct":
        hint = DOMAIN_HINT_BY_DATASET.get(dataset, "egocentric video")
        return (
            f"Domain: {hint}.\n"
            "Answer from the frames. Return only A, B, C, or D.\n\n"
            f"Question: {question}\n"
            f"{option_lines}\n"
            "Answer:"
        )

    return (
        "Answer the multiple-choice question using only one letter: A, B, C, or D.\n"
        "Do not explain.\n\n"
        f"Question: {question}\n"
        f"{option_lines}\n"
        "Answer:"
    )


def call_model(client, model, content, max_tokens):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp.choices[0].message.content


def build_image_content(testbed_dir, sample, max_frames):
    dataset = sample["dataset"]
    content = []
    for frame_path in sample_frames(sample.get("video_path", []), max_frames):
        abs_path = resolve_frame_path(testbed_dir, dataset, frame_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(abs_path)}"},
        })
    return content


def infer_once(client, model, testbed_dir, sample, prompt_mode, max_frames, max_tokens, original_order):
    parsed_options = parse_options(sample.get("options", ""))
    option_lines, display_to_original = format_options(parsed_options, original_order)
    content = build_image_content(testbed_dir, sample, max_frames)
    prompt = build_prompt(sample, prompt_mode, option_lines)
    content.append({"type": "text", "text": prompt})
    raw_output = call_model(client, model, content, max_tokens)
    display_answer = extract_answer(raw_output)
    if display_answer is None:
        return None, raw_output, prompt
    return display_to_original[display_answer], raw_output, prompt


def vote_infer(client, model, testbed_dir, sample, prompt_mode, max_frames, max_tokens):
    orders = [
        ["A", "B", "C", "D"],
        ["B", "C", "D", "A"],
        ["C", "D", "A", "B"],
    ]
    attempts = []
    for order in orders:
        answer, raw_output, prompt = infer_once(
            client, model, testbed_dir, sample, prompt_mode, max_frames, max_tokens, order
        )
        attempts.append({
            "order": order,
            "answer": answer,
            "raw_output": raw_output,
            "prompt": prompt,
        })

    votes = Counter(attempt["answer"] for attempt in attempts if attempt["answer"] in VALID_ANSWERS)
    if not votes:
        return None, attempts, "vote_no_parse"

    answer, count = votes.most_common(1)[0]
    if count >= 2:
        return answer, attempts, "vote_majority"
    return answer, attempts, "vote_plurality"


def fallback_answer(fallback_index, sample, idx):
    if not fallback_index:
        return None
    row = fallback_index.get(row_key(sample, idx))
    return row.get("answer") if row else None


def make_output_row(template, sample, idx, answer):
    if template:
        row = dict(template[idx - 1])
    else:
        row = {
            "id": sample.get("id", idx),
            "question_id": sample.get("question_id", ""),
            "dataset": sample.get("dataset", ""),
        }
    row["answer"] = answer
    return row


def summarize(outputs, raw_outputs, fallback_rows):
    lines = []
    answers = Counter(row.get("answer", "") for row in outputs)
    statuses = Counter(row.get("status", "") for row in raw_outputs)
    domains = Counter(DOMAIN_BY_DATASET.get(row.get("dataset", ""), row.get("dataset", "")) for row in outputs)
    invalid = [row for row in outputs if row.get("answer") not in VALID_ANSWERS]
    empty = [row for row in outputs if not row.get("answer")]

    lines.append(f"num: {len(outputs)}")
    lines.append(f"empty: {len(empty)}")
    lines.append(f"invalid: {len(invalid)}")
    lines.append("answer_dist: " + " ".join(f"{key}={answers.get(key, 0)}" for key in LETTERS))
    lines.append("domain_dist: " + " ".join(f"{key}={domains.get(key, 0)}" for key in sorted(domains)))
    lines.append("status_dist: " + " ".join(f"{key}={value}" for key, value in sorted(statuses.items())))

    if fallback_rows and len(fallback_rows) == len(outputs):
        changed = Counter()
        total = 0
        for row, fallback_row in zip(outputs, fallback_rows):
            if row.get("answer") != fallback_row.get("answer"):
                total += 1
                changed[DOMAIN_BY_DATASET.get(row.get("dataset", ""), row.get("dataset", ""))] += 1
        lines.append(f"changed_vs_fallback_total: {total}")
        lines.append(
            "changed_vs_fallback_by_domain: "
            + " ".join(f"{key}={changed.get(key, 0)}" for key in ["Surgery", "Industry", "XSports", "Animal"])
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Domain-aware EgoCross inference router.")
    parser.add_argument("--testbed-dir", default="/share/home/group9/data/egocross_full/egocross_testbed")
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="egocross")
    parser.add_argument("--default-prompt-mode", choices=["direct", "strict_direct", "domain_direct"], default="strict_direct")
    parser.add_argument("--default-max-frames", type=int, default=8)
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Dataset/domain route: DOMAIN=prompt_mode[:max_frames], e.g. XSports=domain_direct:8.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--only-datasets", default=None)
    parser.add_argument("--fallback-submission", default=None)
    parser.add_argument("--vote-domains", default=None, help="Comma-separated domains/datasets using 3-way option-order voting.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N selected samples; requires fallback for full output.")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    testbed_dir = Path(args.testbed_dir)
    input_json = Path(args.input_json) if args.input_json else testbed_dir / "egocross_testbed_imgs.json"

    samples = load_rows(input_json)
    template = load_rows(args.template) if args.template else None
    fallback_rows = load_rows(args.fallback_submission)
    fallback_index = index_rows(fallback_rows) if fallback_rows else None
    selectors = parse_selectors(args.only_datasets)
    vote_selectors = parse_selectors(args.vote_domains)
    choose_route = parse_routes(args.route, args.default_prompt_mode, args.default_max_frames)

    if (selectors or args.limit) and not fallback_index:
        raise SystemExit("--only-datasets and --limit require --fallback-submission to keep a full valid output.")

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    outputs = []
    raw_outputs = []
    processed = 0

    for idx, sample in enumerate(samples, start=1):
        dataset = sample["dataset"]
        should_process = selector_matches(dataset, selectors)
        if args.limit and processed >= args.limit:
            should_process = False

        if not should_process:
            answer = fallback_answer(fallback_index, sample, idx)
            if answer is None:
                raise RuntimeError(f"No fallback answer for skipped sample {row_key(sample, idx)}")
            outputs.append(make_output_row(template, sample, idx, answer))
            raw_outputs.append({
                "id": sample.get("id", idx),
                "question_id": sample.get("question_id", ""),
                "dataset": dataset,
                "answer": answer,
                "status": "skipped_fallback",
            })
            continue

        prompt_mode, max_frames = choose_route(dataset)
        use_vote = bool(vote_selectors) and selector_matches(dataset, vote_selectors)
        status = "ok"
        raw_payload = None
        prompt = None

        try:
            if use_vote:
                answer, raw_payload, status = vote_infer(
                    client, args.model, testbed_dir, sample, prompt_mode, max_frames, args.max_tokens
                )
            else:
                answer, raw_payload, prompt = infer_once(
                    client,
                    args.model,
                    testbed_dir,
                    sample,
                    prompt_mode,
                    max_frames,
                    args.max_tokens,
                    ["A", "B", "C", "D"],
                )
                if answer is None:
                    status = "parse_failed"

            if answer is None:
                fallback = fallback_answer(fallback_index, sample, idx)
                if fallback is not None:
                    answer = fallback
                    status = status + "_fallback"
                else:
                    answer = "A"
                    status = status + "_default_a"
        except Exception as exc:
            raw_payload = f"ERROR: {exc}"
            fallback = fallback_answer(fallback_index, sample, idx)
            if fallback is not None:
                answer = fallback
                status = "error_fallback"
            else:
                answer = "A"
                status = "error_default_a"

        row = make_output_row(template, sample, idx, answer)
        outputs.append(row)
        raw_outputs.append({
            "id": row.get("id", sample.get("id", idx)),
            "question_id": row.get("question_id", sample.get("question_id", "")),
            "dataset": row.get("dataset", dataset),
            "answer": answer,
            "status": status,
            "prompt_mode": prompt_mode,
            "max_frames": max_frames,
            "vote": use_vote,
            "prompt": prompt,
            "raw_output": raw_payload,
        })
        processed += 1
        print(
            f"[{idx}/{len(samples)}] {row.get('question_id', '')} -> {answer} "
            f"| {prompt_mode} max{max_frames} status={status}",
            flush=True,
        )
        if args.sleep:
            time.sleep(args.sleep)

    predictions_path = output_dir / "predictions.json"
    raw_path = output_dir / "raw_outputs.json"
    summary_path = output_dir / "metrics_summary.txt"

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, indent=2, ensure_ascii=False)

    summary = summarize(outputs, raw_outputs, fallback_rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    if not args.no_zip:
        with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(predictions_path, arcname="predictions.json")

    print(summary, end="")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
