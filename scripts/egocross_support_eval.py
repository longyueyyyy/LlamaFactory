#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from urllib import error as urllib_error
from urllib import request as urllib_request

import egocross_router_infer as router


DOMAIN_FILES = {
    "Animal": "train_animal.json",
    "Surgery": "train_surgery.json",
    "Industry": "train_industry.json",
    "XSports": "train_xsports.json",
}


def clean_user_text(text):
    text = str(text or "").replace("<image>", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_support_question_options(user_text):
    clean = clean_user_text(user_text)
    matches = list(re.finditer(r"(?<![A-Za-z0-9])([ABCD])\s*[:\).]\s*", clean))
    if len(matches) < 4:
        raise ValueError(f"Could not find four A/B/C/D options in support prompt: {clean[:300]!r}")

    # Use the first ordered A, B, C, D block.
    block = None
    for start in range(len(matches) - 3):
        letters = [match.group(1) for match in matches[start : start + 4]]
        if letters == router.LETTERS:
            block = matches[start : start + 4]
            break
    if block is None:
        raise ValueError(f"Could not find ordered A/B/C/D options in support prompt: {clean[:300]!r}")

    question = clean[: block[0].start()].strip()
    options = {}
    for idx, match in enumerate(block):
        end = block[idx + 1].start() if idx + 1 < len(block) else len(clean)
        options[match.group(1)] = clean[match.end() : end].strip()

    if not question or any(not options.get(letter) for letter in router.LETTERS):
        raise ValueError(f"Malformed support prompt after parsing: {clean[:300]!r}")
    return question, options


def infer_dataset_from_images(images, domain):
    joined = "/" + "/".join(str(item).replace("\\", "/") for item in images) + "/"
    for dataset, dataset_domain in router.DOMAIN_BY_DATASET.items():
        if f"/{dataset}/" in joined:
            return dataset

    for dataset, dataset_domain in router.DOMAIN_BY_DATASET.items():
        if dataset_domain == domain:
            return dataset
    return domain


def support_row_to_sample(row, domain, idx):
    messages = row.get("messages") or []
    user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
    assistant_msg = next((msg for msg in reversed(messages) if msg.get("role") == "assistant"), None)
    if not user_msg or not assistant_msg:
        raise ValueError(f"Support row missing user/assistant message: domain={domain} idx={idx}")

    question, options = parse_support_question_options(user_msg.get("content", ""))
    answer = router.extract_answer(assistant_msg.get("content", ""))
    if answer not in router.VALID_ANSWERS:
        raise ValueError(f"Invalid support answer: domain={domain} idx={idx} answer={answer!r}")

    images = row.get("images") or []
    if not images:
        raise ValueError(f"Support row has no images: domain={domain} idx={idx}")

    dataset = infer_dataset_from_images(images, domain)
    question_type = router.normalize_question_type(row.get("question_type", ""), question)
    return {
        "id": f"{domain}_{idx}",
        "question_id": f"{domain}_{idx}",
        "dataset": dataset,
        "domain": domain,
        "question_text": question,
        "question_type": question_type,
        "options": options,
        "video_path": images,
        "gold_answer": answer,
    }


def load_support_samples(support_dir, only_domains=None):
    support_dir = Path(support_dir)
    selectors = {item.strip().lower() for item in only_domains.split(",") if item.strip()} if only_domains else None
    samples = []
    for domain, file_name in DOMAIN_FILES.items():
        if selectors and domain.lower() not in selectors:
            continue
        path = support_dir / file_name
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for idx, row in enumerate(rows, start=1):
            samples.append(support_row_to_sample(row, domain, idx))
    return samples


def safe_div(num, den):
    return num / den if den else 0.0


def add_bucket(bucket, key, correct):
    item = bucket[key]
    item["total"] += 1
    item["correct"] += int(correct)


def finalize_bucket(bucket):
    return {
        key: {
            "correct": value["correct"],
            "total": value["total"],
            "acc": round(safe_div(value["correct"], value["total"]), 6),
        }
        for key, value in sorted(bucket.items())
    }


def is_single_letter_output(text):
    return isinstance(text, str) and re.fullmatch(r"\s*[ABCD]\s*", text.strip()) is not None


def answer_only_format_ok(raw_payload):
    if is_single_letter_output(raw_payload):
        return True
    if not isinstance(raw_payload, list):
        return False

    raw_outputs = []
    for frame_attempt in raw_payload:
        for attempt in frame_attempt.get("attempts", []):
            raw_outputs.append(attempt.get("raw_output"))
    return bool(raw_outputs) and all(is_single_letter_output(item) for item in raw_outputs)


def build_metrics(rows, args):
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    answer_only = sum(1 for row in rows if row.get("answer_only_format"))
    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    by_dataset = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_status = Counter(row["status"] for row in rows)
    answers = Counter(row.get("answer") or "" for row in rows)
    gold = Counter(row.get("gold_answer") or "" for row in rows)
    used_frames = Counter(str(row.get("used_max_frames", "")) for row in rows)

    for row in rows:
        add_bucket(by_domain, row["domain"], row["correct"])
        add_bucket(by_dataset, row["dataset"], row["correct"])
        add_bucket(by_type, row["question_type"], row["correct"])

    return {
        "strategy": {
            "prompt_mode": args.prompt_mode,
            "max_frames": args.max_frames,
            "frame_sampling": args.frame_sampling,
            "max_tokens": args.max_tokens,
            "vote": bool(args.vote),
            "frame_route": args.frame_route or [],
            "only_domains": args.only_domains,
        },
        "overall": {
            "correct": correct,
            "total": total,
            "acc": round(safe_div(correct, total), 6),
            "answer_only_format_rate": round(safe_div(answer_only, total), 6),
        },
        "by_domain": finalize_bucket(by_domain),
        "by_dataset": finalize_bucket(by_dataset),
        "by_question_type": finalize_bucket(by_type),
        "status_dist": dict(sorted(by_status.items())),
        "answer_dist": dict(sorted(answers.items())),
        "gold_dist": dict(sorted(gold.items())),
        "used_max_frames_dist": dict(sorted(used_frames.items())),
    }


def metrics_text(metrics):
    lines = []
    overall = metrics["overall"]
    lines.append(f"overall_acc: {overall['acc']:.6f} ({overall['correct']}/{overall['total']})")
    lines.append(f"answer_only_format_rate: {overall['answer_only_format_rate']:.6f}")
    lines.append(
        "strategy: "
        + " ".join(f"{key}={value}" for key, value in metrics["strategy"].items() if value not in [None, [], ""])
    )
    lines.append("by_domain:")
    for key, value in metrics["by_domain"].items():
        lines.append(f"  {key}: {value['acc']:.6f} ({value['correct']}/{value['total']})")
    lines.append("by_question_type:")
    for key, value in metrics["by_question_type"].items():
        lines.append(f"  {key}: {value['acc']:.6f} ({value['correct']}/{value['total']})")
    lines.append("status_dist: " + json.dumps(metrics["status_dist"], ensure_ascii=False, sort_keys=True))
    lines.append("answer_dist: " + json.dumps(metrics["answer_dist"], ensure_ascii=False, sort_keys=True))
    lines.append("gold_dist: " + json.dumps(metrics["gold_dist"], ensure_ascii=False, sort_keys=True))
    lines.append("used_max_frames_dist: " + json.dumps(metrics["used_max_frames_dist"], ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"


class SimpleOpenAIClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_chat_completion))

    def create_chat_completion(self, model, messages, max_tokens, temperature):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        content = data["choices"][0]["message"]["content"]
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def make_client(base_url):
    try:
        from openai import OpenAI
        return OpenAI(base_url=base_url, api_key="dummy")
    except ModuleNotFoundError:
        return SimpleOpenAIClient(base_url)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fixed EgoCross inference strategy on labeled support set.")
    parser.add_argument("--support-dir", default="/share/home/group9/data/egocross")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="egocross")
    parser.add_argument("--prompt-mode", choices=sorted(router.PROMPT_MODES), default="direct")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--frame-sampling", choices=["uniform", "endpoint", "tail_dense"], default="uniform")
    parser.add_argument("--frame-route", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--vote", action="store_true", help="Use 3-way option-order voting for every selected support sample.")
    parser.add_argument("--only-domains", default=None, help="Comma-separated support domains, e.g. Industry,XSports.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to write non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_support_samples(args.support_dir, args.only_domains)
    if args.limit:
        samples = samples[: args.limit]

    client = make_client(args.base_url)
    support_dir = Path(args.support_dir)
    frame_routes = router.parse_frame_routes(args.frame_route)
    raw_rows = []

    for idx, sample in enumerate(samples, start=1):
        max_frames = router.apply_frame_routes(sample, args.max_frames, frame_routes)
        answer = None
        raw_payload = None
        prompt = None
        frame_attempts = []
        used_max_frames = max_frames
        status = "ok"

        try:
            if args.vote:
                answer, raw_payload, status, used_max_frames = router.vote_with_frame_retry(
                    client,
                    args.model,
                    support_dir,
                    sample,
                    args.prompt_mode,
                    max_frames,
                    args.frame_sampling,
                    args.max_tokens,
                )
                frame_attempts = raw_payload
            else:
                answer, raw_payload, prompt, used_max_frames, frame_attempts, status = router.infer_with_frame_retry(
                    client,
                    args.model,
                    support_dir,
                    sample,
                    args.prompt_mode,
                    max_frames,
                    args.frame_sampling,
                    args.max_tokens,
                )
        except Exception as exc:
            raw_payload = f"ERROR: {exc}"
            status = "error"

        correct = answer == sample["gold_answer"]
        raw_rows.append({
            "id": sample["id"],
            "question_id": sample["question_id"],
            "dataset": sample["dataset"],
            "domain": sample["domain"],
            "question_type": sample["question_type"],
            "gold_answer": sample["gold_answer"],
            "answer": answer,
            "correct": correct,
            "answer_only_format": answer_only_format_ok(raw_payload),
            "status": status,
            "prompt_mode": args.prompt_mode,
            "max_frames": max_frames,
            "used_max_frames": used_max_frames,
            "frame_sampling": args.frame_sampling,
            "frame_attempts": frame_attempts,
            "prompt": prompt,
            "raw_output": raw_payload,
        })
        print(
            f"[{idx}/{len(samples)}] {sample['question_id']} gold={sample['gold_answer']} pred={answer} "
            f"correct={int(correct)} max{used_max_frames}/{max_frames} status={status}",
            flush=True,
        )

    metrics = build_metrics(raw_rows, args)
    with open(output_dir / "support_predictions.json", "w", encoding="utf-8") as f:
        json.dump(raw_rows, f, indent=2, ensure_ascii=False)
    with open(output_dir / "support_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    text = metrics_text(metrics)
    with open(output_dir / "support_metrics.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print(text, end="")
    print(f"Saved support eval to {output_dir}")


if __name__ == "__main__":
    main()
