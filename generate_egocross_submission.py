import argparse
import base64
import json
import re
import time
from pathlib import Path


DOMAIN_BY_DATASET = {
    "CholecTrack20": "Surgery",
    "EgoSurgery": "Surgery",
    "ENIGMA": "Industry",
    "ExtrameSportFPV": "XSports",
    "EgoPet": "Animal",
}

DOMAIN_HINT_BY_DATASET = {
    "CholecTrack20": "laparoscopic surgery instrument and anatomy video",
    "EgoSurgery": "first-person surgery video",
    "ENIGMA": "industrial assembly egocentric video",
    "ExtrameSportFPV": "first-person extreme sports video",
    "EgoPet": "pet-mounted first-person animal video",
}


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def normalize_options(options):
    if isinstance(options, dict):
        lines = []
        for key in ["A", "B", "C", "D"]:
            value = options.get(key) or options.get(key.lower())
            if value is not None:
                lines.append(f"{key}) {value}")
        return "\n".join(lines)

    if isinstance(options, list):
        lines = []
        letters = ["A", "B", "C", "D"]
        for i, value in enumerate(options[:4]):
            text = value
            if isinstance(value, dict):
                text = value.get("text") or value.get("answer") or value.get("option") or str(value)
            lines.append(f"{letters[i]}) {text}")
        return "\n".join(lines)

    return str(options)


def extract_answer(text: str):
    text = text.strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"([ABCD])", text)
    if match:
        return match.group(1)
    return None


def row_key(row, fallback_idx=None):
    if row.get("id") is not None:
        return ("id", str(row["id"]))
    if row.get("question_id"):
        return ("question_id", str(row["question_id"]))
    return ("idx", str(fallback_idx))


def load_submission(path):
    if not path:
        return None

    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    return {row_key(row, idx): row for idx, row in enumerate(rows, start=1)}


def parse_dataset_selectors(value):
    if not value:
        return None

    selectors = {item.strip().lower() for item in value.split(",") if item.strip()}
    return selectors or None


def selected_dataset(dataset, selectors):
    if not selectors:
        return True

    domain = DOMAIN_BY_DATASET.get(dataset, dataset)
    return dataset.lower() in selectors or domain.lower() in selectors


def is_context_length_error(exc):
    text = str(exc).lower()
    return "exceeds model" in text and "maximum context length" in text


def frame_retry_sequence(max_frames):
    start = max_frames or 8
    sequence = []
    for value in [start, 4, 2, 1]:
        if value <= start and value not in sequence:
            sequence.append(value)
    return sequence


def resolve_frame_path(testbed_dir: Path, dataset: str, frame_path: str) -> str:
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


def sample_frame_paths(frame_paths, max_frames):
    if max_frames and len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        return [frame_paths[int(i * step)] for i in range(max_frames)]
    return list(frame_paths)


def build_content(testbed_dir, sample, frame_paths, prompt):
    content = []
    for frame_path in frame_paths:
        abs_path = resolve_frame_path(testbed_dir, sample["dataset"], frame_path)
        img_b64 = encode_image(abs_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    content.append({"type": "text", "text": prompt})
    return content


def build_prompt(sample, prompt_mode):
    question = sample.get("question_text", "")
    options = normalize_options(sample.get("options", ""))
    dataset = sample.get("dataset", "")

    if prompt_mode == "strict_direct":
        return (
            "Return exactly one character: A, B, C, or D.\n"
            "Do not output any other words, punctuation, or explanation.\n\n"
            f"Question: {question}\n"
            f"{options}\n"
            "Answer:"
        )

    if prompt_mode == "domain_direct":
        hint = DOMAIN_HINT_BY_DATASET.get(dataset, "egocentric video")
        return (
            "Answer this multiple-choice question from the provided frames.\n"
            f"Domain hint: {hint}.\n"
            "Return only one letter: A, B, C, or D. Do not explain.\n\n"
            f"Question: {question}\n"
            f"{options}\n"
            "Answer:"
        )

    return (
        "Answer the multiple-choice question using only one letter: A, B, C, or D.\n"
        "Do not explain.\n\n"
        f"Question: {question}\n"
        f"{options}\n"
        "Answer:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed-dir", default="/share/home/group9/data/egocross_full/egocross_testbed")
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output", default="submission_full_sft_32k_200k.json")
    parser.add_argument("--template", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="egocross")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use all frames")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt-mode", choices=["direct", "strict_direct", "domain_direct"], default="direct")
    parser.add_argument(
        "--only-datasets",
        default=None,
        help="Comma-separated dataset names or domain aliases to run, e.g. ENIGMA,ExtrameSportFPV or Industry,XSports.",
    )
    parser.add_argument(
        "--fallback-submission",
        default=None,
        help="Full submission used for skipped samples and API/parse failures.",
    )
    parser.add_argument("--raw-output", default=None)
    parser.add_argument("--frame-retry", action="store_true", help="Retry context-length errors with fewer frames.")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    selectors = parse_dataset_selectors(args.only_datasets)
    if selectors and not args.fallback_submission:
        parser.error("--only-datasets requires --fallback-submission so skipped rows keep valid answers.")

    testbed_dir = Path(args.testbed_dir)
    input_json = Path(args.input_json) if args.input_json else testbed_dir / "egocross_testbed_imgs.json"


    with open(input_json) as f:
        samples = json.load(f)

    template = None
    if args.template:
        with open(args.template) as f:
            template = json.load(f)

    fallback = load_submission(args.fallback_submission)

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    outputs = []
    raw_outputs = []


    for idx, sample in enumerate(samples, start=1):
        dataset = sample["dataset"]
        sample_key = row_key(sample, idx)
        fallback_row = dict(fallback[sample_key]) if fallback and sample_key in fallback else None

        if not selected_dataset(dataset, selectors):
            if not fallback_row:
                raise RuntimeError(f"No fallback row for skipped sample {sample_key}.")
            outputs.append(fallback_row)
            raw_outputs.append({
                "id": fallback_row.get("id", sample.get("id", idx)),
                "question_id": fallback_row.get("question_id", sample.get("question_id", "")),
                "dataset": fallback_row.get("dataset", dataset),
                "answer": fallback_row.get("answer", ""),
                "prompt_mode": args.prompt_mode,
                "status": "skipped_fallback",
                "raw_output": "",
            })
            print(
                f"[{idx}/{len(samples)}] {fallback_row.get('question_id', '')} -> "
                f"{fallback_row.get('answer', '')} | skipped fallback",
                flush=True,
            )
            continue

        frame_paths = sample.get("video_path", [])

        prompt = build_prompt(sample, args.prompt_mode)

        status = "ok"
        used_max_frames = args.max_frames
        attempts = []
        try:
            retry_frames = frame_retry_sequence(args.max_frames) if args.frame_retry else [args.max_frames]
            for attempt_frames in retry_frames:
                used_max_frames = attempt_frames
                try:
                    selected_frames = sample_frame_paths(frame_paths, attempt_frames)
                    content = build_content(testbed_dir, sample, selected_frames, prompt)
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=args.max_tokens,
                        temperature=0,
                    )
                    raw_answer = resp.choices[0].message.content
                    answer = extract_answer(raw_answer)
                    attempts.append({
                        "max_frames": attempt_frames,
                        "status": "ok" if answer is not None else "parse_failed",
                    })
                    if answer is None:
                        if fallback_row:
                            answer = fallback_row.get("answer", "A")
                            status = "parse_fallback"
                        else:
                            answer = "A"
                            status = "parse_default_a"
                    elif attempt_frames != args.max_frames:
                        status = "ok_after_frame_retry"
                    break
                except Exception as e:
                    attempts.append({
                        "max_frames": attempt_frames,
                        "status": "context_length_error" if is_context_length_error(e) else "error",
                        "error": str(e),
                    })
                    if args.frame_retry and is_context_length_error(e) and attempt_frames != retry_frames[-1]:
                        continue
                    raise
        except Exception as e:
            raw_answer = f"ERROR: {e}"
            if fallback_row:
                answer = fallback_row.get("answer", "A")
                status = "error_fallback"
            else:
                answer = "A"
                status = "error_default_a"

        if template:
            row = dict(template[idx - 1])
            row["answer"] = answer
        else:
            row = {
                "id": sample.get("id", idx),
                "question_id": sample.get("question_id", ""),
                "dataset": dataset,
                "answer": answer,
            }

        outputs.append(row)
        raw_outputs.append({
            "id": row.get("id", sample.get("id", idx)),
            "question_id": row.get("question_id", sample.get("question_id", "")),
            "dataset": row.get("dataset", dataset),
            "answer": answer,
            "prompt_mode": args.prompt_mode,
            "status": status,
            "max_frames": args.max_frames,
            "used_max_frames": used_max_frames,
            "frame_attempts": attempts,
            "prompt": prompt,
            "raw_output": raw_answer,
        })


        print(
            f"[{idx}/{len(samples)}] {row.get('question_id', sample.get('question_id', ''))} "
            f"-> {answer} | max{used_max_frames}/{args.max_frames} | status={status} | raw={raw_answer!r}",
            flush=True,
        )

        if args.sleep:
            time.sleep(args.sleep)

    with open(args.output, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    if args.raw_output:
        with open(args.raw_output, "w") as f:
            json.dump(raw_outputs, f, indent=2, ensure_ascii=False)
        print(f"Saved raw outputs to {args.raw_output}")

    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
