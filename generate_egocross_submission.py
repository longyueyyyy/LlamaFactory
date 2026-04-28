import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


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


def extract_answer(text: str) -> str:
    text = text.strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"([ABCD])", text)
    if match:
        return match.group(1)
    return "A"


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



def build_prompt(sample):
    question = sample.get("question_text", "")
    options = normalize_options(sample.get("options", ""))

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
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    testbed_dir = Path(args.testbed_dir)
    input_json = Path(args.input_json) if args.input_json else testbed_dir / "egocross_testbed_imgs.json"


    with open(input_json) as f:
        samples = json.load(f)

    template = None
    if args.template:
        with open(args.template) as f:
            template = json.load(f)

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    outputs = []


    for idx, sample in enumerate(samples, start=1):
        dataset = sample["dataset"]
        frame_paths = sample.get("video_path", [])

        if args.max_frames and len(frame_paths) > args.max_frames:
            step = len(frame_paths) / args.max_frames
            frame_paths = [frame_paths[int(i * step)] for i in range(args.max_frames)]

        content = []
        for frame_path in frame_paths:
            abs_path = resolve_frame_path(testbed_dir, dataset, frame_path)
            img_b64 = encode_image(abs_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })

        content.append({"type": "text", "text": build_prompt(sample)})

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=8,
                temperature=0,
            )
            raw_answer = resp.choices[0].message.content
            answer = extract_answer(raw_answer)
        except Exception as e:
            raw_answer = f"ERROR: {e}"
            answer = "A"

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


        print(f"[{idx}/{len(samples)}] {sample.get('question_id', '')} -> {answer} | raw={raw_answer!r}", flush=True)

        if args.sleep:
            time.sleep(args.sleep)

    with open(args.output, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
