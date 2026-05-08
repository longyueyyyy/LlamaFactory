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
SUPPORT_FILES = {
    "Animal": "train_animal.json",
    "Surgery": "train_surgery.json",
    "Industry": "train_industry.json",
    "XSports": "train_xsports.json",
}
ENHANCED_SUPPORT_FILES = {
    "Animal": "train_animal_enhanced.json",
    "Surgery": "train_surgery_enhanced.json",
    "Industry": "train_industry_enhanced.json",
    "XSports": "train_xsports_enhanced.json",
}
DOMAIN_ALIASES = {
    "animal": "Animal",
    "surgery": "Surgery",
    "industry": "Industry",
    "xsports": "XSports",
    "x-sports": "XSports",
    "extreme sports": "XSports",
}
QUESTION_TYPE_ALIASES = {
    "object counting": "counting",
    "counting": "counting",
    "not visible": "not_visible",
    "object not visible": "not_visible",
    "instrument not visible": "not_visible",
    "spatial localization": "region_localization",
    "region localization": "region_localization",
    "temporal localization": "temporal_localization",
    "action temporal localization": "temporal_localization",
    "prediction": "next_interaction",
    "next interaction prediction": "next_interaction",
    "next direction prediction": "next_direction",
    "direction prediction": "next_direction",
    "action identification": "action_identification",
    "special action identification": "action_identification",
    "action sequence identification": "action_sequence",
    "sport identification": "sport_identification",
    "animal identification": "animal_identification",
}
QUESTION_TYPE_GROUPS = {
    "counting": "counting",
    "not_visible": "visibility",
    "region_localization": "localization",
    "temporal_localization": "localization",
    "next_interaction": "prediction",
    "next_direction": "prediction",
    "action_identification": "identification",
    "action_sequence": "sequence",
    "sport_identification": "identification",
    "animal_identification": "identification",
    "object_interaction": "identification",
    "tool_interaction": "identification",
    "phase_sequence": "sequence",
    "unknown": "unknown",
}
EVIDENCE_BY_TYPE = {
    "counting": "count grouped object types, not repeated instances.",
    "not_visible": "check which option is absent across the frames.",
    "region_localization": "use the object's approximate position in the view.",
    "temporal_localization": "compare early, middle, and late frames for the first action time.",
    "next_interaction": "use the last visible interaction to predict the next one.",
    "next_direction": "use motion direction from the latest frames.",
    "action_identification": "identify the visible action in the requested interval.",
    "action_sequence": "match the ordered actions visible over time.",
    "sport_identification": "identify the sport from scene and motion cues.",
    "animal_identification": "identify the animal from visible body and scene cues.",
    "object_interaction": "identify the object being interacted with.",
    "tool_interaction": "identify the tool being manipulated.",
    "phase_sequence": "match the surgical phase order.",
    "unknown": "match the visual evidence to the options.",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "be", "being", "by", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or", "the",
    "this", "to", "up", "was", "were", "what", "when", "where", "which", "with",
    "within", "video", "segment", "clip", "question", "answer", "option",
}
PROTECTED_OUTPUT_DIR_NAMES = {
    "baseline_max8",
    "weighted_answer_only_full_i4_x4_lr5e6_ep1_direct_max8_vid006_2f",
    "router_baseline_strong_weighted_weak_direct_max8_vid006_2f",
}


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


def normalize_domain(value):
    text = str(value or "").strip()
    if not text:
        return ""
    if text in DOMAIN_BY_DATASET:
        return DOMAIN_BY_DATASET[text]
    return DOMAIN_ALIASES.get(text.lower(), text)


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


def parse_frame_routes(route_args):
    routes = []
    for item in route_args or []:
        if "=" not in item:
            raise ValueError(f"Frame route must use SUBSTRING=max_frames: {item}")
        selector, value = item.split("=", 1)
        selector = selector.strip()
        if not selector:
            raise ValueError(f"Empty frame route selector: {item}")
        routes.append((selector.lower(), int(value)))
    return routes


def apply_frame_routes(sample, max_frames, frame_routes):
    haystack = json.dumps(sample, ensure_ascii=False).lower()
    for selector, routed_frames in frame_routes:
        if selector in haystack:
            return routed_frames
    return max_frames


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


def strip_option_label(text):
    return re.sub(r"^\s*[ABCDabcd]\s*[\):\.\-]\s*", "", str(text)).strip()


def format_options(parsed_options, original_order, strip_labels=False):
    by_letter = dict(parsed_options)
    display_to_original = {}
    lines = []
    for display_letter, original_letter in zip(LETTERS, original_order):
        display_to_original[display_letter] = original_letter
        value = by_letter[original_letter]
        if strip_labels:
            value = strip_option_label(value)
        lines.append(f"{display_letter}) {value}")
    return "\n".join(lines), display_to_original


def extract_answer(text):
    upper = str(text).strip().upper()
    answer_matches = re.findall(r"(?:FINAL\s+)?ANSWER\s*[:：]\s*([ABCD])\b", upper)
    if answer_matches:
        return answer_matches[-1]
    last_line_matches = re.findall(r"(?m)^\s*([ABCD])\s*$", upper)
    if last_line_matches:
        return last_line_matches[-1]
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


def text_tokens(*parts):
    text = " ".join(str(part or "").lower() for part in parts)
    tokens = re.findall(r"[a-z0-9]+", text)
    return {token for token in tokens if len(token) >= 3 and token not in STOPWORDS}


def options_text(options):
    if isinstance(options, dict):
        return " ".join(str(options.get(letter) or options.get(letter.lower()) or "") for letter in LETTERS)
    if isinstance(options, list):
        values = []
        for item in options:
            if isinstance(item, dict):
                values.append(str(item.get("text") or item.get("answer") or item.get("option") or item))
            else:
                values.append(str(item))
        return " ".join(values)
    return str(options or "")


def normalize_question_type(value, question_text=""):
    raw = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    raw = re.sub(r"\s+", " ", raw)
    if raw in QUESTION_TYPE_ALIASES:
        return QUESTION_TYPE_ALIASES[raw]

    text = (raw + " " + str(question_text or "").lower()).strip()
    if "how many" in text or "number of" in text or "distinct types" in text:
        return "counting"
    if "not visible" in text or "not shown" in text or "not seen" in text:
        return "not_visible"
    if "which region" in text or "located" in text or "where" in text:
        return "region_localization"
    if "at what approximate time" in text or "timestamp" in text or "first interact" in text:
        return "temporal_localization"
    if "next direction" in text or "direction of movement" in text:
        return "next_direction"
    if "next type of interaction" in text or "predicted next" in text:
        return "next_interaction"
    if "immediately follows" in text or "next phase" in text or "key action that will begin" in text:
        return "phase_sequence"
    if "sequence of actions" in text:
        return "action_sequence"
    if "what action is being performed" in text:
        return "action_identification"
    if "extreme sport" in text:
        return "sport_identification"
    if "type of animal" in text:
        return "animal_identification"
    if "object is the cat interacting" in text:
        return "object_interaction"
    if "which tool" in text or "surgical instrument" in text:
        return "tool_interaction"
    return "unknown"


def question_type_group(question_type):
    return QUESTION_TYPE_GROUPS.get(question_type, "unknown")


def extract_question_and_options(user_text):
    clean = str(user_text or "").replace("<image>", "")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    option_start = None
    for idx, line in enumerate(lines):
        if re.match(r"^[ABCDabcd]\s*[\):\.\-]", line):
            option_start = idx
            break

    if option_start is None:
        return clean.strip(), ""

    question = " ".join(lines[:option_start]).strip()
    options = lines[option_start:option_start + 4]
    return question, options


def extract_strict_answer(text):
    match = re.search(r"\b([ABCD])\b", str(text or "").strip().upper())
    return match.group(1) if match else None


def clean_reasoning_text(text, max_words=24):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^(descriptions? of (the )?pictures?|analysis of options|reasoning consistency)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\b(final\s+)?answer\s*[:：].*$", "", text, flags=re.I).strip()
    if not text:
        return ""

    sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words]).rstrip(" ,;:") + "."
    return sentence.rstrip()


def section_text(text, label):
    pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?:\n\s*\n|$)"
    match = re.search(pattern, str(text or ""), flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def enhanced_answer_line(text, answer):
    pattern = rf"(?im)^\s*{re.escape(answer)}\s*[\):\.\-]?\s*(.+)$"
    matches = re.findall(pattern, str(text or ""))
    return matches[-1].strip() if matches else ""


def compress_enhanced_reasoning(text, answer, question_type):
    candidates = [
        enhanced_answer_line(text, answer),
        section_text(text, "Reasoning Consistency"),
        section_text(text, "Descriptions of the pictures"),
        section_text(text, "Analysis of Options"),
    ]
    for candidate in candidates:
        cleaned = clean_reasoning_text(candidate)
        if cleaned:
            return cleaned
    return EVIDENCE_BY_TYPE.get(question_type, EVIDENCE_BY_TYPE["unknown"])


def resolve_support_frame_path(support_dir, frame_path):
    raw = str(frame_path).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return str(path)

    rel = raw.lstrip("/")
    candidates = [support_dir / rel]
    marker = "/frames/"
    if marker in "/" + rel:
        rel_from_frames = ("frames/" + rel.split(marker, 1)[1]).lstrip("/")
        candidates.append(support_dir / rel_from_frames)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Cannot resolve support frame path: {frame_path}. Tried: "
        + "; ".join(str(candidate) for candidate in candidates)
    )


def infer_domain_from_images(images):
    joined = " ".join(str(item).replace("\\", "/") for item in images)
    for dataset, domain in DOMAIN_BY_DATASET.items():
        if f"/{dataset}/" in f"/{joined}/":
            return domain
    return ""


def load_enhanced_reasoning_by_domain(support_dir, use_enhanced):
    if not use_enhanced:
        return {domain: {} for domain in SUPPORT_FILES}

    support_dir = Path(support_dir)
    enhanced = {domain: {} for domain in SUPPORT_FILES}
    for domain, file_name in ENHANCED_SUPPORT_FILES.items():
        path = support_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Few-shot enhanced support file not found: {path}")
        rows = load_rows(path)
        for idx, row in enumerate(rows, start=1):
            messages = row.get("messages") or []
            assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), {})
            enhanced[domain][idx] = assistant_msg.get("content", "")
    return enhanced


def build_support_index(support_dir, use_enhanced=False):
    support_dir = Path(support_dir)
    if not support_dir.exists():
        raise FileNotFoundError(f"Few-shot support dir not found: {support_dir}")

    by_domain = {domain: [] for domain in SUPPORT_FILES}
    enhanced_by_domain = load_enhanced_reasoning_by_domain(support_dir, use_enhanced)
    for domain, file_name in SUPPORT_FILES.items():
        path = support_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Few-shot support file not found: {path}")

        rows = load_rows(path)
        for idx, row in enumerate(rows, start=1):
            messages = row.get("messages") or []
            user_msg = next((msg for msg in messages if msg.get("role") == "user"), {})
            assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), {})
            question, options = extract_question_and_options(user_msg.get("content", ""))
            answer = extract_strict_answer(assistant_msg.get("content", ""))
            images = row.get("images") or []
            row_domain = normalize_domain(row.get("domain")) or infer_domain_from_images(images) or domain
            row_domain = normalize_domain(row_domain)
            if row_domain != domain:
                row_domain = domain
            if answer not in VALID_ANSWERS:
                raise ValueError(f"Invalid support answer in {path} row {idx}: {assistant_msg.get('content')!r}")
            if not images:
                raise ValueError(f"Missing support images in {path} row {idx}")

            resolved_images = [resolve_support_frame_path(support_dir, image) for image in images]
            question_type = normalize_question_type("", question)
            if options:
                parsed_options = parse_options(options)
                option_lines, _ = format_options(parsed_options, LETTERS, strip_labels=True)
            else:
                option_lines = ""
            enhanced_text = enhanced_by_domain.get(domain, {}).get(idx, "")
            reasoning = compress_enhanced_reasoning(enhanced_text, answer, question_type) if enhanced_text else ""
            example = {
                "support_id": f"{domain}:{idx}",
                "domain": domain,
                "question_type": question_type,
                "question_type_group": question_type_group(question_type),
                "question": question,
                "options": options,
                "option_lines": option_lines,
                "answer": answer,
                "reasoning": reasoning,
                "has_enhanced_reasoning": bool(reasoning),
                "images": resolved_images,
                "num_images": len(resolved_images),
                "tokens": text_tokens(question, options_text(options)),
            }
            by_domain[domain].append(example)

    return by_domain


def fewshot_target_metadata(sample):
    parsed_options = parse_options(sample.get("options", ""))
    option_lines, _ = format_options(parsed_options, LETTERS, strip_labels=True)
    question = str(sample.get("question_text", "") or "")
    question_type = normalize_question_type(sample.get("question_type", ""), question)
    domain = normalize_domain(DOMAIN_BY_DATASET.get(sample.get("dataset", ""), sample.get("dataset", "")))
    return {
        "domain": domain,
        "question": question,
        "question_type": question_type,
        "question_type_group": question_type_group(question_type),
        "option_lines": option_lines,
        "tokens": text_tokens(question, options_text(sample.get("options", ""))),
    }


def rank_fewshot_examples(support_index, sample, k):
    target = fewshot_target_metadata(sample)
    candidates = support_index.get(target["domain"], [])
    scored = []
    for candidate in candidates:
        overlap = len(target["tokens"] & candidate["tokens"])
        score = overlap
        if candidate["question_type"] == target["question_type"]:
            score += 100
        elif candidate["question_type_group"] == target["question_type_group"]:
            score += 35
        score -= min(candidate["num_images"], 200) / 1000.0
        scored.append((score, candidate["support_id"], candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:max(k, 0)]], target


def image_content_from_paths(frame_paths, max_frames):
    content = []
    for frame_path in sample_frames(frame_paths, max_frames):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame_path)}"},
        })
    return content


def fewshot_evidence(question_type, mode):
    if mode == "none":
        return ""
    return EVIDENCE_BY_TYPE.get(question_type, EVIDENCE_BY_TYPE["unknown"])


def example_reasoning(example, evidence_mode, output_format):
    if output_format == "short_reason_answer":
        return example.get("reasoning") or fewshot_evidence(example["question_type"], "compressed")
    return fewshot_evidence(example["question_type"], evidence_mode)


def build_fewshot_prompt(target, output_format):
    if output_format == "short_reason_answer":
        return (
            "Now answer the target question in the same format as the examples.\n"
            f"Domain: {target['domain']}. Type: {target['question_type']}.\n"
            "Use exactly two lines. Keep the reasoning to one short visual sentence.\n"
            "The second line must be exactly: Answer: X, where X is A, B, C, or D.\n\n"
            f"Question: {target['question']}\n"
            f"{target['option_lines']}\n"
            "Reasoning:"
        )

    return (
        "Now answer the target question.\n"
        f"Domain: {target['domain']}. Type: {target['question_type']}.\n"
        "Return exactly one character: A, B, C, or D.\n"
        "No explanation. No punctuation.\n\n"
        f"Question: {target['question']}\n"
        f"{target['option_lines']}\n"
        "Answer:"
    )


def build_fewshot_content(
    testbed_dir,
    sample,
    max_frames,
    support_index,
    fewshot_k,
    fewshot_frames,
    evidence_mode,
    output_format,
):
    examples, target = rank_fewshot_examples(support_index, sample, fewshot_k)
    if output_format == "short_reason_answer":
        guide_text = (
            "Use the short examples as answer-style guides for this egocentric video QA task. "
            "Each example has its own frames, a one-sentence visual reasoning line, and the final answer. "
            "For the target, follow the same two-line format."
        )
    else:
        guide_text = (
            "Use the short examples as answer-style guides for this egocentric video QA task. "
            "Each example has its own frames, question, and correct answer. "
            "For the target, output only A, B, C, or D."
        )
    content = [{
        "type": "text",
        "text": guide_text,
    }]
    selected = []

    for ex_idx, example in enumerate(examples, start=1):
        used_images = sample_frames(example["images"], fewshot_frames)
        content.extend(image_content_from_paths(used_images, None))
        reasoning = example_reasoning(example, evidence_mode, output_format)
        if output_format == "short_reason_answer":
            answer_block = f"Reasoning: {reasoning}\nAnswer: {example['answer']}"
        else:
            evidence_text = f" Evidence: {reasoning}" if reasoning else ""
            answer_block = f"{evidence_text}\nAnswer: {example['answer']}"
        content.append({
            "type": "text",
            "text": (
                f"Example {ex_idx}. Domain: {example['domain']}. "
                f"Type: {example['question_type']}.\n"
                f"Question: {example['question']}\n"
                f"{example['option_lines']}\n"
                f"{answer_block}"
            ),
        })
        selected.append({
            "support_id": example["support_id"],
            "domain": example["domain"],
            "question_type": example["question_type"],
            "question": example["question"],
            "answer": example["answer"],
            "reasoning": reasoning,
            "has_enhanced_reasoning": example.get("has_enhanced_reasoning", False),
            "num_images_total": example["num_images"],
            "num_images_used": len(used_images),
            "images_used": used_images,
        })

    content.extend(build_image_content(testbed_dir, sample, max_frames))
    prompt = build_fewshot_prompt(target, output_format)
    content.append({"type": "text", "text": prompt})
    return content, prompt, selected, target


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


def is_context_length_error(exc):
    text = str(exc).lower()
    return "exceeds model" in text and "maximum context length" in text


def frame_retry_sequence(max_frames):
    start = max_frames or 8
    candidates = [start, 12, 8, 6, 4, 2, 1]
    sequence = []
    for value in candidates:
        if value <= start and value not in sequence:
            sequence.append(value)
    return sequence


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


def infer_fewshot_once(
    client,
    model,
    testbed_dir,
    sample,
    max_frames,
    max_tokens,
    support_index,
    fewshot_k,
    fewshot_frames,
    evidence_mode,
    output_format,
):
    content, prompt, selected, target = build_fewshot_content(
        testbed_dir,
        sample,
        max_frames,
        support_index,
        fewshot_k,
        fewshot_frames,
        evidence_mode,
        output_format,
    )
    raw_output = call_model(client, model, content, max_tokens)
    answer = extract_answer(raw_output)
    return answer, raw_output, prompt, selected, target


def infer_with_frame_retry(client, model, testbed_dir, sample, prompt_mode, max_frames, max_tokens):
    attempts = []
    for attempt_frames in frame_retry_sequence(max_frames):
        try:
            answer, raw_output, prompt = infer_once(
                client,
                model,
                testbed_dir,
                sample,
                prompt_mode,
                attempt_frames,
                max_tokens,
                ["A", "B", "C", "D"],
            )
            attempts.append({
                "max_frames": attempt_frames,
                "status": "ok" if answer is not None else "parse_failed",
                "answer": answer,
                "raw_output": raw_output,
                "prompt": prompt,
            })
            status = "ok" if answer is not None else "parse_failed"
            if attempt_frames != max_frames:
                status = f"{status}_after_frame_retry"
            return answer, raw_output, prompt, attempt_frames, attempts, status
        except Exception as exc:
            attempts.append({
                "max_frames": attempt_frames,
                "status": "context_length_error" if is_context_length_error(exc) else "error",
                "error": str(exc),
            })
            if not is_context_length_error(exc):
                raise

    raise RuntimeError("All frame retry attempts exceeded context length.")


def infer_fewshot_with_frame_retry(
    client,
    model,
    testbed_dir,
    sample,
    max_frames,
    max_tokens,
    support_index,
    fewshot_k,
    fewshot_frames,
    evidence_mode,
    output_format,
):
    attempts = []
    for attempt_frames in frame_retry_sequence(max_frames):
        try:
            answer, raw_output, prompt, selected, target = infer_fewshot_once(
                client,
                model,
                testbed_dir,
                sample,
                attempt_frames,
                max_tokens,
                support_index,
                fewshot_k,
                fewshot_frames,
                evidence_mode,
                output_format,
            )
            attempts.append({
                "max_frames": attempt_frames,
                "status": "ok" if answer is not None else "parse_failed",
                "answer": answer,
                "raw_output": raw_output,
                "prompt": prompt,
                "target_question_type": target["question_type"],
                "fewshot_examples": selected,
            })
            status = "ok" if answer is not None else "parse_failed"
            if attempt_frames != max_frames:
                status = f"{status}_after_frame_retry"
            return answer, raw_output, prompt, attempt_frames, attempts, status, selected, target
        except Exception as exc:
            attempts.append({
                "max_frames": attempt_frames,
                "status": "context_length_error" if is_context_length_error(exc) else "error",
                "error": str(exc),
            })
            if not is_context_length_error(exc):
                raise

    raise RuntimeError("All frame retry attempts exceeded context length.")


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


def vote_with_frame_retry(client, model, testbed_dir, sample, prompt_mode, max_frames, max_tokens):
    frame_attempts = []
    for attempt_frames in frame_retry_sequence(max_frames):
        try:
            answer, attempts, status = vote_infer(
                client, model, testbed_dir, sample, prompt_mode, attempt_frames, max_tokens
            )
            frame_attempts.append({
                "max_frames": attempt_frames,
                "status": status,
                "attempts": attempts,
            })
            if attempt_frames != max_frames:
                status = f"{status}_after_frame_retry"
            return answer, frame_attempts, status, attempt_frames
        except Exception as exc:
            frame_attempts.append({
                "max_frames": attempt_frames,
                "status": "context_length_error" if is_context_length_error(exc) else "error",
                "error": str(exc),
            })
            if not is_context_length_error(exc):
                raise

    raise RuntimeError("All frame retry attempts exceeded context length.")


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
    fewshot_count = sum(1 for row in raw_outputs if row.get("fewshot"))
    if fewshot_count:
        type_dist = Counter(
            row.get("target_question_type") or "unknown"
            for row in raw_outputs
            if row.get("fewshot")
        )
        lines.append(f"fewshot_processed: {fewshot_count}")
        lines.append("fewshot_question_type_dist: " + " ".join(f"{key}={value}" for key, value in sorted(type_dist.items())))

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


def support_index_summary(support_index):
    if not support_index:
        return {}
    return {domain: len(rows) for domain, rows in sorted(support_index.items())}


def write_fewshot_config(output_dir, args, support_index):
    config = {
        "fewshot_support_dir": args.fewshot_support_dir,
        "fewshot_k": args.fewshot_k,
        "fewshot_frames": args.fewshot_frames,
        "fewshot_domains": args.fewshot_domains,
        "fewshot_evidence_mode": args.fewshot_evidence_mode,
        "fewshot_output_format": args.fewshot_output_format,
        "fewshot_use_enhanced": args.fewshot_use_enhanced,
        "support_index_counts": support_index_summary(support_index),
        "notes": [
            "Few-shot mode is enabled only for samples matching fewshot_domains.",
            "Support examples are sampled to fewshot_frames images each.",
            "Target samples still use default-max-frames plus frame-route overrides.",
        ],
    }
    with open(output_dir / "fewshot_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


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
    parser.add_argument(
        "--frame-route",
        action="append",
        default=[],
        help="Question-id/dataset substring max-frame override, e.g. ExtrameSportFPV_VID006=2.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--only-datasets", default=None)
    parser.add_argument("--fallback-submission", default=None)
    parser.add_argument("--vote-domains", default=None, help="Comma-separated domains/datasets using 3-way option-order voting.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N selected samples; requires fallback for full output.")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--fewshot-support-dir", default=None, help="Directory containing train_animal/train_surgery/train_industry/train_xsports JSON files.")
    parser.add_argument("--fewshot-k", type=int, default=0, help="Number of same-domain few-shot examples to prepend. Default disables few-shot.")
    parser.add_argument("--fewshot-frames", type=int, default=2, help="Frames per few-shot support example.")
    parser.add_argument("--fewshot-domains", default="Industry,XSports", help="Comma-separated datasets/domains that use few-shot examples.")
    parser.add_argument("--fewshot-evidence-mode", choices=["none", "compressed"], default="compressed")
    parser.add_argument("--fewshot-output-format", choices=["direct", "short_reason_answer"], default="direct")
    parser.add_argument("--fewshot-use-enhanced", action="store_true", help="Use train_*_enhanced.json to distill one short reasoning sentence for support examples.")
    parser.add_argument("--fewshot-output-dir-suffix", default="", help="Optional suffix appended to --output-dir for safer experiment naming.")
    args = parser.parse_args()

    output_dir = Path(str(args.output_dir) + args.fewshot_output_dir_suffix)
    if output_dir.name in PROTECTED_OUTPUT_DIR_NAMES:
        raise SystemExit(f"Refusing to write protected EgoCross output directory: {output_dir}")
    fewshot_enabled = args.fewshot_k > 0
    if fewshot_enabled and output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Few-shot runs must use a fresh output directory, already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    testbed_dir = Path(args.testbed_dir)
    input_json = Path(args.input_json) if args.input_json else testbed_dir / "egocross_testbed_imgs.json"

    samples = load_rows(input_json)
    template = load_rows(args.template) if args.template else None
    fallback_rows = load_rows(args.fallback_submission)
    fallback_index = index_rows(fallback_rows) if fallback_rows else None
    selectors = parse_selectors(args.only_datasets)
    vote_selectors = parse_selectors(args.vote_domains)
    fewshot_selectors = parse_selectors(args.fewshot_domains)
    choose_route = parse_routes(args.route, args.default_prompt_mode, args.default_max_frames)
    frame_routes = parse_frame_routes(args.frame_route)
    support_index = None

    if fewshot_enabled:
        if not args.fewshot_support_dir:
            raise SystemExit("--fewshot-k requires --fewshot-support-dir")
        if args.fewshot_frames < 1:
            raise SystemExit("--fewshot-frames must be >= 1")
        if args.fewshot_output_format == "short_reason_answer" and args.max_tokens == 8:
            args.max_tokens = 64
        support_index = build_support_index(args.fewshot_support_dir, use_enhanced=args.fewshot_use_enhanced)
        write_fewshot_config(output_dir, args, support_index)

    if (selectors or args.limit) and not fallback_index:
        raise SystemExit("--only-datasets and --limit require --fallback-submission to keep a full valid output.")

    client = None
    outputs = []
    raw_outputs = []
    selected_examples_log = []
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

        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=args.base_url, api_key="dummy")

        prompt_mode, max_frames = choose_route(dataset)
        max_frames = apply_frame_routes(sample, max_frames, frame_routes)
        use_vote = bool(vote_selectors) and selector_matches(dataset, vote_selectors)
        use_fewshot = bool(support_index) and selector_matches(dataset, fewshot_selectors)
        if use_fewshot:
            use_vote = False
        status = "ok"
        raw_payload = None
        prompt = None
        used_max_frames = max_frames
        frame_attempts = []
        fewshot_examples = []
        target_question_type = None

        try:
            if use_fewshot:
                answer, raw_payload, prompt, used_max_frames, frame_attempts, status, fewshot_examples, target = (
                    infer_fewshot_with_frame_retry(
                        client,
                        args.model,
                        testbed_dir,
                        sample,
                        max_frames,
                        args.max_tokens,
                        support_index,
                        args.fewshot_k,
                        args.fewshot_frames,
                        args.fewshot_evidence_mode,
                        args.fewshot_output_format,
                    )
                )
                target_question_type = target["question_type"]
                selected_examples_log.append({
                    "id": sample.get("id", idx),
                    "question_id": sample.get("question_id", ""),
                    "dataset": dataset,
                    "target_question_type": target_question_type,
                    "fewshot_examples": fewshot_examples,
                })
            elif use_vote:
                answer, raw_payload, status, used_max_frames = vote_with_frame_retry(
                    client, args.model, testbed_dir, sample, prompt_mode, max_frames, args.max_tokens
                )
                frame_attempts = raw_payload
            else:
                answer, raw_payload, prompt, used_max_frames, frame_attempts, status = infer_with_frame_retry(
                    client,
                    args.model,
                    testbed_dir,
                    sample,
                    prompt_mode,
                    max_frames,
                    args.max_tokens,
                )
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
            "used_max_frames": used_max_frames,
            "frame_attempts": frame_attempts,
            "vote": use_vote,
            "fewshot": use_fewshot,
            "fewshot_k": args.fewshot_k if use_fewshot else 0,
            "fewshot_frames": args.fewshot_frames if use_fewshot else 0,
            "fewshot_output_format": args.fewshot_output_format if use_fewshot else None,
            "fewshot_use_enhanced": args.fewshot_use_enhanced if use_fewshot else False,
            "target_question_type": target_question_type,
            "fewshot_examples": fewshot_examples,
            "prompt": prompt,
            "raw_output": raw_payload,
        })
        processed += 1
        print(
            f"[{idx}/{len(samples)}] {row.get('question_id', '')} -> {answer} "
            f"| {'fewshot' if use_fewshot else prompt_mode} max{used_max_frames}/{max_frames} status={status}",
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
    if fewshot_enabled:
        with open(output_dir / "fewshot_selected_examples.json", "w", encoding="utf-8") as f:
            json.dump(selected_examples_log, f, indent=2, ensure_ascii=False)

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
