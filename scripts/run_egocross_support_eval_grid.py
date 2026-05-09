#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_CANDIDATES = [
    {
        "name": "direct_tail_dense_max8_vid006_4f",
        "prompt_mode": "direct",
        "max_frames": 8,
        "frame_sampling": "tail_dense",
        "frame_routes": ["VID006=4"],
    },
    {
        "name": "direct_tail_dense_max12_vid006_4f",
        "prompt_mode": "direct",
        "max_frames": 12,
        "frame_sampling": "tail_dense",
        "frame_routes": ["VID006=4"],
    },
    {
        "name": "direct_tail_dense_max16_vid006_4f",
        "prompt_mode": "direct",
        "max_frames": 16,
        "frame_sampling": "tail_dense",
        "frame_routes": ["VID006=4"],
    },
    {
        "name": "direct_endpoint_max12_vid006_4f",
        "prompt_mode": "direct",
        "max_frames": 12,
        "frame_sampling": "endpoint",
        "frame_routes": ["VID006=4"],
    },
    {
        "name": "direct_uniform_max12_vid006_4f",
        "prompt_mode": "direct",
        "max_frames": 12,
        "frame_sampling": "uniform",
        "frame_routes": ["VID006=4"],
    },
    {
        "name": "strict_direct_tail_dense_max12_vid006_4f",
        "prompt_mode": "strict_direct",
        "max_frames": 12,
        "frame_sampling": "tail_dense",
        "frame_routes": ["VID006=4"],
    },
]


SUMMARY_COLUMNS = [
    "name",
    "run_status",
    "overall",
    "answer_only",
    "Animal",
    "Surgery",
    "Industry",
    "XSports",
    "status_dist",
    "used_max_frames_dist",
    "output_dir",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small EgoCross support-set inference strategy grid.")
    parser.add_argument("--support-dir", default="/share/home/group9/data/egocross")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="egocross")
    parser.add_argument("--out-root", default="egocross_outputs/support_eval")
    parser.add_argument("--log-dir", default="logs/support_eval")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-domains", default=None)
    parser.add_argument("--candidates", default=None, help="Comma-separated candidate names. Default runs all presets.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Do not overwrite; create a timestamped output dir when needed.")
    return parser.parse_args()


def selected_candidates(names):
    if not names:
        return DEFAULT_CANDIDATES
    wanted = {item.strip() for item in names.split(",") if item.strip()}
    known = {candidate["name"] for candidate in DEFAULT_CANDIDATES}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f"Unknown candidates: {', '.join(unknown)}\nKnown candidates: {', '.join(sorted(known))}")
    return [candidate for candidate in DEFAULT_CANDIDATES if candidate["name"] in wanted]


def is_non_empty_dir(path):
    return path.is_dir() and any(path.iterdir())


def build_command(args, candidate, output_dir):
    cmd = [
        sys.executable,
        "scripts/egocross_support_eval.py",
        "--support-dir",
        args.support_dir,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--prompt-mode",
        candidate["prompt_mode"],
        "--max-frames",
        str(candidate["max_frames"]),
        "--frame-sampling",
        candidate["frame_sampling"],
        "--max-tokens",
        str(args.max_tokens),
        "--output-dir",
        str(output_dir),
    ]
    for route in candidate.get("frame_routes", []):
        cmd.extend(["--frame-route", route])
    if args.only_domains:
        cmd.extend(["--only-domains", args.only_domains])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def stream_command(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def read_metrics(output_dir):
    path = output_dir / "support_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def domain_acc(metrics, domain):
    if not metrics:
        return ""
    return metrics.get("by_domain", {}).get(domain, {}).get("acc", "")


def summary_row(name, run_status, output_dir):
    metrics = read_metrics(output_dir)
    overall = metrics.get("overall", {}) if metrics else {}
    return {
        "name": name,
        "run_status": run_status,
        "overall": overall.get("acc", ""),
        "answer_only": overall.get("answer_only_format_rate", ""),
        "Animal": domain_acc(metrics, "Animal"),
        "Surgery": domain_acc(metrics, "Surgery"),
        "Industry": domain_acc(metrics, "Industry"),
        "XSports": domain_acc(metrics, "XSports"),
        "status_dist": json.dumps(metrics.get("status_dist", {}) if metrics else {}, ensure_ascii=False, sort_keys=True),
        "used_max_frames_dist": json.dumps(
            metrics.get("used_max_frames_dist", {}) if metrics else {}, ensure_ascii=False, sort_keys=True
        ),
        "output_dir": str(output_dir),
    }


def write_summary(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(SUMMARY_COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(column, "")) for column in SUMMARY_COLUMNS) + "\n")


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    log_dir = Path(args.log_dir)
    summary_path = Path(args.summary) if args.summary else out_root / "_grid_summary.tsv"
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for candidate in selected_candidates(args.candidates):
        name = candidate["name"]
        output_dir = out_root / name
        log_path = log_dir / f"{name}.log"
        run_status = "ok"

        if (output_dir / "support_metrics.json").exists() and not args.rerun:
            print(f"[SKIP] {name}: existing support_metrics.json")
            summary_rows.append(summary_row(name, "skipped_existing_metrics", output_dir))
            write_summary(summary_path, summary_rows)
            continue

        if is_non_empty_dir(output_dir):
            if args.rerun:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = out_root / f"{name}_{stamp}"
                log_path = log_dir / f"{name}_{stamp}.log"
            else:
                print(f"[SKIP] {name}: non-empty output dir without metrics: {output_dir}")
                summary_rows.append(summary_row(name, "skipped_non_empty_dir", output_dir))
                write_summary(summary_path, summary_rows)
                continue

        cmd = build_command(args, candidate, output_dir)
        print()
        print(f"[RUN] {name}")
        print("      " + " ".join(cmd))

        if args.dry_run:
            summary_rows.append(summary_row(name, "dry_run", output_dir))
            write_summary(summary_path, summary_rows)
            continue

        status = stream_command(cmd, log_path)
        if status != 0:
            run_status = f"failed_{status}"
            print(f"[FAIL] {name}: exit {status}, see {log_path}")

        summary_rows.append(summary_row(name, run_status, output_dir))
        write_summary(summary_path, summary_rows)

    print()
    print(f"[DONE] summary: {summary_path}")
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
