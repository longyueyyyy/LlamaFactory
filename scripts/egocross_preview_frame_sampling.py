#!/usr/bin/env python3
import argparse
from pathlib import Path

import egocross_support_eval as support_eval
import egocross_router_infer as router


def frame_label(path):
    raw = str(path).replace("\\", "/")
    parts = raw.split("/")
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return raw


def parse_args():
    parser = argparse.ArgumentParser(description="Preview EgoCross frame sampling choices without calling the model.")
    parser.add_argument("--support-dir", default="/share/home/group9/data/egocross")
    parser.add_argument("--eval-json", default=None)
    parser.add_argument("--sampling", default="tail_dense,query_diverse,query_diverse_tail")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--frame-route", action="append", default=[])
    parser.add_argument("--only-domains", default=None)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = (
        support_eval.load_eval_samples(args.eval_json, args.only_domains)
        if args.eval_json
        else support_eval.load_support_samples(args.support_dir, args.only_domains)
    )
    if args.limit:
        samples = samples[: args.limit]
    frame_routes = router.parse_frame_routes(args.frame_route)
    samplings = [item.strip() for item in args.sampling.split(",") if item.strip()]

    for sample in samples:
        max_frames = router.apply_frame_routes(sample, args.max_frames, frame_routes)
        print(
            f"{sample['id']}\t{sample['domain']}\t{sample['dataset']}\t"
            f"{sample.get('video_id', '')}\t{sample['question_type']}\tframes={len(sample.get('video_path', []))}\tmax={max_frames}"
        )
        for sampling in samplings:
            chosen = router.sample_frames(sample.get("video_path", []), max_frames, sampling, sample=sample)
            labels = ", ".join(frame_label(path) for path in chosen)
            print(f"  {sampling}: {labels}")


if __name__ == "__main__":
    main()
