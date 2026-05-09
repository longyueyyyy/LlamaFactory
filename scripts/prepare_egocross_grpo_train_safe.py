#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_SOURCE = "/share/home/group9/why/rl_grpo_v2/output/egocross_grpo_answer_v7/v3-20260507-113212"
DEFAULT_OUTPUT = "saves/egocross/qwen3vl4b/grpo_v3_train_rope_default"


def link_or_copy(source, target, copy_files):
    if target.exists() or target.is_symlink():
        return
    if copy_files:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    else:
        os.symlink(source, target, target_is_directory=source.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Create a train-safe EgoCross GRPO model view for transformers.")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files instead of symlinking. Symlink is preferred to avoid duplicating model weights.",
    )
    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Rewrite output config.json if it already exists.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise SystemExit(f"Source model does not exist: {source}")
    if not (source / "config.json").exists():
        raise SystemExit(f"Source config.json does not exist: {source / 'config.json'}")

    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "config.json":
            continue
        link_or_copy(item, output / item.name, args.copy_files)

    output_config = output / "config.json"
    if output_config.exists() and not args.overwrite_config:
        with open(output_config, encoding="utf-8") as f:
            existing = json.load(f)
        rope = existing.get("text_config", {}).get("rope_scaling")
        if rope == {"rope_type": "default"}:
            print(f"Train-safe config already exists: {output_config}")
            return
        raise SystemExit(
            f"Refusing to overwrite existing config with rope_scaling={rope!r}. "
            "Pass --overwrite-config to rewrite it."
        )

    with open(source / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    config.setdefault("text_config", {})
    config["text_config"]["rope_scaling"] = {"rope_type": "default"}

    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote train-safe model view: {output}")
    print("text_config.rope_scaling =", config["text_config"]["rope_scaling"])


if __name__ == "__main__":
    main()
