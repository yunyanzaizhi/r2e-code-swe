#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


ENV_PATTERNS = [
    ("code_swe", re.compile(r"code[_-]?swe|SWE-Bench|R2E-Gym", re.I)),
    ("programming", re.compile(r"env\.env_name=Programming|env_name['\"]?:\s*['\"]?Programming|quixbugs|programming_history", re.I)),
    ("sokoban", re.compile(r"env\.env_name=Sokoban|sokoban|recent_window|structured_summary|full_history|long_horizon|history_strategy|master_run|wait_and_run|watchdog", re.I)),
    ("alfworld", re.compile(r"env\.env_name=alfworld|AlfredTWEnv|alfworld", re.I)),
    ("webshop", re.compile(r"env\.env_name=Webshop|webshop", re.I)),
]

KNOWN_ENVS = {env_name for env_name, _ in ENV_PATTERNS} | {"unknown_env", "mixed_env"}


def slug(value: str) -> str:
    value = (value or "unknown").lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value).strip("_")
    return value or "unknown"


def read_sample(path: Path, max_chars: int = 250_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def classify_env(path: Path, text: str) -> str:
    haystack = f"{path.as_posix()}\n{text}"
    explicit_patterns = [
        r"env\.env_name=([A-Za-z0-9_./-]+)",
        r"['\"]env_name['\"]\s*:\s*['\"]([^'\"]+)['\"]",
    ]
    for pattern in explicit_patterns:
        explicit = re.search(pattern, haystack)
        if not explicit:
            continue
        raw = explicit.group(1).strip("'\"")
        if "alfworld" in raw.lower():
            return "alfworld"
        if "sokoban" in raw.lower():
            return "sokoban"
        if "programming" in raw.lower():
            return "programming"
        if "webshop" in raw.lower():
            return "webshop"
        if "code_swe" in raw.lower():
            return "code_swe"
        return slug(raw)
    for env_name, pattern in ENV_PATTERNS:
        if pattern.search(haystack):
            return env_name
    return "unknown_env"


def classify_gpu(path: Path, text: str) -> str:
    haystack = f"{path.name}\n{text}"
    match = re.search(r"trainer\.n_gpus_per_node=([0-9]+)", haystack)
    if not match:
        match = re.search(r"['\"]?n_gpus_per_node['\"]?\s*[:=]\s*([0-9]+)", haystack)
    if not match:
        match = re.search(r"([0-9]+)\s*gpu", haystack, flags=re.I)
    if not match:
        visible = re.search(r"CUDA_VISIBLE_DEVICES=([0-9,]+)", haystack)
        if visible:
            count = len([x for x in visible.group(1).split(",") if x.strip()])
            match = re.match(r"([0-9]+)", str(count))
    if not match:
        return "unknown_gpu"
    count = int(match.group(1))
    if count == 1:
        return "single_gpu"
    if count == 2:
        return "dual_gpu"
    return f"multi_gpu_{count}"


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 2
    while True:
        candidate = parent / f"{stem}__{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def should_skip(path: Path, log_roots: list[Path], reclassify_existing: bool = False) -> bool:
    parts = set(path.parts)
    if "_manifest" in parts:
        return True
    if reclassify_existing:
        return not any(root in path.parents or path == root for root in log_roots)
    for env_name, _ in ENV_PATTERNS:
        if env_name in parts:
            return True
    if any(bucket in parts for bucket in ("single_gpu", "dual_gpu", "unknown_gpu")):
        return True
    if any(part.startswith("multi_gpu_") for part in path.parts):
        return True
    return not any(root in path.parents or path == root for root in log_roots)


def target_filename(path: Path, scan_root: Path, canonical_root: Path) -> str:
    if canonical_root in path.parents and len(path.parents) >= 3:
        try:
            rel = path.relative_to(canonical_root)
        except ValueError:
            rel = Path()
        if len(rel.parts) >= 3 and rel.parts[0] in KNOWN_ENVS:
            return slug(path.name)
    relative_hint = "__".join(path.relative_to(scan_root).parts)
    target_name = slug(relative_hint)
    if not target_name.endswith(".log"):
        target_name += ".log"
    return target_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize experiment logs by environment and GPU count.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--log-root", default="experiments/logs", help="Canonical output log root.")
    parser.add_argument(
        "--include-root",
        action="append",
        default=["experiments/logs", "experiments/experiments/logs"],
        help="Existing log root to scan. Can be provided multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reclassify-existing",
        action="store_true",
        help="Also scan already-organized logs and move them if env/GPU classification changed.",
    )
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    canonical_root = (project_root / args.log_root).resolve()
    include_roots = [(project_root / item).resolve() for item in args.include_root]
    manifest_dir = canonical_root / "_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"log_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    moves = []
    seen = set()
    for scan_root in include_roots:
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.log")):
            path = path.resolve()
            if path in seen or should_skip(path, include_roots, args.reclassify_existing):
                continue
            seen.add(path)
            text = read_sample(path)
            env_name = classify_env(path, text)
            gpu_bucket = classify_gpu(path, text)
            target_name = target_filename(path, scan_root, canonical_root)
            target = canonical_root / env_name / gpu_bucket / target_name
            if target.resolve() == path:
                continue
            target = unique_target(target)
            moves.append({"src": str(path), "dst": str(target), "env": env_name, "gpu": gpu_bucket})

    if not args.dry_run:
        for move in moves:
            src = Path(move["src"])
            dst = Path(move["dst"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        manifest_path.write_text(json.dumps({"moves": moves}, indent=2, ensure_ascii=False), encoding="utf-8")

    for move in moves:
        prefix = "DRY" if args.dry_run else "MOVE"
        print(f"{prefix}\t{move['env']}\t{move['gpu']}\t{move['src']}\t{move['dst']}")
    print(f"manifest\t{manifest_path}")
    print(f"count\t{len(moves)}")


if __name__ == "__main__":
    main()
