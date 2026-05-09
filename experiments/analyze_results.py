#!/usr/bin/env python3
"""Analyze Sokoban history-management experiment logs."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
LOG_ROOT = Path(os.environ.get("EXPERIMENT_LOG_ROOT", PROJECT_ROOT / "experiments/logs"))
RESULT_DIR = Path(os.environ.get("EXPERIMENT_RESULT_DIR", PROJECT_ROOT / "experiments/results"))


def gpu_bucket_from_count(gpu_count: str) -> str:
    if gpu_count == "1":
        return "single_gpu"
    if gpu_count == "2":
        return "dual_gpu"
    if not gpu_count or gpu_count == "unknown":
        return "unknown_gpu"
    return f"multi_gpu_{gpu_count}"


GPU_BUCKET = os.environ.get("EXPERIMENT_GPU_BUCKET") or gpu_bucket_from_count(os.environ.get("GPU_COUNT", "2"))
SOKOBAN_LOG_DIR = LOG_ROOT / "sokoban" / GPU_BUCKET
SOKOBAN_UNKNOWN_LOG_DIR = LOG_ROOT / "sokoban" / "unknown_gpu"
LONG_HORIZON_SUMMARY_LOG = SOKOBAN_UNKNOWN_LOG_DIR / "long_horizon_stability_summary.log"
EXPECTED_TOTAL_STEPS = 50

STEP_METRIC_KEYS = [
    "episode/success_rate",
    "val/success_rate",
    "episode/length/mean",
    "episode/reward/mean",
    "prompt_length/mean",
    "response_length/mean",
    "timing_s/step",
    "timing_s/gen",
    "perf/throughput",
    "episode/valid_action_ratio",
    "val/text/test_score",
]

RESULT_FIELD_ORDER = [
    "strategy",
    "source_log",
    "completed_steps",
    "expected_total_steps",
    "is_complete",
    "train_success_rate_last",
    "train_success_rate_max",
    "val_success_rate_last",
    "val_success_rate_max",
    "train_episode_length_mean",
    "train_reward_mean",
    "prompt_tokens_mean",
    "response_tokens_mean",
    "epoch_time_mean",
    "valid_action_ratio_mean",
    "train_tail_mean",
    "train_tail_std",
    "val_tail_mean",
    "val_tail_std",
    "peak_to_late_drop",
    "late_len_mean",
    "horizon_cap_ratio",
    "late_prompt_mean",
    "late_step_time_s",
    "late_valid_action_ratio",
    "late_stage_note",
]

SOKOBAN_EXPERIMENTS = [
    {
        "name": "K=3",
        "paths": [
            SOKOBAN_LOG_DIR / "history_strategy_logs_20260419_215236__recent_window_k3.log",
            SOKOBAN_LOG_DIR / "recent_window_k3.log",
        ],
        "max_steps": 15,
    },
    {
        "name": "K=5",
        "paths": [
            SOKOBAN_LOG_DIR / "history_strategy_logs_20260419_215236__recent_window_k5.log",
            SOKOBAN_LOG_DIR / "recent_window_k5.log",
        ],
        "max_steps": 15,
    },
    {
        "name": "Full History",
        "paths": [
            SOKOBAN_LOG_DIR / "history_strategy_logs_20260419_215236__full_history.log",
            SOKOBAN_LOG_DIR / "full_history.log",
        ],
        "max_steps": 15,
    },
    {
        "name": "Structured Summary",
        "paths": [
            SOKOBAN_LOG_DIR / "history_strategy_logs_20260419_215236__structured_summary.log",
            SOKOBAN_LOG_DIR / "structured_summary.log",
        ],
        "max_steps": 15,
    },
]

LONG_HORIZON_EXPERIMENTS = SOKOBAN_EXPERIMENTS 

def resolve_log_path(path_candidates: list[Path]) -> Path:
    for path in path_candidates:
        if path.exists():
            return path
    return path_candidates[0]


def _extract_step_metric(line: str, key: str):
    match = re.search(re.escape(key) + r":(-?\d+(?:\.\d+)?)", line)
    if match:
        return float(match.group(1))
    return None


def parse_step_rows(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    rows_by_step: dict[int, dict[str, Any]] = {}
    content = log_path.read_text(errors="ignore")
    for line in content.splitlines():
        if "step:" not in line:
            continue

        step_match = re.search(r"step:(\d+)", line)
        if not step_match:
            continue

        step = int(step_match.group(1))
        row = rows_by_step.setdefault(step, {"step": step})
        for key in STEP_METRIC_KEYS:
            value = _extract_step_metric(line, key)
            if value is not None:
                row[key] = value

    return [rows_by_step[step] for step in sorted(rows_by_step)]


def parse_log_file(log_path: str | Path) -> dict[str, list[float]]:
    metrics = {
        "epochs": [],
        "train_success_rate": [],
        "val_success_rate": [],
        "train_episode_length": [],
        "train_reward": [],
        "epoch_time": [],
        "prompt_tokens": [],
        "response_tokens": [],
        "valid_action_ratio": [],
    }

    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Warning: {log_path} not found")
        return metrics

    for row in parse_step_rows(log_path):
        metrics["epochs"].append(row["step"])
        if "episode/success_rate" in row:
            metrics["train_success_rate"].append(row["episode/success_rate"])
        if "val/success_rate" in row:
            metrics["val_success_rate"].append(row["val/success_rate"])
        if "episode/length/mean" in row:
            metrics["train_episode_length"].append(row["episode/length/mean"])
        if "episode/reward/mean" in row:
            metrics["train_reward"].append(row["episode/reward/mean"])
        if "prompt_length/mean" in row:
            metrics["prompt_tokens"].append(row["prompt_length/mean"])
        if "response_length/mean" in row:
            metrics["response_tokens"].append(row["response_length/mean"])
        if "timing_s/step" in row:
            metrics["epoch_time"].append(row["timing_s/step"])
        if "episode/valid_action_ratio" in row:
            metrics["valid_action_ratio"].append(row["episode/valid_action_ratio"])

    return metrics


def summarize_metrics(metrics: dict[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for key, values in metrics.items():
        if key == "epochs":
            continue
        if values:
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_max"] = max(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_last"] = values[-1]
            summary[f"{key}_count"] = len(values)
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_max"] = None
            summary[f"{key}_min"] = None
            summary[f"{key}_last"] = None
            summary[f"{key}_count"] = 0

    return summary


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _safe_pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return pstdev(values)


def compute_long_horizon_proxy(rows: list[dict[str, Any]], max_steps: int) -> dict[str, Any]:
    train_rows = [row for row in rows if "episode/success_rate" in row]
    val_rows = [row for row in rows if "val/success_rate" in row]
    note_parts: list[str] = []

    if len(train_rows) >= 5:
        tail = train_rows[-max(1, len(train_rows) // 5) :]
        train_tail = [row["episode/success_rate"] for row in tail]
        peak = max(row["episode/success_rate"] for row in train_rows)
        late_len_values = [row["episode/length/mean"] for row in tail if "episode/length/mean" in row]
        prompt_values = [row["prompt_length/mean"] for row in tail if "prompt_length/mean" in row]
        step_time_values = [row["timing_s/step"] for row in tail if "timing_s/step" in row]
        valid_action_values = [row["episode/valid_action_ratio"] for row in tail if "episode/valid_action_ratio" in row]

        train_mean = _safe_mean(train_tail)
        train_std = _safe_pstdev(train_tail)
        peak_to_late_drop = peak - train_mean if train_mean is not None else None
        late_len_mean = _safe_mean(late_len_values)
        horizon_cap_ratio = None
        if late_len_values:
            threshold = 0.9 * max_steps
            horizon_cap_ratio = sum(value >= threshold for value in late_len_values) / len(late_len_values)
        late_prompt_mean = _safe_mean(prompt_values)
        late_step_time = _safe_mean(step_time_values)
        late_valid_action_ratio = _safe_mean(valid_action_values)
    elif train_rows:
        tail = train_rows
        train_mean = train_rows[0].get("episode/success_rate")
        train_std = 0.0
        peak_to_late_drop = 0.0
        late_len_mean = train_rows[0].get("episode/length/mean")
        threshold = 0.9 * max_steps
        horizon_cap_ratio = 1.0 if late_len_mean is not None and late_len_mean >= threshold else 0.0
        late_prompt_mean = train_rows[0].get("prompt_length/mean")
        late_step_time = train_rows[0].get("timing_s/step")
        late_valid_action_ratio = train_rows[0].get("episode/valid_action_ratio")
        note_parts.append("insufficient_train_steps")
    else:
        tail = []
        train_mean = None
        train_std = None
        peak_to_late_drop = None
        late_len_mean = None
        horizon_cap_ratio = None
        late_prompt_mean = None
        late_step_time = None
        late_valid_action_ratio = None
        note_parts.append("no_train_metrics")

    if len(val_rows) >= 3:
        val_tail = val_rows[-max(1, len(val_rows) // 3) :]
        val_tail_scores = [row["val/success_rate"] for row in val_tail]
        val_mean = _safe_mean(val_tail_scores)
        val_std = _safe_pstdev(val_tail_scores)
    elif val_rows:
        val_mean = val_rows[-1]["val/success_rate"]
        val_std = 0.0
        note_parts.append("insufficient_val_steps")
    else:
        val_mean = None
        val_std = None
        note_parts.append("no_val_metrics")

    return {
        "train_tail_mean": train_mean,
        "train_tail_std": train_std,
        "val_tail_mean": val_mean,
        "val_tail_std": val_std,
        "peak_to_late_drop": peak_to_late_drop,
        "late_len_mean": late_len_mean,
        "horizon_cap_ratio": horizon_cap_ratio,
        "late_prompt_mean": late_prompt_mean,
        "late_step_time_s": late_step_time,
        "late_valid_action_ratio": late_valid_action_ratio,
        "tail_steps": [row["step"] for row in tail],
        "note": ";".join(note_parts) if note_parts else "ok",
    }


def summarize_experiment_result(strategy: str, log_path: Path, expected_total_steps: int, max_steps: int) -> dict[str, Any]:
    rows = parse_step_rows(log_path)
    metrics = parse_log_file(log_path)
    summary = summarize_metrics(metrics)
    proxy = compute_long_horizon_proxy(rows, max_steps)
    max_logged_step = max((row["step"] for row in rows), default=-1)
    completed_steps = max_logged_step + 1 if max_logged_step >= 0 else 0

    result = {
        "strategy": strategy,
        "source_log": str(log_path),
        "completed_steps": completed_steps,
        "expected_total_steps": expected_total_steps,
        "is_complete": completed_steps >= expected_total_steps,
        "train_success_rate_last": summary.get("train_success_rate_last"),
        "train_success_rate_max": summary.get("train_success_rate_max"),
        "val_success_rate_last": summary.get("val_success_rate_last"),
        "val_success_rate_max": summary.get("val_success_rate_max"),
        "train_episode_length_mean": summary.get("train_episode_length_mean"),
        "train_reward_mean": summary.get("train_reward_mean"),
        "prompt_tokens_mean": summary.get("prompt_tokens_mean"),
        "response_tokens_mean": summary.get("response_tokens_mean"),
        "epoch_time_mean": summary.get("epoch_time_mean"),
        "valid_action_ratio_mean": summary.get("valid_action_ratio_mean"),
        "train_tail_mean": proxy.get("train_tail_mean"),
        "train_tail_std": proxy.get("train_tail_std"),
        "val_tail_mean": proxy.get("val_tail_mean"),
        "val_tail_std": proxy.get("val_tail_std"),
        "peak_to_late_drop": proxy.get("peak_to_late_drop"),
        "late_len_mean": proxy.get("late_len_mean"),
        "horizon_cap_ratio": proxy.get("horizon_cap_ratio"),
        "late_prompt_mean": proxy.get("late_prompt_mean"),
        "late_step_time_s": proxy.get("late_step_time_s"),
        "late_valid_action_ratio": proxy.get("late_valid_action_ratio"),
        "late_stage_note": proxy.get("note"),
    }
    return result


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_rate_pair(last_value: Any, max_value: Any) -> str:
    return f"{_format_number(last_value)} / {_format_number(max_value)}"


def generate_comparison_table(results: list[dict[str, Any]]) -> str:
    header = (
        "| Strategy | Complete | Steps | Train SR last/max | Val SR last/max | Prompt mean | Step time | "
        "Late val mean±std | Late prompt | Late step time | Late valid ratio |\n"
    )
    divider = (
        "|---|---|---:|---|---|---:|---:|---|---:|---:|---:|\n"
    )
    rows = []
    for result in results:
        late_val = f"{_format_number(result['val_tail_mean'])} ± {_format_number(result['val_tail_std'])}"
        rows.append(
            "| {strategy} | {complete} | {steps}/{expected} | {train_sr} | {val_sr} | {prompt} | {step_time} | {late_val} | {late_prompt} | {late_step} | {late_valid} |".format(
                strategy=result["strategy"],
                complete="yes" if result["is_complete"] else "no",
                steps=result["completed_steps"],
                expected=result["expected_total_steps"],
                train_sr=_format_rate_pair(result["train_success_rate_last"], result["train_success_rate_max"]),
                val_sr=_format_rate_pair(result["val_success_rate_last"], result["val_success_rate_max"]),
                prompt=_format_number(result["prompt_tokens_mean"], 1),
                step_time=_format_number(result["epoch_time_mean"], 1),
                late_val=late_val,
                late_prompt=_format_number(result["late_prompt_mean"], 1),
                late_step=_format_number(result["late_step_time_s"], 1),
                late_valid=_format_number(result["late_valid_action_ratio"]),
            )
        )
    return header + divider + "\n".join(rows) + "\n"


def _format_proxy_value(value: Any, digits: int = 3) -> str:
    return _format_number(value, digits)


def _build_dynamic_interpretation(proxy_results: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["Interpretation"]

    completed_sokoban = {
    name: proxy
    for name, proxy in proxy_results.items()
    if proxy["note"] == "ok"
}

    if completed_sokoban:
        best_stability = max(
            completed_sokoban.items(),
            key=lambda item: ((item[1]["val_tail_mean"] or float("-inf")), -((item[1]["peak_to_late_drop"] or float("inf")))),
        )
        fastest = min(
            completed_sokoban.items(),
            key=lambda item: item[1]["late_step_time_s"] if item[1]["late_step_time_s"] is not None else float("inf"),
        )
        shortest_prompt = min(
            completed_sokoban.items(),
            key=lambda item: item[1]["late_prompt_mean"] if item[1]["late_prompt_mean"] is not None else float("inf"),
        )

        lines.append(
            f"- Best completed Sokoban late-stage proxy: {best_stability[0]} (late val success {_format_proxy_value(best_stability[1]['val_tail_mean'])}, peak_to_late_drop {_format_proxy_value(best_stability[1]['peak_to_late_drop'])})."
        )
        lines.append(
            f"- Fastest completed Sokoban strategy: {fastest[0]} (late step time {_format_proxy_value(fastest[1]['late_step_time_s'], 1)}s)."
        )
        lines.append(
            f"- Shortest late-stage context among completed Sokoban runs: {shortest_prompt[0]} (late prompt {_format_proxy_value(shortest_prompt[1]['late_prompt_mean'], 1)})."
        )

    for name, proxy in proxy_results.items():
        if proxy["note"] != "ok":
            lines.append(f"- {name}: {proxy['note']}; current logs are insufficient for a firm late-stage conclusion.")

    return lines


def generate_long_horizon_summary(experiments: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    proxy_results: dict[str, dict[str, Any]] = {}
    lines = [
        "Long-horizon Stability Summary",
        "Generated from existing training logs",
        "",
        "Method",
        "- This is a proxy analysis from existing step-level logs, not true per-episode horizon binning.",
        "- late_success_mean: mean success rate over the final 20% of available training steps.",
        "- late_success_std: standard deviation over the final 20% of available training steps.",
        "- peak_to_late_drop: peak success rate minus late_success_mean.",
        "- horizon_cap_ratio: fraction of final 20% steps whose episode_length_mean is at least 90% of max_steps.",
        "",
        "Table",
        "strategy\ttrain_tail_mean\ttrain_tail_std\tval_tail_mean\tval_tail_std\tpeak_to_late_drop\tlate_len_mean\thorizon_cap_ratio\tlate_prompt_mean\tlate_step_time_s\tnote",
    ]

    for experiment in experiments:
        log_path = resolve_log_path(experiment["paths"])
        rows = parse_step_rows(log_path)
        proxy = compute_long_horizon_proxy(rows, experiment["max_steps"])
        proxy["source_log"] = str(log_path)
        proxy_results[experiment["name"]] = proxy

        lines.append(
            "\t".join(
                [
                    experiment["name"],
                    _format_proxy_value(proxy["train_tail_mean"]),
                    _format_proxy_value(proxy["train_tail_std"]),
                    _format_proxy_value(proxy["val_tail_mean"]),
                    _format_proxy_value(proxy["val_tail_std"]),
                    _format_proxy_value(proxy["peak_to_late_drop"]),
                    _format_proxy_value(proxy["late_len_mean"]),
                    _format_proxy_value(proxy["horizon_cap_ratio"]),
                    _format_proxy_value(proxy["late_prompt_mean"], 1),
                    _format_proxy_value(proxy["late_step_time_s"], 1),
                    proxy["note"],
                ]
            )
        )

    lines.append("")
    lines.extend(_build_dynamic_interpretation(proxy_results))
    return "\n".join(lines) + "\n", proxy_results


def write_results_csv(results: list[dict[str, Any]], output_path: Path):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELD_ORDER)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in RESULT_FIELD_ORDER})


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SOKOBAN_LOG_DIR, exist_ok=True)
    os.makedirs(SOKOBAN_UNKNOWN_LOG_DIR, exist_ok=True)

    results: list[dict[str, Any]] = []
    detailed_metrics: dict[str, Any] = {}
    comparison_summary: dict[str, Any] = {}

    for experiment in SOKOBAN_EXPERIMENTS:
        log_path = resolve_log_path(experiment["paths"])
        result = summarize_experiment_result(
            strategy=experiment["name"],
            log_path=log_path,
            expected_total_steps=EXPECTED_TOTAL_STEPS,
            max_steps=experiment["max_steps"],
        )
        metrics = parse_log_file(log_path)
        results.append(result)
        detailed_metrics[experiment["name"]] = metrics
        comparison_summary[experiment["name"]] = result

        print(
            f"{experiment['name']}: complete={result['is_complete']} steps={result['completed_steps']}/{result['expected_total_steps']} "
            f"val_last={result['val_success_rate_last']} val_max={result['val_success_rate_max']} "
            f"prompt_mean={result['prompt_tokens_mean']} step_time={result['epoch_time_mean']} source={log_path}"
        )

    table = generate_comparison_table(results)
    long_horizon_summary, long_horizon_results = generate_long_horizon_summary(LONG_HORIZON_EXPERIMENTS)

    (RESULT_DIR / "comparison_table.md").write_text(
        "# Sokoban History Management Experiment Results\n\n" + table
    )
    (RESULT_DIR / "comparison_summary.json").write_text(json.dumps(comparison_summary, indent=2, default=str))
    (RESULT_DIR / "detailed_metrics.json").write_text(json.dumps(detailed_metrics, indent=2, default=str))
    (RESULT_DIR / "sokoban_comparison_results.json").write_text(json.dumps(results, indent=2, default=str))
    write_results_csv(results, RESULT_DIR / "sokoban_comparison_results.csv")
    (RESULT_DIR / "long_horizon_stability_summary.json").write_text(json.dumps(long_horizon_results, indent=2, default=str))
    LONG_HORIZON_SUMMARY_LOG.write_text(long_horizon_summary)

    print("\n=== Comparison Table ===")
    print(table)
    print("=== Long-horizon Stability Summary ===")
    print(long_horizon_summary)
    print(f"Results saved to {RESULT_DIR}/")


if __name__ == "__main__":
    main()
