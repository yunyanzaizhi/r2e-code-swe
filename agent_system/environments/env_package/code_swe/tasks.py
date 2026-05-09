import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class CodeSWETask:
    task_id: str
    dataset_name: str
    repo: str
    base_commit: str
    problem_statement: str
    test_spec: Dict[str, Any]
    gold_patch_optional: Optional[str] = None
    raw_record: Dict[str, Any] = field(default_factory=dict)
    repo_path: Optional[str] = None
    repo_url: Optional[str] = None


def _json_or_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _first(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _safe_task_id(value: Any) -> str:
    return str(value or "unknown")


def normalize_task_record(
    record: Dict[str, Any],
    dataset_name: str,
    split: str = "train",
    index: int = 0,
) -> CodeSWETask:
    """Normalize R2E-Gym, SWE-Bench, and local records into one task schema."""
    dataset_l = dataset_name.lower()
    repo = _first(record, ["repo", "repo_name", "repository"], "unknown/repo")
    base_commit = _first(record, ["base_commit", "commit_hash", "commit", "sha"], "HEAD")
    problem_statement = _first(record, ["problem_statement", "issue", "prompt"], "")
    docker_image = record.get("docker_image") or record.get("image_name")

    fail_to_pass = _json_or_value(record.get("FAIL_TO_PASS"), [])
    pass_to_pass = _json_or_value(record.get("PASS_TO_PASS"), [])
    expected_output_json = _json_or_value(record.get("expected_output_json"), None)

    if "swe-bench" in dataset_l or "swebench" in dataset_l:
        task_id = str(record.get("instance_id") or f"{dataset_name}:{split}:{index}")
    elif record.get("task_id"):
        task_id = str(record["task_id"])
    else:
        task_id = f"{dataset_name}:{split}:{index}:{repo}:{base_commit}"

    test_spec = {
        "FAIL_TO_PASS": fail_to_pass if isinstance(fail_to_pass, list) else [],
        "PASS_TO_PASS": pass_to_pass if isinstance(pass_to_pass, list) else [],
        "run_tests": record.get("run_tests"),
        "test_patch": record.get("test_patch"),
        "test_command": record.get("test_command"),
        "install_command": record.get("install_command"),
        "expected_output_json": expected_output_json if isinstance(expected_output_json, dict) else None,
        "docker_image": docker_image,
    }

    return CodeSWETask(
        task_id=_safe_task_id(task_id),
        dataset_name=dataset_name,
        repo=str(repo),
        base_commit=str(base_commit),
        problem_statement=str(problem_statement),
        test_spec=test_spec,
        gold_patch_optional=record.get("patch") or record.get("gold_patch") or record.get("parsed_commit_content"),
        raw_record=dict(record),
        repo_path=record.get("repo_path"),
        repo_url=record.get("repo_url"),
    )


def load_tasks_from_hf(
    dataset_name: str,
    split: str,
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> List[CodeSWETask]:
    """Load and normalize Hugging Face task rows. Import is lazy for lightweight tests."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required to load Hugging Face code/SWE tasks.") from exc

    hf_split = split if streaming or max_samples is None else f"{split}[:{max_samples}]"
    ds = load_dataset(dataset_name, split=hf_split, streaming=streaming)
    rows = ds if streaming else iter(ds)
    tasks = []
    for idx, row in enumerate(rows):
        if max_samples is not None and idx >= max_samples:
            break
        tasks.append(normalize_task_record(dict(row), dataset_name=dataset_name, split=split, index=idx))
    return tasks


def load_tasks_from_jsonl(path: str, dataset_name: str = "local", split: str = "train") -> List[CodeSWETask]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            tasks.append(normalize_task_record(json.loads(line), dataset_name=dataset_name, split=split, index=idx))
    return tasks


def load_tasks_from_config(config: Any, is_train: bool = True) -> List[CodeSWETask]:
    split = getattr(config, "train_split" if is_train else "val_split", None)
    split = split or ("train" if is_train else "test")
    dataset_name = getattr(config, "train_dataset_name" if is_train else "val_dataset_name", None)
    dataset_name = dataset_name or getattr(config, "dataset_name", None)
    dataset_path = getattr(config, "train_dataset_path" if is_train else "val_dataset_path", None)
    dataset_path = dataset_path or getattr(config, "dataset_path", None)
    max_samples = getattr(config, "max_train_samples" if is_train else "max_val_samples", None)
    streaming = bool(getattr(config, "streaming", False))

    if dataset_path:
        return load_tasks_from_jsonl(dataset_path, dataset_name=dataset_name or "local", split=split)
    if not dataset_name:
        raise ValueError("code_swe.dataset_name or code_swe.dataset_path must be configured.")
    return load_tasks_from_hf(dataset_name=dataset_name, split=split, max_samples=max_samples, streaming=streaming)
