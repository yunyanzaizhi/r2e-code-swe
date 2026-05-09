import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


class SplitPolicyError(ValueError):
    """Raised when a dataset split is unsafe for the requested mode."""


@dataclass
class R2ECodeSWETask:
    task_id: str
    dataset_name: str
    split: str
    repo_name: str
    repo: str
    docker_image: str
    base_commit: str
    problem_statement: str
    test_spec: Dict[str, Any]
    gold_patch_optional: Optional[str] = None
    raw_record: Dict[str, Any] = field(default_factory=dict)

    @property
    def repo_label(self) -> str:
        return self.repo_name or self.repo or "unknown/repo"


def _first(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def _json_or_value(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _safe_task_id(value: Any) -> str:
    text = str(value or "unknown")
    return text.replace("\n", " ").strip()


def validate_r2e_split_policy(
    dataset_name: str,
    split: str,
    mode: str,
    allow_train_on_dev: bool = False,
) -> None:
    dataset_l = (dataset_name or "").lower()
    split_l = (split or "").lower()
    mode_l = (mode or "").lower()

    if mode_l != "train":
        return

    if "swe-bench-lite" in dataset_l and split_l == "test":
        raise SplitPolicyError(
            "R2E-Gym/SWE-Bench-Lite split=test is evaluation-only and must not be used for training."
        )

    if split_l == "dev_10pr_v1":
        message = "dev_10pr_v1 is for smoke/debug/validation only, not training."
        if allow_train_on_dev:
            warnings.warn(message + " allow_train_on_dev=True bypassed this guard.", RuntimeWarning)
            return
        raise SplitPolicyError(message + " Set allow_train_on_dev=true only for an intentional debug run.")

    if "r2e-gym" in dataset_l and split_l != "train":
        warnings.warn(
            f"Training with split={split!r} for dataset={dataset_name!r}; expected split='train'.",
            RuntimeWarning,
        )


def normalize_r2e_task_record(
    record: Dict[str, Any],
    dataset_name: str,
    split: str = "train",
    index: int = 0,
) -> R2ECodeSWETask:
    repo_name = _first(record, ["repo_name", "repository_name"], "")
    repo = _first(record, ["repo", "repository"], repo_name or "unknown/repo")
    docker_image = _first(record, ["docker_image", "image_name"], "")
    base_commit = _first(record, ["base_commit", "commit_hash", "commit", "sha"], "HEAD")
    problem_statement = _first(record, ["problem_statement", "issue", "prompt"], "")

    expected_output_json = _json_or_value(record.get("expected_output_json"), None)
    fail_to_pass = _json_or_value(record.get("FAIL_TO_PASS"), [])
    pass_to_pass = _json_or_value(record.get("PASS_TO_PASS"), [])

    task_id = _first(record, ["task_id", "instance_id"], None)
    if task_id is None:
        id_tail = docker_image or f"{repo}:{base_commit}"
        task_id = f"{dataset_name}:{split}:{index}:{id_tail}"

    test_spec = {
        "FAIL_TO_PASS": fail_to_pass if isinstance(fail_to_pass, list) else [],
        "PASS_TO_PASS": pass_to_pass if isinstance(pass_to_pass, list) else [],
        "run_tests": record.get("run_tests"),
        "run_tests_regression": record.get("run_tests_regression"),
        "test_patch": record.get("test_patch"),
        "expected_output_json": expected_output_json if isinstance(expected_output_json, dict) else None,
    }

    return R2ECodeSWETask(
        task_id=_safe_task_id(task_id),
        dataset_name=str(dataset_name),
        split=str(split),
        repo_name=str(repo_name or repo),
        repo=str(repo),
        docker_image=str(docker_image),
        base_commit=str(base_commit),
        problem_statement=str(problem_statement),
        test_spec=test_spec,
        gold_patch_optional=_first(record, ["patch", "gold_patch", "parsed_commit_content"], None),
        raw_record=dict(record),
    )


def load_r2e_tasks_from_hf(
    dataset_name: str,
    split: str,
    max_samples: Optional[int] = None,
    streaming: bool = False,
    mode: str = "train",
    allow_train_on_dev: bool = False,
) -> List[R2ECodeSWETask]:
    validate_r2e_split_policy(dataset_name, split, mode, allow_train_on_dev)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for R2E code/SWE tasks.") from exc

    if streaming:
        ds = load_dataset(dataset_name, split=split, streaming=True)
        rows = iter(ds)
    else:
        hf_split = split if max_samples is None else f"{split}[:{max_samples}]"
        rows = iter(load_dataset(dataset_name, split=hf_split))

    tasks: List[R2ECodeSWETask] = []
    for idx, row in enumerate(rows):
        if max_samples is not None and idx >= max_samples:
            break
        tasks.append(normalize_r2e_task_record(dict(row), dataset_name, split, idx))
    return tasks


def load_r2e_tasks_from_jsonl(
    path: str,
    dataset_name: str = "local",
    split: str = "train",
    mode: str = "train",
    allow_train_on_dev: bool = False,
) -> List[R2ECodeSWETask]:
    validate_r2e_split_policy(dataset_name, split, mode, allow_train_on_dev)
    tasks: List[R2ECodeSWETask] = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            tasks.append(normalize_r2e_task_record(json.loads(line), dataset_name, split, idx))
    return tasks


def load_r2e_tasks_from_config(config: Any, is_train: bool = True) -> List[R2ECodeSWETask]:
    split = getattr(config, "train_split" if is_train else "val_split", None)
    split = split or ("train" if is_train else "dev_10pr_v1")
    dataset_name = getattr(config, "train_dataset_name" if is_train else "val_dataset_name", None)
    dataset_name = dataset_name or getattr(config, "dataset_name", "R2E-Gym/R2E-Gym-Lite")
    dataset_path = getattr(config, "train_dataset_path" if is_train else "val_dataset_path", None)
    dataset_path = dataset_path or getattr(config, "dataset_path", None)
    max_samples = getattr(config, "max_train_samples" if is_train else "max_val_samples", None)
    streaming = bool(getattr(config, "streaming", False))
    allow_train_on_dev = bool(getattr(config, "allow_train_on_dev", False))
    mode = "train" if is_train else "eval"

    if dataset_path:
        return load_r2e_tasks_from_jsonl(dataset_path, dataset_name, split, mode, allow_train_on_dev)
    return load_r2e_tasks_from_hf(dataset_name, split, max_samples, streaming, mode, allow_train_on_dev)
