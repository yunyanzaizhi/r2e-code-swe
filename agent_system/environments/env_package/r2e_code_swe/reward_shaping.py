import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


EDITOR_WRITE_COMMANDS = {"create", "str_replace", "insert"}
EDITOR_VIEW_COMMANDS = {"view"}


def _strip_line_suffix(path: str) -> str:
    return re.sub(r":\d+(?::\d+)?$", "", str(path or "").strip())


def classify_r2e_path(path: str) -> str:
    """Classify an R2E /testbed path for reward shaping and submit gating."""
    clean = _strip_line_suffix(path)
    if not clean:
        return "unknown"
    if not (clean == "/testbed" or clean.startswith("/testbed/")):
        return "outside_testbed"

    rel = clean[len("/testbed") :].lstrip("/")
    if not rel:
        return "workspace"

    rel_parts = PurePosixPath(rel).parts
    name = rel_parts[-1]
    lowered = rel.lower()
    lower_name = name.lower()

    if rel in {".git", ".venv"} or lowered.startswith((".git/", ".venv/")):
        return "r2e_aux"
    if rel in {"r2e_tests", "install.sh", "datasets", "Makefile"}:
        return "r2e_aux"
    if lowered.startswith("r2e_tests/"):
        return "r2e_aux"
    if lower_name.startswith("process_") and lower_name.endswith(".py"):
        return "r2e_aux"

    if rel == "test.py" or re.match(r"^(reproduce|repro|debug|scratch)[A-Za-z0-9_.-]*\.py$", name):
        return "repro"

    if any(part in {"tests", "test", "testing"} for part in rel_parts[:-1]):
        return "test"
    if lower_name.startswith("test_") or lower_name.endswith("_test.py"):
        return "test"

    return "source"


def editor_path_kind(action: Dict[str, Any]) -> str:
    params = action.get("parameters") or {}
    return classify_r2e_path(str(params.get("path") or ""))


def is_editor_write(action: Dict[str, Any]) -> bool:
    if action.get("tool_name") != "str_replace_editor":
        return False
    command = str((action.get("parameters") or {}).get("command") or "")
    return command in EDITOR_WRITE_COMMANDS


def is_noop_str_replace(action: Dict[str, Any]) -> bool:
    if action.get("tool_name") != "str_replace_editor":
        return False
    params = action.get("parameters") or {}
    if str(params.get("command") or "") != "str_replace":
        return False
    if "old_str" not in params or "new_str" not in params:
        return False
    old_str = str(params.get("old_str") or "").expandtabs()
    new_str = str(params.get("new_str") or "").expandtabs()
    return old_str == new_str


def is_editor_view(action: Dict[str, Any]) -> bool:
    if action.get("tool_name") != "str_replace_editor":
        return False
    command = str((action.get("parameters") or {}).get("command") or "")
    return command in EDITOR_VIEW_COMMANDS


def patch_source_files(patch: str) -> List[str]:
    source_files: List[str] = []
    seen: Set[str] = set()
    for line in str(patch or "").splitlines():
        if not line.startswith("diff --git "):
            continue
        match = re.match(r"diff --git a/(.*?) b/(.*)$", line)
        if not match:
            continue
        candidate = match.group(2)
        path = "/testbed/" + candidate.lstrip("/")
        if classify_r2e_path(path) == "source" and candidate not in seen:
            seen.add(candidate)
            source_files.append(candidate)
    return source_files


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def is_repo_exploration_command(command: str) -> bool:
    return _contains_any(command, ["find ", "ls ", "pwd", "git status", "git ls-files"])


def is_issue_search_command(command: str) -> bool:
    return _contains_any(command, ["grep ", "grep -", "rg ", "find ", "fd "])


def is_validation_command(command: str) -> bool:
    return _contains_any(
        command,
        [
            "pytest",
            "tox",
            "unittest",
            "nosetests",
            "run_tests",
            "r2e_tests",
            "python -m pytest",
        ],
    )


@dataclass
class R2ERewardShapingConfig:
    enabled: bool = True
    positive_cap: float = 0.40
    negative_cap: float = -0.35
    repo_explored: float = 0.01
    issue_search: float = 0.02
    source_view: float = 0.01
    source_view_cap: float = 0.03
    first_source_edit: float = 0.04
    additional_source_edit: float = 0.01
    additional_source_edit_cap: float = 0.02
    source_patch_present: float = 0.06
    validation_after_edit: float = 0.12
    clean_submit: float = 0.08
    submit_before_source_edit_penalty: float = -0.08
    submit_before_validation_after_source_edit_penalty: float = -0.12
    edit_test_penalty: float = -0.12
    edit_r2e_aux_penalty: float = -0.12
    repeated_failed_action_penalty: float = -0.06
    repeated_no_progress_action_penalty: float = -0.08
    max_steps_no_source_edit_penalty: float = -0.10
    max_steps_no_validation_after_edit_penalty: float = -0.08
    editor_no_occurrences_penalty: float = -0.04
    editor_multiple_occurrences_penalty: float = -0.04

    @classmethod
    def from_obj(cls, obj: Any) -> "R2ERewardShapingConfig":
        if obj is None:
            return cls()
        data: Dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            if isinstance(obj, dict) and field_name in obj:
                data[field_name] = obj[field_name]
            elif hasattr(obj, field_name):
                data[field_name] = getattr(obj, field_name)
        return cls(**data)

    def new_state(self) -> "R2ERewardShapingState":
        return R2ERewardShapingState()


@dataclass
class R2ERewardShapingState:
    repo_explored_seen: bool = False
    issue_search_seen: bool = False
    source_files_viewed: Set[str] = field(default_factory=set)
    source_edit_count: int = 0
    source_patch_seen: bool = False
    validation_after_edit_seen: bool = False
    clean_submit_seen: bool = False
    last_potential: float = 0.0
    negative_total: float = 0.0

    def potential(self, config: R2ERewardShapingConfig) -> float:
        value = 0.0
        if self.repo_explored_seen:
            value += config.repo_explored
        if self.issue_search_seen:
            value += config.issue_search
        value += min(len(self.source_files_viewed) * config.source_view, config.source_view_cap)
        if self.source_edit_count > 0:
            value += config.first_source_edit
            value += min((self.source_edit_count - 1) * config.additional_source_edit, config.additional_source_edit_cap)
        if self.source_patch_seen:
            value += config.source_patch_present
        if self.validation_after_edit_seen:
            value += config.validation_after_edit
        if self.clean_submit_seen:
            value += config.clean_submit
        return min(value, config.positive_cap)

    def apply_penalty(self, config: R2ERewardShapingConfig, penalty: float) -> float:
        if penalty >= 0:
            return 0.0
        remaining = config.negative_cap - self.negative_total
        applied = max(penalty, remaining)
        self.negative_total += applied
        return applied

    def update(
        self,
        config: R2ERewardShapingConfig,
        action: Dict[str, Any],
        result_info: Dict[str, Any],
        patch_text: Optional[str] = None,
        auto_submitted: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        if not config.enabled:
            return 0.0, {
                "terminal_r2e_reward": float(result_info.get("reward", 0.0) or 0.0),
                "shaping_delta": 0.0,
                "penalty": 0.0,
                "phi_before": self.last_potential,
                "phi_after": self.last_potential,
                "events": [],
            }

        before = self.potential(config)
        events: List[str] = []
        penalty = 0.0
        tool_name = action.get("tool_name")
        params = action.get("parameters") or {}
        execution_success = bool(result_info.get("tool_execution_success", result_info.get("is_action_valid", True)))
        fail_reason = result_info.get("tool_execution_fail_reason") or result_info.get("fail_reason")

        if tool_name == "bash" and execution_success:
            command = str(params.get("cmd") or "")
            if is_repo_exploration_command(command) and not self.repo_explored_seen:
                self.repo_explored_seen = True
                events.append("repo_explored")
            if is_issue_search_command(command) and not self.issue_search_seen:
                self.issue_search_seen = True
                events.append("issue_relevant_search")
            if self.source_edit_count > 0 and is_validation_command(command) and not self.validation_after_edit_seen:
                self.validation_after_edit_seen = True
                events.append("validation_after_source_edit")

        if tool_name == "str_replace_editor":
            path = str(params.get("path") or "")
            path_kind = classify_r2e_path(path)
            if is_editor_view(action) and execution_success and path_kind == "source":
                clean_path = _strip_line_suffix(path)
                if clean_path not in self.source_files_viewed:
                    self.source_files_viewed.add(clean_path)
                    events.append("unique_source_file_viewed")
            if (
                is_editor_write(action)
                and not is_noop_str_replace(action)
                and execution_success
                and path_kind == "source"
                and not result_info.get("semantic_noop_edit")
            ):
                self.source_edit_count += 1
                events.append("successful_source_edit")

        if fail_reason == "submit_before_successful_source_edit":
            penalty += config.submit_before_source_edit_penalty
            events.append("submit_before_source_edit")
        elif fail_reason == "submit_before_validation_after_source_edit":
            penalty += config.submit_before_validation_after_source_edit_penalty
            events.append("submit_before_validation_after_source_edit")
        elif fail_reason == "submit_before_passing_validation_after_source_edit":
            penalty += config.submit_before_validation_after_source_edit_penalty
            events.append("submit_before_passing_validation_after_source_edit")
        elif fail_reason == "test_edit_blocked":
            penalty += config.edit_test_penalty
            events.append("test_edit_blocked")
        elif fail_reason == "r2e_aux_edit_blocked":
            penalty += config.edit_r2e_aux_penalty
            events.append("r2e_aux_edit_blocked")
        elif fail_reason == "repeated_failed_action":
            penalty += config.repeated_failed_action_penalty
            events.append("repeated_failed_action")
        elif fail_reason == "repeated_no_progress_action":
            penalty += config.repeated_no_progress_action_penalty
            events.append("repeated_no_progress_action")
        elif fail_reason == "repeated_no_progress_action_limit":
            penalty += config.repeated_no_progress_action_penalty
            events.append("repeated_no_progress_action_limit")
        elif fail_reason == "editor_error":
            editor_recovery_hint = result_info.get("editor_recovery_hint")
            if editor_recovery_hint == "missing_or_misindented_old_str":
                penalty += config.editor_no_occurrences_penalty
                events.append("editor_no_occurrences")
            elif editor_recovery_hint == "non_unique_old_str":
                penalty += config.editor_multiple_occurrences_penalty
                events.append("editor_multiple_occurrences")

        if tool_name == "submit":
            source_files = patch_source_files(patch_text or "")
            if source_files and not self.source_patch_seen:
                self.source_patch_seen = True
                events.append("source_patch_present")
            if execution_success and self.source_edit_count > 0 and not self.clean_submit_seen:
                self.clean_submit_seen = True
                events.append("clean_submit_after_source_edit")
            if auto_submitted and self.source_edit_count <= 0:
                penalty += config.max_steps_no_source_edit_penalty
                events.append("max_steps_no_source_edit")
            elif auto_submitted and self.source_edit_count > 0 and not self.validation_after_edit_seen:
                penalty += config.max_steps_no_validation_after_edit_penalty
                events.append("max_steps_no_validation_after_edit")

        after = self.potential(config)
        shaping_delta = after - before
        applied_penalty = self.apply_penalty(config, penalty)
        self.last_potential = after
        total = shaping_delta + applied_penalty
        return total, {
            "terminal_r2e_reward": float(result_info.get("reward", 0.0) or 0.0),
            "shaping_delta": shaping_delta,
            "penalty": applied_penalty,
            "phi_before": before,
            "phi_after": after,
            "events": events,
        }
