from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    import gym
except ImportError:
    class _Env:
        pass

    class gym:
        Env = _Env

from .runtime import R2ERepoRuntime, R2ERuntimeConfig, R2EToolResult
from .reward_shaping import (
    R2ERewardShapingConfig,
    R2ERewardShapingState,
    classify_r2e_path,
    is_editor_write,
    is_noop_str_replace,
    is_validation_command,
)
from .tasks import R2ECodeSWETask, load_r2e_tasks_from_config


class R2ECodeSWEEnv(gym.Env):
    def __init__(
        self,
        tasks: List[R2ECodeSWETask],
        runtime_config: Optional[R2ERuntimeConfig] = None,
        env_num: int = 1,
        group_n: int = 1,
        max_steps: int = 20,
        invalid_action_penalty: float = 0.0,
        auto_submit_on_max_steps: bool = True,
        require_successful_edit_before_submit: bool = True,
        require_validation_before_submit: bool = True,
        max_repeated_failed_actions: int = 1,
        max_repeated_failed_action_blocks: int = 3,
        max_repeated_no_progress_actions: int = 3,
        reward_shaping_config: Optional[R2ERewardShapingConfig] = None,
        is_train: bool = True,
    ):
        if not tasks:
            raise ValueError("R2ECodeSWEEnv requires at least one task.")
        self.tasks = tasks
        self.runtime_config = runtime_config or R2ERuntimeConfig()
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.max_steps = max_steps
        self.invalid_action_penalty = invalid_action_penalty
        self.auto_submit_on_max_steps = auto_submit_on_max_steps
        self.require_successful_edit_before_submit = require_successful_edit_before_submit
        self.require_validation_before_submit = require_validation_before_submit
        self.max_repeated_failed_actions = max_repeated_failed_actions
        self.max_repeated_failed_action_blocks = max_repeated_failed_action_blocks
        self.max_repeated_no_progress_actions = max_repeated_no_progress_actions
        self.reward_shaping_config = reward_shaping_config or R2ERewardShapingConfig()
        self.is_train = is_train
        self._cursor = 0
        self.current_tasks: List[R2ECodeSWETask] = []
        self.runtimes: List[R2ERepoRuntime] = []
        self.step_counts: List[int] = []
        self.dones: List[bool] = []
        self.successful_edit_counts: List[int] = []
        self.successful_source_edit_counts: List[int] = []
        self.validation_after_source_edit_counts: List[int] = []
        self.reward_shaping_states: List[R2ERewardShapingState] = []
        self.last_failed_action_signatures: List[Optional[str]] = []
        self.repeated_failed_action_counts: List[int] = []
        self.seen_no_progress_action_signatures: List[set] = []
        self.repeated_no_progress_action_counts: List[Dict[str, int]] = []

    def reset(self, kwargs: Any = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.close()
        selected = [self.tasks[(self._cursor + idx) % len(self.tasks)] for idx in range(self.env_num)]
        self._cursor = (self._cursor + self.env_num) % len(self.tasks)
        self.current_tasks = [task for task in selected for _ in range(self.group_n)]
        self.step_counts = [0 for _ in range(self.num_processes)]
        self.dones = [False for _ in range(self.num_processes)]
        self.successful_edit_counts = [0 for _ in range(self.num_processes)]
        self.successful_source_edit_counts = [0 for _ in range(self.num_processes)]
        self.validation_after_source_edit_counts = [0 for _ in range(self.num_processes)]
        self.reward_shaping_states = [self.reward_shaping_config.new_state() for _ in range(self.num_processes)]
        self.last_failed_action_signatures = [None for _ in range(self.num_processes)]
        self.repeated_failed_action_counts = [0 for _ in range(self.num_processes)]
        self.seen_no_progress_action_signatures = [set() for _ in range(self.num_processes)]
        self.repeated_no_progress_action_counts = [{} for _ in range(self.num_processes)]
        self.runtimes = []

        observations: List[str] = []
        infos: List[Dict[str, Any]] = []
        for idx, task in enumerate(self.current_tasks):
            runtime = R2ERepoRuntime(self.runtime_config)
            runtime.setup(task, replica_id=str(idx))
            self.runtimes.append(runtime)
            observations.append(self._initial_observation(task, runtime))
            infos.append(self._base_info(idx, task, runtime))
            self._write_event(idx, {"event": "reset", "observation": observations[-1], "info": infos[-1]})
        return observations, infos

    def _initial_observation(self, task: R2ECodeSWETask, runtime: R2ERepoRuntime) -> str:
        setup = "Workspace ready." if not runtime.setup_error else f"Workspace setup failed: {runtime.setup_error}"
        lines = [
            f"Task {task.task_id}",
            f"Repository: {task.repo_label}",
            f"Docker image: {task.docker_image}",
            "Workspace: /testbed",
            setup,
        ]
        if not runtime.setup_error and hasattr(runtime, "workspace_overview"):
            try:
                overview = runtime.workspace_overview()
            except Exception as exc:
                overview = f"Workspace root preview unavailable: {type(exc).__name__}: {exc}"
            if overview:
                lines.extend(["", overview])
        return "\n".join(lines)

    def step(self, actions: List[Dict[str, Any]]) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}.")
        observations: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for idx, action in enumerate(actions):
            task = self.current_tasks[idx]
            runtime = self.runtimes[idx]
            if self.dones[idx]:
                result = R2EToolResult("Episode already completed.", done=True, is_action_valid=True)
                shaping_reward = 0.0
                shaping_info = self._empty_reward_breakdown()
            else:
                self.step_counts[idx] += 1
                result = self._execute_action(idx, runtime, action)
                self._record_action_result(idx, action, result)
                shaping_reward, shaping_info = self._apply_reward_shaping(idx, action, result)
                if not result.done and self.auto_submit_on_max_steps and self.step_counts[idx] >= self.max_steps:
                    submit_result = runtime.submit()
                    submit_result.observation = "Maximum step count reached; auto-submitting current workspace.\n" + submit_result.observation
                    submit_shaping, submit_breakdown = self._apply_reward_shaping(
                        idx,
                        {"tool_name": "submit", "parameters": {}},
                        submit_result,
                        auto_submitted=True,
                    )
                    shaping_reward += submit_shaping
                    shaping_info = self._merge_reward_breakdowns(shaping_info, submit_breakdown)
                    result = submit_result
                self.dones[idx] = result.done

            terminal_reward = float(result.reward)
            reward = terminal_reward + shaping_reward
            if not result.is_action_valid and not result.done:
                reward -= float(self.invalid_action_penalty)
            info = self._base_info(idx, task, runtime)
            info.update(result.info)
            info.update(shaping_info)
            info.update(
                {
                    "is_action_valid": bool(result.is_action_valid),
                    "tool_calling": 1 if result.is_action_valid else 0,
                    "tool_execution_success": bool(info.get("tool_execution_success", result.is_action_valid)),
                    "terminal_r2e_reward": terminal_reward,
                    "shaping_reward": shaping_reward,
                    "total_reward": reward,
                    "successful_source_edit_count": self.successful_source_edit_counts[idx],
                }
            )
            info["won"] = bool(info.get("won", reward == 1.0))
            observations.append(result.observation)
            rewards.append(reward)
            dones.append(bool(self.dones[idx]))
            infos.append(info)
            self._write_event(
                idx,
                {
                    "event": "step",
                    "step": self.step_counts[idx],
                    "action": action,
                    "reward": reward,
                    "done": bool(self.dones[idx]),
                    "observation": result.observation,
                    "info": info,
                },
            )
        return observations, rewards, dones, infos

    def _action_signature(self, action: Dict[str, Any]) -> str:
        return json.dumps(action or {}, ensure_ascii=False, sort_keys=True, default=str)

    def _ensure_no_progress_signature_state(self, idx: int) -> None:
        while len(self.seen_no_progress_action_signatures) <= idx:
            self.seen_no_progress_action_signatures.append(set())
        while len(self.repeated_no_progress_action_counts) <= idx:
            self.repeated_no_progress_action_counts.append({})

    def _no_progress_action_signature(self, idx: int, action: Dict[str, Any]) -> Optional[str]:
        tool_name = action.get("tool_name")
        params = action.get("parameters") or {}
        if tool_name == "bash":
            command = str(params.get("cmd") or "").strip()
            if not command:
                return None
            return json.dumps(
                {
                    "tool_name": "bash",
                    "cmd": command,
                    "source_edit_count": self.successful_source_edit_counts[idx],
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        if tool_name == "str_replace_editor":
            if str(params.get("command") or "") != "view":
                return None
            path = str(params.get("path") or "").strip()
            if not path:
                return None
            return json.dumps(
                {
                    "tool_name": "str_replace_editor",
                    "command": "view",
                    "path": path,
                    "view_range": params.get("view_range"),
                    "source_edit_count": self.successful_source_edit_counts[idx],
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        return None

    def action_mask(self, idx: int, final_step: bool = False) -> Dict[str, Any]:
        if final_step:
            return {
                "allowed_tools": ["submit"],
                "masked_tools": {
                    "bash": "final_step_forces_submit",
                    "str_replace_editor": "final_step_forces_submit",
                    "validate": "final_step_forces_submit",
                },
                "allow_bash": False,
                "allow_str_replace_editor": False,
                "allow_validate": False,
                "allow_submit": True,
                "submit_reason": "final_forced_submit",
            }

        source_edits = self.successful_source_edit_counts[idx] if idx < len(self.successful_source_edit_counts) else 0
        validations = (
            self.validation_after_source_edit_counts[idx]
            if idx < len(self.validation_after_source_edit_counts)
            else 0
        )
        has_source_edit = source_edits > 0
        has_required_validation = (not self.require_validation_before_submit) or validations > 0
        allow_validate = has_source_edit
        allow_submit = (not self.require_successful_edit_before_submit or has_source_edit) and has_required_validation

        allowed_tools = ["bash", "str_replace_editor"]
        masked_tools: Dict[str, str] = {}
        if allow_validate:
            allowed_tools.append("validate")
        else:
            masked_tools["validate"] = "requires_successful_source_edit"
        if allow_submit:
            allowed_tools.append("submit")
            submit_reason = "allowed"
        elif has_source_edit and self.require_validation_before_submit and validations <= 0:
            masked_tools["submit"] = "requires_validation_after_source_edit"
            submit_reason = "requires_validation_after_source_edit"
        else:
            masked_tools["submit"] = "requires_successful_source_edit_and_validation"
            submit_reason = "requires_successful_source_edit_and_validation"

        return {
            "allowed_tools": allowed_tools,
            "masked_tools": masked_tools,
            "allow_bash": True,
            "allow_str_replace_editor": True,
            "allow_validate": allow_validate,
            "allow_submit": allow_submit,
            "validate_reason": masked_tools.get("validate", "allowed"),
            "submit_reason": submit_reason,
            "successful_source_edit_count": source_edits,
            "validation_after_source_edit_count": validations,
        }

    def _edit_command_succeeded(self, action: Dict[str, Any], result: R2EToolResult) -> bool:
        if action.get("tool_name") != "str_replace_editor":
            return False
        command = str((action.get("parameters") or {}).get("command") or "")
        if command not in {"create", "str_replace", "insert"}:
            return False
        if is_noop_str_replace(action):
            return False
        return bool(result.info.get("tool_execution_success", result.is_action_valid))

    def _source_edit_command_succeeded(self, action: Dict[str, Any], result: R2EToolResult) -> bool:
        if not self._edit_command_succeeded(action, result):
            return False
        path = str((action.get("parameters") or {}).get("path") or "")
        return classify_r2e_path(path) == "source"

    def _record_action_result(self, idx: int, action: Dict[str, Any], result: R2EToolResult) -> None:
        if self._edit_command_succeeded(action, result):
            self.successful_edit_counts[idx] += 1
        if self._source_edit_command_succeeded(action, result):
            self.successful_source_edit_counts[idx] += 1

        if self._validation_command_after_source_edit(idx, action, result):
            self.validation_after_source_edit_counts[idx] += 1

        execution_success = bool(result.info.get("tool_execution_success", result.is_action_valid))
        if result.is_action_valid and execution_success:
            no_progress_signature = self._no_progress_action_signature(idx, action)
            if no_progress_signature is not None:
                self._ensure_no_progress_signature_state(idx)
                self.seen_no_progress_action_signatures[idx].add(no_progress_signature)
                self.repeated_no_progress_action_counts[idx].setdefault(no_progress_signature, 0)
            self.last_failed_action_signatures[idx] = None
            self.repeated_failed_action_counts[idx] = 0
            return

        signature = self._action_signature(action)
        if self.last_failed_action_signatures[idx] == signature:
            self.repeated_failed_action_counts[idx] += 1
        else:
            self.last_failed_action_signatures[idx] = signature
            self.repeated_failed_action_counts[idx] = 1

    def _validation_command_after_source_edit(self, idx: int, action: Dict[str, Any], result: R2EToolResult) -> bool:
        if action.get("tool_name") not in {"bash", "validate"}:
            return False
        if self.successful_source_edit_counts[idx] <= 0:
            return False
        command = str((action.get("parameters") or {}).get("cmd") or "")
        if not is_validation_command(command):
            return False
        # A failing test command is still useful validation feedback. Count only
        # commands that reached the R2E bash runtime, not adapter-side blocks.
        return result.is_action_valid and "exit_code" in result.info

    def _repeated_no_progress_block(self, idx: int, action: Dict[str, Any]) -> Optional[R2EToolResult]:
        signature = self._no_progress_action_signature(idx, action)
        if signature is None:
            return None
        self._ensure_no_progress_signature_state(idx)
        if signature not in self.seen_no_progress_action_signatures[idx]:
            return None
        count = self.repeated_no_progress_action_counts[idx].get(signature, 0) + 1
        self.repeated_no_progress_action_counts[idx][signature] = count
        limit = self.max_repeated_no_progress_actions
        reached_limit = limit >= 0 and count >= limit
        params = action.get("parameters") or {}
        tool_name = str(action.get("tool_name") or "")
        if tool_name == "bash":
            target = str(params.get("cmd") or "the same bash command")
            repeated_message = (
                "Repeated no-progress tool call blocked: you already ran this exact bash command "
                "without making a source edit. Inspect a relevant file/range, search a different symbol, "
                "or make a real str_replace instead of repeating the same command."
            )
        else:
            target = str(params.get("path") or "the same file")
            repeated_message = (
                "Repeated no-progress tool call blocked: you already viewed this exact file/range "
                "without making a source edit. Inspect a different range, search a different symbol, "
                "run a targeted test, or make a real str_replace before viewing it again."
            )
        if reached_limit:
            message = (
                "Repeated no-progress limit reached: the same tool call was requested too many times "
                "without a source edit. Ending this episode so training receives a clear stuck-trajectory signal."
            )
            fail_reason = "repeated_no_progress_action_limit"
        else:
            message = repeated_message
            fail_reason = "repeated_no_progress_action"
        return R2EToolResult(
            message,
            done=reached_limit,
            is_action_valid=True,
            info={
                "fail_reason": fail_reason,
                "stderr": message,
                "edit_path": target,
                "no_progress_tool_name": tool_name,
                "exit_reason": fail_reason if reached_limit else None,
                "repeated_no_progress_action": True,
                "repeated_no_progress_action_count": count,
                "max_repeated_no_progress_actions": limit,
                "tool_execution_success": False,
                "tool_execution_fail_reason": fail_reason,
            },
        )

    def _repeated_failure_block(self, idx: int, action: Dict[str, Any]) -> Optional[R2EToolResult]:
        if self.max_repeated_failed_actions < 0:
            return None
        signature = self._action_signature(action)
        if self.last_failed_action_signatures[idx] != signature:
            return None
        if self.repeated_failed_action_counts[idx] < self.max_repeated_failed_actions:
            return None
        if self._submit_requires_validation(idx, action):
            return self._blocked_submit_before_validation(repeated=True)
        if self._validate_requires_source_edit(idx, action):
            return self._blocked_validate_before_source_edit(repeated=True)
        block_count = max(1, self.repeated_failed_action_counts[idx] - self.max_repeated_failed_actions + 1)
        limit = self.max_repeated_failed_action_blocks
        reached_limit = limit >= 0 and block_count >= limit
        if reached_limit:
            message = (
                "Repeated failed-action limit reached: the same failed tool call was blocked too many times. "
                "Ending this episode so training receives a clear stuck-trajectory signal."
            )
            fail_reason = "repeated_failed_action_limit"
        else:
            message = (
                "Repeated failed tool call blocked. Do not repeat the same failed action. "
                "If grep failed, include both a pattern and an actual path from the workspace preview, "
                "for example: { grep -RIn -- 'term' . | head -50; status=${PIPESTATUS[0]}; test \"$status\" -eq 0 -o \"$status\" -eq 141; }. "
                "If a path was not found, run pwd && find . -maxdepth 3 -type f | head -80 from /testbed. "
                "If a file view was clipped, use str_replace_editor view with a narrow view_range before editing."
            )
            fail_reason = "repeated_failed_action"
        return R2EToolResult(
            message,
            done=reached_limit,
            is_action_valid=True,
            info={
                "fail_reason": fail_reason,
                "stderr": message,
                "exit_reason": fail_reason if reached_limit else None,
                "repeated_failed_action_block_count": block_count,
                "max_repeated_failed_action_blocks": limit,
                "tool_execution_success": False,
                "tool_execution_fail_reason": fail_reason,
            },
        )

    def _validate_requires_source_edit(self, idx: int, action: Dict[str, Any]) -> bool:
        return action.get("tool_name") == "validate" and self.successful_source_edit_counts[idx] <= 0

    def _blocked_validate_before_source_edit(self, repeated: bool = False) -> R2EToolResult:
        prefix = "Validate still masked" if repeated else "Validate masked"
        message = (
            f"{prefix}: make at least one successful source edit before using validate. "
            "Use bash to inspect files or str_replace_editor to make a source edit first."
        )
        return R2EToolResult(
            message,
            is_action_valid=True,
            info={
                "fail_reason": "validate_before_successful_source_edit",
                "stderr": message,
                "tool_execution_success": False,
                "tool_execution_fail_reason": "validate_before_successful_source_edit",
                "validate_masked": True,
            },
        )

    def _blocked_validate_non_test_command(self, command: str) -> R2EToolResult:
        message = (
            "Validate masked: validate requires a test-like command such as python -m pytest -q, "
            "pytest -q, tox, unittest, r2e_tests, or run_tests. Use bash for search/view shell commands."
        )
        return R2EToolResult(
            message,
            is_action_valid=True,
            info={
                "fail_reason": "validate_requires_test_command",
                "stderr": message,
                "validate_cmd": command,
                "tool_execution_success": False,
                "tool_execution_fail_reason": "validate_requires_test_command",
                "validate_masked": True,
            },
        )

    def _submit_requires_validation(self, idx: int, action: Dict[str, Any]) -> bool:
        return (
            action.get("tool_name") == "submit"
            and self.require_validation_before_submit
            and self.successful_source_edit_counts[idx] > 0
            and self.validation_after_source_edit_counts[idx] <= 0
        )

    def _blocked_submit_before_validation(self, repeated: bool = False) -> R2EToolResult:
        prefix = "Submit still blocked" if repeated else "Submit blocked"
        message = (
            f"{prefix}: run a validation command after your source edit before submitting. "
            "Your next action should be a single bash validation command, for example:\n"
            "<function=bash>\n"
            "<parameter=cmd>python -m pytest -q</parameter>\n"
            "</function>\n"
            "A failing or timed-out validation command is useful feedback; inspect its output, revise the patch if needed, "
            "then submit only after validation has run."
        )
        return R2EToolResult(
            message,
            is_action_valid=True,
            info={
                "fail_reason": "submit_before_validation_after_source_edit",
                "stderr": message,
                "repeated_submit_before_validation": bool(repeated),
                "tool_execution_success": False,
                "tool_execution_fail_reason": "submit_before_validation_after_source_edit",
            },
        )

    def _blocked_editor_write(self, path_kind: str, path: str) -> Optional[R2EToolResult]:
        if path_kind == "test":
            message = (
                "Cannot modify test files. You may view tests to understand expected behavior, "
                "but fixes must edit source files under /testbed."
            )
            return R2EToolResult(
                message,
                is_action_valid=True,
                info={
                    "fail_reason": "test_edit_blocked",
                    "stderr": message,
                    "edit_path_kind": path_kind,
                    "edit_path": path,
                    "test_edit_blocked": True,
                    "tool_execution_success": False,
                    "tool_execution_fail_reason": "test_edit_blocked",
                },
            )
        if path_kind == "r2e_aux":
            message = "Cannot modify R2E/runtime auxiliary files. Fix the repository source code instead."
            return R2EToolResult(
                message,
                is_action_valid=True,
                info={
                    "fail_reason": "r2e_aux_edit_blocked",
                    "stderr": message,
                    "edit_path_kind": path_kind,
                    "edit_path": path,
                    "r2e_aux_edit_blocked": True,
                    "tool_execution_success": False,
                    "tool_execution_fail_reason": "r2e_aux_edit_blocked",
                },
            )
        return None

    def _blocked_noop_edit(self, action: Dict[str, Any], path_kind: str, path: str) -> Optional[R2EToolResult]:
        if not is_noop_str_replace(action):
            return None
        message = (
            "No-op str_replace blocked: old_str and new_str are identical, so no patch would be created. "
            "Change new_str to make a real source edit, or inspect more context before editing."
        )
        return R2EToolResult(
            message,
            is_action_valid=True,
            info={
                "fail_reason": "noop_edit",
                "stderr": message,
                "edit_path_kind": path_kind,
                "edit_path": path,
                "noop_edit_blocked": True,
                "tool_execution_success": False,
                "tool_execution_fail_reason": "noop_edit",
            },
        )

    def _execute_action(self, idx: int, runtime: R2ERepoRuntime, action: Dict[str, Any]) -> R2EToolResult:
        if not action.get("tool_name"):
            error = action.get("error") or "Invalid action: missing tool call."
            return R2EToolResult(error, is_action_valid=False, info={"fail_reason": "invalid_action", "stderr": error})

        no_progress = self._repeated_no_progress_block(idx, action)
        if no_progress is not None:
            return no_progress

        repeated = self._repeated_failure_block(idx, action)
        if repeated is not None:
            return repeated

        tool_name = action["tool_name"]
        params = action.get("parameters") or {}
        if tool_name == "bash":
            return runtime.run_bash(params.get("cmd", ""), cwd=params.get("cwd"))
        if tool_name == "validate":
            command = str(params.get("cmd") or "")
            if self._validate_requires_source_edit(idx, action):
                return self._blocked_validate_before_source_edit()
            if not is_validation_command(command):
                return self._blocked_validate_non_test_command(command)
            result = runtime.run_bash(command, cwd=params.get("cwd"))
            result.info["validation_action"] = True
            result.info["validation_cmd"] = command
            return result
        if tool_name == "str_replace_editor":
            path = str(params.get("path") or "")
            path_kind = classify_r2e_path(path)
            if is_editor_write(action):
                blocked = self._blocked_editor_write(path_kind, path)
                if blocked is not None:
                    return blocked
                noop_blocked = self._blocked_noop_edit(action, path_kind, path)
                if noop_blocked is not None:
                    return noop_blocked
            result = runtime.run_editor(params)
            result.info.setdefault("edit_path_kind", path_kind)
            result.info.setdefault("edit_path", path)
            return result
        if tool_name == "submit":
            if (
                self.require_successful_edit_before_submit
                and not getattr(runtime, "setup_error", None)
                and self.successful_source_edit_counts[idx] <= 0
            ):
                message = (
                    "Submit blocked: make at least one successful source edit before submitting. "
                    "Inspect files with bash or str_replace_editor view, then use str_replace_editor create, str_replace, or insert."
                )
                return R2EToolResult(
                    message,
                    is_action_valid=True,
                    info={
                        "fail_reason": "submit_before_successful_source_edit",
                        "stderr": message,
                        "tool_execution_success": False,
                        "tool_execution_fail_reason": "submit_before_successful_source_edit",
                    },
                )
            if not getattr(runtime, "setup_error", None) and self._submit_requires_validation(idx, action):
                return self._blocked_submit_before_validation()
            return runtime.submit()
        message = f"Invalid action: unknown tool '{tool_name}'."
        return R2EToolResult(message, is_action_valid=False, info={"fail_reason": "unknown_tool", "stderr": message})

    def _empty_reward_breakdown(self) -> Dict[str, Any]:
        return {
            "reward_breakdown": {
                "terminal_r2e_reward": 0.0,
                "shaping_delta": 0.0,
                "penalty": 0.0,
                "phi_before": 0.0,
                "phi_after": 0.0,
                "events": [],
            },
            "shaping_reward": 0.0,
        }

    def _merge_reward_breakdowns(self, first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
        a = dict(first.get("reward_breakdown") or {})
        b = dict(second.get("reward_breakdown") or {})
        return {
            "reward_breakdown": {
                "terminal_r2e_reward": b.get("terminal_r2e_reward", a.get("terminal_r2e_reward", 0.0)),
                "shaping_delta": float(a.get("shaping_delta", 0.0)) + float(b.get("shaping_delta", 0.0)),
                "penalty": float(a.get("penalty", 0.0)) + float(b.get("penalty", 0.0)),
                "phi_before": a.get("phi_before", 0.0),
                "phi_after": b.get("phi_after", a.get("phi_after", 0.0)),
                "events": list(a.get("events") or []) + list(b.get("events") or []),
            },
            "shaping_reward": float(first.get("shaping_reward", 0.0)) + float(second.get("shaping_reward", 0.0)),
        }

    def _apply_reward_shaping(
        self,
        idx: int,
        action: Dict[str, Any],
        result: R2EToolResult,
        auto_submitted: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        if idx >= len(self.reward_shaping_states):
            return 0.0, self._empty_reward_breakdown()
        patch_text = None
        patch_path = result.info.get("patch_path")
        if patch_path:
            try:
                patch_text = Path(patch_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                patch_text = None
        result_info = dict(result.info)
        result_info.setdefault("is_action_valid", result.is_action_valid)
        shaping_reward, breakdown = self.reward_shaping_states[idx].update(
            self.reward_shaping_config,
            action,
            result_info,
            patch_text=patch_text,
            auto_submitted=auto_submitted,
        )
        return shaping_reward, {
            "reward_breakdown": breakdown,
            "shaping_reward": shaping_reward,
        }

    def _base_info(self, idx: int, task: R2ECodeSWETask, runtime: R2ERepoRuntime) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "dataset_name": task.dataset_name,
            "split": task.split,
            "repo": task.repo,
            "repo_name": task.repo_name,
            "docker_image": task.docker_image,
            "base_commit": task.base_commit,
            "setup_error": runtime.setup_error,
            "trajectory_dir": str(runtime.episode_dir) if runtime.episode_dir else None,
            "step_count": self.step_counts[idx] if idx < len(self.step_counts) else 0,
            "won": False,
            "validation_after_source_edit_count": (
                self.validation_after_source_edit_counts[idx]
                if idx < len(self.validation_after_source_edit_counts)
                else 0
            ),
            "action_mask": self.action_mask(idx) if idx < len(getattr(self, "current_tasks", [])) else {},
        }

    def _write_event(self, idx: int, event: Dict[str, Any]) -> None:
        runtime = self.runtimes[idx]
        if runtime.episode_dir is None:
            return
        path = runtime.episode_dir / "trajectory.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        for runtime in getattr(self, "runtimes", []):
            runtime.close()
        self.runtimes = []


def build_r2e_code_swe_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    env_config: Any = None,
    max_steps: int = 20,
    resources_per_worker: Any = None,
    is_train: bool = True,
) -> R2ECodeSWEEnv:
    del seed, resources_per_worker
    r2e_config = getattr(env_config, "r2e_code_swe", env_config)
    tasks = load_r2e_tasks_from_config(r2e_config, is_train=is_train)
    runtime_config = R2ERuntimeConfig.from_obj(getattr(r2e_config, "runtime", None))
    return R2ECodeSWEEnv(
        tasks=tasks,
        runtime_config=runtime_config,
        env_num=env_num,
        group_n=group_n,
        max_steps=max_steps,
        invalid_action_penalty=float(getattr(r2e_config, "invalid_action_penalty", 0.0)),
        auto_submit_on_max_steps=bool(getattr(r2e_config, "auto_submit_on_max_steps", True)),
        require_successful_edit_before_submit=bool(getattr(r2e_config, "require_successful_edit_before_submit", True)),
        require_validation_before_submit=bool(getattr(r2e_config, "require_validation_before_submit", True)),
        max_repeated_failed_actions=int(getattr(r2e_config, "max_repeated_failed_actions", 1)),
        max_repeated_failed_action_blocks=int(getattr(r2e_config, "max_repeated_failed_action_blocks", 3)),
        max_repeated_no_progress_actions=int(getattr(r2e_config, "max_repeated_no_progress_actions", 3)),
        reward_shaping_config=R2ERewardShapingConfig.from_obj(getattr(r2e_config, "reward_shaping", None)),
        is_train=is_train,
    )
