from dataclasses import dataclass
from typing import Any, Dict, Optional

from .runtime import CommandResult, WorkspaceRuntime, _truncate
from .tasks import CodeSWETask


@dataclass
class RewardResult:
    reward: float
    won: bool
    skipped: bool
    fail_reason: Optional[str]
    info: Dict[str, Any]
    observation: str


class TestRewardEvaluator:
    def __init__(self, max_output_chars: int = 12000):
        self.max_output_chars = max_output_chars

    def evaluate(self, runtime: WorkspaceRuntime, task: CodeSWETask) -> RewardResult:
        patch, patch_path = runtime.export_patch()
        test_patch_error = runtime.apply_test_patch()
        install_result = runtime.run_install_if_configured()
        test_command = runtime.get_test_command()

        base_info: Dict[str, Any] = {
            "task_id": task.task_id,
            "dataset_name": task.dataset_name,
            "repo": task.repo,
            "test_command": test_command,
            "patch_path": patch_path,
            "patch_chars": len(patch),
            "setup_error": runtime.setup_error,
            "test_patch_error": test_patch_error,
        }

        if runtime.setup_error:
            observation = f"Submit received, but workspace setup failed: {runtime.setup_error}\nPatch path: {patch_path}"
            base_info.update({"exit_code": None, "stdout": "", "stderr": "", "fail_reason": "setup_failed"})
            return RewardResult(0.0, False, True, "setup_failed", base_info, observation)

        if test_patch_error:
            observation = f"Submit received, but applying the dataset test_patch failed:\n{test_patch_error}\nPatch path: {patch_path}"
            base_info.update({"exit_code": None, "stdout": "", "stderr": test_patch_error, "fail_reason": "test_patch_failed"})
            return RewardResult(0.0, False, True, "test_patch_failed", base_info, observation)

        if install_result is not None and install_result.exit_code != 0:
            observation = (
                "Submit received, but install_command failed. This task is treated as failed/skipped under no-Docker runtime.\n"
                + install_result.observation
                + f"\nPatch path: {patch_path}"
            )
            base_info.update(
                {
                    "exit_code": install_result.exit_code,
                    "stdout": install_result.stdout,
                    "stderr": install_result.stderr,
                    "fail_reason": "install_failed",
                }
            )
            return RewardResult(0.0, False, True, "install_failed", base_info, observation)

        if not test_command:
            observation = (
                "Submit received, but no local test command could be derived. "
                "Docker metadata is ignored by this no-Docker environment.\n"
                f"Patch path: {patch_path}"
            )
            base_info.update({"exit_code": None, "stdout": "", "stderr": "", "fail_reason": "no_test_command"})
            return RewardResult(0.0, False, True, "no_test_command", base_info, observation)

        result: CommandResult = runtime.run_bash(
            test_command,
            timeout=runtime.config.reward_timeout,
            enforce_policy=False,
        )
        success = result.exit_code == 0
        fail_reason = None if success else (result.fail_reason or "tests_failed")
        base_info.update(
            {
                "exit_code": result.exit_code,
                "stdout": _truncate(result.stdout, self.max_output_chars),
                "stderr": _truncate(result.stderr, self.max_output_chars),
                "timed_out": result.timed_out,
                "fail_reason": fail_reason,
            }
        )
        observation = (
            f"Submit received. Patch exported to: {patch_path}\n"
            f"Reward: {1.0 if success else 0.0}\n"
            f"Test command: {test_command}\n"
            f"{result.observation}"
        )
        return RewardResult(1.0 if success else 0.0, success, False, fail_reason, base_info, observation)
