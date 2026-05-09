from typing import Any, Dict, List, Optional, Tuple

try:
    import gym
except ImportError:
    class _Env:
        pass

    class gym:
        Env = _Env

from .reward import TestRewardEvaluator
from .runtime import RuntimeConfig, WorkspaceRuntime
from .tasks import CodeSWETask, load_tasks_from_config


class CodeSWEEnv(gym.Env):
    def __init__(
        self,
        tasks: List[CodeSWETask],
        runtime_config: Optional[RuntimeConfig] = None,
        env_num: int = 1,
        group_n: int = 1,
        max_steps: int = 20,
        invalid_action_penalty: Optional[float] = None,
        auto_submit_on_max_steps: bool = True,
        is_train: bool = True,
    ):
        if not tasks:
            raise ValueError("CodeSWEEnv requires at least one task.")
        if env_num < 1 or group_n < 1:
            raise ValueError("env_num and group_n must be positive integers.")
        self.tasks = tasks
        self.runtime_config = runtime_config or RuntimeConfig()
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.max_steps = max_steps
        self.invalid_action_penalty = (
            self.runtime_config.invalid_action_penalty
            if invalid_action_penalty is None
            else invalid_action_penalty
        )
        self.auto_submit_on_max_steps = auto_submit_on_max_steps
        self.is_train = is_train
        self._cursor = 0
        self.runtimes: List[WorkspaceRuntime] = []
        self.current_tasks: List[CodeSWETask] = []
        self.step_counts: List[int] = []
        self.dones: List[bool] = []
        self.evaluator = TestRewardEvaluator(max_output_chars=self.runtime_config.max_output_chars)

    def reset(self, kwargs: Any = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.close()
        selected = []
        for i in range(self.env_num):
            selected.append(self.tasks[(self._cursor + i) % len(self.tasks)])
        self._cursor = (self._cursor + self.env_num) % len(self.tasks)

        self.current_tasks = [task for task in selected for _ in range(self.group_n)]
        self.runtimes = []
        self.step_counts = [0 for _ in range(self.num_processes)]
        self.dones = [False for _ in range(self.num_processes)]
        observations: List[str] = []
        infos: List[Dict[str, Any]] = []

        for idx, task in enumerate(self.current_tasks):
            runtime = WorkspaceRuntime(self.runtime_config)
            runtime.setup(task, replica_id=str(idx))
            self.runtimes.append(runtime)
            observations.append(self._initial_observation(task, runtime))
            infos.append(self._base_info(idx, task, runtime))

        return observations, infos

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
                observations.append("Episode already completed.")
                rewards.append(0.0)
                dones.append(True)
                info = self._base_info(idx, task, runtime)
                info.update({"won": False, "is_action_valid": True, "tool_calling": 0})
                infos.append(info)
                continue

            self.step_counts[idx] += 1
            obs, reward, done, info = self._execute_action(idx, action)

            if not done and self.auto_submit_on_max_steps and self.step_counts[idx] >= self.max_steps:
                reward_result = self.evaluator.evaluate(runtime, task)
                obs = "Maximum step count reached; auto-submitting current workspace.\n" + reward_result.observation
                reward = reward_result.reward
                done = True
                info.update(reward_result.info)
                info.update({"won": reward_result.won, "skipped": reward_result.skipped})

            self.dones[idx] = done
            observations.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)

        return observations, rewards, dones, infos

    def _execute_action(self, idx: int, action: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        task = self.current_tasks[idx]
        runtime = self.runtimes[idx]
        info = self._base_info(idx, task, runtime)
        info.update({"tool_calling": 0, "won": False, "skipped": False})

        if not action.get("tool_name"):
            message = action.get("error") or "Invalid action: missing tool call."
            info.update({"is_action_valid": False, "fail_reason": "invalid_action"})
            return message, -float(self.invalid_action_penalty), False, info

        tool_name = action["tool_name"]
        params = action.get("parameters") or {}

        if tool_name == "bash":
            result = runtime.run_bash(params.get("cmd", ""), cwd=params.get("cwd"))
            info.update(
                {
                    "is_action_valid": result.is_action_valid,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "fail_reason": result.fail_reason,
                    "tool_calling": 1 if result.is_action_valid else 0,
                }
            )
            reward = 0.0 if result.is_action_valid else -float(self.invalid_action_penalty)
            return result.observation, reward, False, info

        if tool_name == "str_replace_editor":
            result = runtime.run_editor(params)
            info.update(
                {
                    "is_action_valid": result.is_action_valid,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "fail_reason": result.fail_reason,
                    "tool_calling": 1 if result.is_action_valid else 0,
                }
            )
            reward = 0.0 if result.is_action_valid else -float(self.invalid_action_penalty)
            return result.observation, reward, False, info

        if tool_name == "submit":
            reward_result = self.evaluator.evaluate(runtime, task)
            info.update(reward_result.info)
            info.update(
                {
                    "is_action_valid": True,
                    "tool_calling": 1,
                    "won": reward_result.won,
                    "skipped": reward_result.skipped,
                }
            )
            return reward_result.observation, reward_result.reward, True, info

        info.update({"is_action_valid": False, "fail_reason": "unknown_tool"})
        return f"Invalid action: unknown tool '{tool_name}'.", -float(self.invalid_action_penalty), False, info

    def _initial_observation(self, task: CodeSWETask, runtime: WorkspaceRuntime) -> str:
        setup = "Workspace ready." if not runtime.setup_error else f"Workspace setup failed: {runtime.setup_error}"
        return (
            f"Task {task.task_id}\n"
            f"Repository: {task.repo}\n"
            f"Workspace: /testbed\n"
            f"{setup}\n\n"
            f"Issue:\n{task.problem_statement}"
        )

    def _base_info(self, idx: int, task: CodeSWETask, runtime: WorkspaceRuntime) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "dataset_name": task.dataset_name,
            "repo": task.repo,
            "base_commit": task.base_commit,
            "workspace_path": str(runtime.workspace_path) if runtime.workspace_path else None,
            "setup_error": runtime.setup_error,
            "step_count": self.step_counts[idx] if idx < len(self.step_counts) else 0,
            "won": False,
        }

    def close(self) -> None:
        for runtime in getattr(self, "runtimes", []):
            runtime.close()
        self.runtimes = []


def build_code_swe_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    env_config: Any = None,
    max_steps: int = 20,
    resources_per_worker: Any = None,
    is_train: bool = True,
) -> CodeSWEEnv:
    code_config = getattr(env_config, "code_swe", env_config)
    tasks = load_tasks_from_config(code_config, is_train=is_train)
    runtime_config = RuntimeConfig.from_obj(getattr(code_config, "runtime", None))
    if hasattr(code_config, "repo_url_map") and getattr(code_config, "repo_url_map"):
        runtime_config.repo_url_map.update(dict(getattr(code_config, "repo_url_map")))
    return CodeSWEEnv(
        tasks=tasks,
        runtime_config=runtime_config,
        env_num=env_num,
        group_n=group_n,
        max_steps=max_steps,
        invalid_action_penalty=getattr(code_config, "invalid_action_penalty", None),
        auto_submit_on_max_steps=bool(getattr(code_config, "auto_submit_on_max_steps", True)),
        is_train=is_train,
    )
