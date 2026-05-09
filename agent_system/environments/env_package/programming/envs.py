import gym
import os
import subprocess
import re
import sys
import shutil
import tempfile


class ProgrammingEnv(gym.Env):
    def __init__(self, data_root, max_steps=10, reward_mode="binary", env_num=1, group_n=1):
        self.data_root = data_root
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        if not isinstance(env_num, int) or not isinstance(group_n, int) or env_num < 1 or group_n < 1:
            raise ValueError("env_num and group_n must be positive integers")
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.step_count = 0

        self.tasks = self._load_tasks()
        self._workspace_root = tempfile.mkdtemp(prefix="programming_env_")

        # 保存每个 buggy 文件的原始内容，避免被模型输出污染后无法恢复
        self.original_codes = {}
        program_dir = os.path.join(self.data_root, "python_programs")
        for fname in self.tasks:
            path = os.path.join(program_dir, fname)
            with open(path, "r") as f:
                self.original_codes[fname] = f.read()

    def _load_tasks(self):
        path = os.path.join(self.data_root, "python_programs")
        files = [f for f in os.listdir(path) if f.endswith(".py")]
        files.sort()
        return files

    def reset(self):
        self.step_count = 0

        if self.env_num > len(self.tasks):
            raise ValueError(
                f"env_num ({self.env_num}) exceeds available tasks ({len(self.tasks)})"
            )

        selected_tasks = self.tasks[:self.env_num]
        self.cur_files = [
            task
            for task in selected_tasks
            for _ in range(self.group_n)
        ]
        self.workspaces = []
        self.codes = []
        obs = []
        infos = []

        if os.path.exists(self._workspace_root):
            shutil.rmtree(self._workspace_root)
        os.makedirs(self._workspace_root, exist_ok=True)

        for index, cur_file in enumerate(self.cur_files):
            workspace = os.path.join(self._workspace_root, str(index))
            shutil.copytree(self.data_root, workspace)
            file_path = os.path.join(workspace, "python_programs", cur_file)

            with open(file_path, "w") as f:
                f.write(self.original_codes[cur_file])

            with open(file_path, "r") as f:
                code = f.read()

            self.workspaces.append(workspace)
            self.codes.append(code)
            obs.append(f"""Fix the bug in this Python source file.

File to edit: python_programs/{cur_file}

You must output the complete fixed content of this file only.
Do not edit tests.
Do not output Markdown fences like ```python.

Current buggy code:
{code}
""")
            infos.append({"file": cur_file})

        self.cur_file = self.cur_files[0]
        self.code = self.codes[0]

        return obs, infos

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} actions, got {len(actions)}"
            )

        self.step_count += 1

        obs_list = []
        rewards = []
        dones = []
        infos = []

        for index, new_code in enumerate(actions):
            cur_file = self.cur_files[index]
            workspace = self.workspaces[index]
            file_path = os.path.join(workspace, "python_programs", cur_file)

            if not new_code.strip():
                success = False
                test_output = (
                    "Invalid action: empty code. "
                    "Please output complete fixed Python source code."
                )
                result = None
            else:
                with open(file_path, "w") as f:
                    f.write(new_code)

                try:
                    test_name = "test_" + cur_file
                    test_path = os.path.join("python_testcases", test_name)

                    env = os.environ.copy()
                    python_paths = [
                        workspace,
                        os.path.join(workspace, "python_testcases"),
                        env.get("PYTHONPATH", ""),
                    ]
                    env["PYTHONPATH"] = os.pathsep.join(
                        path for path in python_paths if path
                    )
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", "-q", test_path],
                        cwd=workspace,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=10,
                        text=True,
                        env=env,
                    )
                    success = result.returncode == 0
                    test_output = (result.stdout + "\n" + result.stderr).strip()

                except subprocess.TimeoutExpired as e:
                    success = False
                    test_output = f"Test execution timeout: {repr(e)}"
                    result = "timeout"
                except OSError as e:
                    raise RuntimeError("Failed to start pytest for ProgrammingEnv") from e
                except Exception as e:
                    success = False
                    test_output = f"Test execution error: {repr(e)}"
                    result = None

            feedback = self._build_feedback(success, test_output, result)
            reward = self._compute_reward(feedback)
            done = success or self.step_count >= self.max_steps

            if success:
                obs = "All tests passed."
            else:
                obs = "Tests failed.\n\n" + test_output[-4000:]

            info = {
                "won": success,
                "file": cur_file,
                **feedback,
            }
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return obs_list, rewards, dones, infos

    def _build_feedback(self, success, test_output, result):
        n_pass, n_fail = self._parse_test_counts(test_output)
        if success:
            category = "pass"
            sub_error = None
        elif result == "timeout":
            category = "timeout"
            sub_error = "Timeout"
        elif result is None:
            category = "error"
            sub_error = "ExecutionError"
        else:
            sub_error = self._extract_sub_error(test_output)
            if result.returncode == 124 or "Timeout" in test_output:
                category = "timeout"
                sub_error = sub_error or "Timeout"
            elif sub_error == "SyntaxError":
                category = "syntax_error"
            elif "failed" in test_output or n_pass > 0 or n_fail > 0:
                category = "failure"
            else:
                category = "error"

        total = n_pass + n_fail
        pass_ratio = n_pass / total if total else (1.0 if success else 0.0)
        adaptive = -0.3 + 1.3 * pass_ratio
        coarse = self._coarse_reward(category)

        return {
            "feedback_category": category,
            "sub_error": sub_error,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "pass_ratio": pass_ratio,
            "reward_components": {
                "coarse": coarse,
                "adaptive": adaptive,
                "final": self._compute_reward_value(category, adaptive),
            },
        }

    def _compute_reward(self, feedback):
        if self.reward_mode == "binary":
            return 1.0 if feedback["feedback_category"] == "pass" else 0.0
        if self.reward_mode == "rltf_scalar":
            return feedback["reward_components"]["final"]
        raise ValueError(f"Unsupported programming reward_mode: {self.reward_mode}")

    def _compute_reward_value(self, category, adaptive):
        if self.reward_mode == "binary":
            return 1.0 if category == "pass" else 0.0
        if category in {"pass", "failure"}:
            return adaptive
        return self._coarse_reward(category)

    def _coarse_reward(self, category):
        return {
            "pass": 1.0,
            "failure": -0.3,
            "error": -0.6,
            "timeout": -0.6,
            "syntax_error": -1.0,
        }.get(category, -0.6)

    def _parse_test_counts(self, test_output):
        passed = 0
        failed = 0
        for count, status in re.findall(r"(\d+)\s+(passed|failed|error|errors)", test_output):
            if status == "passed":
                passed += int(count)
            else:
                failed += int(count)
        return passed, failed

    def close(self):
        if os.path.exists(self._workspace_root):
            shutil.rmtree(self._workspace_root)

    def _extract_sub_error(self, test_output):
        for error_name in (
            "SyntaxError",
            "IndentationError",
            "IndexError",
            "TypeError",
            "ValueError",
            "EOFError",
            "TimeoutError",
            "NameError",
            "KeyError",
            "ImportError",
            "ZeroDivisionError",
            "RecursionError",
            "AssertionError",
        ):
            if error_name in test_output:
                return error_name
        return None


def build_programming_envs(
    seed=0,
    env_num=1,
    group_n=1,
    data_root=None,
    max_steps=10,
    reward_mode="binary",
    resources_per_worker=None,
    is_train=True,
):
    return ProgrammingEnv(
        data_root=data_root,
        max_steps=max_steps,
        reward_mode=reward_mode,
        env_num=env_num,
        group_n=group_n,
    )

