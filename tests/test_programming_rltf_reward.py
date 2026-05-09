
import math
import os
import pytest

from agent_system.environments.env_package.programming.envs import ProgrammingEnv, build_programming_envs


def write_quixbugs_task(tmp_path, test_source):
    programs = tmp_path / "python_programs"
    tests = tmp_path / "python_testcases"
    programs.mkdir()
    tests.mkdir()
    (programs / "bug.py").write_text("def add(a, b):\n    return 0\n")
    (tests / "test_bug.py").write_text(test_source)
    return tmp_path


def test_rltf_scalar_reward_uses_partial_test_pass_rate(tmp_path):
    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\n"
        "def test_positive():\n"
        "    assert add(1, 2) == 3\n\n"
        "def test_negative():\n"
        "    assert add(-1, -2) == -3\n",
    )
    env = ProgrammingEnv(data_root=str(data_root), reward_mode="rltf_scalar")
    env.reset()

    _, rewards, dones, infos = env.step([
        "def add(a, b):\n"
        "    if a > 0:\n"
        "        return a + b\n"
        "    return 0\n"
    ])

    assert dones == [False]
    assert math.isclose(rewards[0], 0.35, abs_tol=1e-9)
    assert infos[0]["feedback_category"] == "failure"
    assert infos[0]["n_pass"] == 1
    assert infos[0]["n_fail"] == 1
    assert math.isclose(infos[0]["pass_ratio"], 0.5, abs_tol=1e-9)
    assert math.isclose(infos[0]["reward_components"]["adaptive"], 0.35, abs_tol=1e-9)


def test_rltf_scalar_reward_penalizes_syntax_errors(tmp_path):
    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )
    env = ProgrammingEnv(data_root=str(data_root), reward_mode="rltf_scalar")
    env.reset()

    _, rewards, dones, infos = env.step(["def add(:\n    return 3\n"])

    assert dones == [False]
    assert rewards == [-1.0]
    assert infos[0]["feedback_category"] == "syntax_error"
    assert infos[0]["sub_error"] == "SyntaxError"
    assert infos[0]["n_pass"] == 0
    assert infos[0]["n_fail"] >= 1


def test_binary_reward_mode_preserves_existing_pass_fail_behavior(tmp_path):
    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )
    env = ProgrammingEnv(data_root=str(data_root), reward_mode="binary")
    env.reset()

    _, rewards, _, infos = env.step(["def add(a, b):\n    return a - b\n"])

    assert rewards == [0.0]
    assert infos[0]["feedback_category"] == "failure"


def test_build_programming_envs_passes_reward_mode(tmp_path):
    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )

    env = build_programming_envs(data_root=str(data_root), reward_mode="rltf_scalar")

    assert env.reward_mode == "rltf_scalar"


def test_rltf_reward_uses_current_python_for_pytest_when_path_lacks_pytest(tmp_path, monkeypatch):
    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = ProgrammingEnv(data_root=str(data_root), reward_mode="rltf_scalar")
    env.reset()

    _, rewards, dones, infos = env.step(["def add(a, b):\n    return a + b\n"])

    assert rewards == [1.0]
    assert dones == [True]
    assert infos[0]["feedback_category"] == "pass"


def test_rltf_reward_classifies_pytest_timeout(tmp_path, monkeypatch):
    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )

    def raise_timeout(*args, **kwargs):
        import subprocess
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=10)

    monkeypatch.setattr("agent_system.environments.env_package.programming.envs.subprocess.run", raise_timeout)
    env = ProgrammingEnv(data_root=str(data_root), reward_mode="rltf_scalar")
    env.reset()

    _, rewards, dones, infos = env.step(["def add(a, b):\n    return a + b\n"])

    assert dones == [False]
    assert rewards == [-0.6]
    assert infos[0]["feedback_category"] == "timeout"
    assert infos[0]["sub_error"] == "Timeout"


def test_pytest_start_oserror_fails_fast(tmp_path, monkeypatch):
    import pytest

    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )

    def raise_oserror(*args, **kwargs):
        raise OSError("pytest cannot start")

    monkeypatch.setattr("agent_system.environments.env_package.programming.envs.subprocess.run", raise_oserror)
    env = ProgrammingEnv(data_root=str(data_root), reward_mode="rltf_scalar")
    env.reset()

    with pytest.raises(RuntimeError, match="Failed to start pytest for ProgrammingEnv"):
        env.step(["def add(a, b):\n    return a + b\n"])


def test_programming_env_reset_honors_env_num_and_group_n(tmp_path):
    data_root = write_quixbugs_task(tmp_path, test_source="def test_add():\\n    assert add(1, 2) == 3\\n")
    env = build_programming_envs(
        data_root=str(data_root),
        env_num=1,
        group_n=2,
        reward_mode="rltf_scalar",
    )

    obs, infos = env.reset()

    assert len(obs) == 2
    assert len(infos) == 2
    assert all("file" in info for info in infos)


def test_programming_env_step_handles_batched_actions(tmp_path):
    data_root = write_quixbugs_task(tmp_path, test_source="def test_add():\\n    assert add(1, 2) == 3\\n")
    env = build_programming_envs(
        data_root=str(data_root),
        env_num=1,
        group_n=2,
        reward_mode="rltf_scalar",
    )
    obs, _ = env.reset()

    next_obs, rewards, dones, infos = env.step(["", ""])

    assert len(obs) == 2
    assert len(next_obs) == 2
    assert len(rewards) == 2
    assert len(dones) == 2
    assert len(infos) == 2
    assert rewards == [-0.6, -0.6]


def write_two_quixbugs_tasks(tmp_path):
    programs = tmp_path / "python_programs"
    tests = tmp_path / "python_testcases"
    programs.mkdir()
    tests.mkdir()
    (programs / "a_bug.py").write_text("def fix_a():\n    return 0\n")
    (programs / "b_bug.py").write_text("def fix_b():\n    return 0\n")
    (tests / "test_a_bug.py").write_text(
        "from python_programs.a_bug import fix_a\n\n"
        "def test_a():\n"
        "    assert fix_a() == 1\n"
    )
    (tests / "test_b_bug.py").write_text(
        "from python_programs.b_bug import fix_b\n\n"
        "def test_b():\n"
        "    assert fix_b() == 2\n"
    )
    return tmp_path


def test_programming_env_selects_env_num_tasks_and_repeats_by_group_n(tmp_path):
    data_root = write_two_quixbugs_tasks(tmp_path)
    env = build_programming_envs(
        data_root=str(data_root),
        env_num=2,
        group_n=2,
        reward_mode="rltf_scalar",
    )

    _, infos = env.reset()

    assert [info["file"] for info in infos] == [
        "a_bug.py",
        "a_bug.py",
        "b_bug.py",
        "b_bug.py",
    ]


def test_programming_env_batched_non_empty_actions_are_isolated(tmp_path):
    data_root = write_two_quixbugs_tasks(tmp_path)
    env = build_programming_envs(
        data_root=str(data_root),
        env_num=2,
        group_n=1,
        reward_mode="rltf_scalar",
    )
    env.reset()

    _, rewards, dones, infos = env.step([
        "def fix_a():\n    return 1\n",
        "def fix_b():\n    return 0\n",
    ])

    assert rewards == [1.0, -0.3]
    assert dones == [True, False]
    assert [info["file"] for info in infos] == ["a_bug.py", "b_bug.py"]


def test_programming_env_step_rejects_wrong_action_count(tmp_path):
    data_root = write_quixbugs_task(tmp_path, "from python_programs.bug import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    env = build_programming_envs(
        data_root=str(data_root),
        env_num=1,
        group_n=2,
        reward_mode="rltf_scalar",
    )
    env.reset()

    with pytest.raises(ValueError, match="Expected 2 actions, got 1"):
        env.step([""])


def test_programming_env_rejects_non_positive_env_sizes(tmp_path):
    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )

    with pytest.raises(ValueError, match="env_num and group_n must be positive integers"):
        build_programming_envs(data_root=str(data_root), env_num=0, group_n=1)

    with pytest.raises(ValueError, match="env_num and group_n must be positive integers"):
        build_programming_envs(data_root=str(data_root), env_num=1, group_n=0)


def test_programming_env_close_removes_workspace_root(tmp_path):
    data_root = write_quixbugs_task(
        tmp_path,
        "from python_programs.bug import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )
    env = build_programming_envs(data_root=str(data_root), env_num=1, group_n=1)
    workspace_root = env._workspace_root

    assert os.path.exists(workspace_root)

    env.close()

    assert not os.path.exists(workspace_root)

