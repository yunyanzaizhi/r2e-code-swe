import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_system.environments.env_package.code_swe.envs import CodeSWEEnv
from agent_system.environments.env_package.code_swe.projection import code_swe_projection
from agent_system.environments.env_package.code_swe.runtime import RuntimeConfig, WorkspaceRuntime
from agent_system.environments.env_package.code_swe.tasks import normalize_task_record


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "tiny_repo"
    repo.mkdir()
    (repo / "hello.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", "hello.py")
    _git(repo, "commit", "-m", "base")
    return repo


def test_normalize_r2e_lite_record_handles_commit_schema():
    record = {
        "repo_name": "aiohttp",
        "docker_image": "namanjain12/aiohttp_final:abc123",
        "commit_hash": "abc123",
        "problem_statement": "[ISSUE]\nFix parser\n[/ISSUE]",
        "expected_output_json": json.dumps({"test_parser": "PASSED"}),
        "parsed_commit_content": "{}",
    }

    task = normalize_task_record(record, dataset_name="R2E-Gym/R2E-Gym-Lite", split="dev_10pr_v1", index=7)

    assert task.task_id == "R2E-Gym/R2E-Gym-Lite:dev_10pr_v1:7:aiohttp:abc123"
    assert task.repo == "aiohttp"
    assert task.base_commit == "abc123"
    assert task.test_spec["expected_output_json"] == {"test_parser": "PASSED"}
    assert task.raw_record["docker_image"] == "namanjain12/aiohttp_final:abc123"


def test_normalize_swebench_record_handles_json_lists():
    record = {
        "repo": "astropy/astropy",
        "instance_id": "astropy__astropy-12907",
        "base_commit": "d16bfe0",
        "problem_statement": "Bug report",
        "FAIL_TO_PASS": "[\"tests/test_a.py::test_bug\"]",
        "PASS_TO_PASS": ["tests/test_a.py::test_regression"],
        "patch": "diff --git a/a.py b/a.py\n",
        "test_patch": "diff --git a/tests/test_a.py b/tests/test_a.py\n",
        "run_tests": "python -m pytest tests/test_a.py",
        "docker_image": "metadata-only",
    }

    task = normalize_task_record(record, dataset_name="R2E-Gym/SWE-Bench-Lite", split="test", index=0)

    assert task.task_id == "astropy__astropy-12907"
    assert task.repo == "astropy/astropy"
    assert task.base_commit == "d16bfe0"
    assert task.gold_patch_optional.startswith("diff --git")
    assert task.test_spec["FAIL_TO_PASS"] == ["tests/test_a.py::test_bug"]
    assert task.test_spec["PASS_TO_PASS"] == ["tests/test_a.py::test_regression"]
    assert task.test_spec["docker_image"] == "metadata-only"


def test_projection_parses_xml_action_and_flags_bad_format():
    actions, valids = code_swe_projection([
        "<function=bash>\n<parameter=cmd>sed -n '1,20p' hello.py</parameter>\n</function>",
        "not a tool call",
    ])

    assert actions[0]["tool_name"] == "bash"
    assert actions[0]["parameters"]["cmd"] == "sed -n '1,20p' hello.py"
    assert valids == [1, 0]
    assert actions[1]["tool_name"] == ""
    assert "format" in actions[1]["error"].lower()


def test_runtime_blocks_sensitive_paths_and_docker(tiny_repo: Path, tmp_path: Path):
    task = normalize_task_record(
        {
            "task_id": "tiny",
            "repo": "tiny/repo",
            "repo_path": str(tiny_repo),
            "base_commit": "HEAD",
            "problem_statement": "Fix add",
            "test_command": f"{sys.executable} -c \"import hello; assert hello.add(1, 2) == 3\"",
        },
        dataset_name="local",
        split="train",
        index=0,
    )
    runtime = WorkspaceRuntime(RuntimeConfig(workspace_root=str(tmp_path / "workspaces")))
    runtime.setup(task, replica_id="0")

    docker_result = runtime.run_bash("docker ps")
    home_result = runtime.run_bash("cat /home/caiting/.ssh/id_rsa")

    assert docker_result.exit_code == 126
    assert home_result.exit_code == 126
    assert "blocked" in docker_result.observation.lower()
    assert "blocked" in home_result.observation.lower()


def test_editor_supports_view_replace_insert_undo(tiny_repo: Path, tmp_path: Path):
    task = normalize_task_record(
        {
            "task_id": "tiny",
            "repo": "tiny/repo",
            "repo_path": str(tiny_repo),
            "base_commit": "HEAD",
            "problem_statement": "Fix add",
            "test_command": f"{sys.executable} -c \"import hello; assert hello.add(1, 2) == 3\"",
        },
        dataset_name="local",
        split="train",
        index=0,
    )
    runtime = WorkspaceRuntime(RuntimeConfig(workspace_root=str(tmp_path / "workspaces")))
    runtime.setup(task, replica_id="0")

    view = runtime.run_editor({"command": "view", "path": "/testbed/hello.py"})
    replaced = runtime.run_editor(
        {
            "command": "str_replace",
            "path": "/testbed/hello.py",
            "old_str": "return a - b",
            "new_str": "return a + b",
        }
    )
    inserted = runtime.run_editor(
        {
            "command": "insert",
            "path": "/testbed/hello.py",
            "insert_line": 0,
            "new_str": "# generated by test",
        }
    )
    undone = runtime.run_editor({"command": "undo_edit", "path": "/testbed/hello.py"})

    assert "cat -n" in view.observation
    assert "edited" in replaced.observation
    assert "edited" in inserted.observation
    assert "undone" in undone.observation
    assert runtime.read_text("/testbed/hello.py").startswith("def add")


def test_code_swe_env_fix_and_submit_returns_patch_reward(tiny_repo: Path, tmp_path: Path):
    env = CodeSWEEnv(
        tasks=[
            normalize_task_record(
                {
                    "task_id": "tiny",
                    "repo": "tiny/repo",
                    "repo_path": str(tiny_repo),
                    "base_commit": "HEAD",
                    "problem_statement": "Fix add so it returns the sum.",
                    "test_command": f"{sys.executable} -c \"import hello; assert hello.add(1, 2) == 3\"",
                },
                dataset_name="local",
                split="train",
                index=0,
            )
        ],
        runtime_config=RuntimeConfig(workspace_root=str(tmp_path / "workspaces")),
        env_num=1,
        group_n=1,
        max_steps=3,
    )

    obs, infos = env.reset()
    assert "Fix add" in obs[0]
    assert infos[0]["task_id"] == "tiny"

    edit_action = {
        "tool_name": "str_replace_editor",
        "parameters": {
            "command": "str_replace",
            "path": "/testbed/hello.py",
            "old_str": "return a - b",
            "new_str": "return a + b",
        },
    }
    obs, rewards, dones, infos = env.step([edit_action])
    assert rewards == [0.0]
    assert dones == [False]
    assert infos[0]["is_action_valid"] is True

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])

    assert rewards == [1.0]
    assert dones == [True]
    assert infos[0]["won"] is True
    assert Path(infos[0]["patch_path"]).exists()
    assert "return a + b" in Path(infos[0]["patch_path"]).read_text(encoding="utf-8")
