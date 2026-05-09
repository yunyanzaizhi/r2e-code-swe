import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .tasks import CodeSWETask


DEFAULT_REPO_URLS = {
    "aiohttp": "https://github.com/aio-libs/aiohttp.git",
    "bokeh": "https://github.com/bokeh/bokeh.git",
    "coveragepy": "https://github.com/nedbat/coveragepy.git",
    "datalad": "https://github.com/datalad/datalad.git",
    "numpy": "https://github.com/numpy/numpy.git",
    "orange3": "https://github.com/biolab/orange3.git",
    "pandas": "https://github.com/pandas-dev/pandas.git",
    "pillow": "https://github.com/python-pillow/Pillow.git",
    "pyramid": "https://github.com/Pylons/pyramid.git",
    "scrapy": "https://github.com/scrapy/scrapy.git",
    "sympy": "https://github.com/sympy/sympy.git",
    "tornado": "https://github.com/tornadoweb/tornado.git",
}


@dataclass
class RuntimeConfig:
    workspace_root: str = "/tmp/verl_agent_code_swe"
    repo_cache_dir: Optional[str] = None
    patches_dir: Optional[str] = None
    prepare_repo: bool = True
    allow_network_clone: bool = False
    cleanup_workspaces: bool = True
    command_timeout: int = 30
    reward_timeout: int = 300
    max_output_chars: int = 12000
    max_file_chars: int = 20000
    max_view_lines: int = 400
    allow_install: bool = False
    apply_test_patch: bool = True
    derive_pytest_from_swebench: bool = True
    use_dataset_run_tests: bool = False
    default_test_command: Optional[str] = None
    invalid_action_penalty: float = 0.0
    prepare_r2e_from_fix_commit: bool = True
    repo_url_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_REPO_URLS))
    blocked_commands: List[str] = field(
        default_factory=lambda: [
            "sudo",
            "su",
            "docker",
            "podman",
            "ssh",
            "scp",
            "sftp",
            "curl",
            "wget",
            "nc",
            "ncat",
            "netcat",
        ]
    )
    blocked_patterns: List[str] = field(
        default_factory=lambda: [
            r"rm\s+-[^\n]*[rf][^\n]*/\s*(?:$|\s)",
            r"/(?:home|root|Users)(?:/|\s|$)",
            r"\.ssh(?:/|\s|$)",
            r"/etc/(?:shadow|sudoers)(?:\s|$)",
            r"/var/run/docker\.sock",
            r"\bHF_HOME\b",
            r"\bTRANSFORMERS_CACHE\b",
        ]
    )

    @classmethod
    def from_obj(cls, obj: Any) -> "RuntimeConfig":
        if obj is None:
            return cls()
        data = {}
        for field_name in cls.__dataclass_fields__:
            if isinstance(obj, dict) and field_name in obj:
                data[field_name] = obj[field_name]
            elif hasattr(obj, field_name):
                data[field_name] = getattr(obj, field_name)
        if "repo_url_map" in data and data["repo_url_map"] is None:
            data.pop("repo_url_map")
        return cls(**data)


@dataclass
class CommandResult:
    observation: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    is_action_valid: bool = True
    fail_reason: Optional[str] = None


def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return (
        text[:keep]
        + "\n--- <output clipped; use targeted view/search commands for more> ---\n"
        + text[-keep:]
    )


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = os.path.basename(path)
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


class WorkspaceRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.task: Optional[CodeSWETask] = None
        self.episode_dir: Optional[Path] = None
        self.workspace_path: Optional[Path] = None
        self.setup_error: Optional[str] = None
        self.edit_history: Dict[str, List[str]] = {}

    def setup(self, task: CodeSWETask, replica_id: str) -> None:
        self.task = task
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.task_id)[:120]
        self.episode_dir = Path(self.config.workspace_root) / f"{safe_task_id}_{replica_id}_{uuid.uuid4().hex[:8]}"
        self.workspace_path = self.episode_dir / "testbed"
        self.edit_history = {}
        self.setup_error = None

        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "home").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "tmp").mkdir(parents=True, exist_ok=True)

        try:
            if self.config.prepare_repo:
                self._prepare_workspace(task)
            self._restore_r2e_old_non_test_files(task)
            self._remove_python_caches()
        except Exception as exc:
            self.setup_error = f"{type(exc).__name__}: {exc}"
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            (self.workspace_path / "README.code_swe_setup_failed.txt").write_text(self.setup_error, encoding="utf-8")

    def close(self) -> None:
        if self.config.cleanup_workspaces and self.episode_dir and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)

    def _prepare_workspace(self, task: CodeSWETask) -> None:
        if task.repo_path:
            self._copy_repo(Path(task.repo_path), self.workspace_path)
            self._checkout(task.base_commit)
            return

        repo_url = task.repo_url or self.config.repo_url_map.get(task.repo)
        if repo_url is None and "/" in task.repo:
            repo_url = f"https://github.com/{task.repo}.git"
        if repo_url is None:
            raise RuntimeError(f"No repo_path or repo_url configured for repo '{task.repo}'.")

        cache_repo = None
        if self.config.repo_cache_dir:
            cache_repo = Path(self.config.repo_cache_dir) / re.sub(r"[^A-Za-z0-9_.-]+", "__", task.repo)
            if not cache_repo.exists():
                if not self.config.allow_network_clone:
                    raise RuntimeError(f"Repo cache missing for {task.repo}: {cache_repo}")
                cache_repo.parent.mkdir(parents=True, exist_ok=True)
                self._run_internal(["git", "clone", repo_url, str(cache_repo)], cwd=None, timeout=self.config.reward_timeout)
            self._run_internal(["git", "fetch", "--all", "--tags"], cwd=cache_repo, timeout=self.config.reward_timeout, check=False)
            self._run_internal(["git", "clone", "--no-hardlinks", str(cache_repo), str(self.workspace_path)], cwd=None, timeout=self.config.reward_timeout)
        else:
            if not self.config.allow_network_clone:
                raise RuntimeError(f"Network clone disabled and no repo cache configured for {task.repo}.")
            self._run_internal(["git", "clone", repo_url, str(self.workspace_path)], cwd=None, timeout=self.config.reward_timeout)

        self._checkout(task.base_commit)

    def _copy_repo(self, src: Path, dst: Path) -> None:
        if not src.exists():
            raise RuntimeError(f"repo_path does not exist: {src}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src,
            dst,
            symlinks=True,
            ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc", ".mypy_cache", ".pytest_cache"),
        )

    def _checkout(self, ref: str) -> None:
        if not self._has_git() or not ref or ref == "UNKNOWN":
            return
        if ref == "HEAD":
            self._run_internal(["git", "reset", "--hard", "HEAD"], cwd=self.workspace_path, timeout=self.config.command_timeout, check=False)
        else:
            self._run_internal(["git", "checkout", "-f", ref], cwd=self.workspace_path, timeout=self.config.reward_timeout, check=False)
        self._run_internal(["git", "clean", "-fd"], cwd=self.workspace_path, timeout=self.config.command_timeout, check=False)

    def _restore_r2e_old_non_test_files(self, task: CodeSWETask) -> None:
        if not self.config.prepare_r2e_from_fix_commit:
            return
        parsed = task.raw_record.get("parsed_commit_content")
        if not parsed:
            return
        try:
            parsed_commit = json.loads(parsed) if isinstance(parsed, str) else parsed
        except Exception:
            return
        for file_diff in parsed_commit.get("file_diffs", []):
            header = file_diff.get("header") or {}
            file_info = header.get("file") or {}
            rel_path = file_info.get("path") or file_diff.get("new_path") or file_diff.get("old_path")
            if not rel_path or _is_test_path(rel_path):
                continue
            old_content = file_diff.get("old_file_content", "")
            target = self.resolve_path(rel_path)
            if old_content == "":
                if target.exists():
                    target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(old_content, encoding="utf-8")

    def _remove_python_caches(self) -> None:
        if not self.workspace_path or not self.workspace_path.exists():
            return
        for pyc in self.workspace_path.rglob("*.pyc"):
            pyc.unlink(missing_ok=True)
        for cache in self.workspace_path.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)

    def _has_git(self) -> bool:
        return bool(self.workspace_path and (self.workspace_path / ".git").exists())

    def _run_internal(
        self,
        cmd: List[str],
        cwd: Optional[Path],
        timeout: int,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed ({' '.join(cmd)}): {result.stderr[-2000:]}")
        return result

    def _env(self) -> Dict[str, str]:
        assert self.episode_dir is not None
        home = self.episode_dir / "home"
        tmp = self.episode_dir / "tmp"
        cache = home / ".cache"
        for path in (home, tmp, cache / "pip", cache / "huggingface"):
            path.mkdir(parents=True, exist_ok=True)
        return {
            "PATH": f"{Path(sys.executable).parent}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "PIP_CACHE_DIR": str(cache / "pip"),
            "HF_HOME": str(cache / "huggingface"),
            "TRANSFORMERS_CACHE": str(cache / "huggingface" / "transformers"),
            "PYTHONNOUSERSITE": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

    def _is_blocked_command(self, command: str) -> Optional[str]:
        try:
            parts = shlex.split(command, comments=False, posix=True)
        except ValueError as exc:
            return f"shell parse error: {exc}"
        if parts:
            first = os.path.basename(parts[0]).lower()
            if first in set(self.config.blocked_commands):
                return f"blocked command '{first}'"
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return f"blocked pattern '{pattern}'"
        return None

    def resolve_path(self, path: str) -> Path:
        if self.workspace_path is None:
            raise RuntimeError("Runtime has not been setup.")
        raw = str(path)
        if raw.startswith("/testbed"):
            raw = raw[len("/testbed") :].lstrip("/")
            candidate = self.workspace_path / raw
        else:
            p = Path(raw)
            candidate = p if p.is_absolute() else self.workspace_path / p
        resolved = candidate.resolve()
        workspace = self.workspace_path.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"Path '{path}' escapes the episode workspace.")
        return resolved

    def resolve_cwd(self, cwd: Optional[str]) -> Path:
        if cwd in (None, "", "/testbed"):
            return self.workspace_path
        path = self.resolve_path(cwd)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"cwd does not exist or is not a directory: {cwd}")
        return path

    def run_bash(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        enforce_policy: bool = True,
    ) -> CommandResult:
        if enforce_policy:
            blocked = self._is_blocked_command(command)
            if blocked:
                msg = f"Command blocked by no-Docker runtime policy: {blocked}"
                return CommandResult(observation=msg, exit_code=126, stderr=msg, is_action_valid=False, fail_reason=blocked)
        try:
            run_cwd = self.resolve_cwd(cwd)
        except Exception as exc:
            return CommandResult(observation=str(exc), exit_code=2, stderr=str(exc), is_action_valid=False, fail_reason="invalid_cwd")

        timeout = timeout or self.config.command_timeout
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=str(run_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._env(),
            preexec_fn=os.setsid,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            timed_out = False
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            timed_out = True
            exit_code = 124
            stderr = (stderr or "") + f"\nCommand timed out after {timeout}s."

        stdout_t = _truncate(stdout or "", self.config.max_output_chars)
        stderr_t = _truncate(stderr or "", self.config.max_output_chars)
        obs = f"Exit code: {exit_code}\n[stdout]\n{stdout_t}\n[stderr]\n{stderr_t}".strip()
        return CommandResult(
            observation=obs,
            exit_code=exit_code,
            stdout=stdout_t,
            stderr=stderr_t,
            timed_out=timed_out,
            is_action_valid=True,
            fail_reason="timeout" if timed_out else None,
        )

    def read_text(self, path: str, max_chars: Optional[int] = None) -> str:
        resolved = self.resolve_path(path)
        data = resolved.read_text(encoding="utf-8", errors="replace")
        return _truncate(data, max_chars or self.config.max_file_chars)

    def run_editor(self, params: Dict[str, Any]) -> CommandResult:
        try:
            command = str(params.get("command", "")).strip()
            path = self.resolve_path(str(params.get("path", "")))
            if command == "view":
                return CommandResult(observation=self._editor_view(path, params.get("view_range")))
            if command == "create":
                return CommandResult(observation=self._editor_create(path, params.get("file_text")))
            if command == "str_replace":
                return CommandResult(observation=self._editor_replace(path, params.get("old_str"), params.get("new_str", "")))
            if command == "insert":
                return CommandResult(observation=self._editor_insert(path, params.get("insert_line"), params.get("new_str")))
            if command == "undo_edit":
                return CommandResult(observation=self._editor_undo(path))
            raise ValueError("Unknown editor command. Allowed: view, create, str_replace, insert, undo_edit.")
        except Exception as exc:
            return CommandResult(observation=f"Editor error: {exc}", exit_code=1, stderr=str(exc), is_action_valid=False, fail_reason="editor_error")

    def _parse_view_range(self, value: Any) -> Optional[Tuple[int, int]]:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("view_range must be [start_line, end_line].")
        return int(value[0]), int(value[1])

    def _editor_view(self, path: Path, view_range: Any) -> str:
        if path.is_dir():
            lines = [f"Directory listing for {path} (max depth 2, hidden files omitted):"]
            root_depth = len(path.parts)
            for item in sorted(path.rglob("*")):
                if any(part.startswith(".") for part in item.relative_to(path).parts):
                    continue
                if len(item.parts) - root_depth > 2:
                    continue
                suffix = "/" if item.is_dir() else ""
                lines.append(str(item.relative_to(self.workspace_path)) + suffix)
            return _truncate("\n".join(lines), self.config.max_output_chars)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8", errors="replace").expandtabs()
        lines = text.splitlines()
        parsed_range = self._parse_view_range(view_range)
        if parsed_range:
            start, end = parsed_range
            if start < 1 or start > max(1, len(lines)):
                raise ValueError(f"view_range start must be in [1, {len(lines)}].")
            end = len(lines) if end == -1 else end
            if end < start or end > len(lines):
                raise ValueError(f"view_range end must be >= start and <= {len(lines)}, or -1.")
            indexed = enumerate(lines[start - 1 : end], start=start)
        else:
            indexed = enumerate(lines[: self.config.max_view_lines], start=1)
        body = "\n".join(f"{i:6d} {line}" for i, line in indexed)
        if not parsed_range and len(lines) > self.config.max_view_lines:
            body += "\n<response clipped; pass view_range to inspect more lines>"
        return _truncate(f"Here's the result of running `cat -n` on the file: {path}:\n{body}", self.config.max_file_chars)

    def _save_history(self, path: Path) -> str:
        old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        self.edit_history.setdefault(str(path), []).append(old)
        return old

    def _editor_create(self, path: Path, file_text: Any) -> str:
        if path.exists():
            raise FileExistsError(f"File already exists: {path}")
        if file_text is None:
            raise ValueError("create requires file_text.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.edit_history.setdefault(str(path), []).append("")
        path.write_text(str(file_text), encoding="utf-8")
        return f"File created at {path}.\n{self._editor_view(path, None)}"

    def _editor_replace(self, path: Path, old_str: Any, new_str: Any) -> str:
        if old_str is None:
            raise ValueError("str_replace requires old_str.")
        text = path.read_text(encoding="utf-8", errors="replace").expandtabs()
        old = str(old_str).expandtabs()
        new = "" if new_str is None else str(new_str).expandtabs()
        count = text.count(old)
        if count == 0:
            raise ValueError(f"No occurrences of old_str found in {path}.")
        if count > 1:
            raise ValueError(f"Multiple occurrences of old_str found in {path}; make it unique.")
        self._save_history(path)
        path.write_text(text.replace(old, new), encoding="utf-8")
        return f"The file {path} has been edited.\n{self._editor_view(path, None)}"

    def _editor_insert(self, path: Path, insert_line: Any, new_str: Any) -> str:
        if insert_line is None:
            raise ValueError("insert requires insert_line.")
        if new_str is None:
            raise ValueError("insert requires new_str.")
        text = path.read_text(encoding="utf-8", errors="replace").expandtabs()
        lines = text.split("\n")
        index = int(insert_line)
        if index < 0 or index > len(lines):
            raise ValueError(f"insert_line must be in [0, {len(lines)}].")
        self._save_history(path)
        new_lines = lines[:index] + str(new_str).expandtabs().split("\n") + lines[index:]
        path.write_text("\n".join(new_lines), encoding="utf-8")
        return f"The file {path} has been edited.\n{self._editor_view(path, None)}"

    def _editor_undo(self, path: Path) -> str:
        history = self.edit_history.get(str(path), [])
        if not history:
            raise ValueError(f"No previous edits found for {path}.")
        old = history.pop()
        path.write_text(old, encoding="utf-8")
        return f"Last edit to {path} undone successfully.\n{self._editor_view(path, None)}"

    def export_patch(self) -> Tuple[str, str]:
        patch_dir = Path(self.config.patches_dir) if self.config.patches_dir else self.episode_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_name = "model.patch" if not self.config.patches_dir else f"{self.episode_dir.name}.patch"
        patch_path = patch_dir / patch_name
        if not self._has_git():
            patch_path.write_text("", encoding="utf-8")
            return "", str(patch_path)
        self._run_internal(["git", "add", "-N", "."], cwd=self.workspace_path, timeout=self.config.command_timeout, check=False)
        result = self._run_internal(["git", "diff", "--binary", "HEAD", "--", "."], cwd=self.workspace_path, timeout=self.config.command_timeout, check=False)
        patch = result.stdout or ""
        patch_path.write_text(patch, encoding="utf-8")
        return patch, str(patch_path)

    def apply_test_patch(self) -> Optional[str]:
        if not self.config.apply_test_patch or not self.task:
            return None
        test_patch = self.task.test_spec.get("test_patch")
        if not test_patch or not self._has_git():
            return None
        patch_file = self.episode_dir / "test.patch"
        patch_file.write_text(test_patch, encoding="utf-8")
        result = self._run_internal(["git", "apply", "--whitespace=fix", str(patch_file)], cwd=self.workspace_path, timeout=self.config.command_timeout, check=False)
        if result.returncode != 0:
            return _truncate(result.stderr or result.stdout, self.config.max_output_chars)
        return None

    def get_test_command(self) -> Optional[str]:
        if not self.task:
            return None
        spec = self.task.test_spec
        if spec.get("test_command"):
            return spec["test_command"]
        if self.config.derive_pytest_from_swebench:
            tests = []
            for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                tests.extend(spec.get(key) or [])
            if tests:
                unique = []
                for item in tests:
                    if item not in unique:
                        unique.append(item)
                return "python -m pytest -q " + " ".join(shlex.quote(x) for x in unique)
        if self.config.use_dataset_run_tests and spec.get("run_tests"):
            script_path = self.episode_dir / "run_tests_from_dataset.sh"
            script = str(spec["run_tests"]).replace("/testbed", str(self.workspace_path))
            sanitized_lines = []
            for line in script.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("source /opt/miniconda") or stripped.startswith("conda activate"):
                    continue
                if "git config --global" in stripped:
                    continue
                sanitized_lines.append(line)
            script_path.write_text("\n".join(sanitized_lines) + "\n", encoding="utf-8")
            return f"bash {shlex.quote(str(script_path))}"
        return self.config.default_test_command

    def run_install_if_configured(self) -> Optional[CommandResult]:
        if not self.config.allow_install or not self.task:
            return None
        install_command = self.task.test_spec.get("install_command")
        if not install_command:
            return None
        return self.run_bash(install_command, timeout=self.config.reward_timeout, enforce_policy=False)
