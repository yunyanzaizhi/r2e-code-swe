import base64
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .prompts import clip_text
from .reward_shaping import classify_r2e_path
from .tasks import R2ECodeSWETask


@dataclass
class R2ERuntimeConfig:
    r2e_repo_root: str = "/home/caiting/R2E-Gym"
    backend: str = "docker"
    command_timeout: int = 60
    reward_timeout: int = 300
    max_output_chars: int = 12000
    workspace_overview_max_chars: int = 4000
    patches_dir: Optional[str] = None
    trajectory_dir: str = "experiments/logs/r2e_code_swe/trajectories"
    save_patch: bool = True
    verbose: bool = False

    @classmethod
    def from_obj(cls, obj: Any) -> "R2ERuntimeConfig":
        if obj is None:
            return cls()
        data = {}
        for field_name in cls.__dataclass_fields__:
            if isinstance(obj, dict) and field_name in obj:
                data[field_name] = obj[field_name]
            elif hasattr(obj, field_name):
                data[field_name] = getattr(obj, field_name)
        return cls(**data)


@dataclass
class R2EToolResult:
    observation: str
    reward: float = 0.0
    done: bool = False
    is_action_valid: bool = True
    info: Dict[str, Any] = field(default_factory=dict)


def _is_recoverable_grep_head_sigpipe(command: str, exit_code: str, output: Any) -> bool:
    """grep | head can exit 141 when head closes after producing useful output."""
    exit_text = str(exit_code or "").strip()
    if not (
        exit_text == "141"
        or re.search(r"(?:^|\b)(?:Error:\s*)?Exit code\s+141(?:\b|$)", exit_text, flags=re.IGNORECASE)
    ):
        return False
    if not str(output or "").strip():
        return False
    command_text = str(command or "")
    return bool(
        re.search(r"\bgrep\b", command_text, flags=re.DOTALL)
        and re.search(r"\|\s*head\b", command_text, flags=re.DOTALL)
    )


def _is_full_pytest_command(command: str) -> bool:
    try:
        tokens = shlex.split(str(command or ""))
    except ValueError:
        return False
    if not tokens or any(token in {"&&", ";", "|"} for token in tokens):
        return False
    pytest_index = -1
    if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
        pytest_index = 2
    elif tokens[0] == "pytest":
        pytest_index = 0
    else:
        return False
    pytest_args = tokens[pytest_index + 1 :]
    return all(arg.startswith("-") for arg in pytest_args)


_EDITOR_PATH_ERROR_RE = re.compile(
    r"(?:"
    r"path\b.*\bdoes not exist|"
    r"does not exist|"
    r"no such file|"
    r"not found|"
    r"cannot open|"
    r"can't open|"
    r"not a directory"
    r")",
    flags=re.IGNORECASE,
)


def _is_editor_path_error(output: str) -> bool:
    return bool(_EDITOR_PATH_ERROR_RE.search(str(output or "")))


def _unescape_editor_text_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if "\n" in value:
        return value
    if not any(token in value for token in ("\\n", "\\t", "\\r")):
        return value
    return value.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")


def _safe_name(text: str, max_len: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:max_len] or "unknown"


def _split_git_patch_blocks(patch: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in str(patch or "").splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _patch_block_path(block: List[str]) -> str:
    if not block:
        return ""
    match = re.match(r"diff --git a/(.*?) b/(.*?)\s*$", block[0])
    if not match:
        return ""
    return match.group(2).strip()


def _patch_block_is_new_file(block: List[str]) -> bool:
    return any(line.startswith("new file mode") or line.startswith("--- /dev/null") for line in block)


def filter_submission_patch(raw_patch: str) -> Tuple[str, Dict[str, Any]]:
    """Filter R2E git diff output to source edits suitable for model submission logs.

    R2E DockerRuntime.get_patch() stages all files with git add -A. Official R2E
    trajectory export narrows patches to editor-touched existing files; this adapter
    applies the same conservative spirit by dropping new files and non-source/R2E
    auxiliary paths from the saved submission patch while preserving the raw patch
    separately for debugging.
    """
    raw_patch = str(raw_patch or "")
    kept_blocks: List[str] = []
    kept_files: List[str] = []
    dropped_files: List[Dict[str, str]] = []

    for block in _split_git_patch_blocks(raw_patch):
        path = _patch_block_path(block)
        if not path:
            dropped_files.append({"path": "", "reason": "unparseable_diff_header"})
            continue
        is_new_file = _patch_block_is_new_file(block)
        path_kind = classify_r2e_path("/testbed/" + path.lstrip("/"))
        if is_new_file:
            dropped_files.append({"path": path, "reason": "new_file"})
            continue
        if path_kind != "source":
            dropped_files.append({"path": path, "reason": path_kind})
            continue
        kept_files.append(path)
        kept_blocks.append("".join(block))

    filtered_patch = "".join(kept_blocks)
    stats: Dict[str, Any] = {
        "raw_patch_chars": len(raw_patch),
        "filtered_patch_chars": len(filtered_patch),
        "kept_files": kept_files,
        "dropped_files": dropped_files,
        "dropped_file_count": len(dropped_files),
    }
    return filtered_patch, stats


class R2ERepoRuntime:
    def __init__(self, config: R2ERuntimeConfig):
        self.config = config
        self.task: Optional[R2ECodeSWETask] = None
        self.env = None
        self.setup_error: Optional[str] = None
        self.episode_dir: Optional[Path] = None
        self.patch_path: Optional[Path] = None
        self.test_output_path: Optional[Path] = None
        self.recent_tool_outputs: List[str] = []

    def _ensure_imports(self):
        src = Path(self.config.r2e_repo_root) / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
        return EnvArgs, RepoEnv

    def _check_docker(self) -> None:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Docker is required for r2e_code_swe but is not usable: "
                + (result.stderr or result.stdout).strip()
            )

    def setup(self, task: R2ECodeSWETask, replica_id: str) -> None:
        self.task = task
        self.setup_error = None
        root = Path(self.config.trajectory_dir)
        self.episode_dir = root / f"{_safe_name(task.task_id)}_{replica_id}_{int(time.time())}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.patch_path = None
        self.test_output_path = None
        self.recent_tool_outputs = []

        try:
            self._check_docker()
            EnvArgs, RepoEnv = self._ensure_imports()
            env_args = EnvArgs(ds=task.raw_record, docker_image=task.docker_image or None)
            self.env = RepoEnv(
                env_args,
                backend=self.config.backend,
                verbose=self.config.verbose,
                step_timeout=self.config.command_timeout,
                reward_timeout=self.config.reward_timeout,
            )
            if getattr(self.env.runtime, "container", None) is None:
                raise RuntimeError(
                    f"Docker container did not start for image {task.docker_image}. "
                    "Check Docker daemon proxy, image availability, and disk space."
                )
            self._add_r2e_editor_tool()
        except Exception as exc:
            self.setup_error = f"{type(exc).__name__}: {exc}"
            self.env = None

    def _add_r2e_editor_tool(self) -> None:
        if self.env is None:
            return
        tool_path = Path(self.config.r2e_repo_root) / "src/r2egym/agenthub/tools/str_replace_editor.py"
        if tool_path.exists():
            self.env.add_commands([str(tool_path)])

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None

    def _not_ready(self) -> Optional[R2EToolResult]:
        if self.setup_error:
            return R2EToolResult(
                observation=f"R2E Docker environment setup failed: {self.setup_error}",
                done=True,
                is_action_valid=False,
                info={"fail_reason": "setup_failed", "setup_error": self.setup_error, "skipped": True},
            )
        if self.env is None:
            return R2EToolResult(
                observation="R2E Docker environment is not initialized.",
                done=True,
                is_action_valid=False,
                info={"fail_reason": "setup_failed", "skipped": True},
            )
        return None

    def run_bash(self, command: str, cwd: Optional[str] = None) -> R2EToolResult:
        not_ready = self._not_ready()
        if not_ready:
            return not_ready
        workdir = cwd or "/testbed"
        user_command = str(command or "")
        shell_command = "bash -lc " + shlex.quote("set -o pipefail\n" + user_command)
        output, code = self.env.runtime.run(shell_command, timeout=self.config.command_timeout, workdir=workdir)
        original_code_s = str(code)
        code_s = original_code_s
        output_text = str(output or "")
        self._remember_tool_output(output_text)
        recovery_message = ""
        recovered_grep_sigpipe = _is_recoverable_grep_head_sigpipe(user_command, original_code_s, output_text)
        if recovered_grep_sigpipe:
            code_s = "0"
            recovery_message = (
                "\n[adapter recovery] Normalized grep|head SIGPIPE exit 141 to success "
                "because search output was produced."
            )
        observation = (
            f"Exit code: {code_s}\n[output]\n"
            f"{clip_text(output_text, self.config.max_output_chars, 'command output')}{recovery_message}"
        )
        execution_success = code_s == "0"
        fail_reason = None if execution_success else ("timeout" if code_s == "-1" and "too long" in str(output).lower() else "command_failed")
        if fail_reason == "timeout" and _is_full_pytest_command(user_command):
            observation += (
                "\n[adapter hint] Full pytest timed out. "
                "Run a focused test file from r2e_tests or tests instead."
            )
        info = {
            "exit_code": code_s,
            "stdout": clip_text(output_text, self.config.max_output_chars, "stdout"),
            "stderr": "",
            "fail_reason": fail_reason,
            "tool_execution_success": execution_success,
            "tool_execution_fail_reason": fail_reason,
        }
        if recovered_grep_sigpipe:
            info.update(
                {
                    "original_exit_code": original_code_s,
                    "normalized_exit_code": code_s,
                    "adapter_recovery": "grep_head_sigpipe",
                }
            )
        return R2EToolResult(observation=observation, is_action_valid=True, info=info)

    def workspace_overview(self) -> str:
        """Return a short R2E-native view of the repository root for the initial observation."""
        if self.config.workspace_overview_max_chars <= 0:
            return ""
        if self._not_ready():
            return ""
        output, code = self.env.runtime.run(
            self._build_editor_command({"command": "view", "path": "/testbed"}),
            timeout=self.config.command_timeout,
            workdir="/testbed",
        )
        code_s = str(code)
        output_text = str(output or "")
        if code_s != "0" or output_text.lstrip().startswith("ERROR:"):
            return (
                "Workspace root preview unavailable. "
                "Run <function=bash><parameter=cmd>pwd && find . -maxdepth 2 -type f | head -100</parameter></function> "
                "to inspect the actual repository layout."
            )
        preview = clip_text(output_text, self.config.workspace_overview_max_chars, "workspace preview")
        return (
            "Workspace root preview from str_replace_editor view /testbed:\n"
            f"{preview}\n"
            "Reminder: the repository root is /testbed; do not assume /testbed/<repository-name> exists. "
            "Use the actual paths shown above or inspect with pwd && find . -maxdepth 2 -type f | head -100."
        )

    def _build_editor_command(self, params: Dict[str, Any]) -> str:
        command = str(params.get("command"))
        parts = ["str_replace_editor", shlex.quote(command)]
        for key in ("path", "file_text", "old_str", "new_str", "insert_line", "view_range", "python_only"):
            if key not in params or params[key] is None:
                continue
            value = params[key]
            if key == "view_range" and isinstance(value, (list, tuple)):
                value = json.dumps(list(value))
            parts.extend([f"--{key}", shlex.quote(str(value))])
        return " ".join(parts)

    def _remember_tool_output(self, output: Any) -> None:
        text = str(output or "")
        if not text.strip():
            return
        self.recent_tool_outputs.append(text[-20000:])
        self.recent_tool_outputs = self.recent_tool_outputs[-6:]

    def _path_recovery_command(self, path: str) -> str:
        recent_context = "\n".join(self.recent_tool_outputs[-6:])[-60000:]
        payload = {"path": str(path or ""), "context": recent_context}
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        script = """import base64, json, os, re
from pathlib import Path

payload = json.loads(base64.b64decode(__PAYLOAD_B64__).decode("utf-8"))
logical_root = Path("/testbed")
fs_root = Path(os.environ.get("R2E_PATH_RECOVERY_ROOT", "/testbed"))
requested_logical = Path(payload.get("path") or "/testbed")
context = str(payload.get("context") or "")

def logical_to_fs(path):
    try:
        rel = Path(path).relative_to(logical_root)
        return fs_root.joinpath(*rel.parts)
    except ValueError:
        return Path(path)

def fs_to_logical(path):
    try:
        rel = Path(path).relative_to(fs_root)
        return str(logical_root.joinpath(*rel.parts))
    except ValueError:
        return str(path)

requested = logical_to_fs(requested_logical)
root = fs_root
name = requested_logical.name
stem = requested_logical.stem
suffix = requested_logical.suffix

terms = []
symbols = []
def add(term):
    term = str(term or "").strip()
    if term and term not in terms:
        terms.append(term)

def add_symbol(symbol):
    symbol = str(symbol or "").strip()
    if symbol and symbol not in symbols:
        symbols.append(symbol)
    add(symbol)

def snake_to_camel(text):
    parts = [part for part in re.split(r"[_\\-.]+", str(text or "")) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)

add(name)
add(stem)
camel_stem = snake_to_camel(stem)
if camel_stem:
    add_symbol(camel_stem)
for part in re.split(r"[_\\-.]+", stem):
    if len(part) >= 4:
        add(part)

try:
    relative_parts = requested_logical.relative_to(logical_root).parts
except ValueError:
    relative_parts = requested_logical.parts
for part in relative_parts[:-1]:
    if part and part not in {"", os.sep}:
        add(part)

parent = requested.parent
try:
    parent_exists = parent.exists() and parent.is_dir() and root in [parent, *parent.parents]
except Exception:
    parent_exists = False

for match in re.finditer(r"\\b(?:class|def)\\s+([A-Za-z_][A-Za-z0-9_]*)\\b", context):
    add_symbol(match.group(1))
for match in re.finditer(r"\\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Handler|Connector|Context|Variable)\\b", context):
    add_symbol(match.group(0))

def module_candidates(module):
    parts = [part for part in str(module or "").split(".") if part]
    if not parts:
        return []
    base = root.joinpath(*parts)
    return [base.with_suffix(".py"), base / "__init__.py"]

requested_symbols = set(symbols)
import_paths = set()
for match in re.finditer(r"\\bfrom\\s+([A-Za-z_][A-Za-z0-9_.]*)\\s+import\\s+([^\\n#]+)", context):
    module = match.group(1)
    imported = [name.strip().split(" as ")[0] for name in match.group(2).split(",")]
    for imported_name in imported:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", imported_name):
            add_symbol(imported_name)
    if not requested_symbols or any(name in symbols or name in requested_symbols for name in imported):
        for candidate in module_candidates(module):
            import_paths.add(str(candidate))
for match in re.finditer(r"\\bimport\\s+([A-Za-z_][A-Za-z0-9_.]*)", context):
    for candidate in module_candidates(match.group(1)):
        import_paths.add(str(candidate))

grep_paths = set()
for match in re.finditer(r"(?<![A-Za-z0-9_./-])(/testbed/[A-Za-z0-9_./+-]+\\.py)(?=[:\\s]|$)", context):
    grep_paths.add(str(logical_to_fs(match.group(1))))
for match in re.finditer(r"(?m)(?:^|\\s)(\\.?/?[A-Za-z0-9_./+-]+\\.py):\\d+:", context):
    raw = match.group(1)
    if raw.startswith("/testbed/"):
        grep_paths.add(str(logical_to_fs(raw)))
    elif raw.startswith("./"):
        grep_paths.add(str(root / raw[2:]))
    elif not raw.startswith("/"):
        grep_paths.add(str(root / raw))

skip_dirs = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".tox", ".venv", "venv", "build", "dist"}
candidates = []
seen_candidates = set()
scanned = 0

def path_penalty(path):
    logical = fs_to_logical(path)
    lower = logical.lower()
    parts = [part.lower() for part in Path(logical).parts]
    penalty = 0
    if "/test/" in lower or "/tests/" in lower or Path(logical).name.startswith("test_"):
        penalty += 40
    if any(part in {"doc", "docs", "documentation"} for part in parts):
        penalty += 25
    if "egg-info" in lower or "/build/" in lower or "/dist/" in lower:
        penalty += 35
    if Path(logical).suffix != ".py":
        penalty += 10
    return penalty

def add_candidate(path, rank, reason):
    path = Path(path)
    try:
        if not path.exists() or not path.is_file():
            return
    except OSError:
        return
    logical = fs_to_logical(path)
    if logical in seen_candidates:
        return
    seen_candidates.add(logical)
    penalty = path_penalty(path)
    candidates.append((rank + penalty, penalty, len(logical), logical, reason))

for path in import_paths:
    add_candidate(path, 0, "import")
for path in grep_paths:
    add_candidate(path, 1, "recent_grep_output")

for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    dir_path = Path(dirpath)
    for filename in filenames:
        scanned += 1
        if scanned > 50000:
            break
        file_path = dir_path / filename
        file_text = fs_to_logical(file_path)
        file_lower = file_text.lower()
        parts_lower = [part.lower() for part in file_path.parts]
        rank = 100
        if parent_exists:
            try:
                file_path.relative_to(parent)
                rank = min(rank, 12)
            except ValueError:
                pass
        if str(file_path) in import_paths:
            rank = min(rank, 0)
        if str(file_path) in grep_paths:
            rank = min(rank, 1)
        if name and filename == name:
            rank = min(rank, 0)
        if stem and file_path.stem == stem:
            rank = min(rank, 2)
        if stem and stem.lower() in parts_lower:
            rank = min(rank, 4)
        if stem and stem.lower() in filename.lower():
            rank = min(rank, 6)
        for index, term in enumerate(terms):
            term_lower = term.lower()
            if term_lower and term_lower in file_lower:
                rank = min(rank, 10 + index)
        if file_path.suffix == ".py" and (symbols or terms):
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")[:250000]
            except Exception:
                content = ""
            for index, symbol in enumerate(symbols):
                if re.search(r"\\bclass\\s+" + re.escape(symbol) + r"\\b", content):
                    rank = min(rank, 2 + index)
                elif re.search(r"\\bdef\\s+" + re.escape(symbol) + r"\\b", content):
                    rank = min(rank, 3 + index)
                elif symbol and symbol in content:
                    rank = min(rank, 8 + index)
            content_lower = content.lower()
            for index, term in enumerate(terms):
                term_lower = term.lower()
                if len(term_lower) >= 4 and term_lower in content_lower:
                    rank = min(rank, 20 + index)
        if suffix and file_path.suffix == suffix:
            rank -= 1
        if rank < 100:
            add_candidate(file_path, rank, "scan")
    if scanned > 50000:
        break

candidates.sort()
paths = []
seen = set()
for _, _, _, file_text, _ in candidates:
    if file_text in seen:
        continue
    seen.add(file_text)
    paths.append(file_text)
    if len(paths) >= 20:
        break

print(json.dumps({"candidates": paths, "terms": terms[:8], "symbols": symbols[:8], "scanned": scanned}))
""".replace("__PAYLOAD_B64__", json.dumps(encoded))
        return "bash -lc " + shlex.quote("python3 - <<'PY'\n" + script + "\nPY")

    def _path_recovery_hint(self, path: str) -> Tuple[str, List[str]]:
        candidates: List[str] = []
        if not path:
            return (
                "[path recovery] The requested path does not exist. Do not repeat this path.\n"
                "Candidate files: none found.\n"
                "Run pwd && find . -maxdepth 4 -type f | head -160, then use exact paths from output.",
                candidates,
            )

        probe_output, probe_code = self.env.runtime.run(
            self._path_recovery_command(path),
            timeout=self.config.command_timeout,
            workdir="/testbed",
        )
        if str(probe_code) == "0":
            try:
                data = json.loads(str(probe_output or "{}"))
                raw_candidates = data.get("candidates") if isinstance(data, dict) else []
                if isinstance(raw_candidates, list):
                    candidates = [str(item) for item in raw_candidates if str(item).startswith("/testbed/")]
            except json.JSONDecodeError:
                candidates = []

        lines = [
            "[path recovery] The requested path does not exist. Do not repeat this path.",
        ]
        if candidates:
            lines.append("Candidate files:")
            lines.extend(f"- {candidate}" for candidate in candidates[:20])
            lines.append("Do not repeat the nonexistent path. Use one of the exact source candidates above.")
        else:
            lines.append("Candidate files: none found.")
            lines.append("Run pwd && find . -maxdepth 4 -type f | head -160, then use exact paths from output.")
        return "\n".join(lines), candidates

    def _missing_indent_repair_command(self, params: Dict[str, Any]) -> str:
        payload = {
            "path": str(params.get("path") or ""),
            "old_str": str(params.get("old_str") or ""),
            "new_str": str(params.get("new_str") or ""),
        }
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        script = f"""import base64, json
from pathlib import Path

payload = json.loads(base64.b64decode({json.dumps(encoded)}).decode("utf-8"))
path = Path(payload["path"])
old_str = payload["old_str"]
new_str = payload["new_str"]

def leading_ws(line):
    return line[: len(line) - len(line.lstrip(" \\t"))]

try:
    content = path.read_text(encoding="utf-8", errors="surrogateescape")
except Exception as exc:
    print(json.dumps({{"status": "read_error", "error": f"{{type(exc).__name__}}: {{exc}}"}}))
    raise SystemExit(0)

old_lines = old_str.split("\\n")
file_lines = content.split("\\n")
if not old_lines or not old_str.strip():
    print(json.dumps({{"status": "no_match"}}))
    raise SystemExit(0)

matches = []
width = len(old_lines)
for start in range(0, max(len(file_lines) - width + 1, 0)):
    window = file_lines[start : start + width]
    ok = True
    for have, want in zip(window, old_lines):
        if want.strip() == "":
            if have.strip() != "":
                ok = False
                break
        elif have.lstrip(" \\t") != want.lstrip(" \\t"):
            ok = False
            break
    if ok:
        matches.append((start, window))

if len(matches) != 1:
    print(json.dumps({{"status": "ambiguous" if matches else "no_match", "match_count": len(matches)}}))
    raise SystemExit(0)

start, window = matches[0]
anchor = next((idx for idx, line in enumerate(old_lines) if line.strip()), 0)
old_indent = leading_ws(old_lines[anchor])
file_indent = leading_ws(window[anchor])
if old_indent and file_indent.endswith(old_indent):
    prefix = file_indent[: len(file_indent) - len(old_indent)]
else:
    prefix = file_indent

new_lines = []
for line in new_str.split("\\n"):
    if not prefix or not line.strip() or line.startswith(prefix):
        new_lines.append(line)
    else:
        new_lines.append(prefix + line)

print(json.dumps({{
    "status": "matched",
    "repaired_old_str": "\\n".join(window),
    "repaired_new_str": "\\n".join(new_lines),
    "line_start": start + 1,
    "added_prefix": prefix,
}}))
"""
        return "bash -lc " + shlex.quote("python3 - <<'PY'\n" + script + "\nPY")

    def _view_range_scoped_repair_command(self, params: Dict[str, Any]) -> str:
        payload = {
            "path": str(params.get("path") or ""),
            "old_str": str(params.get("old_str") or ""),
            "new_str": str(params.get("new_str") or ""),
            "view_range": params.get("view_range"),
        }
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        script = f"""import ast, base64, json
from pathlib import Path

payload = json.loads(base64.b64decode({json.dumps(encoded)}).decode("utf-8"))
path = Path(payload["path"])
old_str = payload["old_str"]
new_str = payload["new_str"]
view_range = payload.get("view_range")

try:
    if isinstance(view_range, str):
        view_range = ast.literal_eval(view_range)
    start, end = int(view_range[0]), int(view_range[1])
except Exception:
    print(json.dumps({{"status": "bad_view_range"}}))
    raise SystemExit(0)

try:
    content = path.read_text(encoding="utf-8", errors="surrogateescape")
except Exception as exc:
    print(json.dumps({{"status": "read_error", "error": f"{{type(exc).__name__}}: {{exc}}"}}))
    raise SystemExit(0)

lines = content.split("\\n")
if start < 1 or start > len(lines):
    print(json.dumps({{"status": "bad_view_range"}}))
    raise SystemExit(0)
if end == -1:
    end = len(lines)
end = min(end, len(lines))
if end < start:
    print(json.dumps({{"status": "bad_view_range"}}))
    raise SystemExit(0)

block_lines = lines[start - 1 : end]
block = "\\n".join(block_lines)
replacement_count = block.count(old_str)
if replacement_count < 1:
    print(json.dumps({{"status": "no_match_in_range"}}))
    raise SystemExit(0)

expanded_start = start
expanded_end = end
expanded_block = block
while content.count(expanded_block) != 1 and (expanded_start > 1 or expanded_end < len(lines)):
    if expanded_start > 1:
        expanded_start -= 1
    if expanded_end < len(lines):
        expanded_end += 1
    expanded_block = "\\n".join(lines[expanded_start - 1 : expanded_end])

if content.count(expanded_block) != 1:
    print(json.dumps({{"status": "ambiguous_context", "replacement_count": replacement_count}}))
    raise SystemExit(0)

expanded_new = expanded_block.replace(old_str, new_str)
if expanded_new == expanded_block:
    print(json.dumps({{"status": "noop"}}))
    raise SystemExit(0)

print(json.dumps({{
    "status": "matched",
    "repaired_old_str": expanded_block,
    "repaired_new_str": expanded_new,
    "line_start": expanded_start,
    "line_end": expanded_end,
    "replacement_count": replacement_count,
}}))
"""
        return "bash -lc " + shlex.quote("python3 - <<'PY'\n" + script + "\nPY")

    def _retry_repaired_editor(self, params: Dict[str, Any], repair: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        repaired_params = dict(params)
        repaired_params["old_str"] = repair["repaired_old_str"]
        repaired_params["new_str"] = repair["repaired_new_str"]
        retry_output, retry_code = self.env.runtime.run(
            self._build_editor_command(repaired_params),
            timeout=self.config.command_timeout,
            workdir="/testbed",
        )
        retry_code_s = str(retry_code)
        retry_text = str(retry_output or "")
        retry_invalid = retry_text.lstrip().startswith("ERROR:")
        if retry_code_s == "0" and not retry_invalid:
            return retry_text, retry_code_s
        return None

    def _try_literal_newline_repair(self, params: Dict[str, Any], output_text: str) -> Optional[R2EToolResult]:
        if str(params.get("command") or "") != "str_replace":
            return None
        if params.get("old_str") is None or params.get("new_str") is None:
            return None
        if "No occurrences of" not in str(output_text):
            return None
        repaired_params = dict(params)
        repaired_params["old_str"] = _unescape_editor_text_literal(repaired_params.get("old_str"))
        repaired_params["new_str"] = _unescape_editor_text_literal(repaired_params.get("new_str"))
        if repaired_params["old_str"] == params.get("old_str") and repaired_params["new_str"] == params.get("new_str"):
            return None

        retry_output, retry_code = self.env.runtime.run(
            self._build_editor_command(repaired_params),
            timeout=self.config.command_timeout,
            workdir="/testbed",
        )
        retry_code_s = str(retry_code)
        retry_text = str(retry_output or "")
        retry_invalid = retry_text.lstrip().startswith("ERROR:")
        if retry_code_s != "0" or retry_invalid:
            return None

        retry_text += "\n[editor recovery] Literal backslash-n sequences were converted to real newlines for str_replace."
        observation = f"Exit code: {retry_code_s}\n[output]\n{clip_text(retry_text, self.config.max_output_chars, 'editor output')}"
        return R2EToolResult(
            observation=observation,
            is_action_valid=True,
            info={
                "exit_code": retry_code_s,
                "stdout": clip_text(retry_text, self.config.max_output_chars, "stdout"),
                "stderr": "",
                "fail_reason": None,
                "tool_execution_success": True,
                "tool_execution_fail_reason": None,
                "literal_newline_repair_applied": True,
                "editor_recovery_hint": "literal_backslash_newline_repaired",
            },
        )

    def _try_missing_indent_repair(self, params: Dict[str, Any], output_text: str) -> Optional[R2EToolResult]:
        if str(params.get("command") or "") != "str_replace":
            return None
        if params.get("old_str") is None or params.get("new_str") is None:
            return None
        if "No occurrences of" not in str(output_text):
            return None

        probe_command = self._missing_indent_repair_command(params)
        probe_output, probe_code = self.env.runtime.run(
            probe_command,
            timeout=self.config.command_timeout,
            workdir="/testbed",
        )
        try:
            repair = json.loads(str(probe_output or "{}"))
        except json.JSONDecodeError:
            repair = {"status": "probe_parse_error", "raw": str(probe_output or "")}
        if str(probe_code) != "0" or repair.get("status") != "matched":
            return None

        retry = self._retry_repaired_editor(params, repair)
        if retry is None:
            return None
        retry_text, retry_code_s = retry

        retry_text += (
            "\n[editor recovery] Adapter repaired missing leading indentation in old_str/new_str "
            f"from the original file at line {repair.get('line_start')} and retried once."
        )
        observation = f"Exit code: {retry_code_s}\n[output]\n{clip_text(retry_text, self.config.max_output_chars, 'editor output')}"
        return R2EToolResult(
            observation=observation,
            is_action_valid=True,
            info={
                "exit_code": retry_code_s,
                "stdout": clip_text(retry_text, self.config.max_output_chars, "stdout"),
                "stderr": "",
                "fail_reason": None,
                "tool_execution_success": True,
                "tool_execution_fail_reason": None,
                "indent_repair_applied": True,
                "indent_repair_line_start": repair.get("line_start"),
                "editor_recovery_hint": "indent_repaired_old_str",
            },
        )

    def _try_view_range_scoped_repair(self, params: Dict[str, Any], output_text: str) -> Optional[R2EToolResult]:
        if str(params.get("command") or "") != "str_replace":
            return None
        if params.get("old_str") is None or params.get("new_str") is None or params.get("view_range") is None:
            return None
        if "Multiple occurrences of" not in str(output_text):
            return None

        probe_output, probe_code = self.env.runtime.run(
            self._view_range_scoped_repair_command(params),
            timeout=self.config.command_timeout,
            workdir="/testbed",
        )
        try:
            repair = json.loads(str(probe_output or "{}"))
        except json.JSONDecodeError:
            repair = {"status": "probe_parse_error", "raw": str(probe_output or "")}
        if str(probe_code) != "0" or repair.get("status") != "matched":
            return None

        retry = self._retry_repaired_editor(params, repair)
        if retry is None:
            return None
        retry_text, retry_code_s = retry
        retry_text += (
            "\n[editor recovery] Adapter expanded view_range into a unique old_str block "
            f"covering lines {repair.get('line_start')}-{repair.get('line_end')} and retried once."
        )
        observation = f"Exit code: {retry_code_s}\n[output]\n{clip_text(retry_text, self.config.max_output_chars, 'editor output')}"
        return R2EToolResult(
            observation=observation,
            is_action_valid=True,
            info={
                "exit_code": retry_code_s,
                "stdout": clip_text(retry_text, self.config.max_output_chars, "stdout"),
                "stderr": "",
                "fail_reason": None,
                "tool_execution_success": True,
                "tool_execution_fail_reason": None,
                "range_repair_applied": True,
                "range_repair_line_start": repair.get("line_start"),
                "range_repair_line_end": repair.get("line_end"),
                "range_repair_replacement_count": repair.get("replacement_count"),
                "editor_recovery_hint": "view_range_scoped_old_str",
            },
        )

    def run_editor(self, params: Dict[str, Any]) -> R2EToolResult:
        not_ready = self._not_ready()
        if not_ready:
            return not_ready
        path = str(params.get("path") or "")
        if not (path == "/testbed" or path.startswith("/testbed/")):
            msg = "Editor path must be /testbed or under /testbed."
            return R2EToolResult(
                observation=msg,
                is_action_valid=False,
                info={
                    "fail_reason": "invalid_path",
                    "stderr": msg,
                    "tool_execution_success": False,
                    "tool_execution_fail_reason": "invalid_path",
                },
            )

        output, code = self.env.runtime.run(self._build_editor_command(params), timeout=self.config.command_timeout, workdir="/testbed")
        code_s = str(code)
        output_text = str(output or "")
        self._remember_tool_output(output_text)
        invalid_from_output = output_text.lstrip().startswith("ERROR:")
        execution_success = code_s == "0" and not invalid_from_output
        fail_reason = None if execution_success else "editor_error"
        editor_recovery_hint = None
        if not execution_success:
            repaired = self._try_literal_newline_repair(params, output_text)
            if repaired is not None:
                return repaired
            repaired = self._try_missing_indent_repair(params, output_text)
            if repaired is not None:
                return repaired
            repaired = self._try_view_range_scoped_repair(params, output_text)
            if repaired is not None:
                return repaired
        if not execution_success and "Multiple occurrences of" in output_text and "str_replace" in output_text:
            editor_recovery_hint = "non_unique_old_str"
            output_text += (
                "\n[editor recovery] str_replace old_str was not unique. "
                "Use str_replace_editor view on the target line range, then copy a larger consecutive block "
                "from the original file into old_str so it matches exactly one location. "
                "Do not repeat the same one-line str_replace."
            )
        elif not execution_success and "No occurrences of" in output_text and "str_replace" in output_text:
            editor_recovery_hint = "missing_or_misindented_old_str"
            output_text += (
                "\n[editor recovery] str_replace old_str did not match the file. "
                "R2E requires exact text including leading whitespace; view a narrow range and copy the original lines exactly."
            )
        path_recovery_candidates: List[str] = []
        path_recovery_applied = False
        if not execution_success and _is_editor_path_error(output_text):
            path_recovery_applied = True
            editor_recovery_hint = "path_not_found"
            path_hint, path_recovery_candidates = self._path_recovery_hint(path)
            output_text += "\n" + path_hint
        observation = f"Exit code: {code_s}\n[output]\n{clip_text(output_text, self.config.max_output_chars, 'editor output')}"
        info = {
            "exit_code": code_s,
            "stdout": clip_text(output_text, self.config.max_output_chars, "stdout"),
            "stderr": "",
            "fail_reason": fail_reason,
            "tool_execution_success": execution_success,
            "tool_execution_fail_reason": fail_reason,
        }
        if editor_recovery_hint:
            info["editor_recovery_hint"] = editor_recovery_hint
        if path_recovery_applied:
            info["path_recovery_requested_path"] = path
            info["path_recovery_candidates"] = path_recovery_candidates
        return R2EToolResult(
            observation=observation,
            is_action_valid=True,
            info=info,
        )

    def submit(self) -> R2EToolResult:
        not_ready = self._not_ready()
        if not_ready:
            return not_ready
        assert self.task is not None
        info: Dict[str, Any] = {"fail_reason": None}
        patch = ""
        try:
            raw_patch = self.env.runtime.get_patch()
            patch, patch_filter_stats = filter_submission_patch(raw_patch)
            info.update(
                {
                    "raw_patch_chars": patch_filter_stats["raw_patch_chars"],
                    "patch_chars": patch_filter_stats["filtered_patch_chars"],
                    "patch_filter_kept_files": patch_filter_stats["kept_files"],
                    "patch_filter_dropped_files": patch_filter_stats["dropped_files"],
                    "patch_filter_dropped_file_count": patch_filter_stats["dropped_file_count"],
                }
            )
            if self.config.save_patch:
                patch_dir = Path(self.config.patches_dir) if self.config.patches_dir else self.episode_dir / "patches"
                patch_dir.mkdir(parents=True, exist_ok=True)
                safe_task_id = _safe_name(self.task.task_id)
                self.patch_path = patch_dir / f"{safe_task_id}.patch"
                self.patch_path.write_text(patch, encoding="utf-8")
                raw_patch_path = patch_dir / f"{safe_task_id}.raw.patch"
                raw_patch_path.write_text(raw_patch, encoding="utf-8")
                info["patch_path"] = str(self.patch_path)
                info["raw_patch_path"] = str(raw_patch_path)
        except Exception as exc:
            info.update({"fail_reason": "patch_export_failed", "patch_error": f"{type(exc).__name__}: {exc}"})

        try:
            reward_out = self.env.runtime._calculate_reward(get_test_output=True, timeout=self.config.reward_timeout)
            if isinstance(reward_out, tuple):
                reward, test_output = reward_out
            else:
                reward, test_output = reward_out, ""
            reward = float(reward)
            self.test_output_path = self.episode_dir / "test_output.txt"
            self.test_output_path.write_text(str(test_output), encoding="utf-8", errors="replace")
            info.update(
                {
                    "reward": reward,
                    "won": reward == 1.0,
                    "test_output_path": str(self.test_output_path),
                    "test_output": clip_text(test_output, self.config.max_output_chars, "test output"),
                }
            )
            observation = (
                f"Submit received. Patch exported to: {info.get('patch_path')}.\n"
                f"Reward: {reward}\n"
                f"[test output]\n{clip_text(test_output, self.config.max_output_chars, 'test output')}"
            )
            return R2EToolResult(observation=observation, reward=reward, done=True, is_action_valid=True, info=info)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            info.update({"fail_reason": "reward_failed", "reward_error": error, "won": False, "skipped": True})
            return R2EToolResult(
                observation=f"Submit received, but R2E reward calculation failed: {error}",
                reward=0.0,
                done=True,
                is_action_valid=False,
                info=info,
            )
