import re
import shlex
from typing import Any, Dict, List, Optional

from .tasks import R2ECodeSWETask


R2E_ACTION_PROTOCOL = """Your response must be exactly one XML tool call and nothing else.
Do not output explanations.
Do not output markdown.
Do not output multiple tool calls.
Use only a tool call that is listed under Allowed tool calls now.

Invalid:
- Markdown code fences
- JSON in a Markdown fenced block
- Natural language before or after the tool call
- Multiple tool calls
- Missing parameter tags
- path with line suffix such as /testbed/foo.py:123
"""



def _normalize_tool_mask(tool_mask: Optional[Dict[str, Any]], *, initial: bool = False) -> Dict[str, Any]:
    mask = dict(tool_mask or {})
    allow_bash = bool(mask.get("allow_bash", True))
    allow_editor = bool(mask.get("allow_str_replace_editor", mask.get("allow_editor", True)))
    allow_validate = bool(mask.get("allow_validate", False if initial else True))
    allow_submit = bool(mask.get("allow_submit", False if initial else True))
    return {
        "allow_bash": allow_bash,
        "allow_str_replace_editor": allow_editor,
        "allow_validate": allow_validate,
        "allow_submit": allow_submit,
        "validate_reason": str(mask.get("validate_reason") or "requires_successful_source_edit"),
        "submit_reason": str(mask.get("submit_reason") or "requires_successful_source_edit_and_validation"),
    }


_ISSUE_QUOTED_RE = re.compile(r"`([^`\n]{3,120})`|'([^'\n]{3,120})'|\"([^\"\n]{3,120})\"")
_ISSUE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")
_ISSUE_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b")
_ISSUE_STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "because",
    "before",
    "between",
    "broken",
    "cannot",
    "causes",
    "change",
    "correct",
    "crash",
    "does",
    "during",
    "error",
    "expected",
    "fails",
    "failure",
    "from",
    "handle",
    "handles",
    "instead",
    "issue",
    "only",
    "problem",
    "raise",
    "raises",
    "return",
    "should",
    "using",
    "when",
    "where",
    "with",
    "without",
    "wrong",
}


def _clean_issue_term(term: str) -> str:
    term = sanitize_prompt_text(term)
    term = re.sub(r"[<>&\x00-\x1f]+", " ", term)
    term = re.sub(r"\s+", " ", term).strip(" ,.;:()[]{}")
    return term[:120]


def _add_issue_term(terms: List[str], seen: set, term: str) -> None:
    term = _clean_issue_term(term)
    if len(term) < 3:
        return
    key = term.lower()
    if key in seen or key in _ISSUE_STOPWORDS:
        return
    seen.add(key)
    terms.append(term)


def _looks_issue_specific_identifier(token: str) -> bool:
    if "." in token or "_" in token:
        return True
    if token.endswith(("Error", "Exception", "Warning")):
        return True
    return token[:1].isupper() and any(char.isupper() for char in token[1:])


def issue_search_terms(task: Optional[R2ECodeSWETask], max_terms: int = 4) -> List[str]:
    """Extract issue-specific terms that make useful first-pass grep queries."""
    statement = getattr(task, "problem_statement", "") if task is not None else ""
    statement = sanitize_prompt_text(statement)
    terms: List[str] = []
    seen = set()

    for match in _ISSUE_QUOTED_RE.finditer(statement):
        quoted = next(group for group in match.groups() if group)
        _add_issue_term(terms, seen, quoted)
        if len(terms) >= max_terms:
            return terms

    for match in _ISSUE_IDENTIFIER_RE.finditer(statement):
        token = match.group(0)
        if _looks_issue_specific_identifier(token):
            _add_issue_term(terms, seen, token)
            if len(terms) >= max_terms:
                return terms

    for match in _ISSUE_WORD_RE.finditer(statement):
        token = match.group(0)
        if token.lower() not in _ISSUE_STOPWORDS:
            _add_issue_term(terms, seen, token)
            if len(terms) >= max_terms:
                return terms

    return terms


def _shell_single_quote(text: str) -> str:
    return "'" + str(text).replace("'", "'\"'\"'") + "'"


def _issue_grep_example(task: Optional[R2ECodeSWETask]) -> str:
    terms = issue_search_terms(task)
    keyword = terms[0] if terms else "issue keyword"
    return f"grep -RIn -- {_shell_single_quote(keyword)} . | head -50"


def _as_string_list(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _unique_strings(values: List[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def focused_validation_cmd(task: Optional[R2ECodeSWETask], max_pass_to_pass: int = 2) -> str:
    test_spec = getattr(task, "test_spec", None) or {}
    run_tests = _as_string_list(test_spec.get("run_tests"))
    if run_tests:
        return " && ".join(run_tests)

    selected_tests = _unique_strings(
        _as_string_list(test_spec.get("FAIL_TO_PASS"))
        + _as_string_list(test_spec.get("PASS_TO_PASS"))[:max_pass_to_pass]
    )
    if selected_tests:
        return "python -m pytest -q " + " ".join(shlex.quote(test) for test in selected_tests)
    return "python -m pytest -q"


def format_r2e_tool_mask(
    tool_mask: Optional[Dict[str, Any]],
    *,
    task: Optional[R2ECodeSWETask] = None,
    initial: bool = False,
    final_step: bool = False,
) -> str:
    mask = _normalize_tool_mask(tool_mask, initial=initial)
    if final_step:
        mask.update({
            "allow_bash": False,
            "allow_str_replace_editor": False,
            "allow_validate": False,
            "allow_submit": True,
            "submit_reason": "final_forced_submit",
        })

    allowed = []
    masked = []
    if mask["allow_bash"]:
        allowed.append(
            "<function=bash>\n"
            "<parameter=cmd>pwd && find . -maxdepth 2 -type f | head -100</parameter>\n"
            "</function>"
        )
        allowed.append(
            "<function=bash>\n"
            f"<parameter=cmd>{_issue_grep_example(task)}</parameter>\n"
            "</function>"
        )
    if mask["allow_str_replace_editor"]:
        allowed.append(
            "<function=str_replace_editor>\n"
            "<parameter=command>view</parameter>\n"
            "<parameter=path>/testbed</parameter>\n"
            "</function>"
        )
    if mask["allow_validate"]:
        allowed.append(
            "<function=validate>\n"
            f"<parameter=cmd>{focused_validation_cmd(task)}</parameter>\n"
            "</function>"
        )
    else:
        masked.append("validate: available after a successful source edit; use bash or str_replace_editor first.")
    if mask["allow_submit"]:
        allowed.append("<function=submit></function>")
    else:
        if mask["submit_reason"] == "requires_validation_after_source_edit":
            masked.append("submit: available after a validation command has run after the latest source edit.")
        else:
            masked.append("submit: available after a successful source edit and validation; do not submit yet.")

    allowed_text = "\n\n".join(allowed) if allowed else "No tools are currently allowed except the final forced submit."
    masked_text = "\n".join(f"- {item}" for item in masked) if masked else "- none"
    return f"""Allowed tool calls now:

{allowed_text}

Masked tool calls now:
{masked_text}
"""

R2E_FINAL_ACTION_REMINDER = "Your next response must be exactly one XML tool call and nothing else."


R2E_TOOL_SPEC = """Tool semantics:

1. bash
Execute a shell command inside the R2E Docker repository workspace.
bash commands are executed through bash -lc with pipefail enabled, so shell features such as &&, pipes, redirects, variables, and cd ... && ... work inside one command and failures inside pipelines are visible.
The default bash working directory is /testbed; prefer relative paths or absolute /testbed paths that are shown by the workspace preview.
Use bash for repository inspection and focused searches.

2. str_replace_editor
View, create, edit, insert into, or undo edits to files under /testbed.
Commands: view, create, str_replace, insert, undo_edit.
Required parameters: command, path. The path must be /testbed or under /testbed.
Do not include line suffixes in path. For line ranges, keep command=view and add parameter=view_range. Never set command to view_range.
For create include named parameter=file_text.
For str_replace include named parameter=old_str and named parameter=new_str. Never use bare parameter bodies named old_str or new_str.
For str_replace, old_str must match EXACTLY one or more consecutive lines from the original file, including whitespace.
If old_str is not unique, include enough surrounding context copied from a recent view_range so it matches only one location.
Do not use str_replace with identical old_str and new_str; that is a no-op and will not count as a source edit.
For insert include named parameter=insert_line and named parameter=new_str.

3. validate
Run a focused validation command inside the R2E Docker workspace after a successful source edit.
Required parameter: cmd. Prefer the Recommended focused validate command shown in this prompt. The command must be test-like, for example python -m pytest -q tests/test_file.py::test_name, pytest -q, tox, unittest, r2e_tests, or run_tests.
A failing validation command still counts as useful feedback. Inspect the output and revise before submitting.

4. submit
Export the current git diff as your patch and run R2E unit-test reward. Submit only after at least one successful source edit and after validate has run, unless the final step forces submission.
"""

R2E_WORKFLOW_HINTS = """Search and edit strategy:
- R2E places the repository root directly at /testbed. The current bash cwd is already /testbed. Do not assume a subdirectory named after the repository exists; use the actual paths shown by the workspace preview or by find.
- The default bash cwd is /testbed. Do not cd into a file path; cd into its directory or use the absolute file path directly.
- For text search, include both a pattern and a real path: start with the issue-specific grep example shown in Allowed tool calls. Then try nearby class names, function names, error names, or user-visible strings from the issue. A command like grep -n 'term' without a file or directory will fail.
- After search results, inspect only the relevant lines with str_replace_editor view and view_range, for example <parameter=view_range>[1, 120]</parameter>. Avoid repeatedly viewing an entire large file.
- If an observation says a command was blocked as repeated or output was clipped, change strategy immediately: narrow the path/range, search a different symbol, or inspect a different file.
- If str_replace says old_str is not unique, do not retry the same one-line replacement. View the target range and copy a larger consecutive block into old_str.
- Do not edit setup, install, dependency, or generated helper files unless the issue explicitly asks; fixes should usually touch source files related to the behavior described by the issue.
- After a source edit, use the validate tool before submit, usually with the recommended focused validate command shown below. A failing validation is useful feedback; inspect it and revise.
- Do not submit until at least one successful source edit and one post-edit validation command have happened, unless this is the final forced submission step.
"""

_PROMPT_FENCE_RE = re.compile(r"(`{3,}|~{3,})")


def sanitize_prompt_text(text: str) -> str:
    text = "" if text is None else str(text)
    return _PROMPT_FENCE_RE.sub("[code block delimiter removed]", text)


def clip_prompt_text(text: str, max_chars: int, label: str = "text") -> str:
    return sanitize_prompt_text(clip_text(text, max_chars, label))

_HISTORY_PARAM_ORDER = (
    "cmd",
    "command",
    "path",
    "view_range",
    "file_text",
    "old_str",
    "new_str",
    "insert_line",
    "cwd",
    "python_only",
)


def clip_text(text: str, max_chars: int, label: str = "text") -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return (
        text[:keep]
        + f"\n--- <{label} clipped; inspect a narrower file range or command output> ---\n"
        + text[-keep:]
    )


def _history_param_value(value, max_chars: int) -> str:
    if isinstance(value, (list, tuple)):
        text = "[" + ", ".join(str(item) for item in value) + "]"
    else:
        text = "" if value is None else str(value)
    return clip_prompt_text(text, max_chars, "history action parameter")


def format_r2e_action_for_history(action: dict, max_param_chars: int = 1000) -> str:
    tool_name = str((action or {}).get("tool_name") or "").strip()
    if not tool_name:
        error = str((action or {}).get("error") or "").strip()
        if error:
            clipped_error = clip_prompt_text(error, max_param_chars, "invalid action error")
            return (
                f"Previous response was invalid: {clipped_error}\n"
                "Next response must be exactly one XML tool call and nothing else."
            )
        return "Previous response was invalid. Next response must be exactly one XML tool call and nothing else."
    if tool_name == "submit":
        return "<function=submit></function>"

    params = dict((action or {}).get("parameters") or {})
    lines = [f"<function={tool_name}>"]
    keys = [key for key in _HISTORY_PARAM_ORDER if key in params]
    keys.extend(key for key in params if key not in _HISTORY_PARAM_ORDER)
    for key in keys:
        value = _history_param_value(params.get(key), max_param_chars)
        lines.append(f"<parameter={key}>{value}</parameter>")
    lines.append("</function>")
    return "\n".join(lines)


def format_r2e_history_turn(
    step_num: int,
    observation: str,
    action_summary: str,
    max_observation_chars: int = 4000,
    max_action_chars: int = 1500,
) -> str:
    return (
        f"Previous step {step_num}\n"
        f"Observation:\n{clip_prompt_text(observation, max_observation_chars, 'history observation')}\n"
        f"Tool call:\n{clip_prompt_text(action_summary, max_action_chars, 'history tool call')}"
    )


def build_initial_prompt(
    task: R2ECodeSWETask,
    current_observation: str,
    max_problem_chars: int = 6000,
    max_observation_chars: int = 7000,
    tool_mask: Optional[Dict[str, Any]] = None,
) -> str:
    return f"""{R2E_ACTION_PROTOCOL}
At the first step, use bash to inspect the repository. Do not submit before inspecting files.

You are a repository-level software engineering agent running inside an R2E-Gym Docker environment.

Repository: {task.repo_label}
Task id: {task.task_id}
Dataset: {task.dataset_name}
Split: {task.split}
Docker image: {task.docker_image}
Workspace path available to tools: /testbed

Hard constraints:
- Use only bash, str_replace_editor, validate, and submit. Follow the current action mask exactly.
- bash runs inside the R2E Docker workspace, not on the host.
- Make targeted source changes in /testbed. Do not edit reward metadata or hidden answer files.
- Do not try to read expected_output_json, gold patches, dataset answers, SSH keys, or host paths.
- Keep outputs targeted. If output is clipped, use narrower grep/view_range commands.
- Submit only after at least one successful source edit, unless this is the final allowed step.
- Before submit, run a focused validation command after the source edit so you can inspect failures and revise.

{R2E_WORKFLOW_HINTS}

Recommended focused validate command: {focused_validation_cmd(task)}

{format_r2e_tool_mask(tool_mask, task=task, initial=True)}

{R2E_TOOL_SPEC}

Issue:
{clip_prompt_text(task.problem_statement, max_problem_chars, "issue")}

Current observation:
{clip_prompt_text(current_observation, max_observation_chars, "observation")}

{R2E_FINAL_ACTION_REMINDER}
"""


def build_step_prompt(
    task: R2ECodeSWETask,
    current_observation: str,
    history_context: str,
    history_length: int,
    current_step: int,
    max_steps: Optional[int] = None,
    max_problem_chars: int = 6000,
    max_history_chars: int = 6000,
    max_observation_chars: int = 7000,
    tool_mask: Optional[Dict[str, Any]] = None,
) -> str:
    final_step_instruction = ""
    if max_steps is not None and current_step >= max_steps:
        final_step_instruction = (
            "This is the final allowed step.\n"
            "Your entire response must be exactly:\n"
            "<function=submit></function>\n\n"
        )
    return f"""{R2E_ACTION_PROTOCOL}
{final_step_instruction}\
You are a repository-level software engineering agent running inside an R2E-Gym Docker environment.

Repository: {task.repo_label}
Task id: {task.task_id}
Dataset: {task.dataset_name}
Split: {task.split}
Docker image: {task.docker_image}
Workspace path available to tools: /testbed

Hard constraints:
- Use only bash, str_replace_editor, validate, and submit. Follow the current action mask exactly.
- bash runs inside the R2E Docker workspace, not on the host.
- Make targeted source changes in /testbed. Do not edit reward metadata or hidden answer files.
- Do not try to read expected_output_json, gold patches, dataset answers, SSH keys, or host paths.
- Keep outputs targeted. If output is clipped, use narrower grep/view_range commands.
- Submit only after at least one successful source edit, unless this is the final allowed step.
- Before submit, run a focused validation command after the source edit so you can inspect failures and revise.

{R2E_WORKFLOW_HINTS}

Recommended focused validate command: {focused_validation_cmd(task)}

{format_r2e_tool_mask(tool_mask, task=task, final_step=(max_steps is not None and current_step >= max_steps))}

{R2E_TOOL_SPEC}

Issue:
{clip_prompt_text(task.problem_statement, max_problem_chars, "issue")}

Previous {history_length} step(s):
{clip_prompt_text(history_context, max_history_chars, "history")}

Current step: {current_step}
Current observation:
{clip_prompt_text(current_observation, max_observation_chars, "observation")}

{R2E_FINAL_ACTION_REMINDER}
"""
