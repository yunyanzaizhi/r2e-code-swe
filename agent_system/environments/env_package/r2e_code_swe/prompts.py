import re
from typing import Any, Dict, Optional

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


def format_r2e_tool_mask(tool_mask: Optional[Dict[str, Any]], *, initial: bool = False, final_step: bool = False) -> str:
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
            "<parameter=cmd>grep -RIn -- 'TransferEncodingError' /testbed/aiohttp | head -50</parameter>\n"
            "</function>"
        )
    if mask["allow_str_replace_editor"]:
        allowed.append(
            "<function=str_replace_editor>\n"
            "<parameter=command>view</parameter>\n"
            "<parameter=path>/testbed/aiohttp/http_parser.py</parameter>\n"
            "</function>"
        )
        allowed.append(
            "<function=str_replace_editor>\n"
            "<parameter=command>view</parameter>\n"
            "<parameter=path>/testbed/aiohttp/http_parser.py</parameter>\n"
            "<parameter=view_range>[330, 390]</parameter>\n"
            "</function>"
        )
    if mask["allow_validate"]:
        allowed.append(
            "<function=validate>\n"
            "<parameter=cmd>python -m pytest -q</parameter>\n"
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
Required parameter: cmd. The command must be test-like, for example python -m pytest -q, pytest -q, tox, unittest, r2e_tests, or run_tests.
A failing validation command still counts as useful feedback. Inspect the output and revise before submitting.

4. submit
Export the current git diff as your patch and run R2E unit-test reward. Submit only after at least one successful source edit and after validate has run, unless the final step forces submission.
"""

R2E_WORKFLOW_HINTS = """Search and edit strategy:
- R2E places the repository root directly at /testbed. The current bash cwd is already /testbed. Do not assume a subdirectory named after the repository exists; use the actual paths shown by the workspace preview or by find.
- The default bash cwd is /testbed. Do not cd into a file path such as /testbed/aiohttp/http_parser.py; cd into its directory or use the absolute file path directly.
- For text search, include both a pattern and a real path: grep -RIn -- 'TransferEncodingError' /testbed/aiohttp | head -50. If the package path is unclear, search from the repository root with grep -RIn -- 'term' . | head -50. A command like grep -n 'term' without a file or directory will fail.
- After search results, inspect only the relevant lines with str_replace_editor view and view_range, for example <parameter=view_range>[330, 390]</parameter>. Avoid repeatedly viewing an entire large file.
- If an observation says a command was blocked as repeated or output was clipped, change strategy immediately: narrow the path/range, search a different symbol, or inspect a different file.
- If str_replace says old_str is not unique, do not retry the same one-line replacement. View the target range and copy a larger consecutive block into old_str.
- Do not edit setup, install, dependency, or generated helper files unless the issue explicitly asks; fixes should usually touch source files in the package named by the issue.
- After a source edit, use the validate tool before submit, usually with python -m pytest -q or a focused pytest command. A failing validation is useful feedback; inspect it and revise.
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

{format_r2e_tool_mask(tool_mask, initial=True)}

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

{format_r2e_tool_mask(tool_mask, final_step=(max_steps is not None and current_step >= max_steps))}

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
