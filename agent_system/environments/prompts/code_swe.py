CODE_SWE_TOOL_SPEC = """Available tools:

1. bash
Execute a shell command inside the current repository workspace.
Format:
<function=bash>
<parameter=cmd>grep -R "class Foo" -n .</parameter>
</function>

2. str_replace_editor
View, create, edit, insert into, or undo edits to files under /testbed.
Commands: view, create, str_replace, insert, undo_edit.
Format examples:
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed/path/to/file.py</parameter>
<parameter=view_range>[1, 80]</parameter>
</function>

<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/testbed/path/to/file.py</parameter>
<parameter=old_str>exact old text</parameter>
<parameter=new_str>replacement text</parameter>
</function>

3. submit
Export the current git diff as your patch and run local reward tests.
Format:
<function=submit>
</function>
"""


CODE_SWE_TEMPLATE_NO_HIS = """You are a programming agent working on a real repository-level code issue.

Repository: {repo}
Task id: {task_id}
Workspace path available to tools: /testbed

Hard constraints:
- Use only bash, str_replace_editor, and submit.
- Do not use Docker, Podman, sudo, SSH, curl, or wget.
- Do not access user home directories, SSH keys, HF caches, or paths outside /testbed.
- Make targeted source changes. Do not modify dataset-provided tests unless explicitly necessary for a local reproduction file.
- Submit only when you are ready for the environment to export a patch and run tests.

{tool_spec}

Issue:
{problem_statement}

Current observation:
{current_observation}
"""


CODE_SWE_TEMPLATE = """You are a programming agent working on a real repository-level code issue.

Repository: {repo}
Task id: {task_id}
Workspace path available to tools: /testbed

Hard constraints:
- Use only bash, str_replace_editor, and submit.
- Do not use Docker, Podman, sudo, SSH, curl, or wget.
- Do not access user home directories, SSH keys, HF caches, or paths outside /testbed.
- Keep outputs targeted. If output is clipped, use narrower grep/sed/view_range commands.

{tool_spec}

Issue:
{problem_statement}

Previous {history_length} step(s):
{action_history}

Current step: {current_step}
Current observation:
{current_observation}
"""
