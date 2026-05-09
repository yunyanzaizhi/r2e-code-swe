PROGRAMMING_TEMPLATE_NO_HIS = """You are fixing a Python bug.

The environment gives you the buggy source file and test feedback.

Important rules:
- You are editing ONLY the buggy source file shown in Current observation.
- Do NOT edit tests.
- Do NOT output pytest code.
- Do NOT output examples or usage code.
- Do NOT output Markdown fences like ```python.
- The <code> block must contain the complete fixed source code for the buggy file only.

You must respond EXACTLY in this format:

<think>
Briefly explain the bug and your fix.
</think>
<code>
complete fixed source code for the buggy file only
</code>

Current observation:
{current_observation}
"""

PROGRAMMING_TEMPLATE = """You are fixing a Python bug.

The environment gives you the buggy source file and test feedback.

Important rules:
- You are editing ONLY the buggy source file shown in Current observation.
- Do NOT edit tests.
- Do NOT output pytest code.
- Do NOT output examples or usage code.
- Do NOT output Markdown fences like ```python.
- The <code> block must contain the complete fixed source code for the buggy file only.

You have access to interaction history.

Previous {history_length} steps:
{action_history}

Current step: {current_step}

You must respond EXACTLY in this format:

<think>
Briefly explain the bug and your fix.
</think>
<code>
complete fixed source code for the buggy file only
</code>

Current observation:
{current_observation}
"""

PROGRAMMING_SUMMARY_TEMPLATE = """You are fixing a Python bug.

The environment gives you the buggy source file and test feedback.

Important rules:
- You are editing ONLY the buggy source file shown in Current observation.
- Do NOT edit tests.
- Do NOT output pytest code.
- Do NOT output examples or usage code.
- Do NOT output Markdown fences like ```python.
- The <code> block must contain the complete fixed source code for the buggy file only.

Structured summary of previous attempts:
{structured_summary}

Current step: {current_step}

You must respond EXACTLY in this format:

<think>
Briefly explain the bug and your fix.
</think>
<code>
complete fixed source code for the buggy file only
</code>

Current observation:
{current_observation}
"""