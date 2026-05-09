import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_system.environments.env_manager import attach_r2e_projection_debug
from agent_system.environments.env_package.r2e_code_swe.envs import R2ECodeSWEEnv
from agent_system.environments.env_package.r2e_code_swe.projection import parse_r2e_action
from agent_system.environments.env_package.r2e_code_swe.prompts import (
    build_initial_prompt,
    build_step_prompt,
    focused_validation_cmd,
    format_r2e_action_for_history,
    format_r2e_history_turn,
    issue_search_terms,
)
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.environments.env_package.r2e_code_swe.runtime import (
    R2ERepoRuntime,
    R2ERuntimeConfig,
    R2EToolResult,
    filter_submission_patch,
)
from agent_system.environments.env_package.r2e_code_swe.reward_shaping import (
    R2ERewardShapingConfig,
    classify_r2e_path,
)
from agent_system.environments.env_package.r2e_code_swe.tasks import (
    SplitPolicyError,
    normalize_r2e_task_record,
    validate_r2e_split_policy,
)


def test_normalize_r2egym_lite_record_keeps_docker_metadata_and_reward_only_fields():
    record = {
        "repo_name": "aiohttp",
        "docker_image": "namanjain12/aiohttp_final:abc123",
        "commit_hash": "abc123",
        "problem_statement": "Fix chunk parser.",
        "expected_output_json": json.dumps({"test_x": "PASSED"}),
        "parsed_commit_content": "gold patch content",
    }

    task = normalize_r2e_task_record(
        record,
        dataset_name="R2E-Gym/R2E-Gym-Lite",
        split="dev_10pr_v1",
        index=0,
    )

    assert task.repo_name == "aiohttp"
    assert task.docker_image == "namanjain12/aiohttp_final:abc123"
    assert task.base_commit == "abc123"
    assert task.test_spec["expected_output_json"] == {"test_x": "PASSED"}
    assert task.gold_patch_optional == "gold patch content"
    assert task.raw_record["docker_image"] == "namanjain12/aiohttp_final:abc123"


def test_prompt_never_leaks_expected_output_or_gold_patch():
    record = {
        "repo_name": "aiohttp",
        "docker_image": "namanjain12/aiohttp_final:abc123",
        "commit_hash": "abc123",
        "problem_statement": "Fix chunk parser.",
        "expected_output_json": json.dumps({"SECRET_TEST": "PASSED"}),
        "parsed_commit_content": "SECRET_GOLD_PATCH",
    }
    task = normalize_r2e_task_record(record, "R2E-Gym/R2E-Gym-Lite", "dev_10pr_v1")

    prompt = build_initial_prompt(task, current_observation="Workspace ready.", max_problem_chars=2000)

    assert "Fix chunk parser." in prompt
    assert "namanjain12/aiohttp_final:abc123" in prompt
    assert "SECRET_TEST" not in prompt
    assert "SECRET_GOLD_PATCH" not in prompt


def test_focused_validation_cmd_prefers_dataset_run_tests():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix chunk parser.",
            "run_tests": "python -m pytest -q tests/test_http_parser.py::test_chunk_split",
            "FAIL_TO_PASS": ["tests/test_http_parser.py::test_should_not_be_used"],
        },
        "R2E-Gym/SWE-Bench-Lite",
        "test",
    )

    assert focused_validation_cmd(task) == "python -m pytest -q tests/test_http_parser.py::test_chunk_split"


def test_focused_validation_cmd_uses_fail_to_pass_and_small_pass_to_pass():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix connector limits.",
            "FAIL_TO_PASS": ["tests/test_connector.py::test_limit", "tests/test_client.py::test_close"],
            "PASS_TO_PASS": [
                "tests/test_connector.py::test_regression_a",
                "tests/test_connector.py::test_regression_b",
                "tests/test_connector.py::test_regression_c",
            ],
        },
        "R2E-Gym/SWE-Bench-Lite",
        "test",
    )

    assert focused_validation_cmd(task) == (
        "python -m pytest -q "
        "tests/test_connector.py::test_limit "
        "tests/test_client.py::test_close "
        "tests/test_connector.py::test_regression_a "
        "tests/test_connector.py::test_regression_b"
    )


def test_focused_validation_cmd_falls_back_without_test_spec():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix chunk parser.",
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )

    assert focused_validation_cmd(task) == "python -m pytest -q"


def test_prompts_show_focused_validate_command_from_fail_to_pass():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix chunk parser.",
            "FAIL_TO_PASS": ["tests/test_http_parser.py::test_chunk_split"],
        },
        "R2E-Gym/SWE-Bench-Lite",
        "test",
    )

    initial_prompt = build_initial_prompt(task, current_observation="Workspace ready.")
    assert "Recommended focused validate command: python -m pytest -q tests/test_http_parser.py::test_chunk_split" in initial_prompt
    assert "<function=validate>" not in initial_prompt

    step_prompt = build_step_prompt(
        task,
        current_observation="The file has been edited.",
        history_context="Previous step 1",
        history_length=1,
        current_step=2,
        max_steps=5,
        tool_mask={"allow_validate": True, "allow_submit": False},
    )
    assert "Recommended focused validate command: python -m pytest -q tests/test_http_parser.py::test_chunk_split" in step_prompt
    assert "<parameter=cmd>python -m pytest -q tests/test_http_parser.py::test_chunk_split</parameter>" in step_prompt
    assert "<parameter=cmd>python -m pytest -q</parameter>" not in step_prompt


def test_prompt_enforces_single_safe_tool_call_with_issue_specific_examples():
    task = normalize_r2e_task_record(
        {
            "repo_name": "Orange3",
            "docker_image": "namanjain12/orange3_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": (
                "Saving a workflow with renamed variables raises "
                "IncompatibleContext in Orange widgets."
            ),
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )

    assert issue_search_terms(task)[0] == "IncompatibleContext"

    prompt = build_initial_prompt(task, current_observation="Workspace ready.")

    assert "Your response must be exactly one XML tool call and nothing else." in prompt
    assert "<function=bash>" in prompt
    assert "<parameter=cmd>pwd && find . -maxdepth 2 -type f | head -100</parameter>" in prompt
    assert "<parameter=cmd>grep -RIn -- 'IncompatibleContext' . | head -50</parameter>" in prompt
    assert "<function=str_replace_editor>" in prompt
    assert "<parameter=command>view</parameter>" in prompt
    assert "<parameter=path>/testbed</parameter>" in prompt
    assert "Allowed tool calls now:" in prompt
    assert "<function=validate>" not in prompt
    assert "<function=submit></function>" not in prompt
    assert "Masked tool calls now:" in prompt
    assert "validate: available after a successful source edit" in prompt
    assert "submit: available after a successful source edit and validation" in prompt
    assert "- Markdown code fences" in prompt
    assert "- JSON in a Markdown fenced block" in prompt
    assert "- Natural language before or after the tool call" in prompt
    assert "- Multiple tool calls" in prompt
    assert "- Missing parameter tags" in prompt
    assert "- path with line suffix such as /testbed/foo.py:123" in prompt
    assert "At the first step, use bash to inspect the repository." in prompt
    assert "bash commands are executed through bash -lc with pipefail enabled" in prompt
    assert "The default bash working directory is /testbed" in prompt
    assert "R2E places the repository root directly at /testbed" in prompt
    assert "Do not assume a subdirectory named after the repository exists" in prompt
    assert "Do not cd into a file path" in prompt
    assert "start with the issue-specific grep example" in prompt
    assert "<parameter=view_range>[1, 120]</parameter>" in prompt
    assert "Do not submit until at least one successful source edit" in prompt
    assert "one post-edit validation command" in prompt
    assert "python -m pytest -q" in prompt
    assert "<relevant_test>" not in prompt
    assert "```" not in prompt
    assert prompt.rstrip().endswith("Your next response must be exactly one XML tool call and nothing else.")
    for fixed_example in ("aiohttp", "TransferEncodingError", "http_parser.py"):
        assert fixed_example not in prompt
    for placeholder in ("class Foo", "tool_name", "example.py"):
        assert placeholder not in prompt


def test_step_prompt_uses_current_issue_search_terms_for_allowed_grep():
    task = normalize_r2e_task_record(
        {
            "repo_name": "Orange3",
            "docker_image": "namanjain12/orange3_final:def456",
            "commit_hash": "def456",
            "problem_statement": "TypeError occurs when TableModel handles discrete metas.",
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )

    prompt = build_step_prompt(
        task,
        current_observation="Workspace ready.",
        history_context="Previous step 1",
        history_length=1,
        current_step=2,
        max_steps=5,
    )

    assert issue_search_terms(task)[0] == "TypeError"
    assert "<parameter=cmd>grep -RIn -- 'TypeError' . | head -50</parameter>" in prompt
    assert "grep -RIn -- 'TransferEncodingError'" not in prompt
    assert "/testbed/aiohttp/http_parser.py" not in prompt


def test_prompt_action_mask_exposes_validate_then_submit_by_state():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix chunk parser.",
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )

    needs_validation = build_step_prompt(
        task,
        current_observation="The file /testbed/aiohttp/http_parser.py has been edited.",
        history_context="Previous step 1",
        history_length=1,
        current_step=2,
        max_steps=5,
        tool_mask={
            "allow_validate": True,
            "allow_submit": False,
            "submit_reason": "requires_validation_after_source_edit",
        },
    )

    assert "Allowed tool calls now:" in needs_validation
    assert "<function=validate>" in needs_validation
    assert "<parameter=cmd>python -m pytest -q</parameter>" in needs_validation
    assert "<function=submit></function>" not in needs_validation
    assert "submit: available after a validation command has run" in needs_validation

    ready_to_submit = build_step_prompt(
        task,
        current_observation="Exit code: 1\n[output]\nfocused validation failed",
        history_context="Previous step 1",
        history_length=1,
        current_step=3,
        max_steps=5,
        tool_mask={
            "allow_validate": True,
            "allow_submit": True,
        },
    )

    assert "<function=validate>" in ready_to_submit
    assert "<function=submit></function>" in ready_to_submit


def test_prompt_removes_markdown_fence_tokens_from_issue_text():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix this code block:\n```python\nprint('bug')\n```",
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )

    prompt = build_initial_prompt(task, current_observation="Workspace ready.")

    assert "print('bug')" in prompt
    assert "```" not in prompt
    assert "code block delimiter removed" in prompt


def test_step_prompt_ends_with_strict_xml_reminder_after_observation():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix chunk parser.",
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )

    prompt = build_step_prompt(
        task,
        current_observation="ERROR: path /testbed/foo.py does not exist.",
        history_context="Previous step 1\nObservation:\nWorkspace ready.\nTool call:\n<function=bash>\n<parameter=cmd>pwd</parameter>\n</function>",
        history_length=1,
        current_step=2,
        max_steps=4,
    )

    assert "Current observation:" in prompt
    assert prompt.rstrip().endswith("Your next response must be exactly one XML tool call and nothing else.")
    assert prompt.rfind("Current observation:") < prompt.rfind("Your next response must be exactly one XML tool call")


def test_r2e_history_action_is_canonical_xml_not_python_dict():
    action = {
        "tool_name": "str_replace_editor",
        "parameters": {
            "command": "view",
            "path": "/testbed/aiohttp/http_parser.py",
            "view_range": [225, 275],
        },
    }

    action_summary = format_r2e_action_for_history(action)
    history = format_r2e_history_turn(3, "Viewed file output.", action_summary)

    assert "<function=str_replace_editor>" in history
    assert "<parameter=command>view</parameter>" in history
    assert "<parameter=path>/testbed/aiohttp/http_parser.py</parameter>" in history
    assert "<parameter=view_range>[225, 275]</parameter>" in history
    assert "Tool call:" in history
    assert "{'tool_name'" not in history
    assert "Action 3:" not in history


def test_invalid_r2e_history_keeps_editor_parameter_recovery_hint():
    action_summary = format_r2e_action_for_history(
        {
            "tool_name": "",
            "parameters": {},
            "error": (
                "str_replace_editor str_replace requires a non-empty 'old_str'. "
                "Do not write <parameter>old_str</parameter>."
            ),
        }
    )
    history = format_r2e_history_turn(4, "str_replace_editor str_replace requires a non-empty 'old_str'.", action_summary)

    assert "Previous response was invalid:" in history
    assert "requires a non-empty 'old_str'" in history
    assert "Do not write <parameter>old_str</parameter>" in history
    assert "Next response must be exactly one XML tool call" in history


def test_final_step_prompt_requires_submit():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix chunk parser.",
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )

    prompt = build_step_prompt(
        task,
        current_observation="Last command output.",
        history_context="Action: bash",
        history_length=1,
        current_step=4,
        max_steps=4,
    )

    assert "This is the final allowed step." in prompt
    assert "Your entire response must be exactly:" in prompt
    assert "<function=submit>" in prompt


def test_dev_split_is_rejected_for_training_without_explicit_override():
    with pytest.raises(SplitPolicyError):
        validate_r2e_split_policy(
            dataset_name="R2E-Gym/R2E-Gym-Lite",
            split="dev_10pr_v1",
            mode="train",
            allow_train_on_dev=False,
        )

    validate_r2e_split_policy(
        dataset_name="R2E-Gym/R2E-Gym-Lite",
        split="dev_10pr_v1",
        mode="train",
        allow_train_on_dev=True,
    )


def test_swe_bench_lite_test_is_rejected_for_training():
    with pytest.raises(SplitPolicyError):
        validate_r2e_split_policy(
            dataset_name="R2E-Gym/SWE-Bench-Lite",
            split="test",
            mode="train",
            allow_train_on_dev=True,
        )


def test_parse_r2e_actions_supports_xml_and_json_and_reports_invalid_actions():
    bash = parse_r2e_action(
        "<function=bash>\n<parameter=cmd>pwd</parameter>\n</function>"
    )
    assert bash.is_valid
    assert bash.action == {"tool_name": "bash", "parameters": {"cmd": "pwd"}}

    bash_body = parse_r2e_action("<function=bash>\npwd && ls\n</function>")
    assert bash_body.is_valid
    assert bash_body.action == {"tool_name": "bash", "parameters": {"cmd": "pwd && ls"}}

    submit = parse_r2e_action('{"tool_name": "submit", "parameters": {}}')
    assert submit.is_valid
    assert submit.action == {"tool_name": "submit", "parameters": {}}

    validate = parse_r2e_action(
        "<function=validate>\n<parameter=cmd>python -m pytest -q</parameter>\n</function>"
    )
    assert validate.is_valid
    assert validate.action == {"tool_name": "validate", "parameters": {"cmd": "python -m pytest -q"}}

    invalid = parse_r2e_action("<function=bash></function>")
    assert not invalid.is_valid
    assert invalid.action["tool_name"] == ""
    assert "requires" in invalid.error


def test_parse_r2e_action_rejects_multiple_tool_calls_without_placeholder_error():
    invalid = parse_r2e_action(
        "<function=bash>\n"
        "<parameter=cmd>pwd</parameter>\n"
        "</function>\n"
        "<function=submit>\n"
        "</function>"
    )

    assert not invalid.is_valid
    assert "exactly one" in invalid.error
    assert "tool_name" not in invalid.error


def test_parse_r2e_action_explains_malformed_str_replace_parameter_tags():
    invalid = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/testbed/aiohttp/http_parser.py</parameter>\n"
        "<parameter>old_str</parameter>\n"
        "</function>"
    )

    assert not invalid.is_valid
    assert "requires a non-empty 'old_str'" in invalid.error
    assert "Do not write <parameter>old_str</parameter>" in invalid.error
    assert "<parameter=old_str>" in invalid.error
    assert "<parameter=new_str>" in invalid.error


def test_parse_r2e_action_accepts_markdown_fenced_json_and_bash():
    fenced_json = parse_r2e_action(
        "```json\n"
        '{"tool_name": "bash", "parameters": {"cmd": "grep -r TODO ."}}\n'
        "```"
    )
    assert fenced_json.is_valid
    assert fenced_json.action == {"tool_name": "bash", "parameters": {"cmd": "grep -r TODO ."}}

    fenced_bash = parse_r2e_action(
        "```bash\n"
        "str_replace_editor view /testbed/tests/test_resolver.py\n"
        "```"
    )
    assert fenced_bash.is_valid
    assert fenced_bash.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/tests/test_resolver.py"},
    }



def test_parse_r2e_action_accepts_json_tool_schema_variants():
    flat_tool = parse_r2e_action(
        "```json\n"
        '{"tool": "str_replace_editor", "command": "view", "path": "/testbed/Orange/tree.py"}\n'
        "```"
    )
    assert flat_tool.is_valid
    assert flat_tool.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/Orange/tree.py"},
    }

    wrapped_tool_call = parse_r2e_action(
        "```json\n"
        '{"tool_call": {"tool_name": "str_replace_editor", "parameters": {"command": "str_replace", "path": "/testbed/tests/test_resolver.py", "old_str": "old", "new_str": "new"}}}\n'
        "```"
    )
    assert wrapped_tool_call.is_valid
    assert wrapped_tool_call.action == {
        "tool_name": "str_replace_editor",
        "parameters": {
            "command": "str_replace",
            "path": "/testbed/tests/test_resolver.py",
            "old_str": "old",
            "new_str": "new",
        },
    }


def test_parse_r2e_action_accepts_r2e_style_extracted_json_variants():
    action_alias = parse_r2e_action(
        "```json\n"
        '{"action": "str_replace_editor", "command": "view", "path": "/testbed/tests/test_asyncresolver.py"}\n'
        "```"
    )
    assert action_alias.is_valid
    assert action_alias.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/tests/test_asyncresolver.py"},
    }

    prose_wrapped_json = parse_r2e_action(
        "I will inspect the failing test.\n\n"
        "```json\n"
        '{"action": "str_replace_editor", "command": "view", "path": "/testbed/tests/test_http_parser.py"}\n'
        "```"
    )
    assert prose_wrapped_json.is_valid
    assert prose_wrapped_json.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/tests/test_http_parser.py"},
    }

    single_openai_tool_call = parse_r2e_action(
        "```json\n"
        '{"tool_calls": [{"function": {"name": "bash", "arguments": "{\\\"cmd\\\": \\\"pwd\\\"}"}}]}\n'
        "```"
    )
    assert single_openai_tool_call.is_valid
    assert single_openai_tool_call.action == {"tool_name": "bash", "parameters": {"cmd": "pwd"}}


def test_parse_r2e_action_repairs_common_bash_search_mistakes():
    bad_search = parse_r2e_action(
        "<function=bash>\n"
        '<parameter=cmd>cd /testbed/aiohttp/http_parser.py && grep -n "TransferEncodingError" && find . -maxdepth 2 -type f | head -100</parameter>\n'
        "</function>"
    )
    assert bad_search.is_valid
    assert bad_search.action == {
        "tool_name": "bash",
        "parameters": {
            "cmd": 'cd /testbed/aiohttp && grep -RIn -- "TransferEncodingError" . | head -50 && find . -maxdepth 2 -type f | head -100'
        },
    }

    search_with_path = parse_r2e_action(
        "<function=bash>\n"
        '<parameter=cmd>grep -n "TransferEncodingError" /testbed/aiohttp/http_parser.py</parameter>\n'
        "</function>"
    )
    assert search_with_path.is_valid
    assert search_with_path.action == {
        "tool_name": "bash",
        "parameters": {"cmd": 'grep -n "TransferEncodingError" /testbed/aiohttp/http_parser.py'},
    }


def test_parse_r2e_action_normalizes_editor_paths_and_search_ranges():
    line_suffix = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>/testbed/tests/test_http_parser.py:272</parameter>\n"
        "</function>"
    )
    assert line_suffix.is_valid
    assert line_suffix.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/tests/test_http_parser.py", "view_range": [272, -1]},
    }

    repo_relative = parse_r2e_action(
        '{"action": "str_replace_editor", "command": "view", "path": "tests/test_http_parser.py:227-240"}'
    )
    assert repo_relative.is_valid
    assert repo_relative.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/tests/test_http_parser.py", "view_range": [227, 240]},
    }

    symbolic_view_range = parse_r2e_action(
        '{"action": "str_replace_editor", "command": "view", "path": "/repo/Orange/data.py", "view_range": "DiscreteVariable"}'
    )
    assert symbolic_view_range.is_valid
    assert symbolic_view_range.action == {
        "tool_name": "bash",
        "parameters": {"cmd": "grep -RIn -- 'DiscreteVariable' /testbed/Orange/data.py | head -50"},
    }

    command_view_range = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command>view_range</parameter>\n"
        "<parameter=path>/testbed/aiohttp/http_parser.py</parameter>\n"
        "<parameter>start_line=204</parameter>\n"
        "<parameter>end_line=225</parameter>\n"
        "</function>"
    )
    assert command_view_range.is_valid
    assert command_view_range.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/aiohttp/http_parser.py", "view_range": [204, 225]},
    }

    range_parameter = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>/testbed/aiohttp/http_parser.py</parameter>\n"
        "<parameter>range=227-238</parameter>\n"
        "</function>"
    )
    assert range_parameter.is_valid
    assert range_parameter.action["parameters"]["view_range"] == [227, 238]

def test_parse_r2e_action_accepts_malformed_parameter_tags():
    malformed_equals = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/testbed/Orange/data/pandas_compat.py</parameter>\n"
        '<parameter>old_str="table_to_frame"</parameter>\n'
        '<parameter>new_str="table_from_frame"</parameter>\n'
        "</function>"
    )
    assert malformed_equals.is_valid
    assert malformed_equals.action["parameters"]["old_str"] == "table_to_frame"
    assert malformed_equals.action["parameters"]["new_str"] == "table_from_frame"

    malformed_arrow = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter>command>view</parameter>\n"
        "<parameter>path>/testbed/Orange/tree.py</parameter>\n"
        "</function>"
    )
    assert malformed_arrow.is_valid
    assert malformed_arrow.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/Orange/tree.py"},
    }


def test_parse_r2e_action_accepts_key_value_parameter_tags():
    malformed_tag_value = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command=view</parameter>\n"
        "<parameter=path=/testbed/Orange/util.py</parameter>\n"
        "</function>"
    )
    assert malformed_tag_value.is_valid
    assert malformed_tag_value.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/Orange/util.py"},
    }

    malformed_body_value = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter>command=create</parameter>\n"
        "<parameter>path=/testbed/Orange/util.py</parameter>\n"
        '<parameter>file_text="print(1)"</parameter>\n'
        "</function>"
    )
    assert malformed_body_value.is_valid
    assert malformed_body_value.action["parameters"]["command"] == "create"
    assert malformed_body_value.action["parameters"]["path"] == "/testbed/Orange/util.py"
    assert malformed_body_value.action["parameters"]["file_text"] == "print(1)"


def test_parse_r2e_action_repairs_common_str_replace_parameter_variants():
    malformed_xmlish_values = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/testbed/tests/test_resolver.py</parameter>\n"
        "<parameter>old_str</Original error message></parameter>\n"
        "<parameter>new_str</Test error message></parameter>\n"
        "</function>"
    )
    assert malformed_xmlish_values.is_valid
    assert malformed_xmlish_values.action["parameters"]["old_str"] == "Original error message"
    assert malformed_xmlish_values.action["parameters"]["new_str"] == "Test error message"

    three_bare_values = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/testbed/aiohttp/http_parser.py</parameter>\n"
        "<parameter>old_str</parameter>\n"
        "<parameter>TransferEncodingError</parameter>\n"
        "<parameter>TransferEncodingErrorFixed</parameter>\n"
        "</function>"
    )
    assert three_bare_values.is_valid
    assert three_bare_values.action["parameters"]["old_str"] == "TransferEncodingError"
    assert three_bare_values.action["parameters"]["new_str"] == "TransferEncodingErrorFixed"

    empty_old_str = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/testbed/aiohttp/http_parser.py</parameter>\n"
        "<parameter=old_str>\n</parameter>\n"
        "<parameter=new_str></parameter>\n"
        "</function>"
    )
    assert not empty_old_str.is_valid
    assert "non-empty 'old_str'" in empty_old_str.error


def test_parse_r2e_action_converts_editor_cli_shorthand_from_bash():
    shorthand_view = parse_r2e_action(
        "<function=bash>\n"
        "<parameter=cmd>str_replace_editor view /testbed/Orange/data.py</parameter>\n"
        "</function>"
    )
    assert shorthand_view.is_valid
    assert shorthand_view.action == {
        "tool_name": "str_replace_editor",
        "parameters": {"command": "view", "path": "/testbed/Orange/data.py"},
    }

    shorthand_replace = parse_r2e_action(
        "```bash\n"
        "str_replace_editor replace /testbed/Orange/data.py old_value new_value\n"
        "```"
    )
    assert shorthand_replace.is_valid
    assert shorthand_replace.action == {
        "tool_name": "str_replace_editor",
        "parameters": {
            "command": "str_replace",
            "path": "/testbed/Orange/data.py",
            "old_str": "old_value",
            "new_str": "new_value",
        },
    }



def test_parse_r2e_action_accepts_xml_parameter_schema_variants():
    direct_tags = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<command>str_replace</command>\n"
        "<path>/testbed/Orange/util.py</path>\n"
        "<old_str>old</old_str>\n"
        "<new_str>new</new_str>\n"
        "</function>"
    )
    assert direct_tags.is_valid
    assert direct_tags.action == {
        "tool_name": "str_replace_editor",
        "parameters": {
            "command": "str_replace",
            "path": "/testbed/Orange/util.py",
            "old_str": "old",
            "new_str": "new",
        },
    }

    parameters_container = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameters>\n"
        "  <command>view</command>\n"
        "  <path>/testbed/Orange/data.py</path>\n"
        "  <view_range>DiscreteVariable</view_range>\n"
        "</parameters>\n"
        "</function>"
    )
    assert parameters_container.is_valid
    assert parameters_container.action == {
        "tool_name": "bash",
        "parameters": {"cmd": "grep -RIn -- 'DiscreteVariable' /testbed/Orange/data.py | head -50"},
    }

    comma_kv_body = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter>command=insert, path=/testbed/Orange/base.py, insert_line=25, "
        "new_str=from Orange.data import ContinuousVariable, DiscreteVariable</parameter>\n"
        "</function>"
    )
    assert comma_kv_body.is_valid
    assert comma_kv_body.action == {
        "tool_name": "str_replace_editor",
        "parameters": {
            "command": "insert",
            "path": "/testbed/Orange/base.py",
            "insert_line": "25",
            "new_str": "from Orange.data import ContinuousVariable, DiscreteVariable",
        },
    }

    sequential_key_value_parameters = parse_r2e_action(
        "<function=str_replace_editor>\n"
        "<parameter>command</parameter>\n"
        "<parameter>create</parameter>\n"
        "<parameter>path=/testbed/Orange/util.py</parameter>\n"
        "<parameter>file_text=print(1)</parameter>\n"
        "</function>"
    )
    assert sequential_key_value_parameters.is_valid
    assert sequential_key_value_parameters.action["parameters"]["command"] == "create"
    assert sequential_key_value_parameters.action["parameters"]["path"] == "/testbed/Orange/util.py"
    assert sequential_key_value_parameters.action["parameters"]["file_text"] == "print(1)"

def test_projection_debug_preserves_raw_invalid_response_without_mutating_actions():
    actions = [{"tool_name": "", "parameters": {}, "error": "Action format error"}]

    enriched = attach_r2e_projection_debug(
        actions,
        ["I should inspect the files first."],
        [0],
        max_raw_chars=100,
    )

    assert actions[0].get("_raw_model_response") is None
    assert enriched[0]["_projection_is_valid"] is False
    assert enriched[0]["_raw_model_response"] == "I should inspect the files first."


def test_rollout_io_logger_writes_train_step_episode_step_json(tmp_path):
    config = SimpleNamespace(
        env=SimpleNamespace(
            env_name="r2e_code_swe",
            r2e_code_swe=SimpleNamespace(
                rollout_io=SimpleNamespace(
                    enabled=True,
                    log_dir=str(tmp_path),
                    version="v_test",
                    max_prompt_chars=80,
                    max_response_chars=80,
                    max_observation_chars=80,
                    max_info_chars=200,
                )
            ),
        )
    )
    collector = TrajectoryCollector(config=config, tokenizer=None)

    collector._write_rollout_io_logs(
        global_step=7,
        rollout_step=0,
        obs={"text": ["PROMPT without hidden answers"], "anchor": ["Workspace ready."]},
        next_obs={"anchor": ["tool output"], "text": ["next prompt"]},
        text_actions=["<function=bash>\n<parameter=cmd>pwd</parameter>\n</function>"],
        infos=[
            {
                "task_id": "R2E-Gym/R2E-Gym-Subset:train:0:repo:sha",
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                "split": "train",
                "parsed_action": {"tool_name": "bash", "parameters": {"cmd": "pwd"}},
                "expected_output_json": "SECRET_EXPECTED_OUTPUT",
                "gold_patch_optional": "SECRET_GOLD_PATCH",
            }
        ],
        rewards=[0.0],
        dones=[False],
        traj_uid=["abc123"],
        active_masks=[True],
    )

    log_path = tmp_path / "v_test" / "train_step_000007" / "episode_0000_abc123" / "step_0001.json"
    assert log_path.exists()
    payload = json.loads(log_path.read_text())
    assert payload["train_step"] == 7
    assert payload["episode_index"] == 0
    assert payload["step_index"] == 1
    assert payload["model_input"] == "PROMPT without hidden answers"
    assert payload["raw_model_output"].startswith("<function=bash>")
    assert payload["parsed_action"] == {"tool_name": "bash", "parameters": {"cmd": "pwd"}}
    assert payload["tool_observation"] == "tool output"
    assert payload["reward"] == 0.0
    assert payload["done"] is False
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "SECRET_EXPECTED_OUTPUT" not in serialized
    assert "SECRET_GOLD_PATCH" not in serialized



def test_r2e_runtime_wraps_bash_commands_with_shell():
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["timeout"] = kwargs.get("timeout")
        captured["workdir"] = kwargs.get("workdir")
        return "ok", "0"

    runtime = R2ERepoRuntime(R2ERuntimeConfig(command_timeout=7))
    runtime.env = SimpleNamespace(runtime=SimpleNamespace(run=fake_run))

    user_command = "cd /testbed && printf '%s\n' hello | head -1"
    result = runtime.run_bash(user_command)

    assert result.info["tool_execution_success"] is True
    expected = "set -o pipefail\n" + user_command
    assert captured["command"] == "bash -lc " + shlex.quote(expected)
    assert captured["timeout"] == 7
    assert captured["workdir"] == "/testbed"


def test_r2e_runtime_workspace_overview_uses_r2e_directory_view():
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["timeout"] = kwargs.get("timeout")
        captured["workdir"] = kwargs.get("workdir")
        return "Here's the files and directories up to 2 levels deep:\nOrange/\naiohttp/\npyproject.toml\n", "0"

    runtime = R2ERepoRuntime(R2ERuntimeConfig(command_timeout=9, workspace_overview_max_chars=200))
    runtime.env = SimpleNamespace(runtime=SimpleNamespace(run=fake_run))

    overview = runtime.workspace_overview()

    assert captured["command"] == "str_replace_editor view --path /testbed"
    assert captured["timeout"] == 9
    assert captured["workdir"] == "/testbed"
    assert "Workspace root preview" in overview
    assert "Orange/" in overview
    assert "repository root is /testbed" in overview
    assert "do not assume /testbed/" in overview


def test_r2e_initial_observation_includes_workspace_overview():
    class _OverviewRuntime:
        setup_error = None

        def workspace_overview(self):
            return "Workspace root preview from str_replace_editor view /testbed:\nOrange/\n"

    task = _r2e_task_for_env_policy_tests()
    env = R2ECodeSWEEnv([task], env_num=1, group_n=1)

    observation = env._initial_observation(task, _OverviewRuntime())

    assert "Repository: aiohttp" in observation
    assert "Workspace: /testbed" in observation
    assert "Workspace root preview from str_replace_editor view /testbed" in observation
    assert "Orange/" in observation


def test_r2e_runtime_separates_protocol_validity_from_bash_execution_failure():
    runtime = R2ERepoRuntime(R2ERuntimeConfig(command_timeout=1))
    runtime.env = SimpleNamespace(
        runtime=SimpleNamespace(run=lambda *args, **kwargs: ("", "1"))
    )

    result = runtime.run_bash('grep -n "missing" /testbed/file.py')

    assert result.is_action_valid is True
    assert result.info["tool_execution_success"] is False
    assert result.info["tool_execution_fail_reason"] == "command_failed"
    assert result.info["fail_reason"] == "command_failed"




def test_prompt_includes_r2e_str_replace_uniqueness_guidance():
    task = normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix chunk parser.",
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )

    prompt = build_initial_prompt(task, current_observation="Workspace ready.")

    assert "old_str must match EXACTLY one or more consecutive lines" in prompt
    assert "If old_str is not unique" in prompt
    assert "include enough surrounding context" in prompt
    assert "identical old_str and new_str" in prompt


def test_r2e_runtime_editor_nonunique_error_adds_recovery_hint():
    runtime = R2ERepoRuntime(R2ERuntimeConfig(command_timeout=1, max_output_chars=2000))
    runtime.env = SimpleNamespace(
        runtime=SimpleNamespace(
            run=lambda *args, **kwargs: (
                "ERROR: Multiple occurrences of 'raise IncompatibleContext()' found in /testbed/Orange/widgets/settings.py. Please ensure it is unique before using str_replace.\n",
                "0",
            )
        )
    )

    result = runtime.run_editor(
        {
            "command": "str_replace",
            "path": "/testbed/Orange/widgets/settings.py",
            "old_str": "raise IncompatibleContext()",
            "new_str": "pass",
        }
    )

    assert result.is_action_valid is True
    assert result.info["tool_execution_success"] is False
    assert result.info["tool_execution_fail_reason"] == "editor_error"
    assert result.info["editor_recovery_hint"] == "non_unique_old_str"
    assert "old_str was not unique" in result.observation
    assert "copy a larger consecutive block" in result.observation
    assert "Do not repeat the same one-line str_replace" in result.observation


def test_r2e_runtime_editor_repairs_missing_common_indent_before_retry():
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            return (
                "ERROR: No occurrences of 'if pos >= start_pos:\\n    line = data[start_pos:pos]' found in /testbed/aiohttp/http_parser.py for replacement.\n",
                "0",
            )
        if len(commands) == 2:
            return (
                json.dumps(
                    {
                        "status": "matched",
                        "repaired_old_str": "                if pos >= start_pos:\n                    line = data[start_pos:pos]",
                        "repaired_new_str": "                if pos >= start_pos:\n                    line = data[start_pos:pos]\n                    raise BadHttpMessage(\"Data after `Connection: close`\")",
                        "line_start": 331,
                    }
                ),
                "0",
            )
        return ("The file /testbed/aiohttp/http_parser.py has been edited.", "0")

    runtime = R2ERepoRuntime(R2ERuntimeConfig(command_timeout=1, max_output_chars=2000))
    runtime.env = SimpleNamespace(runtime=SimpleNamespace(run=fake_run))

    result = runtime.run_editor(
        {
            "command": "str_replace",
            "path": "/testbed/aiohttp/http_parser.py",
            "old_str": "if pos >= start_pos:\n    line = data[start_pos:pos]",
            "new_str": "if pos >= start_pos:\n    line = data[start_pos:pos]\n    raise BadHttpMessage(\"Data after `Connection: close`\")",
        }
    )

    assert len(commands) == 3
    assert "python3" in commands[1]
    assert "--old_str '                if pos >= start_pos:" in commands[2]
    assert "raise BadHttpMessage" in commands[2]
    assert result.info["tool_execution_success"] is True
    assert result.info["indent_repair_applied"] is True
    assert result.info["editor_recovery_hint"] == "indent_repaired_old_str"
    assert "repaired missing leading indentation" in result.observation


def test_r2e_runtime_editor_repairs_view_range_scoped_nonunique_replace():
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            return (
                "ERROR: Multiple occurrences of 'raise IncompatibleContext()' found in /testbed/Orange/widgets/settings.py. Please ensure it is unique before using str_replace.\n",
                "0",
            )
        if len(commands) == 2:
            return (
                json.dumps(
                    {
                        "status": "matched",
                        "repaired_old_str": "                    raise IncompatibleContext()\n                else:\n                    raise IncompatibleContext()",
                        "repaired_new_str": "                    pass\n                else:\n                    pass",
                        "line_start": 1042,
                        "line_end": 1044,
                        "replacement_count": 2,
                    }
                ),
                "0",
            )
        return ("The file /testbed/Orange/widgets/settings.py has been edited.", "0")

    runtime = R2ERepoRuntime(R2ERuntimeConfig(command_timeout=1, max_output_chars=2000))
    runtime.env = SimpleNamespace(runtime=SimpleNamespace(run=fake_run))

    result = runtime.run_editor(
        {
            "command": "str_replace",
            "path": "/testbed/Orange/widgets/settings.py",
            "view_range": "[1042, 1044]",
            "old_str": "raise IncompatibleContext()",
            "new_str": "pass",
        }
    )

    assert len(commands) == 3
    assert "python3" in commands[1]
    assert "--old_str '                    raise IncompatibleContext()" in commands[2]
    assert "--new_str '                    pass" in commands[2]
    assert result.info["tool_execution_success"] is True
    assert result.info["range_repair_applied"] is True
    assert result.info["range_repair_replacement_count"] == 2
    assert result.info["editor_recovery_hint"] == "view_range_scoped_old_str"
    assert "expanded view_range into a unique old_str block" in result.observation


def test_r2e_runtime_separates_protocol_validity_from_editor_execution_failure():
    runtime = R2ERepoRuntime(R2ERuntimeConfig(command_timeout=1))
    runtime.env = SimpleNamespace(
        runtime=SimpleNamespace(
            run=lambda *args, **kwargs: (
                "ERROR: The path '/testbed/missing.py' does not exist.\n",
                "0",
            )
        )
    )

    result = runtime.run_editor({"command": "view", "path": "/testbed/missing.py"})

    assert result.is_action_valid is True
    assert result.info["tool_execution_success"] is False
    assert result.info["tool_execution_fail_reason"] == "editor_error"
    assert result.info["fail_reason"] == "editor_error"

    unsafe_path = runtime.run_editor({"command": "view", "path": "/tmp/missing.py"})
    assert unsafe_path.is_action_valid is False
    assert unsafe_path.info["tool_execution_fail_reason"] == "invalid_path"

def _r2e_task_for_env_policy_tests():
    return normalize_r2e_task_record(
        {
            "repo_name": "aiohttp",
            "docker_image": "namanjain12/aiohttp_final:abc123",
            "commit_hash": "abc123",
            "problem_statement": "Fix chunk parser.",
        },
        "R2E-Gym/R2E-Gym-Lite",
        "dev_10pr_v1",
    )


class _PolicyRuntime:
    setup_error = None
    episode_dir = None

    def __init__(self):
        self.submit_count = 0
        self.bash_count = 0
        self.editor_count = 0

    def run_bash(self, cmd, cwd=None):
        self.bash_count += 1
        return R2EToolResult(
            "Exit code: 1\n[output]\nmissing",
            is_action_valid=True,
            info={"tool_execution_success": False, "tool_execution_fail_reason": "command_failed", "fail_reason": "command_failed"},
        )

    def run_editor(self, params):
        self.editor_count += 1
        return R2EToolResult(
            "edited",
            is_action_valid=True,
            info={"tool_execution_success": True, "tool_execution_fail_reason": None, "fail_reason": None},
        )

    def submit(self):
        self.submit_count += 1
        return R2EToolResult("submitted", reward=1.0, done=True, is_action_valid=True, info={"reward": 1.0, "won": True})


class _SuccessfulBashPolicyRuntime(_PolicyRuntime):
    def run_bash(self, cmd, cwd=None):
        self.bash_count += 1
        return R2EToolResult(
            "Exit code: 0\n[output]\nfound results",
            is_action_valid=True,
            info={"exit_code": "0", "tool_execution_success": True, "tool_execution_fail_reason": None, "fail_reason": None},
        )


class _FailingValidationPolicyRuntime(_PolicyRuntime):
    def run_bash(self, cmd, cwd=None):
        self.bash_count += 1
        return R2EToolResult(
            "Exit code: 1\n[output]\nfocused test failed",
            is_action_valid=True,
            info={
                "exit_code": "1",
                "tool_execution_success": False,
                "tool_execution_fail_reason": "command_failed",
                "fail_reason": "command_failed",
            },
        )


def _make_policy_env(runtime, require_validation_before_submit=False):
    task = _r2e_task_for_env_policy_tests()
    env = R2ECodeSWEEnv(
        [task],
        max_steps=5,
        require_successful_edit_before_submit=True,
        require_validation_before_submit=require_validation_before_submit,
        max_repeated_failed_actions=1,
        reward_shaping_config=R2ERewardShapingConfig(enabled=True),
    )
    env.current_tasks = [task]
    env.runtimes = [runtime]
    env.step_counts = [0]
    env.dones = [False]
    env.successful_edit_counts = [0]
    env.successful_source_edit_counts = [0]
    env.validation_after_source_edit_counts = [0]
    env.reward_shaping_states = [env.reward_shaping_config.new_state()]
    env.last_failed_action_signatures = [None]
    env.repeated_failed_action_counts = [0]
    env.seen_no_progress_action_signatures = [set()]
    env.repeated_no_progress_action_counts = [{}]
    return env


def test_r2e_path_classifier_separates_source_test_repro_and_aux_files():
    assert classify_r2e_path("/testbed/aiohttp/http_parser.py") == "source"
    assert classify_r2e_path("/testbed/Orange/data/__init__.py") == "source"
    assert classify_r2e_path("/testbed/tests/test_http_parser.py") == "test"
    assert classify_r2e_path("/testbed/aiohttp/tests/test_parser.py") == "test"
    assert classify_r2e_path("/testbed/test.py") == "repro"
    assert classify_r2e_path("/testbed/reproduce_issue.py") == "repro"
    assert classify_r2e_path("/testbed/r2e_tests") == "r2e_aux"
    assert classify_r2e_path("/testbed/install.sh") == "r2e_aux"
    assert classify_r2e_path("/tmp/foo.py") == "outside_testbed"


def test_r2e_reward_shaping_uses_validation_and_error_penalty_knobs():
    config = R2ERewardShapingConfig.from_obj(
        {
            "submit_before_validation_after_source_edit_penalty": -0.12,
            "repeated_failed_action_penalty": -0.06,
            "repeated_no_progress_action_penalty": -0.08,
            "max_steps_no_validation_after_edit_penalty": -0.08,
            "editor_no_occurrences_penalty": -0.04,
            "editor_multiple_occurrences_penalty": -0.04,
            "negative_cap": -0.35,
        }
    )

    def apply_once(fail_reason, *, editor_recovery_hint=None):
        state = config.new_state()
        total, breakdown = state.update(
            config,
            {"tool_name": "str_replace_editor", "parameters": {"command": "str_replace", "path": "/testbed/pkg/mod.py"}},
            {
                "tool_execution_success": False,
                "tool_execution_fail_reason": fail_reason,
                "fail_reason": fail_reason,
                "editor_recovery_hint": editor_recovery_hint,
            },
        )
        return total, breakdown

    state = config.new_state()
    state.source_edit_count = 1
    total, breakdown = state.update(
        config,
        {"tool_name": "submit", "parameters": {}},
        {
            "tool_execution_success": False,
            "tool_execution_fail_reason": "submit_before_validation_after_source_edit",
            "fail_reason": "submit_before_validation_after_source_edit",
        },
    )
    assert total == pytest.approx(-0.12)
    assert breakdown["events"] == ["submit_before_validation_after_source_edit"]
    assert "clean_submit_after_source_edit" not in breakdown["events"]

    total, breakdown = apply_once("repeated_no_progress_action")
    assert total == pytest.approx(-0.08)
    assert breakdown["events"] == ["repeated_no_progress_action"]

    total, breakdown = apply_once("editor_error", editor_recovery_hint="missing_or_misindented_old_str")
    assert total == pytest.approx(-0.04)
    assert breakdown["events"] == ["editor_no_occurrences"]

    total, breakdown = apply_once("editor_error", editor_recovery_hint="non_unique_old_str")
    assert total == pytest.approx(-0.04)
    assert breakdown["events"] == ["editor_multiple_occurrences"]

    state = config.new_state()
    state.source_edit_count = 1
    total, breakdown = state.update(
        config,
        {"tool_name": "submit", "parameters": {}},
        {"tool_execution_success": True, "tool_execution_fail_reason": None, "fail_reason": None},
        auto_submitted=True,
    )
    assert breakdown["penalty"] == pytest.approx(-0.08)
    assert "max_steps_no_validation_after_edit" in breakdown["events"]



def _mixed_raw_patch_with_r2e_aux_files():
    return """diff --git a/aiohttp/http_parser.py b/aiohttp/http_parser.py
index 1111111..2222222 100644
--- a/aiohttp/http_parser.py
+++ b/aiohttp/http_parser.py
@@ -1,3 +1,3 @@
-old parser
+new parser
diff --git a/Makefile b/Makefile
index 3333333..4444444 100644
--- a/Makefile
+++ b/Makefile
@@ -1 +1 @@
-pip install -e .
+uv sync
diff --git a/install.sh b/install.sh
new file mode 100755
index 0000000..5555555
--- /dev/null
+++ b/install.sh
@@ -0,0 +1 @@
+echo setup
diff --git a/process_aiohttp_updateasyncio.py b/process_aiohttp_updateasyncio.py
new file mode 100644
index 0000000..6666666
--- /dev/null
+++ b/process_aiohttp_updateasyncio.py
@@ -0,0 +1 @@
+print('helper')
diff --git a/r2e_tests b/r2e_tests
new file mode 120000
index 0000000..7777777
--- /dev/null
+++ b/r2e_tests
@@ -0,0 +1 @@
+/root/r2e_tests
diff --git a/tests/test_http_parser.py b/tests/test_http_parser.py
index 8888888..9999999 100644
--- a/tests/test_http_parser.py
+++ b/tests/test_http_parser.py
@@ -1 +1 @@
-old test
+new test
"""


def test_filter_submission_patch_keeps_existing_source_edits_and_drops_r2e_aux_files():
    filtered, stats = filter_submission_patch(_mixed_raw_patch_with_r2e_aux_files())

    assert "diff --git a/aiohttp/http_parser.py b/aiohttp/http_parser.py" in filtered
    assert "new parser" in filtered
    assert "Makefile" not in filtered
    assert "install.sh" not in filtered
    assert "process_aiohttp_updateasyncio.py" not in filtered
    assert "r2e_tests" not in filtered
    assert "tests/test_http_parser.py" not in filtered
    assert stats["kept_files"] == ["aiohttp/http_parser.py"]
    dropped = {item["path"]: item["reason"] for item in stats["dropped_files"]}
    assert dropped["Makefile"] == "r2e_aux"
    assert dropped["install.sh"] == "new_file"
    assert dropped["process_aiohttp_updateasyncio.py"] == "new_file"
    assert dropped["r2e_tests"] == "new_file"
    assert dropped["tests/test_http_parser.py"] == "test"


def test_r2e_runtime_submit_saves_filtered_patch_and_raw_patch_for_debug(tmp_path):
    task = _r2e_task_for_env_policy_tests()

    class _FakeR2ERuntime:
        def get_patch(self):
            return _mixed_raw_patch_with_r2e_aux_files()

        def _calculate_reward(self, **kwargs):
            return 0.0, "tests failed"

    runtime = R2ERepoRuntime(R2ERuntimeConfig(patches_dir=str(tmp_path / "patches")))
    runtime.task = task
    runtime.env = SimpleNamespace(runtime=_FakeR2ERuntime())
    runtime.episode_dir = tmp_path / "episode"
    runtime.episode_dir.mkdir()

    result = runtime.submit()

    patch_path = Path(result.info["patch_path"])
    raw_patch_path = Path(result.info["raw_patch_path"])
    assert patch_path.exists()
    assert raw_patch_path.exists()
    assert "aiohttp/http_parser.py" in patch_path.read_text(encoding="utf-8")
    assert "Makefile" not in patch_path.read_text(encoding="utf-8")
    assert "Makefile" in raw_patch_path.read_text(encoding="utf-8")
    assert result.info["raw_patch_chars"] > result.info["patch_chars"]
    assert result.info["patch_filter_kept_files"] == ["aiohttp/http_parser.py"]


def test_r2e_env_blocks_submit_until_successful_edit():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])

    assert dones == [False]
    assert runtime.submit_count == 0
    assert "Submit blocked" in obs[0]
    assert infos[0]["fail_reason"] == "submit_before_successful_source_edit"

    env.step([{"tool_name": "str_replace_editor", "parameters": {"command": "str_replace", "path": "/testbed/a.py", "old_str": "old", "new_str": "new"}}])
    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])
    assert dones == [True]
    assert runtime.submit_count == 1


def test_r2e_env_blocks_test_file_edit_and_keeps_submit_locked():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)

    obs, rewards, dones, infos = env.step(
        [
            {
                "tool_name": "str_replace_editor",
                "parameters": {
                    "command": "insert",
                    "path": "/testbed/tests/test_http_parser.py",
                    "insert_line": 1,
                    "new_str": "# bad",
                },
            }
        ]
    )

    assert dones == [False]
    assert runtime.editor_count == 0
    assert rewards[0] < 0
    assert "Cannot modify test files" in obs[0]
    assert infos[0]["edit_path_kind"] == "test"
    assert infos[0]["test_edit_blocked"] is True
    assert env.successful_source_edit_counts == [0]

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])
    assert runtime.submit_count == 0
    assert dones == [False]
    assert infos[0]["fail_reason"] == "submit_before_successful_source_edit"


def test_r2e_env_repro_edit_does_not_unlock_submit():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)

    obs, rewards, dones, infos = env.step(
        [
            {
                "tool_name": "str_replace_editor",
                "parameters": {
                    "command": "create",
                    "path": "/testbed/test.py",
                    "file_text": "print('repro')",
                },
            }
        ]
    )

    assert runtime.editor_count == 1
    assert rewards[0] == 0.0
    assert infos[0]["edit_path_kind"] == "repro"
    assert infos[0]["successful_source_edit_count"] == 0

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])
    assert runtime.submit_count == 0
    assert "successful source edit" in obs[0]
    assert infos[0]["fail_reason"] == "submit_before_successful_source_edit"


def test_r2e_env_source_edit_gets_shaping_and_unlocks_submit():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)

    obs, rewards, dones, infos = env.step(
        [
            {
                "tool_name": "str_replace_editor",
                "parameters": {
                    "command": "str_replace",
                    "path": "/testbed/aiohttp/http_parser.py",
                    "old_str": "old",
                    "new_str": "new",
                },
            }
        ]
    )

    assert rewards[0] > 0
    assert env.successful_source_edit_counts == [1]
    assert infos[0]["edit_path_kind"] == "source"
    assert infos[0]["reward_breakdown"]["shaping_delta"] > 0
    assert infos[0]["successful_source_edit_count"] == 1

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])
    assert runtime.submit_count == 1
    assert dones == [True]


def test_r2e_env_blocks_submit_until_validation_after_source_edit():
    runtime = _FailingValidationPolicyRuntime()
    env = _make_policy_env(runtime, require_validation_before_submit=True)

    env.step(
        [
            {
                "tool_name": "str_replace_editor",
                "parameters": {
                    "command": "str_replace",
                    "path": "/testbed/aiohttp/http_parser.py",
                    "old_str": "old",
                    "new_str": "new",
                },
            }
        ]
    )

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])

    assert runtime.submit_count == 0
    assert dones == [False]
    assert "run a validation command after your source edit" in obs[0]
    assert infos[0]["fail_reason"] == "submit_before_validation_after_source_edit"
    assert infos[0]["validation_after_source_edit_count"] == 0

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])

    assert runtime.submit_count == 0
    assert dones == [False]
    assert "Submit still blocked" in obs[0]
    assert "<function=bash>" in obs[0]
    assert "<parameter=cmd>python -m pytest -q</parameter>" in obs[0]
    assert infos[0]["fail_reason"] == "submit_before_validation_after_source_edit"
    assert infos[0]["repeated_submit_before_validation"] is True

    obs, rewards, dones, infos = env.step(
        [{"tool_name": "bash", "parameters": {"cmd": "python -m pytest tests/test_http_parser.py::test_bug"}}]
    )

    assert runtime.bash_count == 1
    assert env.validation_after_source_edit_counts == [1]
    assert infos[0]["validation_after_source_edit_count"] == 1

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])

    assert runtime.submit_count == 1
    assert dones == [True]


def test_r2e_env_validate_tool_is_masked_until_edit_then_unlocks_submit():
    runtime = _FailingValidationPolicyRuntime()
    env = _make_policy_env(runtime, require_validation_before_submit=True)

    obs, rewards, dones, infos = env.step([{"tool_name": "validate", "parameters": {"cmd": "python -m pytest -q"}}])

    assert runtime.bash_count == 0
    assert dones == [False]
    assert infos[0]["fail_reason"] == "validate_before_successful_source_edit"
    assert infos[0]["action_mask"]["allow_validate"] is False
    assert infos[0]["action_mask"]["allow_submit"] is False

    env.step(
        [
            {
                "tool_name": "str_replace_editor",
                "parameters": {
                    "command": "str_replace",
                    "path": "/testbed/aiohttp/http_parser.py",
                    "old_str": "old",
                    "new_str": "new",
                },
            }
        ]
    )

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])

    assert runtime.submit_count == 0
    assert infos[0]["fail_reason"] == "submit_before_validation_after_source_edit"
    assert infos[0]["action_mask"]["allow_validate"] is True
    assert infos[0]["action_mask"]["allow_submit"] is False

    obs, rewards, dones, infos = env.step([{"tool_name": "validate", "parameters": {"cmd": "python -m pytest -q"}}])

    assert runtime.bash_count == 1
    assert env.validation_after_source_edit_counts == [1]
    assert infos[0]["validation_action"] is True
    assert infos[0]["validation_after_source_edit_count"] == 1
    assert infos[0]["action_mask"]["allow_submit"] is True

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])

    assert runtime.submit_count == 1
    assert dones == [True]


def test_r2e_env_noop_str_replace_does_not_count_as_source_edit_or_unlock_submit():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)

    obs, rewards, dones, infos = env.step(
        [
            {
                "tool_name": "str_replace_editor",
                "parameters": {
                    "command": "str_replace",
                    "path": "/testbed/aiohttp/http_parser.py",
                    "old_str": "return old_value",
                    "new_str": "return old_value",
                },
            }
        ]
    )

    assert runtime.editor_count == 0
    assert dones == [False]
    assert rewards[0] <= 0
    assert "No-op str_replace blocked" in obs[0]
    assert infos[0]["fail_reason"] == "noop_edit"
    assert infos[0]["tool_execution_success"] is False
    assert infos[0]["successful_source_edit_count"] == 0
    assert env.successful_source_edit_counts == [0]

    obs, rewards, dones, infos = env.step([{"tool_name": "submit", "parameters": {}}])
    assert runtime.submit_count == 0
    assert dones == [False]
    assert infos[0]["fail_reason"] == "submit_before_successful_source_edit"


def test_r2e_env_blocks_repeated_identical_failed_actions():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)
    action = {"tool_name": "bash", "parameters": {"cmd": "grep -r missing ."}}

    env.step([action])
    obs, rewards, dones, infos = env.step([action])

    assert runtime.bash_count == 1
    assert dones == [False]
    assert "Repeated failed tool call blocked" in obs[0]
    assert infos[0]["fail_reason"] == "repeated_failed_action"


def test_r2e_env_stops_after_repeated_failed_action_limit():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)
    env.max_repeated_failed_action_blocks = 3
    action = {"tool_name": "bash", "parameters": {"cmd": "grep -r missing ."}}

    env.step([action])
    env.step([action])
    env.step([action])
    obs, rewards, dones, infos = env.step([action])

    assert runtime.bash_count == 1
    assert dones == [True]
    assert "Repeated failed-action limit reached" in obs[0]
    assert infos[0]["fail_reason"] == "repeated_failed_action_limit"
    assert infos[0]["repeated_failed_action_block_count"] == 3
    assert infos[0]["max_repeated_failed_action_blocks"] == 3
    assert infos[0]["tool_execution_success"] is False
    assert infos[0]["exit_reason"] == "repeated_failed_action_limit"


def test_r2e_env_blocks_repeated_identical_view_without_source_edit():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)
    action = {
        "tool_name": "str_replace_editor",
        "parameters": {
            "command": "view",
            "path": "/testbed/aiohttp/http_parser.py",
            "view_range": [123, 127],
        },
    }

    env.step([action])
    obs, rewards, dones, infos = env.step([action])

    assert runtime.editor_count == 1
    assert dones == [False]
    assert rewards[0] < 0
    assert "already viewed this exact file/range" in obs[0]
    assert infos[0]["is_action_valid"] is True
    assert infos[0]["tool_execution_success"] is False
    assert infos[0]["fail_reason"] == "repeated_no_progress_action"


def test_r2e_env_blocks_repeated_identical_successful_bash_without_source_edit():
    runtime = _SuccessfulBashPolicyRuntime()
    env = _make_policy_env(runtime)
    action = {"tool_name": "bash", "parameters": {"cmd": "grep -RIn -- 'TransferEncodingError' /testbed/aiohttp | head -50"}}

    env.step([action])
    obs, rewards, dones, infos = env.step([action])

    assert runtime.bash_count == 1
    assert dones == [False]
    assert rewards[0] < 0
    assert "already ran this exact bash command" in obs[0]
    assert infos[0]["is_action_valid"] is True
    assert infos[0]["tool_execution_success"] is False
    assert infos[0]["fail_reason"] == "repeated_no_progress_action"
    assert infos[0]["no_progress_tool_name"] == "bash"


def test_r2e_env_stops_after_repeated_no_progress_view_limit():
    runtime = _PolicyRuntime()
    env = _make_policy_env(runtime)
    env.max_repeated_no_progress_actions = 3
    action = {
        "tool_name": "str_replace_editor",
        "parameters": {
            "command": "view",
            "path": "/testbed/aiohttp/http_parser.py",
            "view_range": [123, 127],
        },
    }

    env.step([action])
    env.step([action])
    env.step([action])
    obs, rewards, dones, infos = env.step([action])

    assert runtime.editor_count == 1
    assert dones == [True]
    assert "Repeated no-progress limit reached" in obs[0]
    assert infos[0]["fail_reason"] == "repeated_no_progress_action_limit"
    assert infos[0]["repeated_no_progress_action_count"] == 3
    assert infos[0]["tool_execution_success"] is False
    assert infos[0]["exit_reason"] == "repeated_no_progress_action_limit"
