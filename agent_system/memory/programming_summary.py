"""Structured summary memory for programming/debugging tasks."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from agent_system.memory.base import BaseMemory


def _shorten(text: str, max_chars: int = 1200) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return "...[truncated]...\n" + text[-max_chars:]


def _extract_file_name(obs: str) -> str:
    m = re.search(r"File to edit:\s*([^\n]+)", obs or "")
    if m:
        return m.group(1).strip()
    return "unknown"


def _classify_failure(text: str) -> str:
    text = text or ""

    if "All tests passed" in text:
        return "passed"
    if "SyntaxError" in text:
        return "syntax_error"
    if "ImportError" in text or "ModuleNotFoundError" in text:
        return "import_error"
    if "TimeoutExpired" in text or "timeout" in text.lower():
        return "timeout"
    if "NameError" in text:
        return "name_error"
    if "IndexError" in text:
        return "index_error"
    if "TypeError" in text:
        return "type_error"
    if "AssertionError" in text or "FAILED" in text:
        return "assertion_failure"
    if "Invalid action" in text:
        return "invalid_action"
    if "Tests failed" in text:
        return "test_failure"
    return "unknown"


def _parse_test_counts(text: str) -> Tuple[int | None, int | None, int | None]:
    """
    Parse pytest summary such as:
      7 failed, 1 passed in 0.06s
      1 failed, 12 passed in 0.08s
      8 passed in 0.05s
    Returns: passed, failed, total
    """
    text = text or ""

    passed = 0
    failed = 0

    m_pass = re.search(r"(\d+)\s+passed", text)
    if m_pass:
        passed = int(m_pass.group(1))

    m_fail = re.search(r"(\d+)\s+failed", text)
    if m_fail:
        failed = int(m_fail.group(1))

    if m_pass or m_fail:
        return passed, failed, passed + failed

    if "All tests passed" in text:
        return 1, 0, 1

    return None, None, None


def _extract_key_error_lines(text: str, max_lines: int = 8) -> str:
    """
    Extract useful error lines from pytest output.
    """
    if not text:
        return ""

    useful_patterns = [
        "AssertionError",
        "SyntaxError",
        "ImportError",
        "ModuleNotFoundError",
        "NameError",
        "IndexError",
        "TypeError",
        "TimeoutExpired",
        "FAILED ",
        "E       ",
        "expected",
        "At index",
        "diff:",
    ]

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(p in stripped for p in useful_patterns):
            lines.append(stripped)

    if not lines:
        return _shorten(text, 800)

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for line in lines:
        if line not in seen:
            deduped.append(line)
            seen.add(line)

    return "\n".join(deduped[-max_lines:])


def _code_signature(code: str) -> str:
    """
    A lightweight signature to detect repeated attempts.
    """
    if not code:
        return "empty"
    code = re.sub(r"\s+", " ", code.strip())
    return code[:300]


def _extract_defined_functions(code: str) -> List[str]:
    if not code:
        return []
    return re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, flags=re.MULTILINE)


def _infer_attempt_note(action: str, result_obs: str) -> str:
    funcs = _extract_defined_functions(action)
    failure_type = _classify_failure(result_obs)
    passed, failed, total = _parse_test_counts(result_obs)

    bits = []
    if funcs:
        bits.append("defined functions: " + ", ".join(funcs[:5]))
    else:
        bits.append("no function definition detected")

    bits.append(f"result: {failure_type}")

    if total:
        bits.append(f"tests: {passed}/{total} passed, {failed} failed")

    return "; ".join(bits)


class ProgrammingSummaryMemory(BaseMemory):
    """
    Compact structured memory for programming/debugging tasks.

    It stores every interaction internally, but fetch() returns a concise
    debugging summary instead of the raw full transcript.
    """

    def __init__(self):
        self._data = None
        self._summaries = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()

        self._data = [[] for _ in range(batch_size)]
        self._summaries = [
            {
                "total_steps": 0,
                "file": "unknown",
                "best_passed": 0,
                "best_total": None,
                "best_step": None,
                "last_failure_type": "unknown",
                "failure_counts": {},
                "recent_attempts": [],
                "repeated_code_count": 0,
                "seen_code_signatures": {},
                "last_error_excerpt": "",
                "last_valid_action": "",
                "last_result": "",
            }
            for _ in range(batch_size)
        ]

        self.keys = None
        self.batch_size = batch_size

    def store(self, record: Dict[str, List[Any]], rewards: List[float] | None = None):
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            step_record = {key: record[key][env_idx] for key in self.keys}
            self._data[env_idx].append(step_record)

            summary = self._summaries[env_idx]
            summary["total_steps"] += 1
            step = summary["total_steps"]

            prev_obs = step_record.get("text_obs", "")
            action = step_record.get("action", "")
            result_obs = step_record.get("result_text_obs", "")

            if summary["file"] == "unknown":
                summary["file"] = _extract_file_name(prev_obs)

            failure_type = _classify_failure(result_obs)
            summary["last_failure_type"] = failure_type
            summary["failure_counts"][failure_type] = summary["failure_counts"].get(failure_type, 0) + 1

            passed, failed, total = _parse_test_counts(result_obs)
            if passed is not None:
                if summary["best_total"] is None or passed > summary["best_passed"]:
                    summary["best_passed"] = passed
                    summary["best_total"] = total
                    summary["best_step"] = step

            code_sig = _code_signature(action)
            summary["seen_code_signatures"][code_sig] = summary["seen_code_signatures"].get(code_sig, 0) + 1
            summary["repeated_code_count"] = summary["seen_code_signatures"][code_sig]

            if action:
                summary["last_valid_action"] = _shorten(action, 1200)

            summary["last_result"] = _shorten(result_obs, 1200)
            summary["last_error_excerpt"] = _extract_key_error_lines(result_obs)

            attempt_note = _infer_attempt_note(action, result_obs)
            summary["recent_attempts"].append(
                {
                    "step": step,
                    "failure_type": failure_type,
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                    "repeated_code_count": summary["repeated_code_count"],
                    "note": attempt_note,
                    "error_excerpt": _extract_key_error_lines(result_obs, max_lines=4),
                }
            )

            if len(summary["recent_attempts"]) > 6:
                summary["recent_attempts"] = summary["recent_attempts"][-6:]

    def fetch(
        self,
        history_length: int = 0,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        # history_length is ignored intentionally: this memory returns a compact summary.
        del history_length, obs_key, action_key

        memory_contexts: List[str] = []
        valid_lengths: List[int] = []

        for env_idx in range(self.batch_size):
            summary = self._summaries[env_idx]
            total_steps = summary["total_steps"]

            if total_steps == 0:
                memory_contexts.append("")
                valid_lengths.append(0)
                continue

            best_progress = "unknown"
            if summary["best_total"]:
                best_progress = (
                    f"{summary['best_passed']}/{summary['best_total']} tests passed"
                    f" at step {summary['best_step']}"
                )

            failure_counts = ", ".join(
                f"{k}: {v}" for k, v in sorted(summary["failure_counts"].items())
            ) or "none"

            lines = [
                "[Debugging Summary]",
                f"File: {summary['file']}",
                f"Total previous attempts: {total_steps}",
                f"Best test progress: {best_progress}",
                f"Latest failure type: {summary['last_failure_type']}",
                f"Failure type counts: {failure_counts}",
                "",
                "[Recent Attempts]",
            ]

            for attempt in summary["recent_attempts"]:
                test_part = ""
                if attempt["total"]:
                    test_part = f" | tests {attempt['passed']}/{attempt['total']} passed"
                repeat_part = ""
                if attempt["repeated_code_count"] > 1:
                    repeat_part = f" | repeated same/similar code x{attempt['repeated_code_count']}"

                lines.append(
                    f"- Step {attempt['step']}: {attempt['failure_type']}{test_part}{repeat_part}; {attempt['note']}"
                )
                if attempt["error_excerpt"]:
                    lines.append("  Key error:")
                    for err_line in attempt["error_excerpt"].splitlines()[:4]:
                        lines.append(f"    {err_line}")

            lines.extend(
                [
                    "",
                    "[Current Debugging Guidance]",
                    self._make_guidance(summary),
                    "",
                    "[Important Reminder]",
                    "Output only the complete fixed source code for the buggy file. Do not output tests, examples, Markdown fences, or explanations inside the <code> block.",
                ]
            )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(1)

        return memory_contexts, valid_lengths

    def _make_guidance(self, summary: Dict[str, Any]) -> str:
        failure_type = summary["last_failure_type"]

        if failure_type == "passed":
            return "The latest attempt passed. Keep the solution minimal and avoid unnecessary changes."

        if failure_type == "syntax_error":
            return "Fix Python syntax first. Remove Markdown fences, stray text, incomplete indentation, and non-code content."

        if failure_type == "import_error":
            return "Preserve the expected function name and module-level definitions. Do not rename the target function."

        if failure_type == "name_error":
            return "A referenced name is missing. Add the required import or define the missing variable/function while preserving the original API."

        if failure_type == "timeout":
            return "The code likely has an infinite loop or excessive recursion. Add correct termination conditions and reduce unnecessary search."

        if failure_type == "index_error":
            return "Check loop/queue/list boundary conditions. Avoid popping from empty containers and handle no-solution cases."

        if failure_type == "type_error":
            return "Check argument types, return types, and function signatures expected by the tests."

        if failure_type == "assertion_failure":
            if summary["best_total"]:
                return "Tests execute but outputs are wrong. Compare expected vs actual values and fix algorithmic logic without changing tests."
            return "Tests execute but assertions fail. Focus on algorithmic correctness and preserve the expected function signature."

        if failure_type == "invalid_action":
            return "The previous response did not contain valid source code. Return a complete Python file inside <code>...</code>."

        return "Use the latest pytest feedback to make a minimal algorithmic fix. Preserve function names and do not modify tests."
