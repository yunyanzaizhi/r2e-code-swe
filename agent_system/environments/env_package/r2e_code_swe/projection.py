import json
import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


TOOL_ALIASES = {
    "bash": "bash",
    "execute_bash": "bash",
    "str_replace_editor": "str_replace_editor",
    "file_editor": "str_replace_editor",
    "validate": "validate",
    "run_validation": "validate",
    "submit": "submit",
    "finish": "submit",
}

EDITOR_COMMANDS = {"view", "create", "str_replace", "insert", "undo_edit"}
EDITOR_COMMAND_ALIASES = {"replace": "str_replace", "view_range": "view"}


@dataclass
class ParsedR2EAction:
    action: Dict[str, Any]
    is_valid: bool
    error: str = ""


def _unwrap_markdown_fence(text: str) -> Tuple[str, str]:
    match = re.fullmatch(r"```([A-Za-z0-9_-]*)\s*\n(.*?)\n?```", text.strip(), flags=re.DOTALL)
    if not match:
        return text, ""
    return match.group(2).strip(), match.group(1).strip().lower()


def _json_object_looks_like_action(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if any(key in data for key in ("tool_name", "tool", "action", "name", "tool_call", "tool_calls")):
        return True
    function = data.get("function")
    return isinstance(function, dict) and "name" in function


def _extract_markdown_json_action(text: str) -> str | None:
    candidates: List[str] = []
    for lang, body in re.findall(r"```([A-Za-z0-9_-]*)\s*\n(.*?)\n?```", text, flags=re.DOTALL):
        if lang.strip().lower() not in {"", "json"}:
            continue
        candidate = body.strip()
        if not candidate.startswith("{"):
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if _json_object_looks_like_action(data):
            candidates.append(candidate)
    if len(candidates) > 1:
        raise ValueError("Action format error: expected exactly one JSON tool call, but found multiple.")
    return candidates[0] if candidates else None


def _extract_xml_action(text: str) -> str:
    matches = re.findall(
        r"<function\s*=\s*[^>/\s]+(?:\s*)/>|<function\s*=\s*[^>]+>.*?</function>",
        text,
        flags=re.DOTALL,
    )
    if len(matches) > 1:
        raise ValueError("Action format error: expected exactly one XML tool call, but found multiple.")
    return matches[0] if matches else text


def _clean_param_value(value: str) -> str:
    value = "" if value is None else str(value)
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _placeholderish(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"old_str", "new_str", "line_number", "replacement_text"}


def _editor_str_replace_parameter_error(name: str) -> str:
    if name == "old_str":
        reason = "requires a non-empty 'old_str'"
    else:
        reason = "requires 'new_str'"
    return (
        f"str_replace_editor str_replace {reason}. "
        "Do not write <parameter>old_str</parameter> or <parameter>new_str</parameter>. "
        "Use named XML tags with content copied from the viewed file: "
        "<parameter=old_str>exact original consecutive lines without line numbers</parameter> "
        "and <parameter=new_str>the edited replacement lines</parameter>."
    )


def _parse_malformed_xmlish_parameter(body: str) -> Tuple[str, str] | None:
    match = re.match(
        r"^(command|cmd|path|file_text|old_str|new_str|insert_line|view_range|range|start_line|end_line)\s*</([^>]+)>$",
        body.strip(),
        flags=re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).strip(), _clean_param_value(match.group(2))


def _parse_key_value_chunks(body: str) -> Dict[str, str]:
    chunks = re.split(r",\s*(?=[A-Za-z_][\w-]*\s*=)", body.strip())
    parsed: Dict[str, str] = {}
    for chunk in chunks:
        match = re.match(r"([A-Za-z_][\w-]*)\s*(?:=|>)\s*(.*)", chunk.strip(), flags=re.DOTALL)
        if match:
            parsed[match.group(1).strip()] = _clean_param_value(match.group(2))
    return parsed


def _parse_xml_action(text: str) -> Dict[str, Any]:
    action_text = _extract_xml_action(text)
    self_closing_match = re.fullmatch(r"\s*<function\s*=\s*([^>/\s]+)\s*/>\s*", action_text, flags=re.DOTALL)
    if self_closing_match:
        return {"tool_name": self_closing_match.group(1).strip(), "parameters": {}}
    fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_text)
    if not fn_match:
        raise ValueError("Action format error: expected exactly one XML tool call or JSON object.")
    tool_name = fn_match.group(1).strip()
    params = {}
    for raw_key, raw_value in re.findall(
        r"<parameter\s*=\s*([^>\n<]+)(?:>(.*?)</parameter>|</parameter>)",
        action_text,
        flags=re.DOTALL,
    ):
        key = raw_key.strip()
        value = raw_value.strip()
        if "=" in key and not value:
            key, value = [part.strip() for part in key.split("=", 1)]
        params[key] = _clean_param_value(value)

    bare_parameter_bodies = []
    for body in re.findall(r"<parameter>(.*?)</parameter>", action_text, flags=re.DOTALL):
        body = body.strip()
        if not body:
            continue
        xmlish = _parse_malformed_xmlish_parameter(body)
        if xmlish:
            params.setdefault(xmlish[0], xmlish[1])
            continue
        key_values = _parse_key_value_chunks(body)
        if key_values:
            for key, value in key_values.items():
                params.setdefault(key, value)
        else:
            bare_parameter_bodies.append(body)

    known_keys = {"command", "cmd", "path", "file_text", "old_str", "new_str", "insert_line", "view_range", "range", "start_line", "end_line", "cwd", "python_only"}
    index = 0
    while index + 1 < len(bare_parameter_bodies):
        key = bare_parameter_bodies[index].strip()
        value = bare_parameter_bodies[index + 1].strip()
        if key in known_keys and value not in known_keys:
            params.setdefault(key, _clean_param_value(value))
            index += 2
        else:
            index += 1

    for index, key in enumerate(bare_parameter_bodies):
        key = key.strip()
        if key == "old_str" and "old_str" not in params and index + 1 < len(bare_parameter_bodies):
            value = bare_parameter_bodies[index + 1].strip()
            if value not in known_keys:
                params["old_str"] = _clean_param_value(value)
        if key == "old_str" and "new_str" not in params:
            if index + 2 < len(bare_parameter_bodies) and bare_parameter_bodies[index + 2].strip() not in known_keys:
                params["new_str"] = _clean_param_value(bare_parameter_bodies[index + 2])
            elif (
                index + 3 < len(bare_parameter_bodies)
                and bare_parameter_bodies[index + 2].strip() == "new_str"
                and bare_parameter_bodies[index + 3].strip() not in known_keys
            ):
                params["new_str"] = _clean_param_value(bare_parameter_bodies[index + 3])

    for key in known_keys:
        for value in re.findall(fr"<{key}>(.*?)</{key}>", action_text, flags=re.DOTALL):
            params.setdefault(key, _clean_param_value(value))

    if not params and TOOL_ALIASES.get(tool_name) in {"bash", "validate"}:
        body_match = re.search(r"<function\s*=\s*[^>]+>(.*?)</function>", action_text, flags=re.DOTALL)
        body = body_match.group(1).strip() if body_match else ""
        if body:
            params["cmd"] = body
    return {"tool_name": tool_name, "parameters": params}


def _decode_json_arguments(value: Any, tool_name: Any = None) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
        if TOOL_ALIASES.get(str(tool_name or "").strip()) in {"bash", "validate"}:
            return {"cmd": stripped}
    raise ValueError("JSON action parameters must be an object.")


def _unwrap_json_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    if "tool_calls" in data:
        calls = data["tool_calls"]
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("JSON action must contain exactly one tool call.")
        data = calls[0]
        if not isinstance(data, dict):
            raise ValueError("JSON tool_calls entry must be an object.")

    if "tool_call" in data:
        data = data["tool_call"]
        if not isinstance(data, dict):
            raise ValueError("JSON tool_call must be an object.")

    function = data.get("function")
    if isinstance(function, dict):
        return {
            "tool_name": function.get("name") or data.get("name") or data.get("tool") or data.get("action"),
            "parameters": function.get("arguments") or function.get("parameters") or data.get("arguments") or data.get("parameters"),
        }
    return data


def _parse_json_action(text: str) -> Dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON action must be an object.")
    data = _unwrap_json_tool_call(data)

    tool_name = data.get("tool_name") or data.get("function") or data.get("name") or data.get("tool") or data.get("action")
    params = data.get("parameters") if "parameters" in data else data.get("arguments")
    if params is None:
        params = {
            key: value
            for key, value in data.items()
            if key not in {"tool_name", "function", "name", "tool", "action", "tool_call", "tool_calls", "parameters", "arguments"}
        }
    else:
        params = _decode_json_arguments(params, tool_name)
    if not isinstance(params, dict):
        raise ValueError("JSON action parameters must be an object.")
    return {"tool_name": tool_name, "parameters": params}


def _parse_editor_cli_shorthand(cmd: str) -> Dict[str, Any] | None:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    if len(parts) < 3 or parts[0] != "str_replace_editor":
        return None

    command = EDITOR_COMMAND_ALIASES.get(parts[1], parts[1])
    if command not in EDITOR_COMMANDS:
        return None

    params: Dict[str, Any] = {"command": command, "path": parts[2]}
    rest = parts[3:]
    if command == "create" and rest:
        params["file_text"] = " ".join(rest)
    elif command == "str_replace" and len(rest) >= 2:
        params["old_str"] = rest[0]
        params["new_str"] = " ".join(rest[1:])
    elif command == "insert" and len(rest) >= 2:
        params["insert_line"] = rest[0]
        params["new_str"] = " ".join(rest[1:])
    elif command == "view" and rest:
        params["view_range"] = rest[0]
    return {"tool_name": "str_replace_editor", "parameters": params}


def _single_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def _normalize_workspace_path(path: str) -> str:
    path = str(path or "").strip()
    if not path:
        return ""
    if path == "/repo":
        path = "/testbed"
    elif path.startswith("/repo/"):
        path = "/testbed/" + path[len("/repo/"):]
    elif path.startswith("testbed/"):
        path = "/" + path
    elif not path.startswith("/"):
        path = "/testbed/" + path.lstrip("./")
    return re.sub(r"/{2,}", "/", path)


def _split_line_suffix(path: str) -> Tuple[str, List[int] | None]:
    match = re.match(r"^(.+?):(\d+)(?:[-:](\d+))?$", path)
    if not match:
        return path, None
    base = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3)) if match.group(3) else -1
    if end != -1 and end < start:
        end = -1
    return base, [start, end]


def _normalize_view_range(value: Any) -> Tuple[List[int] | None, str | None]:
    if value in (None, ""):
        return None, None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [int(value[0]), int(value[1])], None
        except (TypeError, ValueError):
            return None, " ".join(str(part) for part in value if part is not None).strip() or None
    text = str(value).strip()
    numbers = re.findall(r"-1|\d+", text)
    if len(numbers) >= 2:
        return [int(numbers[0]), int(numbers[1])], None
    if len(numbers) == 1 and re.fullmatch(r"\s*[\[(]?\s*\d+\s*[\])]?\s*", text):
        return [int(numbers[0]), -1], None
    return None, text or None


def _safe_grep_head_command(pattern_q: str, path: str = ".", max_lines: int = 50) -> str:
    path_q = shlex.quote(path)
    return (
        f"{{ grep -RIn -- {pattern_q} {path_q} | head -{max_lines}; "
        'status=${PIPESTATUS[0]}; test "$status" -eq 0 -o "$status" -eq 141; }'
    )


def _grep_search_action(search_term: str, path: str) -> Dict[str, Any]:
    cmd = _safe_grep_head_command(_single_quote(search_term), path)
    return {"tool_name": "bash", "parameters": {"cmd": cmd}}


_CD_FILE_PATH_RE = re.compile(
    r"(?P<prefix>(?:^|[;&|]{2})\s*)cd\s+(?P<quote>['\"]?)(?P<path>/testbed/[^\s;&|'\"]+\.[A-Za-z0-9_]+)(?P=quote)\s*&&"
)
_GREP_WITHOUT_PATH_RE = re.compile(
    r"(?P<prefix>(?:^|[;&|]{2})\s*)grep\s+(?P<flags>-[A-Za-z]*n[A-Za-z]*)\s+"
    r"(?P<pattern>'[^']*'|\"[^\"]*\"|[^\s;&|]+)\s*(?=(?:&&|\|\||;|$))"
)
_GREP_HEAD_PIPE_RE = re.compile(
    r"(?P<grep>\bgrep\b(?:(?:'[^']*'|\"[^\"]*\"|\\.|[^|;&\n])+?))"
    r"\s*\|\s*"
    r"(?P<head>head(?:\s+(?:-[0-9]+|-n\s+[0-9]+))?)"
)


def _repair_cd_file_path(cmd: str) -> str:
    def replace(match: re.Match) -> str:
        path = match.group("path").rstrip("/")
        parent = path.rsplit("/", 1)[0] or "/testbed"
        return f"{match.group('prefix')}cd {shlex.quote(parent)} &&"

    return _CD_FILE_PATH_RE.sub(replace, cmd)


def _repair_grep_without_path(cmd: str) -> str:
    def replace(match: re.Match) -> str:
        return f"{match.group('prefix')}{_safe_grep_head_command(match.group('pattern'))} "

    return _GREP_WITHOUT_PATH_RE.sub(replace, cmd).strip()


def _repair_grep_head_pipelines(cmd: str) -> str:
    def replace(match: re.Match) -> str:
        # Already-safe examples intentionally inspect PIPESTATUS after head.
        # Leave those untouched so repeated parsing is idempotent.
        tail = cmd[match.end() : match.end() + 120]
        if "PIPESTATUS" in tail:
            return match.group(0)
        grep_cmd = match.group("grep").strip()
        head_cmd = match.group("head").strip()
        return f'{{ {grep_cmd} | {head_cmd}; status=${{PIPESTATUS[0]}}; test "$status" -eq 0 -o "$status" -eq 141; }}'

    return _GREP_HEAD_PIPE_RE.sub(replace, cmd).strip()


def _repair_bash_command(cmd: str) -> str:
    repaired = _repair_cd_file_path(cmd)
    repaired = _repair_grep_without_path(repaired)
    repaired = _repair_grep_head_pipelines(repaired)
    return repaired.strip()


def _unescape_editor_text_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if "\n" in value:
        return value
    if not any(token in value for token in ("\\n", "\\t", "\\r")):
        return value
    return value.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")


def _normalize_action(raw_action: Dict[str, Any]) -> Dict[str, Any]:
    raw_tool = str(raw_action.get("tool_name") or "").strip()
    tool_name = TOOL_ALIASES.get(raw_tool, "")
    if not tool_name:
        raise ValueError("Unknown tool. Allowed tools: bash, str_replace_editor, validate, submit.")

    params = dict(raw_action.get("parameters") or {})
    if tool_name in {"bash", "validate"}:
        cmd = params.get("cmd", params.get("command"))
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("bash requires a non-empty 'cmd' parameter.")
        if tool_name == "bash":
            editor_action = _parse_editor_cli_shorthand(cmd.strip())
            if editor_action is not None:
                return _normalize_action(editor_action)
        normalized_params = {"cmd": _repair_bash_command(cmd.strip())}
        if params.get("cwd") not in (None, ""):
            normalized_params["cwd"] = params.get("cwd")
        return {"tool_name": tool_name, "parameters": normalized_params}

    if tool_name == "str_replace_editor":
        command = EDITOR_COMMAND_ALIASES.get(str(params.get("command") or "").strip(), str(params.get("command") or "").strip())
        path = _normalize_workspace_path(str(params.get("path") or "").strip())
        path, suffix_range = _split_line_suffix(path)
        if command not in EDITOR_COMMANDS:
            raise ValueError("str_replace_editor command must be one of: view, create, str_replace, insert, undo_edit.")
        if not path:
            raise ValueError("str_replace_editor requires a 'path' parameter.")
        params["command"] = command
        params["path"] = path
        if command == "view":
            if suffix_range is not None and params.get("view_range") in (None, ""):
                params["view_range"] = suffix_range
            if params.get("view_range") in (None, "") and params.get("range") not in (None, ""):
                params["view_range"] = params.get("range")
            if params.get("view_range") in (None, "") and params.get("start_line") not in (None, ""):
                params["view_range"] = [params.get("start_line"), params.get("end_line", -1)]
            view_range, search_term = _normalize_view_range(params.get("view_range"))
            if view_range is not None:
                params["view_range"] = view_range
            elif search_term:
                return _grep_search_action(search_term, path)
            else:
                params.pop("view_range", None)
            for extra_key in ("range", "start_line", "end_line"):
                params.pop(extra_key, None)
        elif suffix_range is not None and command == "insert" and params.get("insert_line") is None:
            params["insert_line"] = suffix_range[0]
        if command == "create" and params.get("file_text") is None:
            raise ValueError("str_replace_editor create requires 'file_text'.")
        for text_key in ("old_str", "new_str", "file_text"):
            if text_key in params:
                params[text_key] = _unescape_editor_text_literal(params[text_key])
        if command == "str_replace":
            old_str = params.get("old_str")
            new_str = params.get("new_str")
            if old_str is None or not str(old_str).strip() or _placeholderish(old_str):
                raise ValueError(_editor_str_replace_parameter_error("old_str"))
            if new_str is None or _placeholderish(new_str):
                raise ValueError(_editor_str_replace_parameter_error("new_str"))
        if command == "insert" and (params.get("insert_line") is None or params.get("new_str") is None):
            raise ValueError("str_replace_editor insert requires 'insert_line' and 'new_str'.")
        return {"tool_name": "str_replace_editor", "parameters": params}

    return {"tool_name": "submit", "parameters": {}}


def parse_r2e_action(text: str) -> ParsedR2EAction:
    stripped = (text or "").strip()
    if not stripped:
        return ParsedR2EAction({"tool_name": "", "parameters": {}, "error": "Action format error: empty model output."}, False, "Action format error: empty model output.")
    try:
        stripped, fence_lang = _unwrap_markdown_fence(stripped)
        if fence_lang in {"bash", "sh", "shell"} and not stripped.startswith(("{", "<function")):
            parsed = {"tool_name": "bash", "parameters": {"cmd": stripped}}
        else:
            json_candidate = None if stripped.startswith("{") else _extract_markdown_json_action(stripped)
            if stripped.startswith("{") or json_candidate is not None:
                parsed = _parse_json_action(json_candidate or stripped)
            else:
                parsed = _parse_xml_action(stripped)
        normalized = _normalize_action(parsed)
        return ParsedR2EAction(normalized, True, "")
    except Exception as exc:
        error = str(exc)
        return ParsedR2EAction({"tool_name": "", "parameters": {}, "error": error}, False, error)


def r2e_code_swe_projection(actions: List[str]) -> Tuple[List[Dict[str, Any]], List[int]]:
    parsed_actions: List[Dict[str, Any]] = []
    valids: List[int] = []
    for action in actions:
        parsed = parse_r2e_action(action)
        parsed_actions.append(parsed.action)
        valids.append(1 if parsed.is_valid else 0)
    return parsed_actions, valids
