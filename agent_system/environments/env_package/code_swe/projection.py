import json
import re
from typing import Any, Dict, List, Tuple


TOOL_ALIASES = {
    "bash": "bash",
    "execute_bash": "bash",
    "str_replace_editor": "str_replace_editor",
    "file_editor": "str_replace_editor",
    "submit": "submit",
    "finish": "submit",
}


def _parse_xml_action(text: str) -> Dict[str, Any]:
    fn_match = re.search(r"<function\s*=\s*([^>]+)>", text)
    if not fn_match:
        raise ValueError("Action format error: expected <function=tool_name>...</function>.")
    function_name = fn_match.group(1).strip()
    params = {}
    for key, value in re.findall(r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>", text, flags=re.DOTALL):
        params[key.strip()] = value.strip()
    return {"tool_name": function_name, "parameters": params}


def _parse_json_action(text: str) -> Dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON action must be an object.")
    tool_name = data.get("tool_name") or data.get("function") or data.get("name")
    params = data.get("parameters") or data.get("arguments") or {}
    if not isinstance(params, dict):
        raise ValueError("JSON action parameters must be an object.")
    return {"tool_name": tool_name, "parameters": params}


def _normalize_action(raw_action: Dict[str, Any]) -> Dict[str, Any]:
    raw_tool = str(raw_action.get("tool_name") or "").strip()
    tool_name = TOOL_ALIASES.get(raw_tool, "")
    if not tool_name:
        raise ValueError(f"Unknown tool '{raw_tool}'. Allowed tools: bash, str_replace_editor, submit.")

    params = dict(raw_action.get("parameters") or {})
    if tool_name == "bash":
        cmd = params.get("cmd", params.get("command"))
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("bash requires a non-empty 'cmd' parameter.")
        params["cmd"] = cmd.strip()
    elif tool_name == "str_replace_editor":
        command = params.get("command")
        path = params.get("path")
        if not command:
            raise ValueError("str_replace_editor requires a 'command' parameter.")
        if not path:
            raise ValueError("str_replace_editor requires a 'path' parameter.")
        params["command"] = str(command).strip()
        params["path"] = str(path).strip()
    elif tool_name == "submit":
        params = params or {}

    return {"tool_name": tool_name, "parameters": params}


def parse_code_swe_action(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Action format error: empty model output.")
    if stripped.startswith("{"):
        parsed = _parse_json_action(stripped)
    else:
        parsed = _parse_xml_action(stripped)
    return _normalize_action(parsed)


def code_swe_projection(actions: List[str]) -> Tuple[List[Dict[str, Any]], List[int]]:
    parsed_actions: List[Dict[str, Any]] = []
    valids: List[int] = []

    for action in actions:
        try:
            parsed_actions.append(parse_code_swe_action(action))
            valids.append(1)
        except Exception as exc:
            parsed_actions.append({"tool_name": "", "parameters": {}, "error": str(exc)})
            valids.append(0)

    return parsed_actions, valids
