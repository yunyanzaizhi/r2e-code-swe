"""Structured Sokoban task summary memory."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from agent_system.memory.base import BaseMemory

GRID_TOKENS = {"#", "_", "O", "X", "P", "√", "S"}
OPPOSITE_ACTIONS = {
    "Up": "Down",
    "Down": "Up",
    "Left": "Right",
    "Right": "Left",
}


def _sorted_positions(positions: set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted(positions, key=lambda item: (item[0], item[1]))


def _format_pos(position: Tuple[int, int] | None) -> str:
    if position is None:
        return "unknown"
    return f"({position[0]},{position[1]})"


def _extract_grid_rows(text_obs: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in text_obs.splitlines():
        tokens = re.findall(r"[#_OXP√S]", line)
        if len(tokens) >= 3:
            rows.append(tokens)

    if not rows:
        return []

    width = max(len(row) for row in rows)
    return [row + ["#"] * (width - len(row)) for row in rows]


def parse_sokoban_grid(text_obs: str) -> Dict[str, Any]:
    rows = _extract_grid_rows(text_obs)
    if not rows:
        return {
            "grid": [],
            "height": 0,
            "width": 0,
            "player": None,
            "boxes": [],
            "boxes_on_target": [],
            "targets": [],
            "remaining_boxes": 0,
            "total_boxes": 0,
            "walls": set(),
        }

    player = None
    boxes: set[Tuple[int, int]] = set()
    boxes_on_target: set[Tuple[int, int]] = set()
    targets: set[Tuple[int, int]] = set()
    walls: set[Tuple[int, int]] = set()

    for row_idx, row in enumerate(rows, start=1):
        for col_idx, token in enumerate(row, start=1):
            position = (row_idx, col_idx)
            if token == "#":
                walls.add(position)
            if token in {"P", "S"}:
                player = position
            if token in {"X", "√"}:
                boxes.add(position)
            if token == "√":
                boxes_on_target.add(position)
            if token in {"O", "S", "√"}:
                targets.add(position)

    return {
        "grid": rows,
        "height": len(rows),
        "width": len(rows[0]) if rows else 0,
        "player": player,
        "boxes": _sorted_positions(boxes),
        "boxes_on_target": _sorted_positions(boxes_on_target),
        "targets": _sorted_positions(targets),
        "remaining_boxes": len(boxes) - len(boxes_on_target),
        "total_boxes": len(boxes),
        "walls": walls,
    }


def _state_signature(state: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        state.get("player"),
        tuple(state.get("boxes", [])),
    )


def _is_wall(state: Dict[str, Any], row: int, col: int) -> bool:
    if row < 1 or col < 1 or row > state["height"] or col > state["width"]:
        return True
    return (row, col) in state["walls"]


def _detect_deadlock_risks(state: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    target_set = set(state["targets"])

    for box in state["boxes"]:
        if box in target_set:
            continue

        row, col = box
        up = _is_wall(state, row - 1, col)
        down = _is_wall(state, row + 1, col)
        left = _is_wall(state, row, col - 1)
        right = _is_wall(state, row, col + 1)

        if (up and left) or (up and right) or (down and left) or (down and right):
            risks.append(f"Box {_format_pos(box)} is in a non-target corner (deadlock risk).")
            continue

        if row in {2, state["height"] - 1} and not any(target[0] == row for target in target_set):
            risks.append(f"Box {_format_pos(box)} is pinned to a border wall row without a target in that row.")
        elif col in {2, state["width"] - 1} and not any(target[1] == col for target in target_set):
            risks.append(f"Box {_format_pos(box)} is pinned to a border wall column without a target in that column.")

    deduped: List[str] = []
    for risk in risks:
        if risk not in deduped:
            deduped.append(risk)
    return deduped


def _format_box_snapshot(state: Dict[str, Any]) -> str:
    on_target = set(state["boxes_on_target"])
    box_parts = []
    for index, position in enumerate(state["boxes"], start=1):
        status = "on-target" if position in on_target else "free"
        box_parts.append(f"B{index}{_format_pos(position)}[{status}]")
    return ", ".join(box_parts) if box_parts else "none"


def _format_target_snapshot(state: Dict[str, Any]) -> str:
    open_targets = [target for target in state["targets"] if target not in set(state["boxes_on_target"])]
    if not open_targets:
        return "none"
    return ", ".join(f"T{index}{_format_pos(position)}" for index, position in enumerate(open_targets, start=1))


def _analyze_transition(prev_state: Dict[str, Any], curr_state: Dict[str, Any], action: str, step: int) -> Dict[str, Any]:
    prev_boxes = set(prev_state["boxes"])
    curr_boxes = set(curr_state["boxes"])
    prev_targets = set(prev_state["boxes_on_target"])
    curr_targets = set(curr_state["boxes_on_target"])

    moved_from = _sorted_positions(prev_boxes - curr_boxes)
    moved_to = _sorted_positions(curr_boxes - prev_boxes)
    onto_target = _sorted_positions(curr_targets - prev_targets)
    off_target = _sorted_positions(prev_targets - curr_targets)
    no_state_change = _state_signature(prev_state) == _state_signature(curr_state)

    if no_state_change:
        description = "no state change; move was blocked, invalid, or a no-op"
        informative = False
    elif len(moved_from) == 1 and len(moved_to) == 1:
        description = f"box {_format_pos(moved_from[0])} -> {_format_pos(moved_to[0])}"
        if onto_target:
            description += f"; onto target {_format_pos(onto_target[0])}"
        if off_target:
            description += f"; off target {_format_pos(off_target[0])}"
        informative = True
    elif onto_target or off_target:
        status_bits = []
        if onto_target:
            status_bits.append("onto target " + ", ".join(_format_pos(position) for position in onto_target))
        if off_target:
            status_bits.append("off target " + ", ".join(_format_pos(position) for position in off_target))
        description = "; ".join(status_bits)
        informative = True
    else:
        description = f"player repositioned to {_format_pos(curr_state['player'])}"
        informative = False

    return {
        "step": step,
        "action": action,
        "description": description,
        "no_state_change": no_state_change,
        "informative": informative,
        "box_moved": bool(moved_from or moved_to),
        "onto_target": onto_target,
        "off_target": off_target,
    }


def _has_ping_pong(actions: List[str]) -> bool:
    if len(actions) < 4:
        return False
    recent = actions[-4:]
    return (
        recent[0] == recent[2]
        and recent[1] == recent[3]
        and OPPOSITE_ACTIONS.get(recent[0]) == recent[1]
    )


def _recommended_focus(summary: Dict[str, Any], state: Dict[str, Any], risks: List[str]) -> str:
    if not state["boxes"]:
        return "Use the current observation; no parsed box state is available from history yet."

    if state["remaining_boxes"] == 0:
        return "All boxes are already on targets. Avoid moves that could knock a box off target."

    if any("deadlock" in risk.lower() or "pinned" in risk.lower() for risk in risks):
        return "Do not push the risky box deeper into the wall/corner. Reposition the player to approach it from a safer side."

    latest = summary["transition_log"][-1] if summary["transition_log"] else None
    if latest and latest["no_state_change"]:
        return f"Break the stalled pattern: stop repeating {latest['action']}, move to a different side of the box, then attempt a new push."

    open_targets = [target for target in state["targets"] if target not in set(state["boxes_on_target"])]
    off_target_boxes = [box for box in state["boxes"] if box not in set(state["boxes_on_target"])]
    if off_target_boxes and open_targets:
        return f"Reposition behind box {_format_pos(off_target_boxes[0])} and plan a push path toward target {_format_pos(open_targets[0])} without sending it into a wall corner."

    return "Use the current observation to line up the next push while keeping escape squares open."


class StructuredSummaryMemory(BaseMemory):
    """Compact, planning-oriented task summary for Sokoban."""

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
                "total_boxes": None,
                "recent_actions": [],
                "repeated_action_streak": 0,
                "invalid_or_noop_moves": 0,
                "transition_log": [],
                "state_visit_counts": {},
                "current_state": None,
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

            action = step_record.get("action", "Unknown")
            summary["recent_actions"].append(action)
            if len(summary["recent_actions"]) > 6:
                summary["recent_actions"].pop(0)

            if len(summary["recent_actions"]) >= 2 and summary["recent_actions"][-1] == summary["recent_actions"][-2]:
                summary["repeated_action_streak"] += 1
            else:
                summary["repeated_action_streak"] = 0

            prev_obs = step_record.get("text_obs", "")
            curr_obs = step_record.get("result_text_obs", prev_obs)
            prev_state = parse_sokoban_grid(prev_obs)
            curr_state = parse_sokoban_grid(curr_obs)
            summary["current_state"] = curr_state
            summary["total_boxes"] = curr_state["total_boxes"] or prev_state["total_boxes"] or summary["total_boxes"] or 0

            state_key = _state_signature(curr_state)
            if curr_state["boxes"]:
                summary["state_visit_counts"][state_key] = summary["state_visit_counts"].get(state_key, 0) + 1

            transition = _analyze_transition(prev_state, curr_state, action, summary["total_steps"])
            if transition["no_state_change"]:
                summary["invalid_or_noop_moves"] += 1

            summary["transition_log"].append(transition)
            if len(summary["transition_log"]) > 10:
                summary["transition_log"] = summary["transition_log"][-10:]

    def fetch(
        self,
        history_length: int = 0,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        del history_length, obs_key, action_key
        memory_contexts: List[str] = []
        valid_lengths: List[int] = []

        for env_idx in range(self.batch_size):
            summary = self._summaries[env_idx]
            total_steps = summary["total_steps"]
            state = summary["current_state"]

            if total_steps == 0 or state is None:
                memory_contexts.append("")
                valid_lengths.append(0)
                continue

            current_state_visits = summary["state_visit_counts"].get(_state_signature(state), 0)
            risks = _detect_deadlock_risks(state)
            if current_state_visits > 1:
                risks.append(f"Current state has repeated {current_state_visits} times; possible loop.")
            if summary["repeated_action_streak"] >= 1:
                last_action = summary["recent_actions"][-1]
                risks.append(f"Repeated action pattern: {last_action} x{summary['repeated_action_streak'] + 1}.")
            if _has_ping_pong(summary["recent_actions"]):
                risks.append("Recent actions show a ping-pong reversal pattern.")

            key_changes = [entry for entry in summary["transition_log"] if entry["informative"]][-3:]
            if not key_changes and summary["transition_log"]:
                key_changes = summary["transition_log"][-1:]

            trace_entries = [
                entry
                for entry in summary["transition_log"]
                if entry["box_moved"] or entry["onto_target"] or entry["off_target"]
            ][-3:]
            if not trace_entries and summary["transition_log"]:
                trace_entries = summary["transition_log"][-1:]
            recommended_focus = _recommended_focus(summary, state, risks)

            lines = [
                "[Goal]",
                "Push every box onto a target. Use this compact summary for planning; it is not a full action transcript.",
                "",
                "[Progress]",
                f"Step {total_steps} | Boxes on targets: {len(state['boxes_on_target'])}/{summary['total_boxes'] or state['total_boxes']} | Remaining boxes: {state['remaining_boxes']} | Invalid / no-op moves: {summary['invalid_or_noop_moves']}",
                "",
                "[State Snapshot]",
                f"Player: {_format_pos(state['player'])}",
                f"Boxes: {_format_box_snapshot(state)}",
                f"Open targets: {_format_target_snapshot(state)}",
                "",
                "[Key State Changes]",
            ]

            if key_changes:
                lines.extend(
                    f"- Step {entry['step']} / {entry['action']}: {entry['description']}" for entry in key_changes
                )
            else:
                lines.append("- No meaningful state changes recorded yet.")

            lines.extend(["", "[Risks]"])
            if risks:
                lines.extend(f"- {risk}" for risk in risks)
            else:
                lines.append("- No obvious loop or deadlock signal from prior states.")

            lines.extend(["", "[Recent Useful Trace]"])
            if trace_entries:
                lines.extend(
                    f"- Step {entry['step']} / {entry['action']}: {entry['description']}" for entry in trace_entries
                )
            else:
                lines.append("- No prior steps recorded.")

            lines.extend(["", "[Recommended Focus]", recommended_focus])
            memory_contexts.append("\n".join(lines))
            valid_lengths.append(1)

        return memory_contexts, valid_lengths
