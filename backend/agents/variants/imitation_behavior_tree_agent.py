"""
ImitationBehaviorTreeAgent

Goal: maximize agreement with a specific player's historical actions in a specific game.

This is useful as a "mimic" baseline and for regression-style tests where the
metric is: how many actions match a target trace.
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional, Tuple, List

from engine import GameState, Action, ActionPayload
from engine.serialization import (
    serialize_game_state,
    deserialize_action,
    deserialize_action_payload,
    deserialize_game_state,
)

from agents.behavior_tree_agent import BehaviorTreeAgent


def _state_key(state: GameState) -> str:
    """
    Canonicalize state into a stable key.

    We use the engine's JSON serialization, then sort keys to be stable.
    """
    return json.dumps(serialize_game_state(state), sort_keys=True, separators=(",", ":"))


class ImitationBehaviorTreeAgent(BehaviorTreeAgent):
    """
    A BehaviorTreeAgent that first tries to replay a known action trace.

    If the current (serialized) state matches a state in the trace, it returns
    the trace action (if still legal). Otherwise it falls back to normal BT.
    """

    def __init__(
        self,
        player_id: str,
        *,
        reference_game_id: str,
        reference_player_id: str,
        action_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__(player_id)
        self.reference_game_id = reference_game_id
        self.reference_player_id = reference_player_id

        # Map: canonical serialized state -> serialized action dict (API format)
        self._action_map: Dict[str, Dict[str, Any]] = action_map or self._build_action_map()

    def _build_action_map(self) -> Dict[str, Dict[str, Any]]:
        # Import here to avoid hard dependency during module import time.
        from api.database import get_steps

        rows = get_steps(self.reference_game_id)
        m: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if row["player_id"] != self.reference_player_id:
                continue
            try:
                state_before = json.loads(row["state_before_json"])
                action = json.loads(row["action_json"])
            except Exception:
                continue

            # Normalize to the engine's canonical state representation so lookup matches runtime.
            try:
                state_obj = deserialize_game_state(state_before)
            except Exception:
                continue
            key = _state_key(state_obj)
            if isinstance(action, dict) and "type" in action:
                m[key] = action
        return m

    def _try_trace_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        key1 = _state_key(state)
        action_dict = self._action_map.get(key1)
        if action_dict is None:
            return None

        try:
            target_action = deserialize_action(action_dict["type"])
            target_payload = None
            if action_dict.get("payload") is not None:
                target_payload = deserialize_action_payload(action_dict["payload"])
        except Exception:
            return None

        # Ensure the target action is still legal in this state.
        for a, p in legal_actions_list:
            if a != target_action:
                continue
            if p is None and target_payload is None:
                return (a, None)
            if p is not None and target_payload is not None:
                # Payload dataclasses compare structurally.
                if p == target_payload:
                    return (a, p)
        return None

    def choose_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
    ):
        traced = self._try_trace_action(state, legal_actions_list)
        if traced is not None:
            return traced
        return super().choose_action(state, legal_actions_list)


