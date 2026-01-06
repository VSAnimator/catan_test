"""
DSPy-based LLM agent using optimized DSPy modules.

This agent uses GEPA-optimized DSPy modules directly instead of system prompts.
"""
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    raise ImportError("dspy-ai is not installed. Install it with: pip install dspy-ai")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_agent import BaseAgent
from engine import (
    GameState,
    Action,
    ActionPayload,
    ResourceType,
    ProposeTradePayload,
    DiscardResourcesPayload,
)
from engine.serialization import (
    state_to_text,
    legal_actions_to_text,
    legal_actions,
)
from agents.llm_agent import LLMAgent
from api.routes import _filter_legal_actions
# Import dspy_ml modules - handle both relative and absolute imports
try:
    from dspy_ml.optimizer import DrillOptimizer
except ImportError:
    # Try relative import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_ml.optimizer import DrillOptimizer


class DSPyLLMAgent(BaseAgent):
    """
    LLM agent that uses optimized DSPy modules directly.
    
    This agent uses GEPA-optimized DSPy modules instead of system prompts.
    The optimized modules contain improved instructions that are used dynamically.
    """
    
    def __init__(
        self,
        player_id: str,
        module_path: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the DSPy LLM agent.
        
        Args:
            player_id: ID of the player this agent controls
            module_path: Path to optimized DSPy module file (.pkl)
            model_name: Optional LLM model name (if module doesn't include it)
            api_key: Optional API key (uses env vars if not provided)
        """
        super().__init__(player_id)
        
        if not DSPY_AVAILABLE:
            raise ImportError("dspy-ai is not installed")
        
        # Load optimized module
        optimizer = DrillOptimizer(model_name=model_name or "gpt-4o-mini", api_key=api_key)
        self.module, self.metadata = optimizer.load(module_path)
        
        # Get game rules (without strategic advice)
        temp_agent = LLMAgent("player_0", exclude_strategic_advice=True)
        self.game_rules = temp_agent._get_default_system_prompt()
        
        # Initialize DSPy LM if needed
        if model_name:
            lm = dspy.LM(model=model_name)
            dspy.configure(lm=lm)
    
    def choose_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload], Optional[str]]:
        """
        Choose an action using the optimized DSPy module.
        
        Args:
            state: Current game state
            legal_actions_list: List of legal actions
            
        Returns:
            (Action, Optional[ActionPayload], Optional[str]) tuple
        """
        if not legal_actions_list:
            raise ValueError("No legal actions available")
        
        # Handle pending trade responses FIRST (must be done before other actions)
        if state.pending_trade_offer is not None:
            offer = state.pending_trade_offer
            current_player = state.players[state.current_player_index]
            
            # Check if this player is a target of the trade and needs to respond
            if current_player.id in offer['target_player_ids']:
                if current_player.id not in state.pending_trade_responses:
                    # Player needs to respond - prioritize this
                    accept_actions = [(a, p) for a, p in legal_actions_list if a == Action.ACCEPT_TRADE]
                    reject_actions = [(a, p) for a, p in legal_actions_list if a == Action.REJECT_TRADE]
                    
                    if accept_actions and not reject_actions:
                        return (accept_actions[0][0], accept_actions[0][1], "Accepting trade (only option available)")
                    elif reject_actions and not accept_actions:
                        return (reject_actions[0][0], reject_actions[0][1], "Rejecting trade (cannot afford)")
            
            # Check if this player is the proposer and needs to select a partner
            elif current_player.id == offer['proposer_id']:
                accepting_players = [pid for pid, accepted in state.pending_trade_responses.items() if accepted]
                if len(accepting_players) > 1:
                    # Multiple accepted - must select partner
                    select_actions = [(a, p) for a, p in legal_actions_list if a == Action.SELECT_TRADE_PARTNER]
                    if select_actions:
                        from engine import SelectTradePartnerPayload
                        return (Action.SELECT_TRADE_PARTNER, SelectTradePartnerPayload(selected_player_id=accepting_players[0]), 
                                f"Selecting trade partner: {accepting_players[0]}")
        
        # Filter out propose_trade actions that were already taken this turn
        filtered_actions = []
        player_actions_this_turn = [
            a for a in state.actions_taken_this_turn 
            if a["player_id"] == self.player_id and a["action"] == "propose_trade"
        ]
        
        for action, payload in legal_actions_list:
            if action == Action.PROPOSE_TRADE:
                # Check if this exact trade was already proposed this turn
                already_proposed = False
                if payload and hasattr(payload, 'give_resources') and hasattr(payload, 'receive_resources'):
                    current_give = {rt.value: count for rt, count in payload.give_resources.items()}
                    current_receive = {rt.value: count for rt, count in payload.receive_resources.items()}
                    
                    for prev_action in player_actions_this_turn:
                        prev_payload = prev_action.get("payload", {})
                        prev_give = prev_payload.get("give_resources", {})
                        prev_receive = prev_payload.get("receive_resources", {})
                        prev_targets = set(prev_payload.get("target_player_ids", []))
                        
                        if (prev_give == current_give and
                            prev_receive == current_receive and
                            prev_targets == set(payload.target_player_ids)):
                            already_proposed = True
                            break
                if not already_proposed:
                    filtered_actions.append((action, payload))
            else:
                filtered_actions.append((action, payload))
        
        legal_actions_list = filtered_actions if filtered_actions else legal_actions_list
        
        # Format inputs for DSPy module
        observation = state_to_text(
            state,
            self.player_id,
            exclude_higher_level_features=False
        )
        
        viable_actions = legal_actions_to_text(legal_actions_list, state=state, player_id=self.player_id)
        
        # Call DSPy module
        try:
            result = self.module(
                game_rules=self.game_rules,
                observation=observation,
                viable_actions=viable_actions
            )
            
            reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
            chosen_action_str = result.chosen_action if hasattr(result, 'chosen_action') else "null"
        except Exception as e:
            print(f"Error calling DSPy module: {e}", flush=True)
            # Fallback to first legal action
            action, payload = legal_actions_list[0]
            return (action, payload, f"Fallback: DSPy module error ({e})")
        
        # Parse chosen_action JSON string
        if not chosen_action_str or chosen_action_str.lower() == "null":
            # Fallback to first legal action
            action, payload = legal_actions_list[0]
            return (action, payload, "Fallback: No action chosen")
        
        try:
            chosen_action_dict = json.loads(chosen_action_str)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing chosen_action JSON: {e}", flush=True)
            # Fallback to first legal action
            action, payload = legal_actions_list[0]
            return (action, payload, f"Fallback: JSON parse error ({e})")
        
        # Convert action dict to Action and ActionPayload
        # Reuse parsing logic from LLMAgent
        try:
            action, payload = self._parse_action_dict(chosen_action_dict, state, legal_actions_list)
            return (action, payload, reasoning)
        except Exception as e:
            print(f"Error parsing action: {e}", flush=True)
            # Fallback to first legal action
            action, payload = legal_actions_list[0]
            return (action, payload, f"Fallback: Action parse error ({e})")
    
    def _parse_action_dict(
        self,
        action_dict: Dict[str, Any],
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload]]:
        """
        Parse action dict to Action and ActionPayload.
        
        Simplified version of LLMAgent._parse_llm_action_response.
        """
        action_type_str = action_dict.get("type", "").lower()
        action_payload_dict = action_dict.get("payload", {})
        
        # Special handling for setup phase
        if state.phase == "setup":
            if action_type_str == "build_settlement":
                if any(a == Action.SETUP_PLACE_SETTLEMENT for a, _ in legal_actions_list):
                    action_type_str = "setup_place_settlement"
            elif action_type_str == "build_road":
                if any(a == Action.SETUP_PLACE_ROAD for a, _ in legal_actions_list):
                    action_type_str = "setup_place_road"
        
        # Normalize action type
        action_type_normalized = action_type_str.replace(" ", "_").replace("-", "_").strip()
        
        # Map to Action enum
        action_type_map = {
            "build_settlement": Action.BUILD_SETTLEMENT,
            "build_city": Action.BUILD_CITY,
            "build_road": Action.BUILD_ROAD,
            "buy_dev_card": Action.BUY_DEV_CARD,
            "play_dev_card": Action.PLAY_DEV_CARD,
            "trade_bank": Action.TRADE_BANK,
            "propose_trade": Action.PROPOSE_TRADE,
            "accept_trade": Action.ACCEPT_TRADE,
            "reject_trade": Action.REJECT_TRADE,
            "select_trade_partner": Action.SELECT_TRADE_PARTNER,
            "move_robber": Action.MOVE_ROBBER,
            "steal_resource": Action.STEAL_RESOURCE,
            "discard_resources": Action.DISCARD_RESOURCES,
            "end_turn": Action.END_TURN,
            "setup_place_settlement": Action.SETUP_PLACE_SETTLEMENT,
            "setup_place_road": Action.SETUP_PLACE_ROAD,
            "start_game": Action.START_GAME,
        }
        
        target_action = action_type_map.get(action_type_normalized)
        
        if not target_action:
            # Try fuzzy matching
            for action, _ in legal_actions_list:
                if action.value.lower() == action_type_normalized:
                    target_action = action
                    break
        
        if not target_action:
            raise ValueError(f"Could not map action type: {action_type_str}")
        
        # Find matching legal action with payload
        # Handle special cases
        if target_action == Action.PROPOSE_TRADE and action_payload_dict:
            if "give_resources" in action_payload_dict and "receive_resources" in action_payload_dict:
                llm_give = action_payload_dict["give_resources"]
                llm_receive = action_payload_dict["receive_resources"]
                llm_target_players = action_payload_dict.get("target_player_ids", [])
                
                # Convert string resource names to ResourceType
                def convert_resource_dict(d):
                    result = {}
                    for k, v in d.items():
                        if isinstance(k, str):
                            for rt in ResourceType:
                                if rt.value == k.lower():
                                    result[rt] = v
                                    break
                            else:
                                raise ValueError(f"Invalid resource type: {k}")
                        else:
                            result[k] = v
                    return result
                
                give_resources = convert_resource_dict(llm_give)
                receive_resources = convert_resource_dict(llm_receive)
                
                if not llm_target_players:
                    raise ValueError("target_player_ids is required for PROPOSE_TRADE")
                
                payload = ProposeTradePayload(
                    target_player_ids=llm_target_players,
                    give_resources=give_resources,
                    receive_resources=receive_resources
                )
                return (target_action, payload)
        
        if target_action == Action.DISCARD_RESOURCES and action_payload_dict and "resources" in action_payload_dict:
            player = next((p for p in state.players if p.id == self.player_id), None)
            if player:
                total_resources = sum(player.resources.values())
                expected_discard = total_resources // 2
                
                llm_resources = action_payload_dict["resources"]
                discard_dict = {}
                for k, v in llm_resources.items():
                    if isinstance(k, str):
                        for rt in ResourceType:
                            if rt.value == k.lower():
                                discard_dict[rt] = v
                                break
                        else:
                            raise ValueError(f"Invalid resource type: {k}")
                    else:
                        discard_dict[k] = v
                
                total_discard = sum(discard_dict.values())
                if total_discard == expected_discard:
                    payload = DiscardResourcesPayload(resources=discard_dict)
                    return (target_action, payload)
        
        # For other actions, try to match payload
        for action, payload in legal_actions_list:
            if action == target_action:
                # For actions with IDs, try to match
                if action_payload_dict:
                    if "road_edge_id" in action_payload_dict or "road_id" in action_payload_dict:
                        road_id = action_payload_dict.get("road_edge_id") or action_payload_dict.get("road_id")
                        if payload and hasattr(payload, "road_edge_id") and payload.road_edge_id == road_id:
                            return (action, payload)
                    elif "intersection_id" in action_payload_dict or "intersection" in action_payload_dict:
                        intersection_id = action_payload_dict.get("intersection_id") or action_payload_dict.get("intersection")
                        if payload and hasattr(payload, "intersection_id") and payload.intersection_id == intersection_id:
                            return (action, payload)
                    elif "tile_id" in action_payload_dict:
                        tile_id = action_payload_dict["tile_id"]
                        if payload and hasattr(payload, "tile_id") and payload.tile_id == tile_id:
                            return (action, payload)
                    elif "other_player_id" in action_payload_dict:
                        other_player_id = action_payload_dict["other_player_id"]
                        if payload and hasattr(payload, "other_player_id") and payload.other_player_id == other_player_id:
                            return (action, payload)
                
                # If no payload matching needed or no match found, use first available
                return (action, payload)
        
        # Fallback: use first legal action of target type
        for action, payload in legal_actions_list:
            if action == target_action:
                return (action, payload)
        
        raise ValueError(f"Could not find matching legal action for {action_type_str}")

