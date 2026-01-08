"""
Cluster-guideline agent using extracted DSPy prompt template.

At runtime:
- Embed the current observation (same embedding model used for clustering: text-embedding-3-small).
- Retrieve the nearest meta cluster centroid.
- Use the extracted DSPy prompt template directly with litellm (avoiding thread-safety issues).
"""
import sys
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import litellm
except ImportError:
    litellm = None
    raise ImportError("litellm not installed; needed for LLM calls and embeddings")

from agents.base_agent import BaseAgent
from agents.llm_agent import LLMAgent
from engine import GameState, Action, ActionPayload
from engine.serialization import state_to_text, legal_actions_to_text
from dspy_ml.dspy_prompt_template import format_prompt


def robust_parse(chosen_action_str: str) -> Any:
    """Robust JSON parsing (same as clustering evaluation)."""
    if not chosen_action_str or str(chosen_action_str).lower() == "null":
        return None
    try:
        return json.loads(chosen_action_str)
    except Exception:
        try:
            cleaned = chosen_action_str.rstrip("}").rstrip() + "}"
            return json.loads(cleaned)
        except Exception:
            m = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", chosen_action_str)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None


class GuidelineClusterAgent(BaseAgent):
    """
    Agent that routes to a meta-cluster guideline based on observation embedding,
    then uses the extracted DSPy prompt template directly with litellm.
    """

    def __init__(
        self,
        player_id: str,
        meta_tree_path: str = "dspy_ml/data/guideline_tree_meta.json",
        model: str = "gpt-5.2",
        exclude_strategic_advice: bool = True,
        exclude_higher_level_features: bool = False,
    ):
        super().__init__(player_id)
        
        if litellm is None:
            raise ImportError("litellm not installed; needed for embeddings")
        
        self.model = model
        self.exclude_strategic_advice = exclude_strategic_advice
        self.exclude_higher_level_features = exclude_higher_level_features
        
        # Get game rules (without strategic advice)
        temp_agent = LLMAgent("player_0", exclude_strategic_advice=True)
        self.game_rules = temp_agent._get_default_system_prompt()
        
        # Resolve path relative to backend directory
        backend_dir = Path(__file__).parent.parent
        self.meta_tree_path = (backend_dir / meta_tree_path).resolve()
        if not self.meta_tree_path.exists():
            raise FileNotFoundError(f"Meta guideline tree not found at {self.meta_tree_path}")
        with open(self.meta_tree_path) as f:
            data = json.load(f)
        self.meta_clusters: List[Dict[str, Any]] = data.get("meta_clusters", [])
        if not self.meta_clusters:
            raise ValueError("No meta_clusters found in meta tree")

        # Precompute centroids using guideline text embeddings
        self._prepare_centroids()

    def _embed_text(self, text: str) -> np.ndarray:
        resp = litellm.embedding(model="text-embedding-3-small", input=text)
        return np.array(resp["data"][0]["embedding"])

    def _prepare_centroids(self):
        self.centroids = []
        for cluster in self.meta_clusters:
            cands = cluster.get("candidates", [])
            if not cands:
                continue
            # Use the top guideline text as centroid proxy
            top_guideline = cands[0].get("guideline", "")
            emb = self._embed_text(top_guideline)
            self.centroids.append({
                "cluster_id": cluster["cluster_id"],
                "guideline": top_guideline,
                "embedding": emb,
            })
        if not self.centroids:
            raise ValueError("No centroids prepared from meta clusters")

    def _retrieve_guideline(self, observation: str) -> str:
        obs_emb = self._embed_text(observation)
        best = None
        best_sim = -1e9
        for c in self.centroids:
            emb = c["embedding"]
            sim = float(np.dot(obs_emb, emb) / (np.linalg.norm(obs_emb) * np.linalg.norm(emb) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best = c
        return best["guideline"] if best else ""

    def choose_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload], Optional[str]]:
        if not legal_actions_list:
            raise ValueError("No legal actions available")

        # Filter out propose_trade actions that were already taken this turn
        # (same logic as base LLMAgent)
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
                    from engine import ResourceType
                    # Normalize current payload to string keys (matching stored format)
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
        
        # Use filtered actions
        legal_actions_list = filtered_actions if filtered_actions else legal_actions_list

        # Format inputs exactly like dataset/clustering evaluation
        observation = state_to_text(
            state,
            self.player_id,
            exclude_higher_level_features=self.exclude_higher_level_features
        )
        viable_actions = legal_actions_to_text(legal_actions_list, state=state, player_id=self.player_id)
        
        # Retrieve guideline
        guideline = self._retrieve_guideline(observation)
        
        # Format prompt using extracted DSPy template
        prompt_dict = format_prompt(
            game_rules=self.game_rules,
            observation=observation,
            viable_actions=viable_actions,
            guideline=guideline
        )
        
        # Call LLM using litellm (no DSPy, avoiding thread-safety issues)
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_dict["system"]},
                    {"role": "user", "content": prompt_dict["user"]}
                ],
                temperature=0.0,  # Deterministic like clustering evaluation
            )
            
            response_text = response.choices[0].message.content
            
            # Parse the response - extract reasoning and chosen_action from the structured format
            reasoning = ""
            chosen_action_str = "null"
            
            # Look for [[ ## reasoning ## ]] and [[ ## chosen_action ## ]] sections
            reasoning_match = re.search(r'\[\[ ## reasoning ## \]\]\s*(.*?)(?=\[\[ ## chosen_action ## \]\]|$)', response_text, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            chosen_action_match = re.search(r'\[\[ ## chosen_action ## \]\]\s*(.*?)(?=\[\[ ## completed ## \]\]|$)', response_text, re.DOTALL)
            if chosen_action_match:
                chosen_action_str = chosen_action_match.group(1).strip()
            
        except Exception as e:
            print(f"Error calling LLM: {e}", flush=True)
            # Fallback to first legal action
            action, payload = legal_actions_list[0]
            return (action, payload, f"Fallback: LLM error ({e})")
        
        # Parse chosen_action JSON string (same as clustering evaluation)
        predicted = robust_parse(chosen_action_str)
        if not predicted:
            # Fallback to first legal action
            action, payload = legal_actions_list[0]
            return (action, payload, "Fallback: No action chosen")
        
        # Convert action dict to Action and ActionPayload
        try:
            action, payload = self._parse_action_dict(predicted, state, legal_actions_list)
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
                from engine import ResourceType, ProposeTradePayload
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
            from engine import ResourceType, DiscardResourcesPayload
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
                
                # If no payload matching needed, or payload matches, return first match
                if not action_payload_dict or payload is None:
                    return (action, payload)
        
        # If no match found, return first legal action of this type
        for action, payload in legal_actions_list:
            if action == target_action:
                return (action, payload)
        
        raise ValueError(f"Could not find matching legal action for {action_type_str}")
