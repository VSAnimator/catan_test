"""
Leaf-level cluster-guideline agent using extracted DSPy prompt template.

At runtime:
- Embed the current observation (same embedding model used for clustering: text-embedding-3-small).
- Retrieve the nearest leaf cluster centroid.
- Use the extracted DSPy prompt template directly with litellm (avoiding thread-safety issues).
"""
import sys
import json
import re
import time
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
from dspy_ml.dataset import DrillDataset


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


class LeafGuidelineAgent(BaseAgent):
    """
    Agent that routes to a leaf-cluster guideline based on observation embedding,
    then uses the extracted DSPy prompt template directly with litellm.
    """

    def __init__(
        self,
        player_id: str,
        leaf_tree_path: str = "dspy_ml/data/guideline_tree.json",
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
        self.leaf_tree_path = (backend_dir / leaf_tree_path).resolve()
        if not self.leaf_tree_path.exists():
            raise FileNotFoundError(f"Leaf guideline tree not found at {self.leaf_tree_path}")
        with open(self.leaf_tree_path) as f:
            data = json.load(f)
        self.leaf_clusters: List[Dict[str, Any]] = data.get("clusters", [])
        if not self.leaf_clusters:
            raise ValueError("No clusters found in leaf tree")

        # Load dataset to get observations for centroid computation
        dataset_path = backend_dir / "dspy_ml" / "data" / "drills_dataset.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        dataset = DrillDataset()
        dataset.load_from_json(str(dataset_path))
        self.drills_by_id = {ex.drill_id: ex for ex in dataset.examples}

        # Precompute centroids using observation embeddings (matching clustering evaluation)
        self._prepare_centroids()

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using litellm (same as GuidelineClusterAgent for consistency)."""
        resp = litellm.embedding(model="text-embedding-3-small", input=text)
        return np.array(resp["data"][0]["embedding"])

    def _prepare_centroids(self):
        """Precompute centroids from leaf clusters (matching clustering evaluation).
        Uses precomputed embeddings if available to avoid API calls.
        """
        # Try to load precomputed embeddings
        backend_dir = Path(__file__).parent.parent
        embeddings_path = backend_dir / "dspy_ml" / "data" / "observation_embeddings.json"
        precomputed_embeddings = None
        if embeddings_path.exists():
            try:
                with open(embeddings_path) as f:
                    precomputed_embeddings = json.load(f)
                print(f"Loaded {len(precomputed_embeddings)} precomputed embeddings", flush=True)
            except Exception as e:
                print(f"Warning: Failed to load precomputed embeddings: {e}", flush=True)
        
        self.centroids = []
        
        for cluster in self.leaf_clusters:
            drill_ids = cluster.get("drill_ids", [])
            if not drill_ids:
                continue
            
            obs_texts = []
            for drill_id in drill_ids:
                if drill_id in self.drills_by_id:
                    obs = self.drills_by_id[drill_id].observation
                    if obs:
                        obs_texts.append(obs)
            
            if not obs_texts:
                continue
            
            # Compute centroid by averaging observation embeddings
            obs_embeddings = []
            for obs in obs_texts:
                if precomputed_embeddings and obs in precomputed_embeddings:
                    # Use precomputed embedding
                    obs_embeddings.append(np.array(precomputed_embeddings[obs]))
                else:
                    # Fall back to API call
                    obs_embeddings.append(self._embed_text(obs))
            
            emb = np.mean(obs_embeddings, axis=0)
            
            # Get best guideline from candidates
            candidates = cluster.get("candidates", [])
            best_guideline = candidates[0]["guideline"] if candidates else ""
            
            self.centroids.append({
                "cluster_id": cluster["cluster_id"],
                "guideline": best_guideline,
                "embedding": emb,
            })
        
        if not self.centroids:
            raise ValueError("No centroids prepared from leaf clusters")

    def _retrieve_guideline(self, observation: str) -> Tuple[str, float]:
        """Retrieve guideline from nearest leaf cluster. Returns (guideline, embedding_time)."""
        embed_start = time.time()
        obs_emb = self._embed_text(observation)
        embedding_time = time.time() - embed_start
        
        retrieval_start = time.time()
        best = None
        best_sim = -1e9
        for c in self.centroids:
            emb = c["embedding"]
            sim = float(np.dot(obs_emb, emb) / (np.linalg.norm(obs_emb) * np.linalg.norm(emb) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best = c
        retrieval_time = time.time() - retrieval_start
        
        return (best["guideline"] if best else "", embedding_time)

    def _parse_action_dict(
        self,
        action_dict: Dict[str, Any],
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload]]:
        """
        Parse action dict to Action and ActionPayload.
        Copied from GuidelineClusterAgent to avoid creating a new instance on every action.
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
        
        if target_action == Action.PLAY_DEV_CARD and action_payload_dict:
            from engine import ResourceType, PlayDevCardPayload
            card_type = action_payload_dict.get("card_type", "").lower()
            
            # Handle year_of_plenty
            if card_type == "year_of_plenty" and "year_of_plenty_resources" in action_payload_dict:
                llm_resources = action_payload_dict["year_of_plenty_resources"]
                # Convert string resource names to ResourceType
                yop_dict = {}
                for k, v in llm_resources.items():
                    if isinstance(k, str):
                        for rt in ResourceType:
                            if rt.value == k.lower():
                                yop_dict[rt] = v
                                break
                        else:
                            raise ValueError(f"Invalid resource type: {k}")
                    else:
                        yop_dict[k] = v
                
                # Try to find matching legal action
                for action, payload in legal_actions_list:
                    if action == target_action and payload:
                        if (hasattr(payload, "card_type") and payload.card_type == "year_of_plenty" and
                            hasattr(payload, "year_of_plenty_resources") and
                            payload.year_of_plenty_resources == yop_dict):
                            return (action, payload)
                
                # If no exact match, construct payload (if legal)
                payload = PlayDevCardPayload(
                    card_type="year_of_plenty",
                    year_of_plenty_resources=yop_dict
                )
                # Verify it's in legal actions
                for action, legal_payload in legal_actions_list:
                    if action == target_action:
                        if legal_payload is None or (hasattr(legal_payload, "card_type") and legal_payload.card_type == "year_of_plenty"):
                            return (target_action, payload)
            
            # Handle monopoly
            elif card_type == "monopoly" and "monopoly_resource_type" in action_payload_dict:
                llm_resource_type = action_payload_dict["monopoly_resource_type"]
                # Convert string to ResourceType
                monopoly_rt = None
                if isinstance(llm_resource_type, str):
                    for rt in ResourceType:
                        if rt.value == llm_resource_type.lower():
                            monopoly_rt = rt
                            break
                    if not monopoly_rt:
                        raise ValueError(f"Invalid resource type: {llm_resource_type}")
                else:
                    monopoly_rt = llm_resource_type
                
                # Try to find matching legal action
                for action, payload in legal_actions_list:
                    if action == target_action and payload:
                        if (hasattr(payload, "card_type") and payload.card_type == "monopoly" and
                            hasattr(payload, "monopoly_resource_type") and
                            payload.monopoly_resource_type == monopoly_rt):
                            return (action, payload)
                
                # If no exact match, construct payload (if legal)
                payload = PlayDevCardPayload(
                    card_type="monopoly",
                    monopoly_resource_type=monopoly_rt
                )
                # Verify it's in legal actions
                for action, legal_payload in legal_actions_list:
                    if action == target_action:
                        if legal_payload is None or (hasattr(legal_payload, "card_type") and legal_payload.card_type == "monopoly"):
                            return (target_action, payload)
            
            # For other card types (knight, road_building, victory_point), just match by card_type
            else:
                for action, payload in legal_actions_list:
                    if action == target_action and payload:
                        if hasattr(payload, "card_type") and payload.card_type == card_type:
                            return (action, payload)
                # If no match, return first legal action of this type
                for action, payload in legal_actions_list:
                    if action == target_action:
                        return (action, payload)
        
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

    def choose_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
        drill_id: Optional[int] = None
    ) -> Tuple[Action, Optional[ActionPayload], Optional[str]]:
        total_start = time.time()
        timing_info = {}
        
        if not legal_actions_list:
            raise ValueError("No legal actions available")
        
        # Try the leaf guideline agent first
        try:
            return self._choose_action_impl(state, legal_actions_list, drill_id, total_start, timing_info)
        except Exception as e:
            # Fall back to default LLM agent on any error
            error_msg = str(e)
            print(f"LeafGuidelineAgent error: {error_msg}, falling back to LLMAgent", flush=True)
            
            try:
                from agents.llm_agent import LLMAgent
                fallback_agent = LLMAgent(
                    self.player_id,
                    model=self.model,
                    exclude_strategic_advice=self.exclude_strategic_advice,
                    exclude_higher_level_features=self.exclude_higher_level_features,
                )
                action, payload, reasoning = fallback_agent.choose_action(state, legal_actions_list)
                # Add note about fallback
                fallback_reasoning = f"[Fallback to LLMAgent due to: {error_msg}]\n\n{reasoning}" if reasoning else f"[Fallback to LLMAgent due to: {error_msg}]"
                return action, payload, fallback_reasoning
            except Exception as fallback_error:
                # If fallback also fails, return the original error
                total_time = time.time() - total_start
                timing_str = f"[Timing: total={total_time:.2f}s, error={error_msg}, fallback_error={str(fallback_error)}]"
                return Action.END_TURN, None, f"Error: {error_msg}\nFallback also failed: {str(fallback_error)}\n\n{timing_str}"
    
    def _choose_action_impl(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
        drill_id: Optional[int],
        total_start: float,
        timing_info: Dict[str, float]
    ) -> Tuple[Action, Optional[ActionPayload], Optional[str]]:

        # Filter out propose_trade actions that were already taken this turn
        # (same logic as base LLMAgent)
        filter_start = time.time()
        filtered_actions = []
        player_actions_this_turn = [
            a for a in state.actions_taken_this_turn 
            if a["player_id"] == self.player_id and a["action"] == "propose_trade"
        ]
        
        for action, payload in legal_actions_list:
            if action == Action.PROPOSE_TRADE:
                # Check if this exact trade was already proposed
                if payload and hasattr(payload, "target_player_ids") and hasattr(payload, "give_resources") and hasattr(payload, "receive_resources"):
                    already_proposed = any(
                        a.get("action_payload", {}).get("target_player_ids") == payload.target_player_ids and
                        a.get("action_payload", {}).get("give_resources") == payload.give_resources and
                        a.get("action_payload", {}).get("receive_resources") == payload.receive_resources
                        for a in player_actions_this_turn
                    )
                    if already_proposed:
                        continue
            filtered_actions.append((action, payload))
        
        if not filtered_actions:
            return Action.END_TURN, None, "No legal actions available after filtering"
        
        timing_info["filter_actions"] = time.time() - filter_start

        # Generate observation
        obs_start = time.time()
        observation = state_to_text(
            state,
            self.player_id,
            exclude_higher_level_features=self.exclude_higher_level_features
        )
        timing_info["generate_observation"] = time.time() - obs_start
        
        # Format viable actions
        format_start = time.time()
        viable_actions = legal_actions_to_text(filtered_actions, state=state, player_id=self.player_id)
        timing_info["format_actions"] = time.time() - format_start
        
        # Get guideline (for drills, use best_guidelines_leaf.json if available; otherwise retrieve)
        guideline_start = time.time()
        file_check_start = time.time()
        if drill_id is not None:
            # Try to load best_guidelines_leaf.json for direct lookup
            backend_dir = Path(__file__).parent.parent
            best_guidelines_path = backend_dir / "dspy_ml" / "data" / "best_guidelines_leaf.json"
            file_check_time = time.time() - file_check_start
            timing_info["file_check"] = file_check_time
            
            if best_guidelines_path.exists():
                file_read_start = time.time()
                with open(best_guidelines_path) as f:
                    best_guidelines = json.load(f)
                file_read_time = time.time() - file_read_start
                timing_info["file_read"] = file_read_time
                
                if str(drill_id) in best_guidelines:
                    guideline = best_guidelines[str(drill_id)]
                    timing_info["embedding_time"] = 0.0
                    timing_info["retrieval_time"] = 0.0
                else:
                    retrieve_start = time.time()
                    guideline, embedding_time = self._retrieve_guideline(observation)
                    timing_info["embedding_time"] = embedding_time
                    timing_info["retrieval_compute"] = time.time() - retrieve_start - embedding_time
            else:
                retrieve_start = time.time()
                guideline, embedding_time = self._retrieve_guideline(observation)
                timing_info["embedding_time"] = embedding_time
                timing_info["retrieval_compute"] = time.time() - retrieve_start - embedding_time
        else:
            retrieve_start = time.time()
            guideline, embedding_time = self._retrieve_guideline(observation)
            timing_info["embedding_time"] = embedding_time
            timing_info["retrieval_compute"] = time.time() - retrieve_start - embedding_time
        timing_info["guideline_lookup"] = time.time() - guideline_start
        
        # Format prompt
        prompt_start = time.time()
        prompt_dict = format_prompt(
            game_rules=self.game_rules,
            observation=observation,
            viable_actions=viable_actions,
            guideline=guideline
        )
        timing_info["format_prompt"] = time.time() - prompt_start
        
        # Call LLM
        llm_start = time.time()
        temperature = 1.0 if self.model.startswith("gpt-5") else None
        
        # Count input tokens (rough estimate: ~4 chars per token)
        system_tokens = len(prompt_dict["system"]) // 4
        user_tokens = len(prompt_dict["user"]) // 4
        input_tokens_estimate = system_tokens + user_tokens
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_dict["system"]},
                    {"role": "user", "content": prompt_dict["user"]}
                ],
                temperature=temperature,
            )
            
            llm_end = time.time()
            timing_info["llm_call"] = llm_end - llm_start
            
            response_text = response.choices[0].message.content
            
            # Extract token usage from response
            output_tokens = 0
            actual_input_tokens = 0
            total_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                actual_input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(response.usage, 'completion_tokens', 0)
                total_tokens = getattr(response.usage, 'total_tokens', 0)
            elif hasattr(response, '_hidden_params') and response._hidden_params:
                # Try alternative location for token usage
                usage = response._hidden_params.get('usage', {})
                actual_input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
            
            timing_info["input_tokens"] = actual_input_tokens if actual_input_tokens > 0 else input_tokens_estimate
            timing_info["output_tokens"] = output_tokens
            timing_info["total_tokens"] = total_tokens
            
            # Parse response (same as GuidelineClusterAgent)
            reasoning = ""
            chosen_action_str = "null"
            
            reasoning_patterns = [
                r'\[\[ ## reasoning ## \]\]\s*(.*?)(?=\[\[ ## chosen_action ## \]\]|\[\[ ## completed ## \]\]|$)',
                r'\[\[ ## reasoning ## \]\]\s*(.*?)(?=\[\[ ## chosen_action ## \]\]|$)',
            ]
            for pattern in reasoning_patterns:
                reasoning_match = re.search(pattern, response_text, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    break
            
            chosen_action_patterns = [
                r'\[\[ ## chosen_action ## \]\]\s*(.*?)(?=\[\[ ## completed ## \]\]|$)',
                r'\[\[ ## chosen_action ## \]\]\s*(.*)',
            ]
            for pattern in chosen_action_patterns:
                chosen_action_match = re.search(pattern, response_text, re.DOTALL)
                if chosen_action_match:
                    chosen_action_str = chosen_action_match.group(1).strip()
                    break
            
            if chosen_action_str == "null" and "chosen_action" not in response_text.lower():
                json_match = re.search(r'\{[^{}]*"type"[^{}]*\}', response_text)
                if json_match:
                    chosen_action_str = json_match.group(0)
            
            # Parse action
            action_dict = robust_parse(chosen_action_str)
            if not action_dict:
                return Action.END_TURN, None, reasoning or "Failed to parse action"
            
            # Convert to Action and ActionPayload (copy parsing logic to avoid creating GuidelineClusterAgent instance)
            parse_start = time.time()
            action, payload = self._parse_action_dict(action_dict, state, filtered_actions)
            timing_info["parse_action"] = time.time() - parse_start
            
            total_time = time.time() - total_start
            timing_info["total_time"] = total_time
            
            # Calculate other time components
            filter_time = timing_info.get('filter_actions', 0)
            obs_time = timing_info.get('generate_observation', 0)
            format_actions_time = timing_info.get('format_actions', 0)
            guideline_time = timing_info.get('guideline_lookup', 0)
            format_prompt_time = timing_info.get('format_prompt', 0)
            parse_time = timing_info.get('parse_action', 0)
            file_check_time = timing_info.get('file_check', 0)
            file_read_time = timing_info.get('file_read', 0)
            retrieval_compute_time = timing_info.get('retrieval_compute', 0)
            
            # Calculate unaccounted time (should be minimal)
            accounted_time = (
                timing_info.get('embedding_time', 0) +
                timing_info.get('llm_call', 0) +
                filter_time + obs_time + format_actions_time + 
                guideline_time + format_prompt_time + parse_time +
                file_check_time + file_read_time + retrieval_compute_time
            )
            unaccounted_time = total_time - accounted_time
            
            # Add detailed timing info to reasoning for debugging
            input_tokens = timing_info.get('input_tokens', 0)
            output_tokens = timing_info.get('output_tokens', 0)
            total_tokens = timing_info.get('total_tokens', 0)
            llm_time = timing_info.get('llm_call', 0)
            embed_time = timing_info.get('embedding_time', 0)
            
            # Calculate tokens per second for LLM
            tokens_per_sec = (total_tokens / llm_time) if llm_time > 0 and total_tokens > 0 else 0
            
            timing_str = (
                f"[Timing: total={total_time:.2f}s | "
                f"embed={embed_time:.2f}s | "
                f"llm={llm_time:.2f}s ({input_tokens}→{output_tokens} tokens, {total_tokens} total, {tokens_per_sec:.1f} tok/s) | "
                f"obs={obs_time:.2f}s | "
                f"format_actions={format_actions_time:.2f}s | "
                f"guideline={guideline_time:.2f}s (retrieval_compute={retrieval_compute_time:.2f}s, file_check={file_check_time:.2f}s, file_read={file_read_time:.2f}s) | "
                f"format_prompt={format_prompt_time:.2f}s | "
                f"parse={parse_time:.2f}s | "
                f"unaccounted={unaccounted_time:.2f}s]"
            )
            reasoning_with_timing = f"{reasoning}\n\n{timing_str}" if reasoning else timing_str
            
            return action, payload, reasoning_with_timing
            
        except Exception as e:
            total_time = time.time() - total_start
            timing_info["total_time"] = total_time
            llm_time = timing_info.get('llm_call', 0)
            embed_time = timing_info.get('embedding_time', 0)
            input_tokens = timing_info.get('input_tokens', 0)
            output_tokens = timing_info.get('output_tokens', 0)
            timing_str = f"[Timing: total={total_time:.2f}s, embed={embed_time:.2f}s, llm={llm_time:.2f}s ({input_tokens}→{output_tokens}), error={str(e)}]"
            return Action.END_TURN, None, f"Error: {str(e)}\n\n{timing_str}"

