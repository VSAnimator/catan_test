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
        """Precompute centroids from leaf clusters (matching clustering evaluation)."""
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
            all_obs_embeddings = np.array([self._embed_text(obs) for obs in obs_texts])
            emb = np.mean(all_obs_embeddings, axis=0)
            
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

    def _retrieve_guideline(self, observation: str) -> str:
        """Retrieve guideline from nearest leaf cluster."""
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
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
        drill_id: Optional[int] = None
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

        # Generate observation
        observation = state_to_text(
            state,
            self.player_id,
            exclude_higher_level_features=self.exclude_higher_level_features
        )
        
        # Format viable actions
        viable_actions = legal_actions_to_text(filtered_actions, state=state, player_id=self.player_id)
        
        # Get guideline (for drills, use best_guidelines_leaf.json if available; otherwise retrieve)
        if drill_id is not None:
            # Try to load best_guidelines_leaf.json for direct lookup
            backend_dir = Path(__file__).parent.parent
            best_guidelines_path = backend_dir / "dspy_ml" / "data" / "best_guidelines_leaf.json"
            if best_guidelines_path.exists():
                with open(best_guidelines_path) as f:
                    best_guidelines = json.load(f)
                if str(drill_id) in best_guidelines:
                    guideline = best_guidelines[str(drill_id)]
                else:
                    guideline = self._retrieve_guideline(observation)
            else:
                guideline = self._retrieve_guideline(observation)
        else:
            guideline = self._retrieve_guideline(observation)
        
        # Format prompt
        prompt_dict = format_prompt(
            game_rules=self.game_rules,
            observation=observation,
            viable_actions=viable_actions,
            guideline=guideline
        )
        
        # Call LLM
        temperature = 1.0 if self.model.startswith("gpt-5") else None
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_dict["system"]},
                    {"role": "user", "content": prompt_dict["user"]}
                ],
                temperature=temperature,
            )
            
            response_text = response.choices[0].message.content
            
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
            
            # Convert to Action and ActionPayload (reuse GuidelineClusterAgent's parsing logic)
            from agents.guideline_cluster_agent import GuidelineClusterAgent
            # Create a temporary instance to call the instance method
            temp_agent = GuidelineClusterAgent("temp_player")
            action, payload = temp_agent._parse_action_dict(action_dict, state, filtered_actions)
            
            return action, payload, reasoning
            
        except Exception as e:
            return Action.END_TURN, None, f"Error: {str(e)}"

