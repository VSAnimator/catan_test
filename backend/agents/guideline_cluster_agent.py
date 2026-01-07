"""
Cluster-guideline LLM agent.

At runtime:
- Embed the current observation (same embedding model used for clustering: text-embedding-3-small).
- Retrieve the nearest meta cluster centroid.
- Inject that cluster's top guideline into the LLM agent call (guideline field).
- Fall back to leaf clusters if desired (not implemented here; can be added).
"""
import sys
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_agent import LLMAgent
from engine import GameState, Action, ActionPayload, ResourceType
from engine.serialization import state_to_text, legal_actions_to_text

try:
    import litellm
except ImportError:
    litellm = None


class GuidelineClusterAgent(LLMAgent):
    """
    LLM agent that routes to a meta-cluster guideline based on observation embedding,
    then injects that guideline into the LLM agent call (guideline field).
    """

    def __init__(
        self,
        player_id: str,
        meta_tree_path: str = "dspy_ml/data/guideline_tree_meta.json",
        model: str = "gpt-5.2",
        exclude_strategic_advice: bool = True,
        exclude_higher_level_features: bool = False,
    ):
        super().__init__(
            player_id,
            exclude_strategic_advice=exclude_strategic_advice,
            exclude_higher_level_features=exclude_higher_level_features,
            model=model,
        )
        if litellm is None:
            raise ImportError("litellm not installed; needed for embeddings")
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

        # Precompute centroids using obs-only embeddings of member drills is not possible here
        # because we do not have the drill observations. Instead, we embed the cluster candidate
        # guidelines themselves as a proxy centroid. This keeps it payload-agnostic and uses only
        # text available at runtime.
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

        observation = state_to_text(
            state,
            self.player_id,
            exclude_higher_level_features=self.exclude_higher_level_features
        )
        viable_actions = legal_actions_to_text(legal_actions_list, state=state, player_id=self.player_id)

        # Retrieve guideline
        guideline = self._retrieve_guideline(observation)

        # Use LLMAgent predict with guideline injection
        prompt = self._build_prompt(observation, viable_actions, guideline)
        action, payload, reasoning = self._predict_action_from_prompt(prompt, legal_actions_list)
        return action, payload, reasoning

