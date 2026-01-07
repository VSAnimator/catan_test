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
from engine import GameState, Action, ActionPayload
from engine.serialization import state_to_text, legal_actions_to_text
from api.routes import _filter_legal_actions

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
        model_name: Optional[str] = None,
        exclude_strategic_advice: bool = True,
        exclude_higher_level_features: bool = False,
    ):
        super().__init__(
            player_id,
            exclude_strategic_advice=exclude_strategic_advice,
            exclude_higher_level_features=exclude_higher_level_features,
            model_name=model_name,
        )
        if litellm is None:
            raise ImportError("litellm not installed; needed for embeddings")
        self.meta_tree_path = Path(meta_tree_path)
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
        self.model_name = model_name or "gpt-5.2"
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

        # Filter propose_trade duplicates like base LLMAgent
        filtered_actions = _filter_legal_actions(legal_actions_list, state, include_propose_trade_duplicates=False)
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

