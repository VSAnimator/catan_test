"""
Hybrid Guideline Agent that uses meta-guidelines where performance is similar to leaf,
and leaf-guidelines where there's a significant performance drop.
"""
import json
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import litellm

from agents.base_agent import BaseAgent
from engine import GameState, Action, ActionPayload
from engine.serialization import state_to_text, legal_actions_to_text, legal_actions
from dspy_ml.dspy_prompt_template import format_prompt


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts using OpenAI embeddings."""
    import openai
    embeddings = []
    for t in texts:
        resp = openai.embeddings.create(model=model, input=t)
        vec = resp.data[0].embedding
        embeddings.append(vec)
    return np.array(embeddings)


def robust_parse(chosen_action_str: str) -> Any:
    """Robustly parse JSON action string."""
    if not chosen_action_str or chosen_action_str == "null":
        return None
    import json
    # Try to extract JSON from the string
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', chosen_action_str)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed = json.loads(json_str)
            # Check if it's missing type/payload wrapper (just payload contents)
            if isinstance(parsed, dict) and "type" not in parsed and ("give_resources" in parsed or "receive_resources" in parsed or "target_player_ids" in parsed or "card_type" in parsed or "year_of_plenty_resources" in parsed or "monopoly_resource_type" in parsed):
                # This looks like a payload without wrapper - try to infer the type
                if "card_type" in parsed or "year_of_plenty_resources" in parsed or "monopoly_resource_type" in parsed:
                    return {"type": "play_dev_card", "payload": parsed}
                elif "give_resources" in parsed or "receive_resources" in parsed:
                    if "target_player_ids" in parsed:
                        return {"type": "propose_trade", "payload": parsed}
                    elif "port_intersection_id" in parsed or parsed.get("port_intersection_id") is None:
                        return {"type": "trade_bank", "payload": parsed}
            return parsed
        except:
            # Try removing extra closing braces
            for i in range(5):
                try:
                    json_str_clean = json_str.rstrip('}')
                    parsed = json.loads(json_str_clean)
                    # Same check for missing wrapper
                    if isinstance(parsed, dict) and "type" not in parsed and ("give_resources" in parsed or "receive_resources" in parsed or "target_player_ids" in parsed or "card_type" in parsed):
                        if "card_type" in parsed or "year_of_plenty_resources" in parsed or "monopoly_resource_type" in parsed:
                            return {"type": "play_dev_card", "payload": parsed}
                        elif "give_resources" in parsed or "receive_resources" in parsed:
                            if "target_player_ids" in parsed:
                                return {"type": "propose_trade", "payload": parsed}
                            elif "port_intersection_id" in parsed or parsed.get("port_intersection_id") is None:
                                return {"type": "trade_bank", "payload": parsed}
                    return parsed
                except:
                    json_str = json_str[:-1]
    return None


class HybridGuidelineAgent(BaseAgent):
    """
    Agent that uses hybrid guidelines (meta where similar performance, leaf otherwise).
    
    Uses best_guidelines_hybrid.json which maps drill_id -> guideline.
    For non-drill gameplay, retrieves guideline based on observation similarity.
    """
    
    def __init__(
        self,
        player_id: str,
        hybrid_guidelines_path: Optional[str] = None,
        model: str = "gpt-5.2",
        exclude_strategic_advice: bool = False,
        exclude_higher_level_features: bool = False,
    ):
        """
        Initialize the hybrid guideline agent.
        
        Args:
            player_id: ID of the player this agent controls
            hybrid_guidelines_path: Path to best_guidelines_hybrid.json (relative to backend/)
            model: LLM model to use
            exclude_strategic_advice: Whether to exclude strategic advice from game rules
            exclude_higher_level_features: Whether to exclude higher-level features from observation
        """
        super().__init__(player_id)
        self.model = model
        self.exclude_strategic_advice = exclude_strategic_advice
        self.exclude_higher_level_features = exclude_higher_level_features
        
        # Resolve hybrid guidelines path
        if hybrid_guidelines_path:
            self.hybrid_guidelines_path = Path(hybrid_guidelines_path)
        else:
            backend_dir = Path(__file__).parent.parent
            self.hybrid_guidelines_path = backend_dir / "dspy_ml" / "data" / "best_guidelines_hybrid.json"
        
        # Load hybrid guidelines
        if not self.hybrid_guidelines_path.exists():
            raise FileNotFoundError(f"Hybrid guidelines not found at {self.hybrid_guidelines_path}")
        
        with open(self.hybrid_guidelines_path) as f:
            self.hybrid_guidelines = json.load(f)
        
        # Load leaf and meta clusters for retrieval fallback
        backend_dir = Path(__file__).parent.parent
        leaf_tree_path = backend_dir / "dspy_ml" / "data" / "guideline_tree.json"
        meta_tree_path = backend_dir / "dspy_ml" / "data" / "guideline_tree_meta.json"
        
        self.leaf_centroids = []
        self.meta_centroids = []
        
        if leaf_tree_path.exists():
            self.leaf_centroids = self._load_centroids(leaf_tree_path, is_meta=False)
        
        if meta_tree_path.exists():
            self.meta_centroids = self._load_centroids(meta_tree_path, is_meta=True)
        
        # Get game rules
        from agents.llm_agent import LLMAgent
        temp_agent = LLMAgent("player_0", exclude_strategic_advice=exclude_strategic_advice)
        self.game_rules = temp_agent._get_default_system_prompt()
    
    def _load_centroids(self, tree_path: Path, is_meta: bool = False) -> List[Dict[str, Any]]:
        """Load centroids from leaf or meta tree."""
        with open(tree_path) as f:
            data = json.load(f)
        
        clusters = data.get("meta_clusters" if is_meta else "clusters", [])
        centroids = []
        
        # Load dataset for observation lookup
        dataset_path = tree_path.parent / "drills_dataset.json"
        if not dataset_path.exists():
            return []
        
        from dspy_ml.dataset import DrillDataset
        dataset = DrillDataset()
        dataset.load_from_json(str(dataset_path))
        drills_by_id = {ex.drill_id: ex for ex in dataset.examples}
        
        for cluster in clusters:
            drill_ids = cluster.get("drill_ids", [])
            if not drill_ids:
                continue
            
            obs_texts = []
            for drill_id in drill_ids:
                if drill_id in drills_by_id:
                    obs = drills_by_id[drill_id].observation
                    if obs:
                        obs_texts.append(obs)
            
            if not obs_texts:
                continue
            
            all_obs_embeddings = embed_texts(obs_texts)
            emb = np.mean(all_obs_embeddings, axis=0)
            
            # Get best guideline
            candidates = cluster.get("candidates", [])
            best_guideline = candidates[0]["guideline"] if candidates else ""
            
            centroids.append({
                "cluster_id": cluster["cluster_id"],
                "guideline": best_guideline,
                "embedding": emb,
            })
        
        return centroids
    
    def _get_centroids(self, is_meta: bool = False) -> List[Dict[str, Any]]:
        """Lazy load centroids."""
        if is_meta:
            if self._meta_centroids is None:
                if self.meta_tree_path.exists():
                    self._meta_centroids = self._load_centroids(self.meta_tree_path, is_meta=True)
                else:
                    self._meta_centroids = []
            return self._meta_centroids
        else:
            if self._leaf_centroids is None:
                if self.leaf_tree_path.exists():
                    self._leaf_centroids = self._load_centroids(self.leaf_tree_path, is_meta=False)
                else:
                    self._leaf_centroids = []
            return self._leaf_centroids
    
    def _retrieve_guideline(self, observation: str) -> str:
        """Retrieve guideline using observation embedding."""
        obs_emb = embed_texts([observation])[0]
        
        # Try leaf first, then meta
        best_guideline = ""
        best_sim = -1
        
        for centroids in [self._get_centroids(is_meta=False), self._get_centroids(is_meta=True)]:
            for c in centroids:
                sim = np.dot(obs_emb, c["embedding"]) / (np.linalg.norm(obs_emb) * np.linalg.norm(c["embedding"]))
                if sim > best_sim:
                    best_sim = sim
                    best_guideline = c["guideline"]
        
        return best_guideline
    
    def choose_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
        drill_id: Optional[int] = None
    ) -> Tuple[Action, Optional[ActionPayload], Optional[str]]:
        """
        Choose an action using hybrid guidelines.
        
        For drill evaluation (when drill_id is provided), uses the hybrid guideline directly.
        For regular gameplay, retrieves guideline based on observation similarity.
        """
        if not legal_actions_list:
            return Action.END_TURN, None, "No legal actions available"
        
        # Generate observation
        observation = state_to_text(
            state,
            self.player_id,
            exclude_higher_level_features=self.exclude_higher_level_features
        )
        
        # Format viable actions
        viable_actions = legal_actions_to_text(legal_actions_list, state=state, player_id=self.player_id)
        
        # Get guideline (for drills, use hybrid directly; for gameplay, retrieve)
        if drill_id is not None and str(drill_id) in self.hybrid_guidelines:
            guideline = self.hybrid_guidelines[str(drill_id)]
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
            
            # Parse response
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
            
            # Convert to Action and ActionPayload
            from agents.guideline_cluster_agent import GuidelineClusterAgent
            action, payload = GuidelineClusterAgent._parse_action_dict(action_dict, legal_actions_list)
            
            return action, payload, reasoning
            
        except Exception as e:
            return Action.END_TURN, None, f"Error: {str(e)}"

