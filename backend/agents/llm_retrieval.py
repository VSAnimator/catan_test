"""
Retrieval system for finding similar game states and examples from the database.
Uses state similarity to find relevant examples for RAG.
"""
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from engine import GameState, ResourceType
from engine.serialization import deserialize_game_state
from api.database import get_db_connection, get_steps


@dataclass
class RetrievedExample:
    """A retrieved example from the database."""
    game_id: str
    step_idx: int
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    action: Dict[str, Any]
    player_id: str
    similarity_score: float
    outcome: Optional[str] = None  # "win", "loss", or None if unknown


class StateRetriever:
    """
    Retrieves similar game states from the database for RAG.
    """
    
    def __init__(self, max_examples: int = 5):
        """
        Initialize the retriever.
        
        Args:
            max_examples: Maximum number of examples to retrieve
        """
        self.max_examples = max_examples
    
    def _compute_state_features(self, state: GameState, player_id: str) -> Dict[str, Any]:
        """
        Extract key features from a game state for similarity comparison.
        
        Args:
            state: Game state
            player_id: ID of the player to extract features for
            
        Returns:
            Dictionary of features
        """
        player = next((p for p in state.players if p.id == player_id), None)
        if not player:
            return {}
        
        # Extract key features
        features = {
            "phase": state.phase,
            "victory_points": player.victory_points,
            "total_resources": sum(player.resources.values()),
            "resource_counts": {rt.value: count for rt, count in player.resources.items()},
            "roads_built": player.roads_built,
            "settlements_built": player.settlements_built,
            "cities_built": player.cities_built,
            "dev_cards_count": len(player.dev_cards),
            "knights_played": player.knights_played,
            "longest_road": player.longest_road,
            "largest_army": player.largest_army,
            "turn_number": state.turn_number,
            "dice_roll": state.dice_roll,
            "has_pending_trade": state.pending_trade_offer is not None,
        }
        
        # Add opponent features (simplified)
        opponents = [p for p in state.players if p.id != player_id]
        if opponents:
            features["max_opponent_vp"] = max(p.victory_points for p in opponents)
            features["avg_opponent_resources"] = sum(
                sum(p.resources.values()) for p in opponents
            ) / len(opponents)
        
        return features
    
    def _compute_similarity(
        self, 
        features1: Dict[str, Any], 
        features2: Dict[str, Any]
    ) -> float:
        """
        Compute similarity score between two feature sets.
        Returns a score between 0 and 1.
        
        Args:
            features1: Features from state 1
            features2: Features from state 2
            
        Returns:
            Similarity score (0-1)
        """
        if not features1 or not features2:
            return 0.0
        
        score = 0.0
        weight_sum = 0.0
        
        # Phase match (high weight)
        if features1.get("phase") == features2.get("phase"):
            score += 2.0
        weight_sum += 2.0
        
        # Victory points similarity (high weight)
        vp1 = features1.get("victory_points", 0)
        vp2 = features2.get("victory_points", 0)
        vp_diff = abs(vp1 - vp2)
        vp_similarity = max(0, 1.0 - vp_diff / 10.0)  # Normalize by max VPs
        score += vp_similarity * 2.0
        weight_sum += 2.0
        
        # Resource similarity
        total_res1 = features1.get("total_resources", 0)
        total_res2 = features2.get("total_resources", 0)
        if total_res1 + total_res2 > 0:
            res_similarity = 1.0 - abs(total_res1 - total_res2) / max(total_res1 + total_res2, 1)
            score += res_similarity * 1.5
            weight_sum += 1.5
        
        # Building counts similarity
        for key in ["roads_built", "settlements_built", "cities_built"]:
            val1 = features1.get(key, 0)
            val2 = features2.get(key, 0)
            if val1 + val2 > 0:
                similarity = 1.0 - abs(val1 - val2) / max(val1 + val2, 1)
                score += similarity * 0.5
                weight_sum += 0.5
        
        # Turn number similarity (closer turns are more similar)
        turn1 = features1.get("turn_number", 0)
        turn2 = features2.get("turn_number", 0)
        turn_diff = abs(turn1 - turn2)
        turn_similarity = max(0, 1.0 - turn_diff / 100.0)  # Normalize by reasonable turn range
        score += turn_similarity * 1.0
        weight_sum += 1.0
        
        # Dice roll match
        if features1.get("dice_roll") == features2.get("dice_roll"):
            score += 0.5
        weight_sum += 0.5
        
        # Pending trade match
        if features1.get("has_pending_trade") == features2.get("has_pending_trade"):
            score += 0.5
        weight_sum += 0.5
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def retrieve_similar_states(
        self,
        current_state: GameState,
        player_id: str,
        num_games: int = 100,
        min_similarity: float = 0.3
    ) -> List[RetrievedExample]:
        """
        Retrieve similar game states from the database.
        
        Args:
            current_state: Current game state
            player_id: ID of the player making the decision
            num_games: Number of recent games to search through
            min_similarity: Minimum similarity score to include
            
        Returns:
            List of retrieved examples, sorted by similarity (highest first)
        """
        # Get current state features
        current_features = self._compute_state_features(current_state, player_id)
        
        # Query database for recent games
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get recent games (limit to num_games most recent)
        cursor.execute("""
            SELECT DISTINCT game_id
            FROM steps
            ORDER BY timestamp DESC
            LIMIT ?
        """, (num_games,))
        
        game_ids = [row[0] for row in cursor.fetchall()]
        
        examples = []
        
        # For each game, find similar states
        for game_id in game_ids:
            steps = get_steps(game_id)
            
            for step in steps:
                try:
                    state_before_json = json.loads(step["state_before_json"])
                    state_before = deserialize_game_state(state_before_json)
                    
                    # Only consider steps where the same player was acting
                    if step.get("player_id") != player_id:
                        continue
                    
                    # Compute similarity
                    step_features = self._compute_state_features(state_before, player_id)
                    similarity = self._compute_similarity(current_features, step_features)
                    
                    if similarity >= min_similarity:
                        # Try to determine outcome (check if player won)
                        outcome = None
                        try:
                            state_after_json = json.loads(step["state_after_json"])
                            state_after = deserialize_game_state(state_after_json)
                            
                            # Check if this player won (either in this step or later)
                            player_won = False
                            for p in state_after.players:
                                if p.id == player_id and p.victory_points >= 10:
                                    player_won = True
                                    break
                            
                            if player_won:
                                outcome = "win"
                            else:
                                # Check if game ended and player lost
                                if state_after.phase == "finished":
                                    winner = next(
                                        (p for p in state_after.players if p.victory_points >= 10),
                                        None
                                    )
                                    if winner and winner.id != player_id:
                                        outcome = "loss"
                        except:
                            pass
                        
                        examples.append(RetrievedExample(
                            game_id=game_id,
                            step_idx=step["step_idx"],
                            state_before=state_before_json,
                            state_after=json.loads(step["state_after_json"]),
                            action=json.loads(step["action_json"]),
                            player_id=step.get("player_id", ""),
                            similarity_score=similarity,
                            outcome=outcome
                        ))
                except Exception as e:
                    # Skip steps that can't be parsed
                    continue
        
        # Sort by similarity (highest first) and return top examples
        examples.sort(key=lambda x: x.similarity_score, reverse=True)
        return examples[:self.max_examples]

