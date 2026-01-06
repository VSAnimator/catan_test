"""
Difficulty analysis for drill classification.

Identifies which drills are "hard" (need guidelines) vs "easy" (baseline LLM can handle).
Key insight: drills WITH human-written guidelines are objectively harder.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import get_db_connection


@dataclass
class DrillDifficultyInfo:
    """Difficulty information for a single drill."""
    drill_id: int
    name: str
    has_human_guideline: bool
    guideline_text: Optional[str]
    predicted_difficulty: str  # "easy", "hard", or "unknown"
    confidence: float  # 0.0-1.0
    
    # Features that indicate difficulty
    action_type: str
    phase: str
    num_viable_actions: int
    is_setup: bool
    is_trade_related: bool
    baseline_accuracy: Optional[float] = None


class DifficultyAnalyzer:
    """
    Analyze drill difficulty based on features and human guidelines.
    
    Strategy:
    1. Drills WITH guidelines → definitely hard
    2. Drills similar to guideline drills → probably hard
    3. Drills with complex actions/states → possibly hard
    4. Everything else → probably easy
    """
    
    def __init__(self):
        self.conn = get_db_connection()
    
    def load_all_drills(self) -> List[Dict[str, Any]]:
        """Load all drills from database with their metadata."""
        cursor = self.conn.cursor()
        
        # Get drills with their guidelines
        cursor.execute("""
            SELECT 
                d.id, 
                d.name,
                d.guideline_text,
                ds.expected_action_json,
                ds.state_json
            FROM drills d
            JOIN drill_steps ds ON d.id = ds.drill_id
            WHERE ds.idx = 0
            ORDER BY d.id
        """)
        
        rows = cursor.fetchall()
        
        drills = []
        for row in rows:
            drill = {
                'drill_id': row['id'],
                'name': row['name'],
                'guideline_text': row['guideline_text'],
                'expected_action': json.loads(row['expected_action_json']),
                'state': json.loads(row['state_json'])
            }
            drills.append(drill)
        
        return drills
    
    def extract_difficulty_features(self, drill: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features that indicate drill difficulty.
        
        Returns:
            Dictionary of features relevant to difficulty prediction
        """
        expected_action = drill['expected_action']
        state = drill['state']
        
        action_type = expected_action.get('type', 'unknown')
        phase = state.get('phase', 'unknown')
        
        # Known difficult action types (from guidelines analysis)
        difficult_actions = {
            'setup_place_road', 'setup_place_settlement',  # Setup is tricky
            'propose_trade',  # Trade evaluation is complex
            'reject_trade', 'accept_trade',  # Trade responses need careful eval
            'move_robber',  # Strategic robber placement
            'play_dev_card',  # Card timing decisions
        }
        
        # Count action complexity indicators
        features = {
            'action_type': action_type,
            'phase': phase,
            'is_setup': phase == 'setup',
            'is_trade_related': 'trade' in action_type.lower(),
            'is_known_difficult_action': action_type in difficult_actions,
            
            # State complexity
            'turn_number': state.get('turn_number', 0),
            'has_pending_trade': state.get('pending_trade_offer') is not None,
            'actions_taken_this_turn': len(state.get('actions_taken_this_turn', [])),
            
            # Resource/strategic complexity
            'current_player_vp': self._get_current_player_vp(state),
            'is_close_to_winning': self._get_current_player_vp(state) >= 8,
        }
        
        return features
    
    def _get_current_player_vp(self, state: Dict[str, Any]) -> int:
        """Get current player's victory points."""
        try:
            current_idx = state.get('current_player_index', 0)
            players = state.get('players', [])
            if current_idx < len(players):
                return players[current_idx].get('victory_points', 0)
        except:
            pass
        return 0
    
    def classify_difficulty(
        self,
        drills: List[Dict[str, Any]],
        baseline_results: Optional[Dict[int, float]] = None
    ) -> List[DrillDifficultyInfo]:
        """
        Classify each drill as easy or hard.
        
        Args:
            drills: List of drill dictionaries
            baseline_results: Optional dict of {drill_id: accuracy} from baseline eval
            
        Returns:
            List of DrillDifficultyInfo objects
        """
        difficulty_infos = []
        
        for drill in drills:
            drill_id = drill['drill_id']
            has_guideline = drill['guideline_text'] is not None
            features = self.extract_difficulty_features(drill)
            
            # Classification logic:
            # 1. Has guideline → HARD (100% confidence)
            if has_guideline:
                predicted_difficulty = "hard"
                confidence = 1.0
            
            # 2. Known difficult action types → HARD (high confidence)
            elif features['is_known_difficult_action']:
                predicted_difficulty = "hard"
                confidence = 0.8
            
            # 3. Setup phase → HARD (medium confidence, setup is tricky)
            elif features['is_setup']:
                predicted_difficulty = "hard"
                confidence = 0.7
            
            # 4. Use baseline accuracy if available
            elif baseline_results and drill_id in baseline_results:
                baseline_acc = baseline_results[drill_id]
                if baseline_acc < 0.5:  # Failed on baseline
                    predicted_difficulty = "hard"
                    confidence = 0.8
                else:
                    predicted_difficulty = "easy"
                    confidence = 0.8
            
            # 5. Default to easy (low confidence)
            else:
                predicted_difficulty = "easy"
                confidence = 0.5
            
            # Create info object
            info = DrillDifficultyInfo(
                drill_id=drill_id,
                name=drill['name'],
                has_human_guideline=has_guideline,
                guideline_text=drill['guideline_text'],
                predicted_difficulty=predicted_difficulty,
                confidence=confidence,
                action_type=features['action_type'],
                phase=features['phase'],
                num_viable_actions=0,  # TODO: get from legal_actions
                is_setup=features['is_setup'],
                is_trade_related=features['is_trade_related'],
                baseline_accuracy=baseline_results.get(drill_id) if baseline_results else None
            )
            
            difficulty_infos.append(info)
        
        return difficulty_infos
    
    def get_statistics(self, difficulty_infos: List[DrillDifficultyInfo]) -> Dict[str, Any]:
        """Get statistics about drill difficulty distribution."""
        total = len(difficulty_infos)
        hard = sum(1 for d in difficulty_infos if d.predicted_difficulty == "hard")
        easy = sum(1 for d in difficulty_infos if d.predicted_difficulty == "easy")
        with_guidelines = sum(1 for d in difficulty_infos if d.has_human_guideline)
        
        return {
            'total_drills': total,
            'hard_drills': hard,
            'easy_drills': easy,
            'drills_with_guidelines': with_guidelines,
            'hard_without_guidelines': hard - with_guidelines,
            'confidence_distribution': {
                'high': sum(1 for d in difficulty_infos if d.confidence >= 0.8),
                'medium': sum(1 for d in difficulty_infos if 0.5 <= d.confidence < 0.8),
                'low': sum(1 for d in difficulty_infos if d.confidence < 0.5)
            }
        }
    
    def save_analysis(self, difficulty_infos: List[DrillDifficultyInfo], output_path: str):
        """Save difficulty analysis to JSON file."""
        output = {
            'drills': [
                {
                    'drill_id': info.drill_id,
                    'name': info.name,
                    'predicted_difficulty': info.predicted_difficulty,
                    'confidence': info.confidence,
                    'has_human_guideline': info.has_human_guideline,
                    'guideline_text': info.guideline_text,
                    'action_type': info.action_type,
                    'phase': info.phase,
                    'is_setup': info.is_setup,
                    'is_trade_related': info.is_trade_related,
                    'baseline_accuracy': info.baseline_accuracy
                }
                for info in difficulty_infos
            ],
            'statistics': self.get_statistics(difficulty_infos)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved difficulty analysis to {output_path}", flush=True)

