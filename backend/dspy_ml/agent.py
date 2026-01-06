"""
DSPy agent for running drills (testing/evaluation only).
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    raise ImportError("dspy-ai is not installed. Install it with: pip install dspy-ai")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .signature import CatanDrillSignature
from .dataset import DrillExample
from api.routes import _canonical_action_dict


class DSPyDrillAgent:
    """Agent for testing drills using DSPy modules."""
    
    def __init__(self, module: Any):
        """
        Initialize the agent.
        
        Args:
            module: DSPy module (optimized or unoptimized)
        """
        if not DSPY_AVAILABLE:
            raise ImportError("dspy-ai is not installed")
        
        self.module = module
    
    def predict(self, example: DrillExample) -> Tuple[str, Dict[str, Any]]:
        """
        Predict action for a drill example.
        
        Args:
            example: Drill example
            
        Returns:
            (reasoning, chosen_action_dict) tuple
        """
        # Call DSPy module
        result = self.module(
            game_rules=example.game_rules,
            observation=example.observation,
            viable_actions=example.viable_actions,
            guideline=example.guideline or ""  # Pass guideline (empty string if None)
        )
        
        reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
        
        # Parse chosen_action JSON string
        chosen_action_str = result.chosen_action if hasattr(result, 'chosen_action') else "null"
        chosen_action_dict = None
        
        if chosen_action_str and chosen_action_str.lower() != "null":
            try:
                chosen_action_dict = json.loads(chosen_action_str)
            except (json.JSONDecodeError, TypeError):
                chosen_action_dict = None
        
        return reasoning, chosen_action_dict
    
    def evaluate(self, example: DrillExample, prediction: Dict[str, Any]) -> float:
        """
        Evaluate prediction against correct actions.
        
        Args:
            example: Drill example
            prediction: Predicted action dict
            
        Returns:
            Accuracy score (1.0 if correct, 0.0 if incorrect)
        """
        if not prediction:
            return 0.0
        
        # Check if prediction matches any correct action
        # Pass state for phase-aware comparison (setup phase mappings)
        canonical_predicted = _canonical_action_dict(prediction, state=example.state)
        for correct_action in example.correct_actions:
            if _canonical_action_dict(correct_action, state=example.state) == canonical_predicted:
                return 1.0
        
        return 0.0
    
    def evaluate_step(self, example: DrillExample) -> Tuple[float, str, Dict[str, Any]]:
        """
        Evaluate a single drill step.
        
        Args:
            example: Drill example
            
        Returns:
            (accuracy, reasoning, chosen_action_dict) tuple
        """
        reasoning, chosen_action_dict = self.predict(example)
        accuracy = self.evaluate(example, chosen_action_dict)
        return accuracy, reasoning, chosen_action_dict

