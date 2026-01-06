"""
DSPy signature definition for Catan drill optimization.
"""
import dspy


class CatanDrillSignature(dspy.Signature):
    """
    DSPy signature for Catan drill action selection.
    
    Inputs:
    - game_rules: Game rules text (without strategic advice)
    - observation: Game state observation text (with higher-level features)
    - viable_actions: Filtered legal actions text
    
    Outputs:
    - reasoning: Agent's reasoning about the decision
    - chosen_action: JSON string of chosen action (type + payload)
    """
    game_rules: str = dspy.InputField(desc="Catan game rules and mechanics (without strategic advice)")
    observation: str = dspy.InputField(desc="Current game state observation with higher-level features")
    viable_actions: str = dspy.InputField(desc="Available viable actions (filtered legal actions)")
    
    reasoning: str = dspy.OutputField(desc="Reasoning about which action to choose")
    chosen_action: str = dspy.OutputField(desc='JSON string of chosen action with keys "type" and "payload": {"type": "build_road", "payload": {"road_edge_id": 14}} or null')

