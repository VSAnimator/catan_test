"""
DSPy signature for distilled Catan drill optimization (no guideline field).

This signature is designed for Goal B: agents that internalize strategic knowledge
without needing guidelines at inference time.
"""
import dspy


class CatanDrillSignatureDistilled(dspy.Signature):
    """
    DSPy signature for Catan drill action selection WITHOUT guideline field.
    
    Strategic knowledge is baked into the base instructions rather than
    provided as a separate input field.
    
    Inputs:
    - game_rules: Game rules + strategic principles (enhanced)
    - observation: Game state observation text (with higher-level features)
    - viable_actions: Filtered legal actions text
    
    Outputs:
    - reasoning: Agent's reasoning about the decision
    - chosen_action: JSON string of chosen action (type + payload)
    """
    game_rules: str = dspy.InputField(desc="Catan game rules, mechanics, and strategic principles")
    observation: str = dspy.InputField(desc="Current game state observation with higher-level features")
    viable_actions: str = dspy.InputField(desc="Available viable actions (filtered legal actions)")
    
    reasoning: str = dspy.OutputField(desc="Reasoning about which action to choose, including strategic considerations")
    chosen_action: str = dspy.OutputField(desc='JSON string of chosen action with keys "type" and "payload": {"type": "build_road", "payload": {"road_edge_id": 14}} or null')

