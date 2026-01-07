#!/usr/bin/env python3
"""
Create initial instructions seeded with strategic principles for distilled GEPA optimization.
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.signature_distilled import CatanDrillSignatureDistilled


def create_seeded_module():
    """Create a module with strategic-principles-enhanced initial instructions."""
    
    # Read the synthesized strategic principles (v2 - situational format)
    principles_path = Path(__file__).parent.parent / 'data' / 'strategic_principles_v2.txt'
    with open(principles_path) as f:
        strategic_principles = f.read()
    
    # Create the module
    module = dspy.ChainOfThought(CatanDrillSignatureDistilled)
    
    # Create enhanced instructions that include strategic principles
    base_instructions = """You are selecting the NEXT single action in a step-by-step Catan "drill" loop.

INPUTS (you will be given these 3 blocks)
- game_rules: rules reference text PLUS a section titled "Strategic Principles" (read and use them)
- observation: the full current game state (resources, VP, board, robber, whose turn, pending trade offers, actions taken this turn, etc.)
- viable_actions: the ONLY actions you are allowed to choose from RIGHT NOW (the list may be filtered to correct/incorrect options for training)

TASK
Pick the best immediate next action. You MUST pick an action that appears in viable_actions (exactly one).

REQUIRED REASONING CHECKLIST (be concise):
- Identify any strategic principles that apply (cite the numbers).
- Prefer immediate, concrete value (completing settlement/city/road-to-settle, winning move, freeing blocked production) over speculative trades or dev plays.
- Only propose trades/dev plays if they create a concrete, immediate payoff this turn.
- Do NOT invent actions outside viable_actions.

"""
    
    strategic_section = f"""
=== STRATEGIC PRINCIPLES (learned from expert analysis) ===

These principles have been extracted from analyzing 30 challenging Catan situations where expert strategic advice significantly improved decision-making. Apply these principles when reasoning about your action. The same principles are also provided in the game_rules input so you can reference them explicitly.

{strategic_principles}

===

"""
    
    output_instructions = """
CRITICAL OUTPUT FORMAT
1) Your final output must be JSON with exactly these keys:
   - "reasoning" (string) - explain your strategic thinking
   - "chosen_action" (string) - JSON string of the action

2) The value of "chosen_action" MUST itself be a JSON string encoding an object with:
   - "type" (the action type as a string)
   - "payload" (dict with action-specific fields, or null if no payload)

Example chosen_action values:
- {"type": "build_road", "payload": {"road_edge_id": 14}}
- {"type": "end_turn", "payload": null}
- {"type": "propose_trade", "payload": {"give_resources": {"wood": 1}, "receive_resources": {"brick": 1}, "target_player_ids": ["player_1"]}}

When reasoning, consider which strategic principles apply to this situation and how they guide your decision. Stay within viable_actions and avoid low-impact trades/dev plays that do not produce immediate value.
"""
    
    seeded_instructions = base_instructions + strategic_section + output_instructions
    
    # Set the instructions on the module's predictor
    if hasattr(module, 'predict') and hasattr(module.predict, 'signature'):
        module.predict.signature.instructions = seeded_instructions
        print(f"✓ Created seeded module")
        print(f"✓ Instructions length: {len(seeded_instructions)} chars")
        print(f"✓ Includes {len(strategic_principles.split(chr(10)))} lines of strategic principles")
    else:
        print("Warning: Could not set instructions on module")
    
    return module, seeded_instructions


def main():
    print("Creating seeded module with strategic principles...")
    print()
    
    module, instructions = create_seeded_module()
    
    print()
    print("=" * 80)
    print("SEEDED INSTRUCTIONS (first 1000 chars):")
    print("=" * 80)
    print(instructions[:1000])
    print("...")
    print("=" * 80)
    
    # Save the module
    output_path = Path(__file__).parent.parent / 'data' / 'seeded_module_initial.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module.save(str(output_path))
    print()
    print(f"✓ Saved seeded module to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

