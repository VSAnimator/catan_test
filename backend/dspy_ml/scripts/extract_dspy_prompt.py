#!/usr/bin/env python3
"""
Extract the prompt template from DSPy ChainOfThought module with CatanDrillSignature.
This allows us to use the prompt directly without DSPy (avoiding thread-safety issues).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.signature import CatanDrillSignature
from agents.llm_agent import LLMAgent

def main():
    # Configure DSPy
    lm = dspy.LM(model="gpt-5.2")
    dspy.configure(lm=lm)
    
    # Create module
    module = dspy.ChainOfThought(CatanDrillSignature)
    
    # Get game rules
    temp_agent = LLMAgent("player_0", exclude_strategic_advice=True)
    game_rules = temp_agent._get_default_system_prompt()
    
    # Example inputs
    example_observation = "Current game state: Player 0 has 3 wood, 2 brick, 1 wheat. Turn 5. Phase: main."
    example_viable_actions = "Available actions:\n1. build_road: road_edge_id=14\n2. build_settlement: intersection_id=25"
    example_guideline = "When you have enough resources, prioritize building roads to expand your territory."
    
    # Call the module
    result = module(
        game_rules=game_rules,
        observation=example_observation,
        viable_actions=example_viable_actions,
        guideline=example_guideline
    )
    
    # Inspect history to get the prompt
    history = dspy.inspect_history(n=1)
    
    print("=" * 80)
    print("DSPy PROMPT TEMPLATE")
    print("=" * 80)
    print()
    
    system_prompt = None
    user_prompt_template = None
    
    if history and len(history) > 0:
        last_call = history[-1]
        
        # Extract messages
        if hasattr(last_call, 'messages'):
            messages = last_call.messages
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if role == 'system':
                    system_prompt = content
                elif role == 'user':
                    user_prompt_template = content
                    # Replace example values with placeholders using the actual values we passed
                    user_prompt_template = user_prompt_template.replace(
                        game_rules, "{game_rules}"
                    ).replace(
                        example_observation, "{observation}"
                    ).replace(
                        example_viable_actions, "{viable_actions}"
                    ).replace(
                        example_guideline, "{guideline}"
                    )
    
    # Save to file
    output_path = Path(__file__).parent.parent / "data" / "dspy_prompt_template.py"
    
    template_code = f'''"""
DSPy prompt template extracted from ChainOfThought(CatanDrillSignature).
This template can be used directly without DSPy to avoid thread-safety issues.
"""

SYSTEM_PROMPT = """{system_prompt}"""

USER_PROMPT_TEMPLATE = """{user_prompt_template}"""

def format_prompt(game_rules: str, observation: str, viable_actions: str, guideline: str) -> dict:
    """Format the prompt with given inputs."""
    return {{
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(
            game_rules=game_rules,
            observation=observation,
            viable_actions=viable_actions,
            guideline=guideline
        )
    }}
'''
    
    with open(output_path, 'w') as f:
        f.write(template_code)
    
    print(f"Saved prompt template to: {output_path}")
    print()
    print("System prompt (first 500 chars):")
    print(system_prompt[:500] if system_prompt else "None")
    print()
    print("User prompt template (first 500 chars):")
    print(user_prompt_template[:500] if user_prompt_template else "None")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()

