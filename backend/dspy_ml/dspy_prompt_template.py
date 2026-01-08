"""
DSPy prompt template extracted from ChainOfThought(CatanDrillSignature).
This template can be used directly without DSPy to avoid thread-safety issues.
"""

SYSTEM_PROMPT = """Your input fields are:
1. `game_rules` (str): Catan game rules and mechanics (without strategic advice)
2. `observation` (str): Current game state observation with higher-level features
3. `viable_actions` (str): Available viable actions (filtered legal actions)
4. `guideline` (str): IMPORTANT strategic guideline you should follow for this situation. If provided, this guidance is critical for making the correct decision. (May be empty if no specific guidance available)
Your output fields are:
1. `reasoning` (str): Reasoning about which action to choose
2. `chosen_action` (str): JSON string of chosen action with keys "type" and "payload": {"type": "build_road", "payload": {"road_edge_id": 14}} or null
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## game_rules ## ]]
{game_rules}

[[ ## observation ## ]]
{observation}

[[ ## viable_actions ## ]]
{viable_actions}

[[ ## guideline ## ]]
{guideline}

[[ ## reasoning ## ]]
{reasoning}

[[ ## chosen_action ## ]]
{chosen_action}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        DSPy signature for Catan drill action selection.
        
        Inputs:
        - game_rules: Game rules text (without strategic advice)
        - observation: Game state observation text (with higher-level features)
        - viable_actions: Filtered legal actions text
        - guideline: Strategic guideline for this specific situation (optional, may be empty)
        
        Outputs:
        - reasoning: Agent's reasoning about the decision
        - chosen_action: JSON string of chosen action (type + payload)"""

USER_PROMPT_TEMPLATE = """[[ ## game_rules ## ]]
{game_rules}

[[ ## observation ## ]]
{observation}

[[ ## viable_actions ## ]]
{viable_actions}

[[ ## guideline ## ]]
{guideline}

[[ ## reasoning ## ]]
{reasoning}

[[ ## chosen_action ## ]]
{chosen_action}

[[ ## completed ## ]]"""


def format_prompt(game_rules: str, observation: str, viable_actions: str, guideline: str) -> dict:
    """Format the prompt with given inputs."""
    # Format the user prompt with inputs (reasoning and chosen_action are outputs, so leave as placeholders)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        game_rules=game_rules,
        observation=observation,
        viable_actions=viable_actions,
        guideline=guideline,
        reasoning="{reasoning}",  # Placeholder for output
        chosen_action="{chosen_action}"  # Placeholder for output
    )
    
    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt
    }

