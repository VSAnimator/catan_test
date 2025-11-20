"""
Random agent that randomly selects valid actions (never trades).
"""
import random
from typing import Tuple, Optional, List
from engine import GameState, Action, ActionPayload, ResourceType, DiscardResourcesPayload
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Random agent that picks valid actions randomly.
    
    This agent:
    - Never trades (filters out TRADE_BANK and TRADE_PLAYER actions)
    - Randomly selects from all other legal actions
    - Generates discard payloads when needed
    """
    
    def choose_action(
        self, 
        state: GameState, 
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload]]:
        """
        Randomly choose an action from legal actions, excluding trades.
        
        Args:
            state: Current game state
            legal_actions_list: List of (Action, Optional[ActionPayload]) tuples
            
        Returns:
            A randomly chosen action (never a trade)
        """
        if not legal_actions_list:
            raise ValueError("No legal actions available")
        
        # Filter out trade actions
        non_trade_actions = [
            (action, payload) 
            for action, payload in legal_actions_list
            if action != Action.TRADE_BANK and action != Action.TRADE_PLAYER
        ]
        
        # If no non-trade actions available, fall back to all actions
        # (shouldn't happen in normal gameplay, but handle gracefully)
        if not non_trade_actions:
            non_trade_actions = legal_actions_list
        
        # Handle DISCARD_RESOURCES actions that have None payload
        # Generate a valid discard payload
        processed_actions = []
        for action, payload in non_trade_actions:
            if action == Action.DISCARD_RESOURCES and payload is None:
                # Generate a discard payload
                player = next((p for p in state.players if p.id == self.player_id), None)
                if player:
                    total_resources = sum(player.resources.values())
                    discard_count = total_resources // 2
                    
                    # Create a list of all resources the player has
                    available_resources = []
                    for resource_type, amount in player.resources.items():
                        available_resources.extend([resource_type] * amount)
                    
                    # Randomly select resources to discard
                    if len(available_resources) >= discard_count:
                        resources_to_discard = random.sample(available_resources, discard_count)
                        
                        # Count resources by type
                        discard_dict = {}
                        for resource in resources_to_discard:
                            discard_dict[resource] = discard_dict.get(resource, 0) + 1
                        
                        # Create payload
                        payload = DiscardResourcesPayload(resources=discard_dict)
                        processed_actions.append((action, payload))
                    else:
                        # Shouldn't happen, but skip if it does
                        continue
                else:
                    # Player not found, skip this action
                    continue
            else:
                processed_actions.append((action, payload))
        
        if not processed_actions:
            raise ValueError("No valid actions available after processing")
        
        # Randomly select from available actions
        return random.choice(processed_actions)

