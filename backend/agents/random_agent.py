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
    ) -> Tuple[Action, Optional[ActionPayload], Optional[str]]:
        """
        Randomly choose an action from legal actions, excluding trades.
        
        Args:
            state: Current game state
            legal_actions_list: List of (Action, Optional[ActionPayload]) tuples
            
        Returns:
            A randomly chosen action (never proposes trades, but can accept/reject)
        """
        if not legal_actions_list:
            raise ValueError("No legal actions available")
        
        # Handle pending trade responses (accept/reject)
        accept_actions = [(a, p) for a, p in legal_actions_list if a == Action.ACCEPT_TRADE]
        reject_actions = [(a, p) for a, p in legal_actions_list if a == Action.REJECT_TRADE]
        if accept_actions or reject_actions:
            # Randomly accept or reject
            if accept_actions and random.random() < 0.5:
                action, payload = accept_actions[0]
                return (action, payload, "Randomly accepting trade offer")
            else:
                action, payload = reject_actions[0] if reject_actions else accept_actions[0]
                return (action, payload, "Randomly rejecting trade offer")
        
        # Handle selecting trade partner (if multiple players accepted)
        select_partner_actions = [(a, p) for a, p in legal_actions_list if a == Action.SELECT_TRADE_PARTNER]
        if select_partner_actions:
            # Randomly choose one of the accepting players
            action, payload = random.choice(select_partner_actions)
            return (action, payload, "Randomly selecting trade partner")
        
        # Filter out trade proposal actions (but allow accept/reject which are handled above)
        non_trade_actions = [
            (action, payload) 
            for action, payload in legal_actions_list
            if action != Action.TRADE_BANK 
            and action != Action.TRADE_PLAYER 
            and action != Action.PROPOSE_TRADE
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
        action, payload = random.choice(processed_actions)
        action_name = action.value.replace("_", " ").title()
        return (action, payload, f"Randomly selected: {action_name}")

