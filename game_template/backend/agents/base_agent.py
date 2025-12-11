"""
Base agent interface for game agents.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
# CUSTOMIZE: Import your game's Action and ActionPayload
# from engine import GameState, Action, ActionPayload
# from engine.serialization import legal_actions


class BaseAgent(ABC):
    """
    Base class for all game agents.
    
    Agents receive the current game state and must return a valid action
    from the list of legal actions available to them.
    """
    
    def __init__(self, player_id: str):
        """
        Initialize the agent.
        
        Args:
            player_id: The ID of the player this agent controls
        """
        self.player_id = player_id
    
    @abstractmethod
    def choose_action(
        self, 
        state,  # GameState
        legal_actions_list: List[Tuple[Any, Optional[Any]]]  # List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Any, Optional[Any], Optional[str]]:  # Tuple[Action, Optional[ActionPayload], Optional[str]]
        """
        Choose an action from the list of legal actions.
        
        Args:
            state: Current game state
            legal_actions_list: List of (Action, Optional[ActionPayload]) tuples
        
        Returns:
            A tuple of (Action, Optional[ActionPayload], Optional[str]) representing:
            - The chosen action
            - The action payload (if any)
            - The agent's reasoning (optional, can be None)
        """
        pass
    
    def get_legal_actions(self, state) -> List[Tuple[Any, Optional[Any]]]:
        """
        Get legal actions for this agent's player.
        
        Args:
            state: Current game state
        
        Returns:
            List of legal actions for this player
        """
        # CUSTOMIZE: Import legal_actions from your serialization module
        # from engine.serialization import legal_actions
        # return legal_actions(state, self.player_id)
        pass

