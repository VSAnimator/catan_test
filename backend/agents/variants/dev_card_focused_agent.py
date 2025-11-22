"""
Dev card focused agent - prioritizes development cards.
"""
from typing import Tuple, Optional, List
from engine import GameState, Action, ActionPayload
from .base_behavior_tree import BaseBehaviorTreeAgent


class DevCardFocusedAgent(BaseBehaviorTreeAgent):
    """
    Development card focused strategy: Buy and play dev cards aggressively.
    Priority: Play Dev Cards > Buy Dev Cards > Cities > Settlements > Roads > Trading
    """
    
    def _choose_strategic_action(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Dev card priority: Play > Buy > Build"""
        # Play dev cards first
        dev_card_play = self._choose_dev_card_to_play(state, player, legal_actions_list)
        if dev_card_play:
            return dev_card_play
        
        # Buy dev cards second
        buy_dev_card_action = self._find_buy_dev_card_action(legal_actions_list)
        if buy_dev_card_action:
            return buy_dev_card_action
        
        # Then build
        city_action = self._find_build_city_action(legal_actions_list)
        if city_action:
            return city_action
        
        settlement_action = self._find_build_settlement_action(legal_actions_list)
        if settlement_action:
            return settlement_action
        
        road_action = self._find_build_road_action(legal_actions_list)
        if road_action:
            return road_action
        
        trade_action = self._choose_trade_action(state, player, legal_actions_list)
        if trade_action:
            return trade_action
        
        return None

