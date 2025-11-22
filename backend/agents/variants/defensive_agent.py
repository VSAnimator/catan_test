"""
Defensive agent - focuses on blocking opponents and defensive play.
"""
from typing import Tuple, Optional, List
from engine import GameState, Action, ActionPayload
from .base_behavior_tree import BaseBehaviorTreeAgent


class DefensiveAgent(BaseBehaviorTreeAgent):
    """
    Defensive strategy: Focus on blocking opponents with robber and roads.
    Priority: Play Knight (robber) > Roads (block) > Cities > Settlements > Dev Cards > Trading
    """
    
    def _choose_strategic_action(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Defensive priority: Knight cards > Roads > Build > Dev Cards"""
        # Play knight cards aggressively to move robber
        knight_actions = [
            (a, p) for a, p in legal_actions_list 
            if a == Action.PLAY_DEV_CARD and p and hasattr(p, 'card_type') and p.card_type == 'knight'
        ]
        if knight_actions:
            return knight_actions[0]
        
        # Roads to block opponents
        road_action = self._find_build_road_action(legal_actions_list)
        if road_action:
            return road_action
        
        # Then build normally
        city_action = self._find_build_city_action(legal_actions_list)
        if city_action:
            return city_action
        
        settlement_action = self._find_build_settlement_action(legal_actions_list)
        if settlement_action:
            return settlement_action
        
        # Other dev cards
        dev_card_play = self._choose_dev_card_to_play(state, player, legal_actions_list)
        if dev_card_play:
            return dev_card_play
        
        buy_dev_card_action = self._find_buy_dev_card_action(legal_actions_list)
        if buy_dev_card_action:
            return buy_dev_card_action
        
        trade_action = self._choose_trade_action(state, player, legal_actions_list)
        if trade_action:
            return trade_action
        
        return None

