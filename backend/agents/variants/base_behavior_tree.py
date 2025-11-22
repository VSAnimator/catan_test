"""
Base behavior tree agent with shared functionality for all variants.
"""
import random
from typing import Tuple, Optional, List, Dict
from engine import GameState, Action, ActionPayload, ResourceType, ProposeTradePayload, SelectTradePartnerPayload, DiscardResourcesPayload, MoveRobberPayload, StealResourcePayload, PlayDevCardPayload
from engine.serialization import legal_actions
from ..base_agent import BaseAgent


class BaseBehaviorTreeAgent(BaseAgent):
    """
    Base class for behavior tree agents with shared functionality.
    Variants can override priority methods to change strategy.
    """
    
    def __init__(self, player_id: str):
        super().__init__(player_id)
        self.preferred_resources = [ResourceType.WHEAT, ResourceType.ORE, ResourceType.SHEEP]
    
    def choose_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload]]:
        """Choose action using behavior tree logic."""
        if not legal_actions_list:
            raise ValueError("No legal actions available")
        
        player = next((p for p in state.players if p.id == self.player_id), None)
        if not player:
            raise ValueError(f"Player {self.player_id} not found")
        
        # Handle setup phase
        if state.phase == "setup":
            return self._handle_setup_phase(state, legal_actions_list)
        
        # Get current victory points
        current_vp = player.victory_points
        
        # 1. WIN CONDITION: Can we win this turn?
        if current_vp >= 9:
            vp_card_action = self._find_vp_card_action(legal_actions_list)
            if vp_card_action:
                return vp_card_action
            
            if current_vp == 9:
                city_action = self._find_build_city_action(legal_actions_list)
                if city_action:
                    return city_action
                
                settlement_action = self._find_build_settlement_action(legal_actions_list)
                if settlement_action:
                    return settlement_action
        
        # 2. HANDLE PENDING TRADES
        trade_response = self._handle_pending_trade(state, player, legal_actions_list)
        if trade_response:
            return trade_response
        
        # 3. HANDLE REQUIRED ACTIONS
        required_action = self._handle_required_actions(state, legal_actions_list)
        if required_action:
            return required_action
        
        # 4. STRATEGIC ACTIONS (variant-specific priority)
        strategic_action = self._choose_strategic_action(state, player, legal_actions_list)
        if strategic_action:
            return strategic_action
        
        # 5. DEFAULT: End turn
        end_turn_action = self._find_end_turn_action(legal_actions_list)
        if end_turn_action:
            return end_turn_action
        
        return legal_actions_list[0]
    
    def _handle_setup_phase(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload]]:
        """Handle setup phase actions."""
        place_settlement_actions = [(a, p) for a, p in legal_actions_list if a == Action.SETUP_PLACE_SETTLEMENT]
        if place_settlement_actions:
            return random.choice(place_settlement_actions)
        
        place_road_actions = [(a, p) for a, p in legal_actions_list if a == Action.SETUP_PLACE_ROAD]
        if place_road_actions:
            return random.choice(place_road_actions)
        
        start_game_actions = [(a, p) for a, p in legal_actions_list if a == Action.START_GAME]
        if start_game_actions:
            return start_game_actions[0]
        
        return legal_actions_list[0]
    
    def _choose_strategic_action(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """
        Choose strategic action based on variant's priority.
        Override in subclasses to change strategy.
        """
        # Default: balanced strategy
        # Cities > Settlements > Roads > Dev Cards > Trading
        
        city_action = self._find_build_city_action(legal_actions_list)
        if city_action:
            return city_action
        
        settlement_action = self._find_build_settlement_action(legal_actions_list)
        if settlement_action:
            return settlement_action
        
        road_action = self._find_build_road_action(legal_actions_list)
        if road_action:
            return road_action
        
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
    
    def _handle_required_actions(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Handle actions that must be taken."""
        # Discard resources
        for action, payload in legal_actions_list:
            if action == Action.DISCARD_RESOURCES:
                player = next((p for p in state.players if p.id == self.player_id), None)
                if player:
                    total_resources = sum(player.resources.values())
                    discard_count = total_resources // 2
                    discard_payload = self._create_discard_payload(player, discard_count)
                    if discard_payload:
                        return (action, discard_payload)
        
        # Move robber
        for action, payload in legal_actions_list:
            if action == Action.MOVE_ROBBER:
                if payload and hasattr(payload, 'tile_id'):
                    return (action, payload)
                best_tile = self._choose_robber_tile(state, payload)
                if best_tile:
                    return (action, MoveRobberPayload(best_tile))
                elif payload:
                    return (action, payload)
        
        # Steal resource
        for action, payload in legal_actions_list:
            if action == Action.STEAL_RESOURCE:
                if payload and hasattr(payload, 'other_player_id'):
                    return (action, payload)
                best_target = self._choose_steal_target(state, payload)
                if best_target:
                    return (action, StealResourcePayload(best_target))
                elif payload:
                    return (action, payload)
        
        return None
    
    def _handle_pending_trade(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Handle pending trade offers."""
        if state.pending_trade_offer is None:
            return None
        
        offer = state.pending_trade_offer
        current_player = state.players[state.current_player_index]
        
        if current_player.id in offer['target_player_ids']:
            if current_player.id in state.pending_trade_responses:
                return None
            
            should_accept = self._evaluate_trade_offer(state, player, offer)
            
            if should_accept:
                accept_actions = [(a, p) for a, p in legal_actions_list if a == Action.ACCEPT_TRADE]
                if accept_actions:
                    return accept_actions[0]
            else:
                reject_actions = [(a, p) for a, p in legal_actions_list if a == Action.REJECT_TRADE]
                if reject_actions:
                    return reject_actions[0]
        
        elif current_player.id == offer['proposer_id']:
            accepting_players = [pid for pid, accepted in state.pending_trade_responses.items() if accepted]
            if len(accepting_players) > 1:
                best_partner = self._choose_best_trade_partner(state, accepting_players)
                if best_partner:
                    for action, payload in legal_actions_list:
                        if action == Action.SELECT_TRADE_PARTNER:
                            if payload and isinstance(payload, SelectTradePartnerPayload):
                                if payload.selected_player_id == best_partner:
                                    return (action, payload)
                    return (Action.SELECT_TRADE_PARTNER, SelectTradePartnerPayload(selected_player_id=best_partner))
        
        return None
    
    def _evaluate_trade_offer(self, state: GameState, player, offer: Dict) -> bool:
        """Evaluate if a trade offer is beneficial."""
        give_resources = offer['receive_resources']
        receive_resources = offer['give_resources']
        
        for resource, amount in give_resources.items():
            if player.resources.get(resource, 0) < amount:
                return False
        
        needed = self._get_needed_resources(state, player)
        
        receive_value = 0
        for resource, amount in receive_resources.items():
            if resource in needed:
                receive_value += amount * 2
            else:
                receive_value += amount
        
        give_value = 0
        for resource, amount in give_resources.items():
            if resource in needed:
                give_value += amount * 2
            else:
                give_value += amount
        
        if receive_value > give_value:
            return True
        
        receiving_needed = any(r in needed for r in receive_resources.keys())
        giving_needed = any(r in needed for r in give_resources.keys())
        
        if receiving_needed and not giving_needed:
            return True
        
        if receive_value == give_value and receiving_needed:
            return True
        
        return False
    
    def _choose_best_trade_partner(self, state: GameState, accepting_players: List[str]) -> Optional[str]:
        """Choose the best trade partner."""
        if not accepting_players:
            return None
        
        best_player = None
        max_resources = -1
        
        for player_id in accepting_players:
            player = next((p for p in state.players if p.id == player_id), None)
            if player:
                total_resources = sum(player.resources.values())
                if total_resources > max_resources:
                    max_resources = total_resources
                    best_player = player_id
        
        return best_player if best_player else accepting_players[0]
    
    def _find_vp_card_action(self, legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to play a victory point card."""
        for action, payload in legal_actions_list:
            if action == Action.PLAY_DEV_CARD:
                if payload and hasattr(payload, 'card_type') and payload.card_type == 'victory_point':
                    return (action, payload)
        return None
    
    def _find_build_city_action(self, legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to build a city."""
        for action, payload in legal_actions_list:
            if action == Action.BUILD_CITY:
                return (action, payload)
        return None
    
    def _find_build_settlement_action(self, legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to build a settlement."""
        for action, payload in legal_actions_list:
            if action == Action.BUILD_SETTLEMENT:
                return (action, payload)
        return None
    
    def _find_build_road_action(self, legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to build a road."""
        for action, payload in legal_actions_list:
            if action == Action.BUILD_ROAD:
                return (action, payload)
        return None
    
    def _find_buy_dev_card_action(self, legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to buy a development card."""
        for action, payload in legal_actions_list:
            if action == Action.BUY_DEV_CARD:
                return (action, payload)
        return None
    
    def _find_end_turn_action(self, legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to end turn."""
        for action, payload in legal_actions_list:
            if action == Action.END_TURN:
                return (action, payload)
        return None
    
    def _choose_dev_card_to_play(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Choose which development card to play."""
        card_priority = ['monopoly', 'year_of_plenty', 'knight', 'road_building']
        
        for card_type in card_priority:
            for action, payload in legal_actions_list:
                if action == Action.PLAY_DEV_CARD:
                    if payload and hasattr(payload, 'card_type') and payload.card_type == card_type:
                        if card_type == 'monopoly':
                            resource = self._choose_monopoly_resource(state, player)
                            if resource:
                                return (action, PlayDevCardPayload(card_type='monopoly', monopoly_resource_type=resource))
                        elif card_type == 'year_of_plenty':
                            resources = self._choose_year_of_plenty_resources(state, player)
                            if resources:
                                return (action, PlayDevCardPayload(card_type='year_of_plenty', year_of_plenty_resources=resources))
                        else:
                            return (action, payload)
        
        return None
    
    def _choose_monopoly_resource(self, state: GameState, player) -> Optional[ResourceType]:
        """Choose which resource to monopolize."""
        needed_resources = self._get_needed_resources(state, player)
        if needed_resources:
            return needed_resources[0]
        return ResourceType.WHEAT
    
    def _choose_year_of_plenty_resources(self, state: GameState, player) -> Optional[Dict[ResourceType, int]]:
        """Choose 2 resources from year of plenty."""
        needed = self._get_needed_resources(state, player)
        if len(needed) >= 2:
            return {needed[0]: 1, needed[1]: 1}
        elif len(needed) == 1:
            return {needed[0]: 2}
        else:
            return {ResourceType.WHEAT: 1, ResourceType.ORE: 1}
    
    def _choose_trade_action(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Choose a trade action."""
        needed = self._get_needed_resources(state, player)
        if needed:
            total_resources = sum(player.resources.values())
            if total_resources >= 4:
                for action, payload in legal_actions_list:
                    if action == Action.TRADE_BANK:
                        return (action, payload)
        
        if needed and len(needed) > 0:
            excess_resources = []
            for resource_type in ResourceType:
                amount = player.resources.get(resource_type, 0)
                if amount >= 4:
                    excess_resources.append((resource_type, amount))
            
            if excess_resources:
                other_players = [p.id for p in state.players if p.id != self.player_id]
                if other_players:
                    excess_resource = excess_resources[0][0]
                    needed_resource = needed[0]
                    
                    if player.resources.get(excess_resource, 0) >= 4:
                        propose_actions = [(a, p) for a, p in legal_actions_list if a == Action.PROPOSE_TRADE]
                        if propose_actions:
                            return (Action.PROPOSE_TRADE, ProposeTradePayload(
                                target_player_ids=other_players,
                                give_resources={excess_resource: 4},
                                receive_resources={needed_resource: 1}
                            ))
        
        return None
    
    def _choose_robber_tile(self, state: GameState, payload: Optional[ActionPayload]) -> Optional[int]:
        """Choose best tile to move robber to."""
        player = next((p for p in state.players if p.id == self.player_id), None)
        if not player:
            return None
        
        our_intersections = [i for i in state.intersections if i.owner == self.player_id]
        our_tiles = set()
        for inter in our_intersections:
            our_tiles.update(inter.adjacent_tiles)
        
        if payload and hasattr(payload, 'tile_id'):
            tile_id = payload.tile_id
            if tile_id != state.robber_tile_id and tile_id not in our_tiles:
                return tile_id
        
        for tile in state.tiles:
            if tile.id != state.robber_tile_id and tile.id not in our_tiles:
                return tile.id
        
        return None
    
    def _choose_steal_target(self, state: GameState, payload: Optional[ActionPayload]) -> Optional[str]:
        """Choose which player to steal from."""
        other_players = [p for p in state.players if p.id != self.player_id]
        if not other_players:
            return None
        
        best_player = max(other_players, key=lambda p: sum(p.resources.values()))
        return best_player.id
    
    def _create_discard_payload(self, player, discard_count: int) -> Optional[ActionPayload]:
        """Create a payload for discarding resources."""
        if discard_count == 0:
            return None
        
        resources_to_discard = {}
        total = sum(player.resources.values())
        remaining = discard_count
        
        for resource_type, amount in player.resources.items():
            if remaining <= 0:
                break
            if amount > 0:
                discard_amount = min(amount, (amount * discard_count) // total)
                if discard_amount > 0:
                    resources_to_discard[resource_type] = discard_amount
                    remaining -= discard_amount
        
        if remaining > 0:
            sorted_resources = sorted(
                [(r, a) for r, a in player.resources.items() if a > 0],
                key=lambda x: -x[1]
            )
            for resource_type, amount in sorted_resources:
                if remaining <= 0:
                    break
                current_discard = resources_to_discard.get(resource_type, 0)
                can_discard = min(remaining, amount - current_discard)
                if can_discard > 0:
                    resources_to_discard[resource_type] = current_discard + can_discard
                    remaining -= can_discard
        
        if sum(resources_to_discard.values()) == discard_count:
            return DiscardResourcesPayload(resources_to_discard)
        
        return None
    
    def _get_needed_resources(self, state: GameState, player) -> List[ResourceType]:
        """Determine which resources we need most."""
        needs = []
        
        if player.resources.get(ResourceType.WHEAT, 0) < 2:
            needs.append(ResourceType.WHEAT)
        if player.resources.get(ResourceType.ORE, 0) < 3:
            needs.append(ResourceType.ORE)
        
        if player.resources.get(ResourceType.WOOD, 0) < 1:
            needs.append(ResourceType.WOOD)
        if player.resources.get(ResourceType.BRICK, 0) < 1:
            needs.append(ResourceType.BRICK)
        if player.resources.get(ResourceType.WHEAT, 0) < 1:
            needs.append(ResourceType.WHEAT)
        if player.resources.get(ResourceType.SHEEP, 0) < 1:
            needs.append(ResourceType.SHEEP)
        
        if player.resources.get(ResourceType.WOOD, 0) < 1:
            needs.append(ResourceType.WOOD)
        if player.resources.get(ResourceType.BRICK, 0) < 1:
            needs.append(ResourceType.BRICK)
        
        if player.resources.get(ResourceType.WHEAT, 0) < 1:
            needs.append(ResourceType.WHEAT)
        if player.resources.get(ResourceType.SHEEP, 0) < 1:
            needs.append(ResourceType.SHEEP)
        if player.resources.get(ResourceType.ORE, 0) < 1:
            needs.append(ResourceType.ORE)
        
        seen = set()
        unique_needs = []
        for need in needs:
            if need not in seen:
                seen.add(need)
                unique_needs.append(need)
        
        return unique_needs

