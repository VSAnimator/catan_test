"""
State-conditioned agent that adapts strategy based on game state and win conditions.
"""
from typing import Tuple, Optional, List
from engine import GameState, Action, ActionPayload, ResourceType
from .base_behavior_tree import BaseBehaviorTreeAgent


class StateConditionedAgent(BaseBehaviorTreeAgent):
    """
    State-conditioned agent that adapts strategy based on:
    - Current VP count and proximity to winning
    - Board position (settlements, cities, roads)
    - Development cards in hand
    - Opponent positions
    - Available win conditions (longest road, largest army, VP cards)
    """
    
    def _choose_strategic_action(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Choose action based on current game state and win conditions."""
        
        # Evaluate current position
        vp = player.victory_points
        settlements = sum(1 for i in state.intersections if i.owner == self.player_id and i.building_type == "settlement")
        cities = sum(1 for i in state.intersections if i.owner == self.player_id and i.building_type == "city")
        roads = sum(1 for e in state.road_edges if e.owner == self.player_id)
        
        # Count dev cards (development_cards is a list, not a dict)
        dev_cards_list = player.development_cards if hasattr(player, 'development_cards') else []
        knight_count = dev_cards_list.count("knight") if isinstance(dev_cards_list, list) else 0
        vp_card_count = dev_cards_list.count("victory_point") if isinstance(dev_cards_list, list) else 0
        
        # Check win conditions
        longest_road_length = self._get_longest_road_length(state, self.player_id)
        largest_army_size = self._get_largest_army_size(state, self.player_id)
        
        # Opponent analysis
        opponents = [p for p in state.players if p.id != self.player_id]
        max_opponent_vp = max(p.victory_points for p in opponents) if opponents else 0
        closest_opponent_vp = max_opponent_vp
        
        # Determine strategy based on state
        if vp >= 8:
            # Very close to winning - prioritize immediate win
            return self._prioritize_winning(state, player, legal_actions_list, vp, vp_card_count)
        elif vp >= 6:
            # Close to winning - aggressive building + VP cards
            return self._prioritize_endgame(state, player, legal_actions_list, vp, vp_card_count, settlements, cities)
        elif vp >= 4:
            # Mid-game - evaluate best win condition path
            return self._prioritize_midgame(state, player, legal_actions_list, longest_road_length, largest_army_size, 
                                          knight_count, vp_card_count, settlements, cities, roads, closest_opponent_vp)
        else:
            # Early game - balanced development
            return self._prioritize_early_game(state, player, legal_actions_list, settlements, cities, roads)
    
    def _prioritize_winning(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
        vp: int,
        vp_card_count: int
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Prioritize actions that win immediately."""
        # Play VP card if available
        if vp_card_count > 0:
            vp_card_action = self._find_vp_card_action(legal_actions_list)
            if vp_card_action:
                return vp_card_action
        
        # Build to 10 VPs
        if vp == 9:
            city_action = self._find_build_city_action(legal_actions_list)
            if city_action:
                return city_action
            
            settlement_action = self._find_build_settlement_action(legal_actions_list)
            if settlement_action:
                return settlement_action
        
        # If at 8 VPs, try to get to 9 or 10
        if vp == 8:
            # Check if we can get longest road or largest army
            longest_road_length = self._get_longest_road_length(state, self.player_id)
            if longest_road_length >= 5:
                # Build roads to secure longest road
                road_action = self._find_build_road_action(legal_actions_list)
                if road_action:
                    return road_action
            
            # Otherwise, build for VPs
            city_action = self._find_build_city_action(legal_actions_list)
            if city_action:
                return city_action
            
            settlement_action = self._find_build_settlement_action(legal_actions_list)
            if settlement_action:
                return settlement_action
        
        return None
    
    def _prioritize_endgame(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
        vp: int,
        vp_card_count: int,
        settlements: int,
        cities: int
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Endgame strategy: secure win conditions and build VPs."""
        # If we have VP cards, consider buying more dev cards
        # But prioritize building if we're close
        
        # Build cities (2 VPs, better production)
        city_action = self._find_build_city_action(legal_actions_list)
        if city_action:
            return city_action
        
        # Build settlements (1 VP, production)
        settlement_action = self._find_build_settlement_action(legal_actions_list)
        if settlement_action:
            return settlement_action
        
        # Buy dev cards if we have resources and not too many already
        if vp_card_count < 2:  # Don't hoard too many
            buy_dev_card_action = self._find_buy_dev_card_action(legal_actions_list)
            if buy_dev_card_action:
                return buy_dev_card_action
        
        # Play useful dev cards
        dev_card_play = self._choose_dev_card_to_play(state, player, legal_actions_list)
        if dev_card_play:
            return dev_card_play
        
        # Roads for longest road if close
        longest_road_length = self._get_longest_road_length(state, self.player_id)
        if longest_road_length >= 4:
            road_action = self._find_build_road_action(legal_actions_list)
            if road_action:
                return road_action
        
        # Trading if needed
        trade_action = self._choose_trade_action(state, player, legal_actions_list)
        if trade_action:
            return trade_action
        
        return None
    
    def _prioritize_midgame(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
        longest_road_length: int,
        largest_army_size: int,
        knight_count: int,
        vp_card_count: int,
        settlements: int,
        cities: int,
        roads: int,
        closest_opponent_vp: int
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Mid-game: evaluate and pursue best win condition."""
        
        # Evaluate which win condition is most achievable
        win_condition_scores = {}
        
        # Longest road evaluation
        if longest_road_length >= 4:
            win_condition_scores['longest_road'] = 3  # Close to 5
        elif longest_road_length >= 3:
            win_condition_scores['longest_road'] = 2
        else:
            win_condition_scores['longest_road'] = 1
        
        # Largest army evaluation
        if knight_count >= 2:
            win_condition_scores['largest_army'] = 3  # Close to 3
        elif knight_count >= 1:
            win_condition_scores['largest_army'] = 2
        else:
            win_condition_scores['largest_army'] = 1
        
        # VP cards evaluation
        if vp_card_count >= 2:
            win_condition_scores['vp_cards'] = 3
        elif vp_card_count >= 1:
            win_condition_scores['vp_cards'] = 2
        else:
            win_condition_scores['vp_cards'] = 1
        
        # Building evaluation (cities and settlements)
        if cities < 2 and settlements >= 2:
            win_condition_scores['cities'] = 3  # Can upgrade settlements
        elif cities < 3:
            win_condition_scores['cities'] = 2
        else:
            win_condition_scores['cities'] = 1
        
        # Choose best win condition
        best_condition = max(win_condition_scores.items(), key=lambda x: x[1])[0]
        
        # Execute strategy for best condition
        if best_condition == 'longest_road' and longest_road_length < 5:
            road_action = self._find_build_road_action(legal_actions_list)
            if road_action:
                return road_action
        
        if best_condition == 'largest_army' and knight_count < 3:
            buy_dev_card_action = self._find_buy_dev_card_action(legal_actions_list)
            if buy_dev_card_action:
                return buy_dev_card_action
        
        if best_condition == 'vp_cards' and vp_card_count < 3:
            buy_dev_card_action = self._find_buy_dev_card_action(legal_actions_list)
            if buy_dev_card_action:
                return buy_dev_card_action
        
        if best_condition == 'cities':
            city_action = self._find_build_city_action(legal_actions_list)
            if city_action:
                return city_action
        
        # Fallback: balanced approach
        city_action = self._find_build_city_action(legal_actions_list)
        if city_action:
            return city_action
        
        settlement_action = self._find_build_settlement_action(legal_actions_list)
        if settlement_action:
            return settlement_action
        
        buy_dev_card_action = self._find_buy_dev_card_action(legal_actions_list)
        if buy_dev_card_action:
            return buy_dev_card_action
        
        road_action = self._find_build_road_action(legal_actions_list)
        if road_action:
            return road_action
        
        trade_action = self._choose_trade_action(state, player, legal_actions_list)
        if trade_action:
            return trade_action
        
        return None
    
    def _prioritize_early_game(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
        settlements: int,
        cities: int,
        roads: int
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Early game: balanced development, focus on production."""
        # Prioritize settlements for production
        if settlements < 3:
            settlement_action = self._find_build_settlement_action(legal_actions_list)
            if settlement_action:
                return settlement_action
        
        # Then cities for better production
        if settlements >= 2 and cities < 2:
            city_action = self._find_build_city_action(legal_actions_list)
            if city_action:
                return city_action
        
        # Roads for expansion
        if roads < 8:  # Need roads for expansion
            road_action = self._find_build_road_action(legal_actions_list)
            if road_action:
                return road_action
        
        # Dev cards for future
        buy_dev_card_action = self._find_buy_dev_card_action(legal_actions_list)
        if buy_dev_card_action:
            return buy_dev_card_action
        
        # Trading
        trade_action = self._choose_trade_action(state, player, legal_actions_list)
        if trade_action:
            return trade_action
        
        return None
    
    def _get_longest_road_length(self, state: GameState, player_id: str) -> int:
        """Calculate the length of the longest continuous road for a player."""
        # Use the same algorithm as the engine
        player_roads = [r for r in state.road_edges if r.owner == player_id]
        if not player_roads:
            return 0
        
        # Build graph of connected roads
        road_graph = {}
        for road in player_roads:
            inter1 = road.intersection1_id
            inter2 = road.intersection2_id
            if inter1 not in road_graph:
                road_graph[inter1] = []
            if inter2 not in road_graph:
                road_graph[inter2] = []
            road_graph[inter1].append(inter2)
            road_graph[inter2].append(inter1)
        
        # Find longest path using DFS
        max_length = 0
        
        def dfs_path_length(node: int, visited_nodes: set, visited_edges: set) -> int:
            max_path = 0
            for neighbor in road_graph.get(node, []):
                edge_key = (min(node, neighbor), max(node, neighbor))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    if neighbor not in visited_nodes:
                        path_len = 1 + dfs_path_length(neighbor, visited_nodes | {neighbor}, visited_edges)
                        max_path = max(max_path, path_len)
                    visited_edges.remove(edge_key)
            return max_path
        
        for start_node in road_graph.keys():
            path_len = dfs_path_length(start_node, {start_node}, set())
            max_length = max(max_length, path_len)
        
        return max_length
    
    def _get_largest_army_size(self, state: GameState, player_id: str) -> int:
        """Get the size of the largest army (played knights) for a player."""
        player = next((p for p in state.players if p.id == player_id), None)
        if not player:
            return 0
        return player.knights_played

