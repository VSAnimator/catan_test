#!/usr/bin/env python3
"""
Behavior Tree-based agent for Catan.

Uses a hierarchical decision tree to make strategic choices:
1. Check for winning moves
2. Evaluate resource needs
3. Build strategically (cities > settlements > roads)
4. Buy and play development cards
5. Handle robber/defense
"""
import random
from typing import Tuple, Optional, List, Dict, Set
from engine import GameState, Action, ActionPayload, ResourceType, ProposeTradePayload, SelectTradePartnerPayload, BuildRoadPayload, BuildSettlementPayload
from engine.serialization import legal_actions
from .base_agent import BaseAgent


class BehaviorTreeAgent(BaseAgent):
    """
    Behavior tree-based agent that makes strategic decisions.
    
    Decision priority:
    1. Win if possible (play VP card, build to 10 VPs)
    2. Build cities (upgrade settlements)
    3. Build settlements (for production)
    4. Build roads (for expansion and longest road)
    5. Buy development cards
    6. Play development cards strategically
    7. Trade if needed
    8. End turn
    """
    
    def __init__(self, player_id: str):
        super().__init__(player_id)
        self.preferred_resources = [ResourceType.WHEAT, ResourceType.ORE, ResourceType.SHEEP]
        # Heuristic weights for placement decisions (used for drills & deterministic behavior)
        # Slightly favor wheat to break ties like (ore+brick+wheat) vs (ore+brick+sheep).
        self._resource_value = {
            ResourceType.ORE: 3.0,
            ResourceType.WHEAT: 3.2,
            ResourceType.SHEEP: 2.0,
            ResourceType.BRICK: 2.0,
            ResourceType.WOOD: 2.0,
        }
        self._pip_count = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        self._port_bonus = {
            "3:1": 0.75,
            "wood": 0.5,
            "brick": 0.5,
            "wheat": 0.5,
            "sheep": 0.5,
            "ore": 0.5,
        }
    
    def choose_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload]]:
        """Choose action using behavior tree logic."""
        if not legal_actions_list:
            raise ValueError("No legal actions available")

        # Stash state for scoring helpers that are called from simple find_* methods.
        self._last_state_for_scoring = state

        # Setup phase: prefer high-quality placements deterministically (important for drills)
        if state.phase == "setup":
            setup_action = self._choose_setup_action(state, legal_actions_list)
            if setup_action:
                return setup_action
        
        player = next((p for p in state.players if p.id == self.player_id), None)
        if not player:
            raise ValueError(f"Player {self.player_id} not found")
        
        # Filter out repeated trades before processing
        legal_actions_list = self._filter_repeated_trades(state, player, legal_actions_list)
        
        # Get current victory points
        current_vp = player.victory_points
        
        # 1. WIN CONDITION: Can we win this turn?
        if current_vp >= 9:
            # Check if we can play a victory point card
            vp_card_action = self._find_vp_card_action(legal_actions_list)
            if vp_card_action:
                return vp_card_action
            
            # Check if we can build to 10 VPs
            if current_vp == 9:
                # Try to build a city (gives 1 VP from settlement upgrade)
                city_action = self._find_build_city_action(legal_actions_list)
                if city_action:
                    return city_action
                
                # Try to build a settlement (gives 1 VP)
                settlement_action = self._find_build_settlement_action(legal_actions_list)
                if settlement_action:
                    return settlement_action
        
        # 2. HANDLE PENDING TRADES (must be handled before other actions)
        trade_response = self._handle_pending_trade(state, player, legal_actions_list)
        if trade_response:
            return trade_response
        
        # 3. HANDLE REQUIRED ACTIONS (discard, robber, etc.)
        required_action = self._handle_required_actions(state, legal_actions_list)
        if required_action:
            return required_action
        
        # 4. STRATEGIC BUILDING: Cities > Settlements > Roads
        # Cities are highest priority (2 VPs, better production)
        city_action = self._find_build_city_action(legal_actions_list)
        if city_action:
            return city_action
        
        # Settlements for production and VPs
        settlement_action = self._find_build_settlement_action(legal_actions_list)
        if settlement_action:
            return settlement_action
        
        # Roads for expansion and longest road
        road_action = self._find_build_road_action(legal_actions_list)
        if road_action:
            return road_action
        
        # 5. DEVELOPMENT CARDS: Buy and play strategically
        # Play useful dev cards first
        dev_card_play = self._choose_dev_card_to_play(state, player, legal_actions_list)
        if dev_card_play:
            return dev_card_play
        
        # Buy dev cards if we have resources
        buy_dev_card_action = self._find_buy_dev_card_action(legal_actions_list)
        if buy_dev_card_action:
            return buy_dev_card_action
        
        # 6. TRADING: Propose trades if we need resources
        trade_action = self._choose_trade_action(state, player, legal_actions_list)
        if trade_action:
            return trade_action
        
        # 7. DEFAULT: End turn
        end_turn_action = self._find_end_turn_action(legal_actions_list)
        if end_turn_action:
            return end_turn_action
        
        # Fallback: return first available action
        return legal_actions_list[0]

    # ---------------------------------------------------------------------
    # Placement heuristics (setup + roads)
    # ---------------------------------------------------------------------

    def _intersection_value(self, state: GameState, intersection_id: int) -> float:
        inter = next((i for i in state.intersections if i.id == intersection_id), None)
        if not inter:
            return 0.0
        tile_by_id = {t.id: t for t in state.tiles}
        score = 0.0
        has_wheat = False
        for tid in inter.adjacent_tiles:
            t = tile_by_id.get(tid)
            if not t or not t.resource_type or not t.number_token:
                continue
            has_wheat = has_wheat or (t.resource_type == ResourceType.WHEAT)
            score += self._resource_value.get(t.resource_type, 1.0) * self._pip_count.get(t.number_token.value, 0)
        if inter.port_type:
            score += self._port_bonus.get(inter.port_type, 0.4)
        # Tiny tie-break toward wheat adjacency (city/dev strategy)
        if has_wheat:
            score += 0.05
        return score

    def _connected_intersections(self, state: GameState, player_id: str, extra_road_edge_id: Optional[int] = None) -> Set[int]:
        """Compute intersections connected to the player's road network (including settlements/cities)."""
        road_by_id = {r.id: r for r in state.road_edges}
        owned_edges = [r for r in state.road_edges if r.owner == player_id]
        if extra_road_edge_id is not None and extra_road_edge_id in road_by_id:
            owned_edges = owned_edges + [road_by_id[extra_road_edge_id]]

        # adjacency via owned roads
        adj: Dict[int, Set[int]] = {}
        for r in owned_edges:
            adj.setdefault(r.intersection1_id, set()).add(r.intersection2_id)
            adj.setdefault(r.intersection2_id, set()).add(r.intersection1_id)

        seeds = set()
        for inter in state.intersections:
            if inter.owner == player_id and inter.building_type:
                seeds.add(inter.id)
        # also include endpoints of owned roads (helps early setup)
        for r in owned_edges:
            seeds.add(r.intersection1_id)
            seeds.add(r.intersection2_id)

        seen = set(seeds)
        stack = list(seeds)
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def _is_buildable_settlement_site(self, state: GameState, intersection_id: int) -> bool:
        """Check distance rule + vacancy (ignores resource cost)."""
        inter = next((i for i in state.intersections if i.id == intersection_id), None)
        if not inter or inter.owner is not None or inter.building_type is not None:
            return False
        inter_by_id = {i.id: i for i in state.intersections}
        for adj_id in inter.adjacent_intersections:
            adj = inter_by_id.get(adj_id)
            if adj and adj.building_type is not None:
                return False
        return True

    def _best_future_settlement_value(self, state: GameState, player_id: str, extra_road_edge_id: Optional[int] = None) -> float:
        connected = self._connected_intersections(state, player_id, extra_road_edge_id=extra_road_edge_id)
        best = 0.0
        for iid in connected:
            if self._is_buildable_settlement_site(state, iid):
                best = max(best, self._intersection_value(state, iid))
        return best

    def _road_value(self, state: GameState, player_id: str, road_edge_id: int) -> float:
        """Score a road by how much it improves reachable buildable settlement quality."""
        before = self._best_future_settlement_value(state, player_id, extra_road_edge_id=None)
        after = self._best_future_settlement_value(state, player_id, extra_road_edge_id=road_edge_id)
        # Prefer roads that actually improve future settlement options.
        delta = after - before
        return after + (2.0 * delta)

    def _choose_setup_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        # Place settlement: pick highest value intersection deterministically.
        settlement_actions = [(a, p) for a, p in legal_actions_list if a == Action.SETUP_PLACE_SETTLEMENT and p is not None]
        if settlement_actions:
            best = None
            best_score = -1e9
            for a, p in settlement_actions:
                assert isinstance(p, BuildSettlementPayload)
                s = self._intersection_value(state, p.intersection_id)
                if s > best_score:
                    best_score = s
                    best = (a, p)
            return best

        # Place road: pick the road that best improves future settlement options.
        road_actions = [(a, p) for a, p in legal_actions_list if a == Action.SETUP_PLACE_ROAD and p is not None]
        if road_actions:
            best = None
            best_score = -1e9
            for a, p in road_actions:
                assert isinstance(p, BuildRoadPayload)
                s = self._road_value(state, self.player_id, p.road_edge_id)
                if s > best_score:
                    best_score = s
                    best = (a, p)
            return best

        # Start game
        for a, p in legal_actions_list:
            if a == Action.START_GAME:
                return (a, p)
        return None
    
    def _filter_repeated_trades(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> List[Tuple[Action, Optional[ActionPayload]]]:
        """Filter out propose_trade actions that were already taken this turn."""
        filtered_actions = []
        player_actions_this_turn = [
            a for a in state.actions_taken_this_turn 
            if a["player_id"] == player.id and a["action"] == "propose_trade"
        ]
        
        for action, payload in legal_actions_list:
            if action == Action.PROPOSE_TRADE:
                # Check if this exact trade was already proposed this turn
                already_proposed = False
                if payload and hasattr(payload, 'give_resources') and hasattr(payload, 'receive_resources'):
                    # Normalize current payload to string keys (matching stored format)
                    current_give = {rt.value: count for rt, count in payload.give_resources.items()}
                    current_receive = {rt.value: count for rt, count in payload.receive_resources.items()}
                    
                    for prev_action in player_actions_this_turn:
                        prev_payload = prev_action.get("payload", {})
                        prev_give = prev_payload.get("give_resources", {})
                        prev_receive = prev_payload.get("receive_resources", {})
                        prev_targets = set(prev_payload.get("target_player_ids", []))
                        
                        if (prev_give == current_give and
                            prev_receive == current_receive and
                            prev_targets == set(payload.target_player_ids)):
                            already_proposed = True
                            break
                if not already_proposed:
                    filtered_actions.append((action, payload))
            else:
                filtered_actions.append((action, payload))
        
        return filtered_actions if filtered_actions else legal_actions_list
    
    def _handle_required_actions(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Handle actions that must be taken (discard, robber, etc.)."""
        # Discard resources if required
        for action, payload in legal_actions_list:
            if action == Action.DISCARD_RESOURCES:
                # Create a discard payload
                player = next((p for p in state.players if p.id == self.player_id), None)
                if player:
                    total_resources = sum(player.resources.values())
                    discard_count = total_resources // 2
                    from engine import DiscardResourcesPayload
                    # Discard resources evenly
                    discard_payload = self._create_discard_payload(player, discard_count)
                    if discard_payload:
                        return (action, discard_payload)
        
        # Move robber if required
        for action, payload in legal_actions_list:
            if action == Action.MOVE_ROBBER:
                # If payload already has tile_id, use it
                if payload and hasattr(payload, 'tile_id'):
                    return (action, payload)
                # Otherwise, choose a tile that doesn't hurt us
                best_tile = self._choose_robber_tile(state, payload)
                if best_tile:
                    from engine import MoveRobberPayload
                    return (action, MoveRobberPayload(best_tile))
                elif payload:
                    return (action, payload)
        
        # Steal resource if required
        for action, payload in legal_actions_list:
            if action == Action.STEAL_RESOURCE:
                # If payload already has other_player_id, use it
                if payload and hasattr(payload, 'other_player_id'):
                    return (action, payload)
                # Otherwise, choose a player with resources
                best_target = self._choose_steal_target(state, payload)
                if best_target:
                    from engine import StealResourcePayload
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
        """Handle pending trade offers (accept/reject/select partner)."""
        if state.pending_trade_offer is None:
            return None
        
        offer = state.pending_trade_offer
        current_player = state.players[state.current_player_index]
        
        # Check if we're a target of the trade
        if current_player.id in offer['target_player_ids']:
            # Check if we've already responded
            if current_player.id in state.pending_trade_responses:
                return None  # Already responded, wait for others
            
            # Evaluate if the trade is beneficial
            should_accept = self._evaluate_trade_offer(state, player, offer)
            
            if should_accept:
                accept_actions = [(a, p) for a, p in legal_actions_list if a == Action.ACCEPT_TRADE]
                if accept_actions:
                    return accept_actions[0]
            else:
                reject_actions = [(a, p) for a, p in legal_actions_list if a == Action.REJECT_TRADE]
                if reject_actions:
                    return reject_actions[0]
        
        # Check if we're the proposer and need to select a partner
        elif current_player.id == offer['proposer_id']:
            accepting_players = [pid for pid, accepted in state.pending_trade_responses.items() if accepted]
            if len(accepting_players) > 1:
                # Multiple accepted - choose the best partner
                best_partner = self._choose_best_trade_partner(state, accepting_players)
                if best_partner:
                    # Find the select action for the best partner
                    for action, payload in legal_actions_list:
                        if action == Action.SELECT_TRADE_PARTNER:
                            if payload and isinstance(payload, SelectTradePartnerPayload):
                                if payload.selected_player_id == best_partner:
                                    return (action, payload)
                    # Fallback: create a new payload for the best partner
                    return (Action.SELECT_TRADE_PARTNER, SelectTradePartnerPayload(selected_player_id=best_partner))
        
        return None
    
    def _evaluate_trade_offer(
        self,
        state: GameState,
        player,
        offer: Dict
    ) -> bool:
        """Evaluate if a trade offer is beneficial to accept."""
        # We give receive_resources, we get give_resources
        give_resources = offer['receive_resources']  # What we give
        receive_resources = offer['give_resources']  # What we get
        
        # Check if we can afford it
        for resource, amount in give_resources.items():
            if player.resources.get(resource, 0) < amount:
                return False  # Can't afford
        
        # Evaluate trade value
        # Get our resource needs
        needed = self._get_needed_resources(state, player)
        
        # Calculate value: resources we need vs resources we're giving
        receive_value = 0
        for resource, amount in receive_resources.items():
            if resource in needed:
                receive_value += amount * 2  # Double value for needed resources
            else:
                receive_value += amount
        
        give_value = 0
        for resource, amount in give_resources.items():
            if resource in needed:
                give_value += amount * 2  # Higher cost if it's something we need
            else:
                give_value += amount
        
        # Accept if we're getting more value than we're giving
        # Or if we're getting resources we need
        if receive_value > give_value:
            return True
        
        # Accept if we're getting resources we need, even at slight loss
        receiving_needed = any(r in needed for r in receive_resources.keys())
        giving_needed = any(r in needed for r in give_resources.keys())
        
        if receiving_needed and not giving_needed:
            return True
        
        # Accept if the trade is roughly equal and helps us
        if receive_value == give_value and receiving_needed:
            return True
        
        return False
    
    def _choose_best_trade_partner(
        self,
        state: GameState,
        accepting_players: List[str]
    ) -> Optional[str]:
        """Choose the best trade partner from multiple accepting players."""
        if not accepting_players:
            return None
        
        # For now, prefer players with more resources (more likely to have what we need later)
        # In a more sophisticated version, we could consider board position, VP, etc.
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
    
    def _find_vp_card_action(
        self,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to play a victory point card."""
        for action, payload in legal_actions_list:
            if action == Action.PLAY_DEV_CARD:
                if payload and hasattr(payload, 'card_type') and payload.card_type == 'victory_point':
                    return (action, payload)
        return None
    
    def _find_build_city_action(
        self,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to build a city."""
        for action, payload in legal_actions_list:
            if action == Action.BUILD_CITY:
                return (action, payload)
        return None
    
    def _find_build_settlement_action(
        self,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to build a settlement."""
        for action, payload in legal_actions_list:
            if action == Action.BUILD_SETTLEMENT:
                return (action, payload)
        return None
    
    def _find_build_road_action(
        self,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to build a road (choose best-scoring road deterministically)."""
        road_actions = [(a, p) for a, p in legal_actions_list if a == Action.BUILD_ROAD and p is not None]
        if not road_actions:
            return None
        best = None
        best_score = -1e9
        for a, p in road_actions:
            assert isinstance(p, BuildRoadPayload)
            s = self._road_value(self._last_state_for_scoring, self.player_id, p.road_edge_id)
            if s > best_score:
                best_score = s
                best = (a, p)
        return best
    
    def _find_buy_dev_card_action(
        self,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Find action to buy a development card."""
        for action, payload in legal_actions_list:
            if action == Action.BUY_DEV_CARD:
                return (action, payload)
        return None
    
    def _find_end_turn_action(
        self,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
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
        """Choose which development card to play strategically."""
        # Priority: Monopoly > Year of Plenty > Knight > Road Building
        # (Don't play VP cards here - handled in win condition)
        
        card_priority = ['monopoly', 'year_of_plenty', 'knight', 'road_building']
        
        for card_type in card_priority:
            for action, payload in legal_actions_list:
                if action == Action.PLAY_DEV_CARD:
                    if payload and hasattr(payload, 'card_type') and payload.card_type == card_type:
                        # For monopoly and year_of_plenty, we need to choose resources
                        if card_type == 'monopoly':
                            # Choose resource we need most
                            resource = self._choose_monopoly_resource(state, player)
                            if resource:
                                from engine import PlayDevCardPayload
                                return (action, PlayDevCardPayload(
                                    card_type='monopoly',
                                    monopoly_resource_type=resource
                                ))
                        elif card_type == 'year_of_plenty':
                            # Choose resources we need
                            resources = self._choose_year_of_plenty_resources(state, player)
                            if resources:
                                from engine import PlayDevCardPayload
                                return (action, PlayDevCardPayload(
                                    card_type='year_of_plenty',
                                    year_of_plenty_resources=resources
                                ))
                        else:
                            return (action, payload)
        
        return None
    
    def _choose_monopoly_resource(
        self,
        state: GameState,
        player
    ) -> Optional[ResourceType]:
        """Choose which resource to monopolize."""
        # Choose resource we need most
        needed_resources = self._get_needed_resources(state, player)
        if needed_resources:
            return needed_resources[0]
        # Default to wheat (most versatile)
        return ResourceType.WHEAT
    
    def _choose_year_of_plenty_resources(
        self,
        state: GameState,
        player
    ) -> Optional[Dict[ResourceType, int]]:
        """Choose 2 resources from year of plenty."""
        needed = self._get_needed_resources(state, player)
        if len(needed) >= 2:
            return {needed[0]: 1, needed[1]: 1}
        elif len(needed) == 1:
            return {needed[0]: 2}
        else:
            # Default: wheat and ore
            return {ResourceType.WHEAT: 1, ResourceType.ORE: 1}
    
    def _choose_trade_action(
        self,
        state: GameState,
        player,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Optional[Tuple[Action, Optional[ActionPayload]]]:
        """Choose a trade action if it helps us build something."""
        # First, try bank trades (simpler, no negotiation)
        needed = self._get_needed_resources(state, player)
        if needed:
            # Check if we have enough resources to trade (4:1)
            total_resources = sum(player.resources.values())
            if total_resources >= 4:
                # Find a bank trade action
                for action, payload in legal_actions_list:
                    if action == Action.TRADE_BANK:
                        # Trade for a resource we need
                        return (action, payload)
        
        # Second, try proposing player trades if we have excess resources
        # Only propose if we're close to building something important
        if needed and len(needed) > 0:
            # Find resources we have in excess (4+ of same type)
            excess_resources = []
            for resource_type in ResourceType:
                amount = player.resources.get(resource_type, 0)
                if amount >= 4:
                    excess_resources.append((resource_type, amount))
            
            if excess_resources:
                # Propose trade to all other players
                other_players = [p.id for p in state.players if p.id != self.player_id]
                if other_players:
                    # Trade excess resources for needed resources
                    # Give: 4 of excess resource, Receive: 1 of needed resource
                    excess_resource = excess_resources[0][0]  # Use first excess resource
                    needed_resource = needed[0]  # Use first needed resource
                    
                    # Check if we have enough to give
                    if player.resources.get(excess_resource, 0) >= 4:
                        propose_actions = [
                            (a, p) for a, p in legal_actions_list 
                            if a == Action.PROPOSE_TRADE
                        ]
                        if propose_actions:
                            # Create a trade proposal
                            # For now, propose to all other players
                            # In a more sophisticated version, we could target specific players
                            from engine import ProposeTradePayload
                            return (Action.PROPOSE_TRADE, ProposeTradePayload(
                                target_player_ids=other_players,
                                give_resources={excess_resource: 4},
                                receive_resources={needed_resource: 1}
                            ))
        
        return None
    
    def _choose_robber_tile(
        self,
        state: GameState,
        payload: Optional[ActionPayload]
    ) -> Optional[int]:
        """Choose best tile to move robber to (avoid our own tiles)."""
        player = next((p for p in state.players if p.id == self.player_id), None)
        if not player:
            return None
        
        # Get our intersections
        our_intersections = [i for i in state.intersections if i.owner == self.player_id]
        our_tiles = set()
        for inter in our_intersections:
            our_tiles.update(inter.adjacent_tiles)
        
        # Prefer tiles we don't own
        # If payload has a tile_id, check if it's good
        if payload and hasattr(payload, 'tile_id'):
            tile_id = payload.tile_id
            if tile_id != state.robber_tile_id and tile_id not in our_tiles:
                return tile_id
        
        # Fallback: return first available tile that's not ours
        for tile in state.tiles:
            if tile.id != state.robber_tile_id and tile.id not in our_tiles:
                return tile.id
        
        return None
    
    def _choose_steal_target(
        self,
        state: GameState,
        payload: Optional[ActionPayload]
    ) -> Optional[str]:
        """Choose which player to steal from."""
        # Prefer stealing from players with more resources
        other_players = [p for p in state.players if p.id != self.player_id]
        if not other_players:
            return None
        
        # Choose player with most resources
        best_player = max(other_players, key=lambda p: sum(p.resources.values()))
        return best_player.id
    
    def _create_discard_payload(
        self,
        player,
        discard_count: int
    ) -> Optional[ActionPayload]:
        """Create a payload for discarding resources."""
        from engine import DiscardResourcesPayload
        
        # Discard resources evenly
        resources_to_discard = {}
        total = sum(player.resources.values())
        
        if discard_count == 0:
            return None
        
        # Distribute discard across resources proportionally
        remaining = discard_count
        for resource_type, amount in player.resources.items():
            if remaining <= 0:
                break
            if amount > 0:
                # Discard proportional amount
                discard_amount = min(amount, (amount * discard_count) // total)
                if discard_amount > 0:
                    resources_to_discard[resource_type] = discard_amount
                    remaining -= discard_amount
        
        # If we still need to discard more, add from resources with most
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
    
    def _get_needed_resources(
        self,
        state: GameState,
        player
    ) -> List[ResourceType]:
        """Determine which resources we need most."""
        # Priority: resources needed for cities > settlements > roads > dev cards
        needs = []
        
        # Cities need: 2 wheat, 3 ore
        if player.resources.get(ResourceType.WHEAT, 0) < 2:
            needs.append(ResourceType.WHEAT)
        if player.resources.get(ResourceType.ORE, 0) < 3:
            needs.append(ResourceType.ORE)
        
        # Settlements need: 1 wood, 1 brick, 1 wheat, 1 sheep
        if player.resources.get(ResourceType.WOOD, 0) < 1:
            needs.append(ResourceType.WOOD)
        if player.resources.get(ResourceType.BRICK, 0) < 1:
            needs.append(ResourceType.BRICK)
        if player.resources.get(ResourceType.WHEAT, 0) < 1:
            needs.append(ResourceType.WHEAT)
        if player.resources.get(ResourceType.SHEEP, 0) < 1:
            needs.append(ResourceType.SHEEP)
        
        # Roads need: 1 wood, 1 brick
        if player.resources.get(ResourceType.WOOD, 0) < 1:
            needs.append(ResourceType.WOOD)
        if player.resources.get(ResourceType.BRICK, 0) < 1:
            needs.append(ResourceType.BRICK)
        
        # Dev cards need: 1 wheat, 1 sheep, 1 ore
        if player.resources.get(ResourceType.WHEAT, 0) < 1:
            needs.append(ResourceType.WHEAT)
        if player.resources.get(ResourceType.SHEEP, 0) < 1:
            needs.append(ResourceType.SHEEP)
        if player.resources.get(ResourceType.ORE, 0) < 1:
            needs.append(ResourceType.ORE)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_needs = []
        for need in needs:
            if need not in seen:
                seen.add(need)
                unique_needs.append(need)
        
        return unique_needs

