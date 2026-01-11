"""
Agent runner for playing games with agents.
"""
import copy
from typing import Dict, List, Optional, Tuple
from engine import GameState, Action, ActionPayload
from engine.serialization import (
    deserialize_game_state,
    serialize_game_state,
    serialize_action,
    serialize_action_payload,
    legal_actions,
)
from .base_agent import BaseAgent
from .random_agent import RandomAgent


class AgentRunner:
    """
    Runs a game with agents, handling automatic gameplay.
    """
    
    def __init__(
        self,
        state: GameState,
        agents: Dict[str, BaseAgent],
        max_turns: int = 1000
    ):
        """
        Initialize the agent runner.
        
        Args:
            state: Initial game state
            agents: Dictionary mapping player_id to agent
            max_turns: Maximum number of turns before stopping
        """
        self.state = state
        self.agents = agents
        self.max_turns = max_turns
        self.turn_count = 0
        self.error: Optional[str] = None
    
    def run_automatic(
        self,
        save_state_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[GameState, bool, Optional[str]]:
        """
        Run the game automatically until completion, error, or max turns.
        
        Args:
            save_state_callback: Optional callback to save state after each action
                                Signature: (game_id: str, state_before: GameState, state_after: GameState, action: dict, player_id: str) -> None
            progress_callback: Optional callback for progress updates
                              Signature: (turn_count: int, action_count: int) -> None
        
        Returns:
            Tuple of (final_state, completed, error_message)
            - completed: True if game finished normally, False if stopped early
            - error_message: None if no error, otherwise error description
        """
        try:
            action_count = 0
            while self.turn_count < self.max_turns:
                # Check if game is finished
                if self.state.phase == "finished":
                    return self.state, True, None
                
                # Check for victory condition (10+ victory points)
                for player in self.state.players:
                    if player.victory_points >= 10:
                        return self.state, True, None
                
                # Get current player
                if self.state.phase == "setup":
                    current_player = self.state.players[self.state.setup_phase_player_index]
                elif self.state.phase == "playing":
                    current_player = self.state.players[self.state.current_player_index]
                else:
                    return self.state, True, None  # Game finished
                
                # Get agent for current player
                agent = self.agents.get(current_player.id)
                if not agent:
                    return self.state, False, f"No agent found for player {current_player.id}"
                
                # Handle special case: when 7 is rolled, all players with 8+ resources must discard
                if (self.state.phase == "playing" and 
                    self.state.dice_roll == 7 and
                    not self.state.waiting_for_robber_move and
                    not self.state.waiting_for_robber_steal):
                    # Check if we're still in discard phase
                    robber_has_been_moved = (
                        self.state.robber_initial_tile_id is not None and 
                        self.state.robber_tile_id != self.state.robber_initial_tile_id
                    )
                    
                    if not robber_has_been_moved:
                        # Check if any player needs to discard
                        players_needing_discard = [
                            p for p in self.state.players
                            if (sum(p.resources.values()) >= 8 and 
                                p.id not in self.state.players_discarded)
                        ]
                        
                        if players_needing_discard:
                            # Process discards for all AI players who need to (skip human players)
                            processed_any = False
                            for player in players_needing_discard:
                                discard_agent = self.agents.get(player.id)
                                if not discard_agent:
                                    # Skip human players - they'll handle it via UI
                                    continue
                                
                                # Get legal actions for this player (should include DISCARD_RESOURCES)
                                legal_actions_list = legal_actions(self.state, player.id)
                                
                                # Filter to only discard actions
                                discard_actions = [
                                    (action, payload) 
                                    for action, payload in legal_actions_list
                                    if action == Action.DISCARD_RESOURCES
                                ]
                                
                                if discard_actions:
                                    # Store state before action
                                    state_before = copy.deepcopy(self.state)
                                    
                                    # Choose a discard action (random agent will pick one)
                                    result = discard_agent.choose_action(self.state, discard_actions)
                                    if len(result) == 5:
                                        action, payload, _, _, _ = result  # Ignore reasoning, raw_response, and parsing_warnings for discard
                                    elif len(result) == 4:
                                        action, payload, _, _ = result  # Ignore reasoning and raw_response for discard
                                    elif len(result) == 3:
                                        action, payload, _ = result  # Ignore reasoning for discard
                                    else:
                                        action, payload = result  # Backward compatibility
                                    
                                    # Apply the action
                                    self.state = self.state.step(action, payload, player_id=player.id)
                                    processed_any = True
                                    
                                    # Save state if callback provided
                                    if save_state_callback:
                                        action_dict = {
                                            "type": serialize_action(action),
                                        }
                                        if payload:
                                            action_dict["payload"] = serialize_action_payload(payload)
                                        save_state_callback(
                                            self.state.game_id,
                                            state_before,
                                            self.state,
                                            action_dict,
                                            player.id
                                        )
                            
                            # Continue to next iteration to check if we can proceed
                            if processed_any:
                                continue
                            # If no AI players processed (only human players), continue anyway
                            # The human players will handle their discards via UI
                            continue
                
                # Get legal actions for current player
                legal_actions_list = legal_actions(self.state, current_player.id)
                
                if not legal_actions_list:
                    # No legal actions - this might be an error or end of game
                    return self.state, False, f"No legal actions available for player {current_player.id}"
                
                # Store state before action
                state_before = copy.deepcopy(self.state)
                
                # Automate action if there's only one option
                # BUT: Don't auto-select DISCARD_RESOURCES - agent needs to provide payload
                if len(legal_actions_list) == 1:
                    action, payload = legal_actions_list[0]
                    if action == Action.DISCARD_RESOURCES:
                        # Don't auto-select discard - let agent choose which resources to discard
                        pass  # Fall through to agent.choose_action below
                    else:
                        reasoning = f"Automated: only one legal action available"
                        # Apply the action immediately for non-discard actions
                        self.state = self.state.step(action, payload, player_id=current_player.id)
                        processed_any = True
                        
                        # Save state if callback provided
                        if save_state_callback:
                            action_dict = {
                                "type": serialize_action(action),
                            }
                            if payload:
                                action_dict["payload"] = serialize_action_payload(payload)
                            save_state_callback(
                                self.state.game_id,
                                state_before,
                                self.state,
                                action_dict,
                                current_player.id
                            )
                        
                        # Continue to next iteration
                        continue
                
                # If we get here, either there are multiple actions or it's a discard action
                # (discard actions always go through agent to generate payload)
                # Agent chooses an action
                    # Retry logic for PROPOSE_TRADE parsing errors
                    max_retries = 3
                    retry_count = 0
                    last_error = None
                    
                    while retry_count < max_retries:
                        try:
                            result = agent.choose_action(self.state, legal_actions_list)
                            if len(result) == 5:
                                action, payload, reasoning, raw_llm_response, parsing_warnings = result
                            elif len(result) == 4:
                                action, payload, reasoning, raw_llm_response = result
                                parsing_warnings = None
                            elif len(result) == 3:
                                action, payload, reasoning = result
                                raw_llm_response = None
                                parsing_warnings = None
                            else:
                                # Backward compatibility: old agents return 2-tuple
                                action, payload = result
                                reasoning = None
                                raw_llm_response = None
                                parsing_warnings = None
                            # Success - break out of retry loop
                            break
                        except ValueError as e:
                            error_str = str(e)
                            # Check if this is a PROPOSE_TRADE parsing error that should trigger retry
                            if "PROPOSE_TRADE" in error_str or ("propose_trade" in error_str.lower() and ("parse" in error_str.lower() or "format" in error_str.lower() or "invalid" in error_str.lower())):
                                retry_count += 1
                                last_error = e
                                if retry_count < max_retries:
                                    print(f"  PROPOSE_TRADE parsing error (attempt {retry_count}/{max_retries}): {error_str}", flush=True)
                                    print(f"  Retrying...", flush=True)
                                    # Continue to retry
                                    continue
                                else:
                                    # Max retries reached
                                    return self.state, False, f"Agent error for player {current_player.id}: Failed to parse PROPOSE_TRADE after {max_retries} attempts. Last error: {error_str}"
                            else:
                                # Not a PROPOSE_TRADE parsing error - don't retry
                                return self.state, False, f"Agent error for player {current_player.id}: {str(e)}"
                        except Exception as e:
                            # Other exceptions - don't retry
                            return self.state, False, f"Agent error for player {current_player.id}: {str(e)}"
                
                # Print reasoning if available
                if reasoning:
                    print(f"[{current_player.name}] Reasoning: {reasoning}")
                
                # Apply the action
                try:
                    self.state = self.state.step(action, payload, player_id=current_player.id)
                except ValueError as e:
                    return self.state, False, f"Invalid action for player {current_player.id}: {str(e)}"
                
                # Save state if callback provided
                if save_state_callback:
                    action_dict = {
                        "type": serialize_action(action),
                    }
                    if payload:
                        action_dict["payload"] = serialize_action_payload(payload)
                    if reasoning:
                        action_dict["reasoning"] = reasoning
                    # Store raw_llm_response if it was set (even if None, to track that we checked)
                    if 'raw_llm_response' in locals():
                        action_dict["raw_llm_response"] = raw_llm_response
                    save_state_callback(
                        self.state.game_id,
                        state_before,
                        self.state,
                        action_dict,
                        current_player.id
                    )
                
                # Increment turn count if we ended a turn
                if action == Action.END_TURN:
                    self.turn_count += 1
                
                action_count += 1
                
                # Progress callback for verbose output (more frequent)
                if progress_callback and action_count % 20 == 0:
                    progress_callback(self.turn_count, action_count)
            
            # Reached max turns
            return self.state, False, f"Reached maximum turn limit ({self.max_turns})"
            
        except Exception as e:
            return self.state, False, f"Unexpected error: {str(e)}"
    
    def run_step(
        self,
        save_state_callback: Optional[callable] = None
    ) -> Tuple[GameState, bool, Optional[str], Optional[str]]:
        """
        Run a single step (one action) of the game.
        
        Args:
            save_state_callback: Optional callback to save state after action
                                Signature: (game_id: str, state_before: GameState, state_after: GameState, action: dict, player_id: str) -> None
        
        Returns:
            Tuple of (new_state, game_continues, error_message, player_id)
            - game_continues: True if game should continue, False if finished/error
            - error_message: None if no error, otherwise error description
            - player_id: ID of player who took the action
        """
        try:
            # Check if game is finished
            if self.state.phase == "finished":
                return self.state, False, None, None
            
            # Check for victory condition
            for player in self.state.players:
                if player.victory_points >= 10:
                    return self.state, False, None, None
            
            # Get current player
            if self.state.phase == "setup":
                current_player = self.state.players[self.state.setup_phase_player_index]
            elif self.state.phase == "playing":
                current_player = self.state.players[self.state.current_player_index]
            else:
                return self.state, False, None, None
            
            # Handle special case: when 7 is rolled, all players with 8+ resources must discard
            if (self.state.phase == "playing" and 
                self.state.dice_roll == 7 and
                not self.state.waiting_for_robber_move and
                not self.state.waiting_for_robber_steal):
                # Check if we're still in discard phase
                robber_has_been_moved = (
                    self.state.robber_initial_tile_id is not None and 
                    self.state.robber_tile_id != self.state.robber_initial_tile_id
                )
                
                if not robber_has_been_moved:
                    # Check if any player needs to discard
                    players_needing_discard = [
                        p for p in self.state.players
                        if (sum(p.resources.values()) >= 8 and 
                            p.id not in self.state.players_discarded)
                    ]
                    
                    if players_needing_discard:
                        # Process discards for ALL AI players who need to discard (not just current player)
                        # Find first AI player who needs to discard
                        player_to_discard = None
                        for player in players_needing_discard:
                            if self.agents.get(player.id):
                                player_to_discard = player
                                break
                        
                        if player_to_discard:
                            # AI player needs to discard - process it
                            discard_agent = self.agents.get(player_to_discard.id)
                            legal_actions_list = legal_actions(self.state, player_to_discard.id)
                            
                            # Filter to only discard actions
                            discard_actions = [
                                (action, payload) 
                                for action, payload in legal_actions_list
                                if action == Action.DISCARD_RESOURCES
                            ]
                            
                            if discard_actions:
                                # Store state before action
                                state_before = copy.deepcopy(self.state)
                                
                                # Choose a discard action
                                result = discard_agent.choose_action(self.state, discard_actions)
                                if len(result) == 5:
                                    action, payload, reasoning, _, _ = result  # Ignore raw_response and parsing_warnings for discard
                                elif len(result) == 4:
                                    action, payload, reasoning, _ = result
                                elif len(result) == 3:
                                    action, payload, reasoning = result
                                else:
                                    # Backward compatibility
                                    action, payload = result
                                    reasoning = None
                                
                                # Print reasoning if available
                                if reasoning:
                                    print(f"[{player_to_discard.name}] Reasoning: {reasoning}")
                                
                                # Apply the action
                                self.state = self.state.step(action, payload, player_id=player_to_discard.id)
                                
                                # Save state if callback provided
                                if save_state_callback:
                                    action_dict = {
                                        "type": serialize_action(action),
                                    }
                                    if payload:
                                        action_dict["payload"] = serialize_action_payload(payload)
                                    if reasoning:
                                        action_dict["reasoning"] = reasoning
                                    save_state_callback(
                                        self.state.game_id,
                                        state_before,
                                        self.state,
                                        action_dict,
                                        player_to_discard.id
                                    )
                                
                                return self.state, True, None, player_to_discard.id
                            else:
                                return self.state, False, f"No discard actions available for player {player_to_discard.id}", None
                        else:
                            # No AI players need to discard (only human players) - they'll handle it via UI
                            # Check if there are still players who need to discard - if so, don't let current player take turn
                            remaining_players_needing_discard = [
                                p for p in self.state.players
                                if (sum(p.resources.values()) >= 8 and 
                                    p.id not in self.state.players_discarded)
                            ]
                            if remaining_players_needing_discard:
                                # Still waiting for players to discard - don't process current player's turn
                                return self.state, True, None, None
            
            # Check if we're still in discard phase (before processing current player's turn)
            if (self.state.phase == "playing" and 
                self.state.dice_roll == 7 and
                not self.state.waiting_for_robber_move and
                not self.state.waiting_for_robber_steal):
                robber_has_been_moved = (
                    self.state.robber_initial_tile_id is not None and 
                    self.state.robber_tile_id != self.state.robber_initial_tile_id
                )
                if not robber_has_been_moved:
                    # Check if any player still needs to discard
                    players_still_needing_discard = [
                        p for p in self.state.players
                        if (sum(p.resources.values()) >= 8 and 
                            p.id not in self.state.players_discarded)
                    ]
                    if players_still_needing_discard:
                        # Still waiting for discards - don't process current player's turn yet
                        return self.state, True, None, None
            
            # Get agent for current player
            agent = self.agents.get(current_player.id)
            if not agent:
                return self.state, False, f"No agent found for player {current_player.id}", None
            
            # Get legal actions for current player
            legal_actions_list = legal_actions(self.state, current_player.id)
            
            if not legal_actions_list:
                return self.state, False, f"No legal actions available for player {current_player.id}", None
            
            # Store state before action
            state_before = copy.deepcopy(self.state)
            
            # Automate action if there's only one option
            if len(legal_actions_list) == 1:
                action, payload = legal_actions_list[0]
                reasoning = f"Automated: only one legal action available"
            else:
                # Agent chooses an action
                # Retry logic for PROPOSE_TRADE parsing errors
                max_retries = 3
                retry_count = 0
                last_error = None
                
                while retry_count < max_retries:
                    try:
                        result = agent.choose_action(self.state, legal_actions_list)
                        if len(result) == 5:
                            action, payload, reasoning, raw_llm_response, parsing_warnings = result
                        elif len(result) == 4:
                            action, payload, reasoning, raw_llm_response = result
                            parsing_warnings = None
                        elif len(result) == 3:
                            action, payload, reasoning = result
                            raw_llm_response = None
                            parsing_warnings = None
                        else:
                            # Backward compatibility: old agents return 2-tuple
                            action, payload = result
                            reasoning = None
                            raw_llm_response = None
                            parsing_warnings = None
                        # Success - break out of retry loop
                        break
                    except ValueError as e:
                        error_str = str(e)
                        # Check if this is a PROPOSE_TRADE parsing error that should trigger retry
                        if "PROPOSE_TRADE" in error_str or ("propose_trade" in error_str.lower() and ("parse" in error_str.lower() or "format" in error_str.lower() or "invalid" in error_str.lower())):
                            retry_count += 1
                            last_error = e
                            if retry_count < max_retries:
                                print(f"  PROPOSE_TRADE parsing error (attempt {retry_count}/{max_retries}): {error_str}", flush=True)
                                print(f"  Retrying...", flush=True)
                                # Continue to retry
                                continue
                            else:
                                # Max retries reached
                                return self.state, False, f"Agent error for player {current_player.id}: Failed to parse PROPOSE_TRADE after {max_retries} attempts. Last error: {error_str}", None
                        else:
                            # Not a PROPOSE_TRADE parsing error - don't retry
                            return self.state, False, f"Agent error for player {current_player.id}: {str(e)}", None
                    except Exception as e:
                        # Other exceptions - don't retry
                        return self.state, False, f"Agent error for player {current_player.id}: {str(e)}", None
            
            # Print reasoning if available
            if reasoning:
                print(f"[{current_player.name}] Reasoning: {reasoning}")
            
            # Apply the action
            try:
                self.state = self.state.step(action, payload, player_id=current_player.id)
            except ValueError as e:
                return self.state, False, f"Invalid action for player {current_player.id}: {str(e)}", None
            
            # Save state if callback provided
            if save_state_callback:
                action_dict = {
                    "type": serialize_action(action),
                }
                if payload:
                    action_dict["payload"] = serialize_action_payload(payload)
                if reasoning:
                    action_dict["reasoning"] = reasoning
                # Store raw_llm_response if it was set (even if None, to track that we checked)
                if 'raw_llm_response' in locals():
                    action_dict["raw_llm_response"] = raw_llm_response
                # Store parsing_warnings if it was set
                if 'parsing_warnings' in locals() and parsing_warnings:
                    action_dict["parsing_warnings"] = parsing_warnings
                save_state_callback(
                    self.state.game_id,
                    state_before,
                    self.state,
                    action_dict,
                    current_player.id
                )
            
            # Increment turn count if we ended a turn
            if action == Action.END_TURN:
                self.turn_count += 1
            
            return self.state, True, None, current_player.id
            
        except Exception as e:
            return self.state, False, f"Unexpected error: {str(e)}", None

