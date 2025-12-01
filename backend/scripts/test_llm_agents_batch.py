#!/usr/bin/env python3
"""
Batch test script to run multiple games with LLM agents in parallel and track parsing errors.
"""
import sys
import os
import multiprocessing
from pathlib import Path

# Load environment variables from ~/.zshrc if OPENAI_API_KEY not set
if 'OPENAI_API_KEY' not in os.environ:
    import subprocess
    try:
        result = subprocess.run(['bash', '-c', 'source ~/.zshrc 2>/dev/null && echo $OPENAI_API_KEY'], 
                              capture_output=True, text=True, shell=False)
        if result.stdout.strip():
            os.environ['OPENAI_API_KEY'] = result.stdout.strip()
    except:
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
import random
from engine import GameState, Player, Action
from agents import LLMAgent
from api.database import init_db, create_game, get_latest_state, flush_all_write_queues
from engine.serialization import deserialize_game_state, legal_actions, serialize_game_state


def get_player_color(index: int) -> str:
    """Get color for player by index."""
    colors = ["#FF0000", "#00AA00", "#2196F3", "#FF8C00"]  # Red, Green, Blue, Yellow-Orange
    return colors[index % len(colors)]


def generate_random_name() -> str:
    """Generate a random player name."""
    adjectives = ["Brave", "Swift", "Clever", "Bold", "Wise", "Fierce", "Noble", "Cunning"]
    nouns = ["Wolf", "Eagle", "Lion", "Bear", "Fox", "Hawk", "Tiger", "Panther"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"


def run_single_llm_game(game_num: int, max_turns: int = 20):
    """Run a single game with 2 LLM agents and track parsing errors."""
    try:
        # Initialize database for this process
        init_db()
        
        # Generate game ID
        game_id = str(uuid.uuid4())
        
        # Create players
        player_names = [generate_random_name() for _ in range(2)]
        players = [
            Player(id=f"player_{i}", name=player_names[i], color=get_player_color(i))
            for i in range(2)
        ]
        
        # Create initial game state
        initial_state = GameState(
            game_id=game_id,
            players=players,
            current_player_index=0,
            phase="setup"
        )
        
        # Initialize the board
        initial_state = initial_state._create_initial_board(initial_state)
        
        # Initialize robber on desert tile
        desert_tile = next((t for t in initial_state.tiles if t.resource_type is None), None)
        if desert_tile:
            initial_state.robber_tile_id = desert_tile.id
        
        # Serialize and save to database
        serialized_state = serialize_game_state(initial_state)
        metadata = {
            "player_names": player_names,
            "num_players": 2,
        }
        create_game(
            game_id,
            rng_seed=None,
            metadata=metadata,
            initial_state_json=serialized_state,
        )
        
        # Get initial state
        state_data = get_latest_state(game_id)
        state = deserialize_game_state(state_data)
        
        # Create LLM agents for both players
        player_ids = [p.id for p in state.players]
        agents = {
            player_ids[0]: LLMAgent(player_ids[0], model="gpt-5.1", enable_retrieval=False),
            player_ids[1]: LLMAgent(player_ids[1], model="gpt-5.1", enable_retrieval=False)
        }
        
        turn_count = 0
        parsing_errors = []
        trade_proposals = []  # Track trade proposals
        
        try:
            while state.phase != "finished" and turn_count < max_turns:
                # Get current player
                if state.phase == "setup":
                    current_player = state.players[state.setup_phase_player_index]
                else:
                    current_player = state.players[state.current_player_index]
                
                # Get legal actions
                legal_actions_list = legal_actions(state, current_player.id)
                
                if not legal_actions_list:
                    break
                
                # Get agent for current player
                agent = agents.get(current_player.id)
                if not agent:
                    break
                
                # Agent chooses action
                try:
                    result = agent.choose_action(state, legal_actions_list)
                    if len(result) == 4:
                        action, payload, reasoning, raw_llm_response = result
                    elif len(result) == 3:
                        action, payload, reasoning = result
                        raw_llm_response = None
                    else:
                        action, payload = result
                        reasoning = None
                        raw_llm_response = None
                    
                    # Check for parsing errors
                    if "Failed to parse" in str(reasoning) or "Error processing" in str(reasoning):
                        legal_action_types = [a.value for a, _ in legal_actions_list]
                        parsing_errors.append({
                            'turn': turn_count,
                            'player': current_player.name,
                            'player_id': current_player.id,
                            'reasoning': reasoning,
                            'action_taken': action.value if action else None,
                            'legal_actions': legal_action_types,
                            'llm_response': agent._last_llm_response if hasattr(agent, '_last_llm_response') else None
                        })
                    
                    # Track trade proposals
                    if action == Action.PROPOSE_TRADE:
                        trade_proposals.append({
                            'turn': turn_count,
                            'player': current_player.name,
                            'player_id': current_player.id,
                            'payload': payload
                        })
                    
                    # Execute action
                    new_state = state.step(action, payload, current_player.id)
                except Exception as e:
                    # Capture exception as parsing error
                    legal_action_types = [a.value for a, _ in legal_actions_list]
                    parsing_errors.append({
                        'turn': turn_count,
                        'player': current_player.name,
                        'player_id': current_player.id,
                        'reasoning': f"Exception: {str(e)}",
                        'action_taken': None,
                        'legal_actions': legal_action_types,
                        'llm_response': agent._last_llm_response if hasattr(agent, '_last_llm_response') else None
                    })
                    break
                
                state = new_state
                
                # Save state to database
                from api.database import add_step, get_step_count
                from engine.serialization import serialize_action, serialize_action_payload
                import json
                state_after_json = serialize_game_state(state)
                action_json = {"type": action.value}
                if payload:
                    payload_dict = serialize_action_payload(payload)
                    action_json['payload'] = payload_dict
                step_idx = get_step_count(game_id)
                add_step(
                    game_id=game_id,
                    step_idx=step_idx,
                    player_id=current_player.id,
                    state_before_json=state_data,
                    state_after_json=state_after_json,
                    action_json=action_json,
                    reasoning=reasoning,
                    batch_write=True
                )
                
                # Update state_data for next iteration
                state_data = get_latest_state(game_id)
                
                turn_count += 1
                
                # Check for victory
                if state.phase == "finished":
                    break
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            parsing_errors.append({
                'turn': turn_count,
                'player': 'unknown',
                'player_id': 'unknown',
                'reasoning': f"Game error: {str(e)}",
                'action_taken': None,
                'legal_actions': [],
                'llm_response': None
            })
        
        # Flush write queues
        flush_all_write_queues()
        
        return {
            'game_num': game_num,
            'game_id': game_id,
            'turns_played': turn_count,
            'phase': state.phase,
            'parsing_errors': parsing_errors,
            'trade_proposals': trade_proposals,
            'winner': max(state.players, key=lambda p: p.victory_points).name if state.phase == "finished" else None
        }
    
    except Exception as e:
        return {
            'game_num': game_num,
            'game_id': None,
            'turns_played': 0,
            'phase': 'error',
            'parsing_errors': [{'reasoning': f"Fatal error: {str(e)}"}],
            'winner': None
        }


def main():
    """Run multiple LLM agent games in parallel."""
    num_games = 8
    max_turns = 100
    num_workers = 8
    
    print(f"Running {num_games} LLM agent games in parallel ({num_workers} workers)")
    print(f"Max turns per game: {max_turns}")
    print("="*60)
    
    # Initialize database in main process
    init_db()
    
    # Run games in parallel
    with multiprocessing.Pool(processes=num_workers, initializer=init_db) as pool:
        results = pool.starmap(run_single_llm_game, [(i, max_turns) for i in range(num_games)])
    
    # Flush all write queues
    flush_all_write_queues()
    
    # Analyze results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    total_errors = 0
    games_with_errors = 0
    total_trade_proposals = 0
    
    for result in results:
        game_errors = len(result['parsing_errors'])
        game_trades = len(result.get('trade_proposals', []))
        total_errors += game_errors
        total_trade_proposals += game_trades
        
        if game_errors > 0:
            games_with_errors += 1
            print(f"\nGame {result['game_num']} ({result['game_id']}):")
            print(f"  Turns: {result['turns_played']}, Phase: {result['phase']}")
            print(f"  ⚠️  {game_errors} parsing error(s):")
            for i, error in enumerate(result['parsing_errors'], 1):
                print(f"    Error {i}:")
                print(f"      Turn: {error.get('turn', 'unknown')}")
                print(f"      Player: {error.get('player', 'unknown')}")
                print(f"      Action: {error.get('action_taken', 'none')}")
                print(f"      Legal actions: {error.get('legal_actions', [])}")
                print(f"      Reasoning: {error.get('reasoning', '')[:150]}...")
        else:
            trade_info = f", {game_trades} trade proposal(s)" if game_trades > 0 else ""
            print(f"Game {result['game_num']}: ✅ No errors ({result['turns_played']} turns, {result['phase']}{trade_info})")
    
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total games: {num_games}")
    print(f"Games with errors: {games_with_errors}")
    print(f"Total parsing errors: {total_errors}")
    print(f"Total trade proposals: {total_trade_proposals}")
    print(f"Success rate: {(num_games - games_with_errors) / num_games * 100:.1f}%")
    
    if total_trade_proposals > 0:
        print("\n" + "="*60)
        print("TRADE PROPOSAL ANALYSIS")
        print("="*60)
        for result in results:
            if result.get('trade_proposals'):
                print(f"\nGame {result['game_num']} ({result['game_id']}):")
                for i, trade in enumerate(result['trade_proposals'], 1):
                    give_str = ", ".join([f"{count} {rt.value}" for rt, count in trade['payload'].give_resources.items()])
                    receive_str = ", ".join([f"{count} {rt.value}" for rt, count in trade['payload'].receive_resources.items()])
                    print(f"  Trade {i} (Turn {trade['turn']}, {trade['player']}): Give {give_str}, Receive {receive_str}")
                    print(f"    Targets: {', '.join(trade['payload'].target_player_ids)}")
    else:
        print("\n⚠️  WARNING: No trade proposals found in any game!")
    
    if total_errors > 0:
        print("\n" + "="*60)
        print("DETAILED ERROR ANALYSIS")
        print("="*60)
        for result in results:
            if result['parsing_errors']:
                print(f"\nGame {result['game_num']} ({result['game_id']}):")
                for i, error in enumerate(result['parsing_errors'], 1):
                    print(f"\n  Error {i}:")
                    print(f"    Turn: {error.get('turn', 'unknown')}")
                    print(f"    Player: {error.get('player', 'unknown')} ({error.get('player_id', 'unknown')})")
                    print(f"    Action taken: {error.get('action_taken', 'none')}")
                    print(f"    Legal actions: {error.get('legal_actions', [])}")
                    print(f"    Reasoning: {error.get('reasoning', '')}")
                    if error.get('llm_response'):
                        print(f"    LLM response: {error['llm_response'][:500]}...")


if __name__ == "__main__":
    main()
