#!/usr/bin/env python3
"""
Test script to run a game with 2 LLM agents and track token usage.
"""
import sys
import os
import asyncio
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
from engine import GameState, Player
from agents import LLMAgent
from api.database import init_db, create_game, get_latest_state
from engine.serialization import deserialize_game_state, legal_actions, serialize_game_state


def get_player_color(index: int) -> str:
    """Get color for player by index."""
    colors = ["#FF0000", "#00AA00", "#2196F3", "#F5F5F5"]  # Red, Green, Blue, White
    return colors[index % len(colors)]


def generate_random_name() -> str:
    """Generate a random player name."""
    adjectives = ["Brave", "Swift", "Clever", "Bold", "Wise", "Fierce", "Noble", "Cunning"]
    nouns = ["Wolf", "Eagle", "Lion", "Bear", "Fox", "Hawk", "Tiger", "Panther"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"


def run_llm_game():
    """Run a single game with 2 LLM agents and track token usage."""
    # Initialize database
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
    
    print(f"Created game: {game_id}", flush=True)
    
    # Get initial state
    state_data = get_latest_state(game_id)
    state = deserialize_game_state(state_data)
    
    # Create LLM agents for both players
    print("Creating LLM agents...", flush=True)
    player_ids = [p.id for p in state.players]
    agents = {
        player_ids[0]: LLMAgent(player_ids[0], model="gpt-5.1", enable_retrieval=False),
        player_ids[1]: LLMAgent(player_ids[1], model="gpt-5.1", enable_retrieval=False)
    }
    
    print(f"Player 0: {state.players[0].name} ({player_ids[0]})", flush=True)
    print(f"Player 1: {state.players[1].name} ({player_ids[1]})", flush=True)
    print("\nStarting game...\n", flush=True)
    
    max_turns = 20  # Run for 20 turns
    turn_count = 0
    
    # Track token usage per agent
    all_usage = {pid: [] for pid in player_ids}
    
    # Track parsing errors and LLM responses
    parsing_errors = []
    llm_responses = []
    
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
                print(f"No legal actions for {current_player.name} - game may be stuck")
                break
            
            # Get agent for current player
            agent = agents.get(current_player.id)
            if not agent:
                print(f"No agent for {current_player.id}")
                break
            
            # Agent chooses action
            print(f"Turn {turn_count}: {current_player.name}'s turn", flush=True)
            try:
                print(f"  Calling LLM agent...", flush=True)
                
                # Capture LLM response for analysis
                import io
                import contextlib
                captured_output = io.StringIO()
                with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
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
                
                # Check if there were any warnings in the output
                output_text = captured_output.getvalue()
                if "Warning:" in output_text or "Failed to parse" in str(reasoning) or "Error processing" in str(reasoning):
                    # Get legal actions for this turn
                    legal_action_types = [a.value for a, _ in legal_actions_list]
                    parsing_errors.append({
                        'turn': turn_count,
                        'player': current_player.name,
                        'player_id': current_player.id,
                        'output': output_text,
                        'reasoning': reasoning,
                        'action_taken': action.value if action else None,
                        'legal_actions': legal_action_types,
                        'llm_response': agent._last_llm_response if hasattr(agent, '_last_llm_response') else None
                    })
                    print(f"  ⚠️  PARSING WARNING DETECTED", flush=True)
                
                # Store LLM response if available
                if hasattr(agent, '_last_llm_response'):
                    llm_responses.append({
                        'turn': turn_count,
                        'player': current_player.name,
                        'response': agent._last_llm_response
                    })
                
                # Debug: check what we got
                from engine import Action
                print(f"  Action type: {type(action)}, value: {action}", flush=True)
                if not isinstance(action, Action):
                    print(f"  ERROR: action is not an Action enum! Got: {action} (type: {type(action)})", flush=True)
                    raise ValueError(f"Invalid action: {action}")
                
                # Track token usage from this turn
                if agent.token_usage_history:
                    latest_usage = agent.token_usage_history[-1]
                    all_usage[current_player.id].append(latest_usage)
                    print(f"  Token usage: {latest_usage['prompt_tokens']} prompt + {latest_usage['completion_tokens']} completion = {latest_usage['total_tokens']} total", flush=True)
                    if reasoning:
                        print(f"  Reasoning: {reasoning[:100]}...", flush=True)
                
                # Execute action (step signature: action, payload, player_id)
                new_state = state.step(action, payload, current_player.id)
            except Exception as e:
                print(f"  Error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Don't break - continue to next turn to see if we can recover
                # But skip saving this failed action
                turn_count += 1
                continue
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
                batch_write=False
            )
            
            # Update state_data for next iteration
            state_data = get_latest_state(game_id)
            
            turn_count += 1
            
            # Check for victory
            if state.phase == "finished":
                winner = max(state.players, key=lambda p: p.victory_points)
                print(f"\nGame finished! Winner: {winner.name} with {winner.victory_points} VPs")
                break
            
            if turn_count % 10 == 0:
                print(f"  Progress: Turn {turn_count}, Phase: {state.phase}", flush=True)
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"\nError during game: {e}")
        import traceback
        traceback.print_exc()
    
    # Print token usage statistics
    print("\n" + "="*60)
    print("TOKEN USAGE STATISTICS")
    print("="*60)
    
    for player_id in player_ids:
        player_name = next((p.name for p in state.players if p.id == player_id), player_id)
        usage_list = all_usage[player_id]
        
        if not usage_list:
            print(f"\n{player_name} ({player_id}): No token usage recorded")
            continue
        
        prompt_tokens = [u['prompt_tokens'] for u in usage_list]
        completion_tokens = [u['completion_tokens'] for u in usage_list]
        total_tokens = [u['total_tokens'] for u in usage_list]
        
        print(f"\n{player_name} ({player_id}):")
        print(f"  Total API calls: {len(usage_list)}")
        print(f"  Prompt tokens:")
        print(f"    Min: {min(prompt_tokens)}, Max: {max(prompt_tokens)}, Avg: {sum(prompt_tokens)/len(prompt_tokens):.1f}, Total: {sum(prompt_tokens)}")
        print(f"  Completion tokens:")
        print(f"    Min: {min(completion_tokens)}, Max: {max(completion_tokens)}, Avg: {sum(completion_tokens)/len(completion_tokens):.1f}, Total: {sum(completion_tokens)}")
        print(f"  Total tokens:")
        print(f"    Min: {min(total_tokens)}, Max: {max(total_tokens)}, Avg: {sum(total_tokens)/len(total_tokens):.1f}, Total: {sum(total_tokens)}")
    
    # Overall statistics
    all_prompt = [u['prompt_tokens'] for usage_list in all_usage.values() for u in usage_list]
    all_completion = [u['completion_tokens'] for usage_list in all_usage.values() for u in usage_list]
    all_total = [u['total_tokens'] for usage_list in all_usage.values() for u in usage_list]
    
    if all_total:
        print(f"\nOVERALL (both agents):")
        print(f"  Total API calls: {len(all_total)}")
        print(f"  Prompt tokens: Min={min(all_prompt)}, Max={max(all_prompt)}, Total={sum(all_prompt)}")
        print(f"  Completion tokens: Min={min(all_completion)}, Max={max(all_completion)}, Total={sum(all_completion)}")
        print(f"  Total tokens: Min={min(all_total)}, Max={max(all_total)}, Total={sum(all_total)}")
    
    print(f"\nGame ID: {game_id}")
    print(f"Turns played: {turn_count}")
    
    # Analyze parsing errors
    if parsing_errors:
        print("\n" + "="*60)
        print("PARSING ERROR ANALYSIS")
        print("="*60)
        print(f"Total parsing errors/warnings: {len(parsing_errors)}")
        for i, error in enumerate(parsing_errors, 1):
            print(f"\nError {i}:")
            print(f"  Turn: {error['turn']}")
            print(f"  Player: {error['player']} ({error['player_id']})")
            print(f"  Action taken: {error['action_taken']}")
            print(f"  Legal actions available: {error.get('legal_actions', [])}")
            print(f"  Reasoning: {error['reasoning'][:200] if error['reasoning'] else 'None'}...")
            print(f"  Output: {error['output'][:800] if error['output'] else 'None'}...")
            if error.get('llm_response'):
                print(f"  Full LLM response: {error['llm_response'][:1000]}...")
    else:
        print("\n✅ No parsing errors detected!")
    
    # Show sample LLM responses
    if llm_responses:
        print("\n" + "="*60)
        print("SAMPLE LLM RESPONSES")
        print("="*60)
        for i, resp in enumerate(llm_responses[:5], 1):  # Show first 5
            print(f"\nResponse {i} (Turn {resp['turn']}, {resp['player']}):")
            print(f"  {resp['response'][:300]}...")
    
    return game_id


if __name__ == "__main__":
    game_id = run_llm_game()

