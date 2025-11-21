#!/usr/bin/env python3
"""
Query and analyze game events across multiple games.

Examples:
  # Find all monopoly card plays
  python -m scripts.query_game_events --action PLAY_DEV_CARD --card-type monopoly --num-games 1000

  # Find all 7-rolls and see what happened
  python -m scripts.query_game_events --action ROLL_DICE --dice-roll 7 --num-games 100

  # Find all city builds
  python -m scripts.query_game_events --action BUILD_CITY --num-games 500
"""
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import get_db_connection, get_steps
from engine import deserialize_game_state, Action
from engine.serialization import serialize_action


@dataclass
class GameEvent:
    """Represents a game event with context."""
    game_id: str
    step_idx: int
    player_id: str
    action_type: str
    action_payload: Optional[Dict]
    state_before: Dict
    state_after: Dict
    timestamp: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class GameEventQuery:
    """Query game events across multiple games."""
    
    def __init__(self):
        self.events: List[GameEvent] = []
    
    def query_games(
        self,
        num_games: int = 100,
        action_type: Optional[str] = None,
        card_type: Optional[str] = None,
        dice_roll: Optional[int] = None,
        player_id: Optional[str] = None,
        min_turn: Optional[int] = None,
        max_turn: Optional[int] = None,
    ) -> List[GameEvent]:
        """Query events across multiple games."""
        self.events = []
        
        # Get recent games
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM games ORDER BY rowid DESC LIMIT ?", (num_games,))
        game_ids = [row[0] for row in cursor.fetchall()]
        
        print(f"Querying {len(game_ids)} games for events...")
        
        for i, game_id in enumerate(game_ids, 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(game_ids)} games...", flush=True)
            
            steps = get_steps(game_id)
            for step in steps:
                action_json = json.loads(step['action_json'])
                action_type_str = action_json.get('type', '')
                
                # Normalize action type for comparison (database stores lowercase)
                action_type_normalized = action_type.lower() if action_type else None
                action_type_str_normalized = action_type_str.lower()
                
                # Filter by action type (case-insensitive)
                if action_type_normalized and action_type_str_normalized != action_type_normalized:
                    continue
                
                # Filter by card type (for play_dev_card)
                if card_type and action_type_str_normalized == 'play_dev_card':
                    payload = action_json.get('payload', {})
                    if payload and payload.get('card_type') != card_type:
                        continue
                
                # Filter by dice roll
                if dice_roll is not None:
                    state_after_json = json.loads(step['state_after_json'])
                    state_after = deserialize_game_state(state_after_json)
                    if state_after.dice_roll != dice_roll:
                        continue
                
                # Filter by player
                if player_id and step['player_id'] != player_id:
                    continue
                
                # Filter by turn number
                state_after_json = json.loads(step['state_after_json'])
                state_after = deserialize_game_state(state_after_json)
                if min_turn is not None and state_after.turn_number < min_turn:
                    continue
                if max_turn is not None and state_after.turn_number > max_turn:
                    continue
                
                # This event matches our query
                event = GameEvent(
                    game_id=game_id,
                    step_idx=step['step_idx'],
                    player_id=step['player_id'],
                    action_type=action_type_str,
                    action_payload=action_json.get('payload'),
                    state_before=json.loads(step['state_before_json']),
                    state_after=json.loads(step['state_after_json']),
                    timestamp=step['timestamp'] if 'timestamp' in step.keys() else None,
                )
                self.events.append(event)
        
        print(f"Found {len(self.events)} matching events")
        return self.events
    
    def analyze_monopoly_card(self) -> Dict[str, Any]:
        """Analyze monopoly card plays to verify correctness."""
        monopoly_events = [e for e in self.events if e.action_type.lower() == 'play_dev_card' and 
                          e.action_payload and e.action_payload.get('card_type') == 'monopoly']
        
        if not monopoly_events:
            return {"error": "No monopoly card events found"}
        
        results = {
            "total_plays": len(monopoly_events),
            "correct": 0,
            "incorrect": 0,
            "issues": []
        }
        
        for event in monopoly_events:
            resource_type = event.action_payload.get('monopoly_resource_type')
            if not resource_type:
                results["incorrect"] += 1
                results["issues"].append({
                    "game_id": event.game_id,
                    "step_idx": event.step_idx,
                    "issue": "No resource type specified in monopoly payload"
                })
                continue
            
            # Get player who played the card
            player_id = event.player_id
            
            # Deserialize states
            state_before = deserialize_game_state(event.state_before)
            state_after = deserialize_game_state(event.state_after)
            
            # Find the player
            player_before = next(p for p in state_before.players if p.id == player_id)
            player_after = next(p for p in state_after.players if p.id == player_id)
            
            # Calculate expected resource gain
            from engine import ResourceType
            resource_enum = ResourceType(resource_type)
            expected_gain = 0
            for other_player in state_before.players:
                if other_player.id != player_id:
                    expected_gain += other_player.resources.get(resource_enum, 0)
            
            # Calculate actual resource gain
            actual_gain = (player_after.resources.get(resource_enum, 0) - 
                          player_before.resources.get(resource_enum, 0))
            
            # Verify all other players lost their resources
            all_others_lost = True
            for other_player_before in state_before.players:
                if other_player_before.id != player_id:
                    other_player_after = next(p for p in state_after.players if p.id == other_player_before.id)
                    before_count = other_player_before.resources.get(resource_enum, 0)
                    after_count = other_player_after.resources.get(resource_enum, 0)
                    if after_count != 0:
                        all_others_lost = False
                        results["issues"].append({
                            "game_id": event.game_id,
                            "step_idx": event.step_idx,
                            "issue": f"Player {other_player_before.id} still has {after_count} {resource_type} after monopoly"
                        })
            
            # Check if gain matches expected
            if actual_gain == expected_gain and all_others_lost:
                results["correct"] += 1
            else:
                results["incorrect"] += 1
                if actual_gain != expected_gain:
                    results["issues"].append({
                        "game_id": event.game_id,
                        "step_idx": event.step_idx,
                        "issue": f"Expected {expected_gain} {resource_type}, got {actual_gain}"
                    })
        
        return results
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the events."""
        if not self.events:
            return {"error": "No events found"}
        
        summary = {
            "total_events": len(self.events),
            "unique_games": len(set(e.game_id for e in self.events)),
            "action_types": defaultdict(int),
            "players": defaultdict(int),
            "turn_distribution": defaultdict(int),
        }
        
        for event in self.events:
            summary["action_types"][event.action_type] += 1
            summary["players"][event.player_id] += 1
            
            state_after = deserialize_game_state(event.state_after)
            turn_bucket = (state_after.turn_number // 10) * 10  # Bucket by 10s
            summary["turn_distribution"][turn_bucket] += 1
        
        # Convert defaultdicts to regular dicts
        summary["action_types"] = dict(summary["action_types"])
        summary["players"] = dict(summary["players"])
        summary["turn_distribution"] = dict(summary["turn_distribution"])
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Query and analyze game events")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to search")
    parser.add_argument("--action", type=str, help="Action type to filter (e.g., PLAY_DEV_CARD, BUILD_CITY)")
    parser.add_argument("--card-type", type=str, help="Card type for PLAY_DEV_CARD (e.g., monopoly, knight)")
    parser.add_argument("--dice-roll", type=int, help="Dice roll value to filter")
    parser.add_argument("--player-id", type=str, help="Player ID to filter")
    parser.add_argument("--min-turn", type=int, help="Minimum turn number")
    parser.add_argument("--max-turn", type=int, help="Maximum turn number")
    parser.add_argument("--analyze", type=str, choices=["monopoly"], help="Run specific analysis")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--limit", type=int, help="Limit number of events to return")
    
    args = parser.parse_args()
    
    query = GameEventQuery()
    events = query.query_games(
        num_games=args.num_games,
        action_type=args.action,
        card_type=args.card_type,
        dice_roll=args.dice_roll,
        player_id=args.player_id,
        min_turn=args.min_turn,
        max_turn=args.max_turn,
    )
    
    if args.limit:
        events = events[:args.limit]
    
    # Print summary
    summary = query.get_event_summary()
    print("\n" + "=" * 80)
    print("=== Event Summary ===")
    print(f"Total events: {summary.get('total_events', 0)}")
    print(f"Unique games: {summary.get('unique_games', 0)}")
    print(f"\nAction types:")
    for action, count in sorted(summary.get('action_types', {}).items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")
    print(f"\nTurn distribution:")
    for turn, count in sorted(summary.get('turn_distribution', {}).items()):
        print(f"  Turn {turn}-{turn+9}: {count} events")
    
    # Run analysis if requested
    if args.analyze == "monopoly":
        print("\n" + "=" * 80)
        print("=== Monopoly Card Analysis ===")
        analysis = query.analyze_monopoly_card()
        print(f"Total plays: {analysis.get('total_plays', 0)}")
        print(f"Correct: {analysis.get('correct', 0)}")
        print(f"Incorrect: {analysis.get('incorrect', 0)}")
        if analysis.get('issues'):
            print(f"\nIssues found ({len(analysis['issues'])}):")
            for issue in analysis['issues'][:10]:  # Show first 10
                print(f"  Game {issue['game_id'][:8]}... Step {issue['step_idx']}: {issue['issue']}")
            if len(analysis['issues']) > 10:
                print(f"  ... and {len(analysis['issues']) - 10} more issues")
    
    # Output to file if requested
    if args.output:
        output_data = {
            "summary": summary,
            "events": [e.to_dict() for e in events],
        }
        if args.analyze:
            output_data["analysis"] = query.analyze_monopoly_card() if args.analyze == "monopoly" else {}
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Show sample events
    if events:
        print("\n" + "=" * 80)
        print("=== Sample Events (first 5) ===")
        for event in events[:5]:
            print(f"\nGame {event.game_id[:8]}... Step {event.step_idx}:")
            print(f"  Player: {event.player_id}")
            print(f"  Action: {event.action_type}")
            if event.action_payload:
                print(f"  Payload: {event.action_payload}")
            state_after = deserialize_game_state(event.state_after)
            print(f"  Turn: {state_after.turn_number}")
            print(f"  Phase: {state_after.phase}")


if __name__ == "__main__":
    main()

