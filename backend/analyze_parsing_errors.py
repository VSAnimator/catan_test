#!/usr/bin/env python3
"""
Analyze parsing errors in game logs.
Find instances where agent reasoning indicates one action but a different action was taken.
"""
import sqlite3
import json
import re
from pathlib import Path

DB_PATH = Path('catan.db')
game_id = '1ec82583-e656-4308-ab2e-62e5fd8f443f'

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all steps with reasoning and actions
cursor.execute('''
    SELECT step_idx, player_id, reasoning, action_json, raw_llm_response
    FROM steps
    WHERE game_id = ?
    ORDER BY step_idx
''', (game_id,))

steps = cursor.fetchall()
print(f"Analyzing {len(steps)} steps for parsing errors...\n")

parsing_errors = []

# Action keywords mapping
action_keywords = {
    'build_settlement': ['build settlement', 'settlement', 'build_settlement', 'setup_place_settlement'],
    'build_city': ['build city', 'city', 'build_city'],
    'build_road': ['build road', 'road', 'build_road', 'setup_place_road'],
    'propose_trade': ['propose trade', 'trade', 'propose_trade'],
    'accept_trade': ['accept trade', 'accept_trade'],
    'reject_trade': ['reject trade', 'reject_trade'],
    'play_dev_card': ['play dev card', 'play_dev_card', 'knight', 'play knight'],
    'move_robber': ['move robber', 'move_robber'],
    'buy_dev_card': ['buy dev card', 'buy_dev_card'],
    'end_turn': ['end turn', 'end_turn']
}

for step in steps:
    step_idx = step['step_idx']
    player_id = step['player_id']
    reasoning = step['reasoning'] or ""
    action_json_str = step['action_json']
    raw_llm = step['raw_llm_response']
    
    if not reasoning or len(reasoning) < 20:  # Skip very short or empty reasoning
        continue
    
    try:
        action_dict = json.loads(action_json_str) if action_json_str else {}
        action_taken = action_dict.get('type', 'unknown')
    except:
        action_taken = 'parse_error'
    
    reasoning_lower = reasoning.lower()
    
    # Find what actions are mentioned in reasoning
    mentioned_actions = []
    for action_type, keywords in action_keywords.items():
        for keyword in keywords:
            if keyword in reasoning_lower:
                mentioned_actions.append(action_type)
                break
    
    # Check if reasoning mentions a different action than what was taken
    # But exclude cases where reasoning explains why NOT to take that action
    if mentioned_actions and action_taken not in mentioned_actions:
        # Check if reasoning clearly indicates one action (not just mentions it)
        reasoning_clear = False
        clear_action = None
        
        for action_type in mentioned_actions:
            # Look for strong positive indicators
            action_pattern = action_pattern = action_type.replace("_", r"[\s_]")
            positive_patterns = [
                r'\b(i will|i should|i must|i\'ll|choose|take|do|pick|select|execute|propose|build|buy|play)\s+' + action_pattern,
                action_pattern + r'\s+(immediately|now|first|next|best|should|will)',
                r'\b(action|choice|decision).*?' + action_pattern + r'.*?(is|will be|should be)',
            ]
            
            # Check for negative indicators (explaining why NOT to do it)
            negative_patterns = [
                r'\b(no|not|cannot|can\'t|don\'t|shouldn\'t|won\'t|isn\'t|aren\'t|missing|unavailable|not available|not present|not listed)\s+.*?' + action_pattern,
                action_pattern + r'.*?\b(not|no|unavailable|missing|cannot)',
            ]
            
            has_positive = any(re.search(pattern, reasoning_lower) for pattern in positive_patterns)
            has_negative = any(re.search(pattern, reasoning_lower) for pattern in negative_patterns)
            
            if has_positive and not has_negative:
                reasoning_clear = True
                clear_action = action_type
                break
        
        if reasoning_clear:
            parsing_errors.append({
                'step_idx': step_idx,
                'player_id': player_id,
                'reasoning': reasoning[:800],
                'action_taken': action_taken,
                'clear_action': clear_action,
                'mentioned_actions': mentioned_actions,
                'raw_llm': raw_llm[:800] if raw_llm else None,
                'action_json': action_json_str[:500] if action_json_str else None
            })
    
    # Also check raw LLM response for mismatches
    if raw_llm:
        try:
            # Try to parse JSON from raw LLM response
            llm_data = json.loads(raw_llm)
            if isinstance(llm_data, dict):
                llm_action = llm_data.get('type') or llm_data.get('action', {}).get('type')
                if llm_action and llm_action != action_taken:
                    parsing_errors.append({
                        'step_idx': step_idx,
                        'player_id': player_id,
                        'reasoning': reasoning[:800] if reasoning else "No reasoning",
                        'action_taken': action_taken,
                        'llm_intended_action': llm_action,
                        'mentioned_actions': mentioned_actions,
                        'raw_llm': raw_llm[:800],
                        'action_json': action_json_str[:500] if action_json_str else None,
                        'error_type': 'llm_response_mismatch'
                    })
        except:
            # If not JSON, look for action mentions in raw text
            raw_lower = raw_llm.lower()
            for action_type, keywords in action_keywords.items():
                for keyword in keywords:
                    if keyword in raw_lower and action_taken != action_type:
                        # Check if it's a clear statement of intent
                        if re.search(r'\b(type|action|choose|select|pick|do|take|execute)\s*[:=]?\s*["\']?' + keyword.replace("_", r"[\s_]"), raw_lower):
                            parsing_errors.append({
                                'step_idx': step_idx,
                                'player_id': player_id,
                                'reasoning': reasoning[:800] if reasoning else "No reasoning",
                                'action_taken': action_taken,
                                'llm_intended_action': action_type,
                                'mentioned_actions': mentioned_actions,
                                'raw_llm': raw_llm[:800],
                                'action_json': action_json_str[:500] if action_json_str else None,
                                'error_type': 'llm_text_mismatch'
                            })
                            break

conn.close()

print(f"Found {len(parsing_errors)} potential parsing errors:\n")
for i, error in enumerate(parsing_errors, 1):
    print(f"{'='*80}")
    print(f"Error #{i}: Step {error['step_idx']}, Player {error['player_id']}")
    print(f"Action Taken: {error['action_taken']}")
    print(f"Clear Action from Reasoning: {error['clear_action']}")
    print(f"All Actions Mentioned: {error['mentioned_actions']}")
    print(f"\nReasoning:\n{error['reasoning']}")
    print(f"\nAction JSON:\n{error['action_json']}")
    if error['raw_llm']:
        print(f"\nRaw LLM Response:\n{error['raw_llm']}")
    print()

