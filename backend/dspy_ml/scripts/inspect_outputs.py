#!/usr/bin/env python3
"""
Inspect actual outputs from both agents to see if they're different or if it's an evaluation issue.
Compare the raw outputs side-by-side for drills where one is correct and the other is wrong.
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.dataset import DrillDataset
from agents.guideline_cluster_agent import GuidelineClusterAgent
from engine.serialization import legal_actions, deserialize_game_state
from engine import GameState, Action


def main():
    # Load comparison results
    comparison_path = Path(__file__).parent.parent / "data" / "agent_comparison.json"
    with open(comparison_path) as f:
        results = json.load(f)
    
    # Find drills where clustering is correct but frontend is wrong, or vice versa
    clustering_correct_frontend_wrong = [
        r for r in results 
        if r['clustering']['correct'] and not r['frontend']['correct'] and r['frontend']['predicted']
    ]
    frontend_correct_clustering_wrong = [
        r for r in results 
        if not r['clustering']['correct'] and r['frontend']['correct'] and r['clustering']['predicted']
    ]
    
    print(f"Drills where clustering is correct but frontend is wrong: {len(clustering_correct_frontend_wrong)}")
    print(f"Drills where frontend is correct but clustering is wrong: {len(frontend_correct_clustering_wrong)}\n")
    
    # Show examples
    print("=" * 80)
    print("EXAMPLES: Clustering correct, Frontend wrong")
    print("=" * 80)
    for r in clustering_correct_frontend_wrong[:5]:
        print(f"\nDrill {r['drill_id']}:")
        print(f"  Expected: {r['expected']}")
        print(f"  Clustering predicted: {r['clustering']['predicted']}")
        print(f"  Frontend predicted: {r['frontend']['predicted']}")
        print(f"  Clustering reasoning: {r['clustering']['reasoning'][:200]}...")
        print(f"  Frontend reasoning: {r['frontend']['reasoning'][:200]}...")
        if r['frontend']['error']:
            print(f"  Frontend error: {r['frontend']['error']}")
    
    print("\n" + "=" * 80)
    print("EXAMPLES: Frontend correct, Clustering wrong")
    print("=" * 80)
    for r in frontend_correct_clustering_wrong[:5]:
        print(f"\nDrill {r['drill_id']}:")
        print(f"  Expected: {r['expected']}")
        print(f"  Clustering predicted: {r['clustering']['predicted']}")
        print(f"  Frontend predicted: {r['frontend']['predicted']}")
        print(f"  Clustering reasoning: {r['clustering']['reasoning'][:200]}...")
        print(f"  Frontend reasoning: {r['frontend']['reasoning'][:200]}...")
        if r['clustering']['raw']:
            print(f"  Clustering raw: {r['clustering']['raw'][:200]}...")
    
    # Check if outputs are actually the same but evaluation differs
    print("\n" + "=" * 80)
    print("Checking if outputs are the same but evaluation differs")
    print("=" * 80)
    
    same_outputs_different_eval = []
    for r in results:
        if r['clustering']['predicted'] and r['frontend']['predicted']:
            # Normalize for comparison (remove type fields from payload, etc.)
            clustering_pred = r['clustering']['predicted']
            frontend_pred = r['frontend']['predicted']
            
            # Simple comparison
            clustering_type = clustering_pred.get('type')
            frontend_type = frontend_pred.get('type')
            
            clustering_payload = clustering_pred.get('payload', {})
            frontend_payload = frontend_pred.get('payload', {})
            
            # Remove 'type' field from payload if present
            if isinstance(clustering_payload, dict) and 'type' in clustering_payload:
                clustering_payload = {k: v for k, v in clustering_payload.items() if k != 'type'}
            if isinstance(frontend_payload, dict) and 'type' in frontend_payload:
                frontend_payload = {k: v for k, v in frontend_payload.items() if k != 'type'}
            
            if (clustering_type == frontend_type and 
                clustering_payload == frontend_payload and
                r['clustering']['correct'] != r['frontend']['correct']):
                same_outputs_different_eval.append(r)
    
    print(f"\nDrills with same outputs but different evaluation: {len(same_outputs_different_eval)}")
    for r in same_outputs_different_eval[:5]:
        print(f"\nDrill {r['drill_id']}:")
        print(f"  Expected: {r['expected']}")
        print(f"  Both predicted: {r['clustering']['predicted']}")
        print(f"  Clustering correct: {r['clustering']['correct']}")
        print(f"  Frontend correct: {r['frontend']['correct']}")


if __name__ == "__main__":
    main()

