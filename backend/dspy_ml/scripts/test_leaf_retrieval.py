#!/usr/bin/env python3
"""
Test retrieval from 23 leaf cluster centroids during drill evaluation.
This verifies that retrieval works correctly before implementing the agent.
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.signature import CatanDrillSignature
from dspy_ml.dataset import DrillDataset
from engine.serialization import state_to_text, legal_actions_to_text, legal_actions
from api.routes import _canonical_action_dict, _filter_legal_actions


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts using OpenAI embeddings."""
    import openai
    embeddings = []
    for t in texts:
        resp = openai.embeddings.create(model=model, input=t)
        vec = resp.data[0].embedding
        embeddings.append(vec)
    return np.array(embeddings)


def robust_parse(chosen_action_str: str) -> Any:
    """Robustly parse JSON action string."""
    if not chosen_action_str or chosen_action_str == "null":
        return None
    import re
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', chosen_action_str)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "type" not in parsed and ("give_resources" in parsed or "receive_resources" in parsed or "target_player_ids" in parsed or "card_type" in parsed or "year_of_plenty_resources" in parsed or "monopoly_resource_type" in parsed):
                if "card_type" in parsed or "year_of_plenty_resources" in parsed or "monopoly_resource_type" in parsed:
                    return {"type": "play_dev_card", "payload": parsed}
                elif "give_resources" in parsed or "receive_resources" in parsed:
                    if "target_player_ids" in parsed:
                        return {"type": "propose_trade", "payload": parsed}
                    elif "port_intersection_id" in parsed or parsed.get("port_intersection_id") is None:
                        return {"type": "trade_bank", "payload": parsed}
            return parsed
        except:
            for i in range(5):
                try:
                    json_str_clean = json_str.rstrip('}')
                    parsed = json.loads(json_str_clean)
                    if isinstance(parsed, dict) and "type" not in parsed and ("give_resources" in parsed or "receive_resources" in parsed or "target_player_ids" in parsed or "card_type" in parsed):
                        if "card_type" in parsed or "year_of_plenty_resources" in parsed or "monopoly_resource_type" in parsed:
                            return {"type": "play_dev_card", "payload": parsed}
                        elif "give_resources" in parsed or "receive_resources" in parsed:
                            if "target_player_ids" in parsed:
                                return {"type": "propose_trade", "payload": parsed}
                            elif "port_intersection_id" in parsed or parsed.get("port_intersection_id") is None:
                                return {"type": "trade_bank", "payload": parsed}
                    return parsed
                except:
                    json_str = json_str[:-1]
    return None


def load_leaf_clusters(leaf_tree_path: Path) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Load leaf clusters and prepare centroids."""
    with open(leaf_tree_path) as f:
        data = json.load(f)
    leaf_clusters = data.get("clusters", [])
    
    # Load dataset for observation lookup
    dataset_path = leaf_tree_path.parent / "drills_dataset.json"
    dataset = DrillDataset()
    dataset.load_from_json(str(dataset_path))
    drills_by_id = {ex.drill_id: ex for ex in dataset.examples}
    
    # Compute centroids (same as clustering evaluation)
    centroids = []
    for cluster in leaf_clusters:
        drill_ids = cluster.get("drill_ids", [])
        if not drill_ids:
            continue
        
        obs_texts = []
        for drill_id in drill_ids:
            if drill_id in drills_by_id:
                obs = drills_by_id[drill_id].observation
                if obs:
                    obs_texts.append(obs)
        
        if not obs_texts:
            continue
        
        all_obs_embeddings = embed_texts(obs_texts)
        emb = np.mean(all_obs_embeddings, axis=0)
        
        # Get best guideline from candidates
        candidates = cluster.get("candidates", [])
        best_guideline = candidates[0]["guideline"] if candidates else ""
        
        centroids.append({
            "cluster_id": cluster["cluster_id"],
            "guideline": best_guideline,
            "embedding": emb,
        })
    
    return centroids, drills_by_id


def retrieve_guideline_leaf(observation: str, centroids: List[Dict[str, Any]]) -> str:
    """Retrieve guideline from leaf clusters using observation embedding."""
    obs_emb = embed_texts([observation])[0]
    
    best_idx = 0
    best_sim = -1
    for i, c in enumerate(centroids):
        sim = np.dot(obs_emb, c["embedding"]) / (np.linalg.norm(obs_emb) * np.linalg.norm(c["embedding"]))
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    
    return centroids[best_idx]["guideline"]


def evaluate_drill_retrieval(
    drill_id: int,
    ex: Any,
    centroids: List[Dict[str, Any]],
    game_rules: str
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate a single drill using leaf-level retrieval."""
    if not ex.state:
        return False, {"drill_id": drill_id, "error": "No state"}
    
    # Get player_id from state
    player_id = ex.state.players[ex.state.current_player_index].id if ex.state.current_player_index is not None else ex.state.players[ex.state.setup_phase_player_index].id if ex.state.setup_phase_player_index is not None else None
    if not player_id:
        return False, {"drill_id": drill_id, "error": "No player_id"}
    
    # Generate observation and legal actions on the fly
    try:
        observation = state_to_text(ex.state, player_id, exclude_higher_level_features=False)
        legal_actions_list = legal_actions(ex.state, player_id)
    except Exception as e:
        return False, {"drill_id": drill_id, "error": f"Failed to generate observation/actions: {e}"}
    
    if not legal_actions_list:
        return False, {"drill_id": drill_id, "error": "No legal actions"}
    
    # Apply filtering (same as DrillDataset)
    if ex.correct_actions:
        action_dicts_to_include = ex.correct_actions.copy()
        if ex.incorrect_actions:
            action_dicts_to_include.extend(ex.incorrect_actions)
        legal_actions_list = _filter_legal_actions(legal_actions_list, action_dicts_to_include)
        if not legal_actions_list:
            return False, {"drill_id": drill_id, "error": "Filter rejected all legal actions"}
    
    viable_actions = legal_actions_to_text(legal_actions_list, state=ex.state, player_id=player_id)
    
    # Retrieve guideline (using retrieval, not direct lookup)
    guideline = retrieve_guideline_leaf(observation, centroids)
    
    # Use DSPy module (matching clustering evaluation)
    import dspy
    lm = dspy.LM(model="gpt-5.2")
    dspy.configure(lm=lm)
    module = dspy.ChainOfThought(CatanDrillSignature)
    
    # Call module
    result = module(
        game_rules=game_rules,
        observation=observation,
        viable_actions=viable_actions,
        guideline=guideline
    )
    
    # Extract results
    predicted = robust_parse(getattr(result, "chosen_action", "null") or "null")
    
    # Compare with expected
    expected_actions = ex.correct_actions if ex.correct_actions else [ex.expected_action]
    
    matched = False
    pred_canon = _canonical_action_dict(predicted, state=ex.state)
    
    def normalize_setup_build(canon):
        if not isinstance(canon, dict):
            return canon
        t = canon.get("type")
        if t == "build_road":
            canon = canon.copy()
            canon["type"] = "setup_place_road"
        elif t == "build_settlement":
            canon = canon.copy()
            canon["type"] = "setup_place_settlement"
        return canon
    
    for ca in expected_actions:
        ca_canon = _canonical_action_dict(ca, state=ex.state)
        if normalize_setup_build(ca_canon) == normalize_setup_build(pred_canon):
            matched = True
            break
    
    return matched, {
        "drill_id": drill_id,
        "expected": expected_actions[0] if expected_actions else {},
        "predicted": predicted,
        "matched": matched,
        "guideline": guideline[:100] if guideline else "",
    }


def main():
    # Load drills from database
    print("Loading drills from database...", flush=True)
    dataset = DrillDataset()
    dataset.load_from_database()
    
    if not dataset.examples:
        print("No drills loaded from database!", flush=True)
        return
    
    print(f"Loaded {len(dataset.examples)} drills from database", flush=True)
    
    # Load leaf clusters
    leaf_tree_path = Path(__file__).parent.parent / "data" / "guideline_tree.json"
    centroids, _ = load_leaf_clusters(leaf_tree_path)
    
    print(f"Loaded {len(centroids)} leaf cluster centroids", flush=True)
    
    # Get game rules
    game_rules = dataset.examples[0].game_rules if dataset.examples else ""
    
    # Evaluate in parallel
    print(f"Evaluating {len(dataset.examples)} drills with leaf-level retrieval...", flush=True)
    
    results = []
    with ProcessPoolExecutor(max_workers=min(8, len(dataset.examples))) as executor:
        futures = {
            executor.submit(evaluate_drill_retrieval, ex.drill_id, ex, centroids, game_rules): ex
            for ex in dataset.examples
        }
        
        for future in as_completed(futures):
            try:
                matched, details = future.result()
                results.append((matched, details))
            except Exception as e:
                ex = futures[future]
                print(f"Error evaluating drill {ex.drill_id}: {e}", flush=True)
                results.append((False, {"drill_id": ex.drill_id, "error": str(e)}))
    
    # Calculate results
    correct = sum(1 for matched, _ in results if matched)
    total = len(results)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"  Leaf-level retrieval: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*80}")
    
    # Save detailed results
    output_path = Path(__file__).parent.parent / "data" / "leaf_retrieval_evaluation.json"
    with open(output_path, "w") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "results": [{"matched": m, **d} for m, d in results],
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()

