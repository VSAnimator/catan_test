#!/usr/bin/env python3
"""
Test using leaf-level cluster guidelines instead of meta-guidelines.
This should achieve very high performance (most clusters at 100%).
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.signature import CatanDrillSignature
from dspy_ml.dataset import DrillDataset
from api.routes import _canonical_action_dict
import litellm


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
    # Try to extract JSON from the string
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', chosen_action_str)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed = json.loads(json_str)
            # Check if it's missing type/payload wrapper (just payload contents)
            if isinstance(parsed, dict) and "type" not in parsed and ("give_resources" in parsed or "receive_resources" in parsed or "target_player_ids" in parsed or "card_type" in parsed or "year_of_plenty_resources" in parsed or "monopoly_resource_type" in parsed):
                # This looks like a payload without wrapper - try to infer the type
                if "card_type" in parsed or "year_of_plenty_resources" in parsed or "monopoly_resource_type" in parsed:
                    return {"type": "play_dev_card", "payload": parsed}
                elif "give_resources" in parsed or "receive_resources" in parsed:
                    if "target_player_ids" in parsed:
                        return {"type": "propose_trade", "payload": parsed}
                    elif "port_intersection_id" in parsed or parsed.get("port_intersection_id") is None:
                        return {"type": "trade_bank", "payload": parsed}
            return parsed
        except:
            # Try removing extra closing braces
            for i in range(5):
                try:
                    json_str_clean = json_str.rstrip('}')
                    parsed = json.loads(json_str_clean)
                    # Same check for missing wrapper
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


def evaluate_drill_leaf(
    drill: Dict[str, Any],
    centroids: List[Dict[str, Any]],
    game_rules: str,
    model: str = "gpt-5.2",
    best_guidelines: Optional[Dict[str, str]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate a single drill using leaf-level guidelines."""
    # With ProcessPoolExecutor, each process can have its own DSPy configuration
    import dspy
    
    observation = drill["observation"]
    viable_actions = drill["viable_actions"]
    drill_id = drill["drill_id"]
    
    # Use best guideline per drill if available, otherwise retrieve
    if best_guidelines and str(drill_id) in best_guidelines:
        guideline = best_guidelines[str(drill_id)]
    else:
        guideline = retrieve_guideline_leaf(observation, centroids)
    
    # Create a NEW DSPy module for each evaluation (matching clustering script)
    lm = dspy.LM(model=model)
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
    
    # Compare with expected (use state for phase-aware comparison)
    expected_actions = drill.get("correct_actions", [])
    if not expected_actions:
        expected_actions = [drill.get("expected_action", {})]
    
    # Get state for phase-aware comparison
    state = drill.get("state")
    
    matched = False
    pred_canon = _canonical_action_dict(predicted, state=state)
    
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
        ca_canon = _canonical_action_dict(ca, state=state)
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
    # Load drills from database (generating states on the fly)
    from dspy_ml.dataset import DrillDataset
    from engine.serialization import state_to_text, legal_actions_to_text, legal_actions
    from agents.llm_agent import LLMAgent
    
    print("Loading drills from database...", flush=True)
    dataset = DrillDataset()
    dataset.load_from_database()
    
    if not dataset.examples:
        print("No drills loaded from database!", flush=True)
        return
    
    print(f"Loaded {len(dataset.examples)} drills from database", flush=True)
    
    # Get game rules (from first example)
    game_rules = dataset.examples[0].game_rules if dataset.examples else ""
    
    # Load leaf clusters (for retrieval fallback)
    leaf_tree_path = Path(__file__).parent.parent / "data" / "guideline_tree.json"
    centroids, _ = load_leaf_clusters(leaf_tree_path)
    
    # Load best guidelines per drill (if available)
    best_guidelines_path = Path(__file__).parent.parent / "data" / "best_guidelines_leaf.json"
    best_guidelines = None
    if best_guidelines_path.exists():
        with open(best_guidelines_path) as f:
            best_guidelines = json.load(f)
        print(f"Loaded best leaf guidelines for {len(best_guidelines)} drills", flush=True)
    else:
        print("Warning: best_guidelines_leaf.json not found, will use retrieval", flush=True)
    
    # Prepare drills - generate observations and viable_actions on the fly from GameState
    # IMPORTANT: Use the same filtering logic as DrillDataset to restrict action space
    from api.routes import _filter_legal_actions
    from engine.serialization import serialize_action, serialize_action_payload
    
    drills = []
    for ex in dataset.examples:
        if not ex.state:
            continue
        
        # Get player_id from state
        player_id = ex.state.players[ex.state.current_player_index].id if ex.state.current_player_index is not None else ex.state.players[ex.state.setup_phase_player_index].id if ex.state.setup_phase_player_index is not None else None
        if not player_id:
            continue
        
        # Generate observation and viable_actions on the fly (matching real gameplay)
        try:
            observation = state_to_text(ex.state, player_id, exclude_higher_level_features=False)
            legal_actions_list = legal_actions(ex.state, player_id)
            
            # Apply same filtering as DrillDataset (restrict to correct + incorrect actions)
            if ex.correct_actions:
                # Build action_dicts_to_include (correct + incorrect)
                action_dicts_to_include = ex.correct_actions.copy()
                if ex.incorrect_actions:
                    action_dicts_to_include.extend(ex.incorrect_actions)
                
                # Filter legal actions to match the restricted space
                legal_actions_list = _filter_legal_actions(legal_actions_list, action_dicts_to_include)
                
                if not legal_actions_list:
                    print(f"Warning: Filter rejected all legal actions for drill {ex.drill_id}, skipping", flush=True)
                    continue
            
            viable_actions = legal_actions_to_text(legal_actions_list, state=ex.state, player_id=player_id)
        except Exception as e:
            print(f"Warning: Failed to generate observation/actions for drill {ex.drill_id}: {e}", flush=True)
            continue
        
        drills.append({
            "drill_id": ex.drill_id,
            "game_rules": ex.game_rules,
            "observation": observation,  # Generated on the fly
            "viable_actions": viable_actions,  # Generated on the fly (with filtering applied)
            "expected_action": ex.expected_action,
            "correct_actions": ex.correct_actions,
            "state": ex.state,  # Keep state for phase-aware comparison
        })
    
    print(f"Evaluating {len(drills)} drills with leaf-level guidelines...", flush=True)
    
    # Evaluate in parallel
    results = []
    with ProcessPoolExecutor(max_workers=min(8, len(drills))) as executor:
        futures = {
            executor.submit(evaluate_drill_leaf, drill, centroids, game_rules, "gpt-5.2", best_guidelines): drill
            for drill in drills
        }
        
        for future in as_completed(futures):
            try:
                matched, details = future.result()
                results.append((matched, details))
            except Exception as e:
                drill = futures[future]
                print(f"Error evaluating drill {drill['drill_id']}: {e}", flush=True)
                results.append((False, {"drill_id": drill["drill_id"], "error": str(e)}))
    
    # Calculate results
    correct = sum(1 for matched, _ in results if matched)
    total = len(results)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"  Leaf-level guidelines: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*80}")
    
    # Group by cluster to see per-cluster performance
    leaf_tree_path = Path(__file__).parent.parent / "data" / "guideline_tree.json"
    with open(leaf_tree_path) as f:
        leaf_data = json.load(f)
    
    # Build drill_id -> cluster_id mapping
    drill_to_cluster = {}
    for cluster in leaf_data.get("clusters", []):
        for drill_id in cluster.get("drill_ids", []):
            drill_to_cluster[drill_id] = cluster["cluster_id"]
    
    # Group results by cluster
    cluster_results = {}
    for matched, details in results:
        drill_id = details.get("drill_id")
        if drill_id is None:
            continue
        cluster_id = drill_to_cluster.get(drill_id)
        if cluster_id is None:
            continue
        if cluster_id not in cluster_results:
            cluster_results[cluster_id] = []
        cluster_results[cluster_id].append(matched)
    
    # Print per-cluster performance
    print(f"\nPer-cluster performance:")
    print(f"Cluster | Size | Correct | Score")
    print(f"{'-'*50}")
    for cluster_id in sorted(cluster_results.keys()):
        size = len(cluster_results[cluster_id])
        correct = sum(cluster_results[cluster_id])
        score = correct / size if size > 0 else 0.0
        print(f"  {cluster_id:2d}   | {size:4d} | {correct:3d}/{size:2d} | {score:.3f}")
    
    # Save detailed results
    output_path = Path(__file__).parent.parent / "data" / "leaf_guidelines_evaluation.json"
    with open(output_path, "w") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "results": [{"matched": m, **d} for m, d in results],
            "per_cluster": {
                str(cid): {
                    "size": len(vals),
                    "correct": sum(vals),
                    "score": sum(vals) / len(vals) if vals else 0.0
                }
                for cid, vals in cluster_results.items()
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()

