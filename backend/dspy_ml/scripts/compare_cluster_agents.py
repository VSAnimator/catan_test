#!/usr/bin/env python3
"""
Compare the clustering evaluation agent (from cluster_guideline_tree_meta.py)
with the frontend/backend GuidelineClusterAgent to identify discrepancies.
Runs both in parallel and shows side-by-side outputs.
"""
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import litellm
import re
from dspy_ml.signature import CatanDrillSignature
from dspy_ml.dataset import DrillDataset
from dspy_ml.dspy_prompt_template import format_prompt
from api.routes import _canonical_action_dict
from engine import GameState, Action, ActionPayload
from engine.serialization import state_to_text, legal_actions_to_text, legal_actions, deserialize_game_state, serialize_action_payload
from agents.guideline_cluster_agent import GuidelineClusterAgent


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts using OpenAI embeddings."""
    embeddings = []
    for t in texts:
        resp = litellm.embedding(model=model, input=t)
        vec = resp["data"][0]["embedding"]
        embeddings.append(vec)
    return np.array(embeddings)


def robust_parse(chosen_action_str: str) -> Any:
    """Robust JSON parsing (same as clustering evaluation)."""
    if not chosen_action_str or str(chosen_action_str).lower() == "null":
        return None
    try:
        return json.loads(chosen_action_str)
    except Exception:
        try:
            cleaned = chosen_action_str.rstrip("}").rstrip() + "}"
            return json.loads(cleaned)
        except Exception:
            import re
            m = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", chosen_action_str)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            m = re.search(r"\{.*?\}", chosen_action_str, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return None


def load_meta_clusters(meta_tree_path: Path) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Load meta clusters and prepare centroids."""
    with open(meta_tree_path) as f:
        data = json.load(f)
    meta_clusters = data.get("meta_clusters", [])
    
    # Load dataset for observation lookup
    dataset_path = meta_tree_path.parent / "drills_dataset.json"
    dataset = DrillDataset()
    dataset.load_from_json(str(dataset_path))
    drills_by_id = {ex.drill_id: ex for ex in dataset.examples}
    
    # Compute centroids (same as clustering evaluation)
    centroids = []
    for cluster in meta_clusters:
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
        
        centroids.append({
            "cluster_id": cluster["cluster_id"],
            "guideline": cluster.get("best_guideline", ""),
            "embedding": emb,
        })
    
    return centroids, drills_by_id


def retrieve_guideline_clustering(observation: str, centroids: List[Dict[str, Any]]) -> str:
    """Retrieve guideline using clustering evaluation method."""
    obs_emb = embed_texts([observation])[0]
    best = None
    best_sim = -1e9
    for c in centroids:
        emb = c["embedding"]
        sim = float(np.dot(obs_emb, emb) / (np.linalg.norm(obs_emb) * np.linalg.norm(emb) + 1e-8))
        if sim > best_sim:
            best_sim = sim
            best = c
    return best["guideline"] if best else ""


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts using OpenAI embeddings."""
    embeddings = []
    for t in texts:
        resp = litellm.embedding(model=model, input=t)
        vec = resp["data"][0]["embedding"]
        embeddings.append(vec)
    return np.array(embeddings)


# Global DSPy module (configured once to avoid thread-safety issues)
_dspy_module = None
_dspy_model = None

def _get_dspy_module(model: str = "gpt-5.2"):
    """Get or create DSPy module (configured once)."""
    global _dspy_module, _dspy_model
    if _dspy_module is None or _dspy_model != model:
        import dspy
        from dspy_ml.signature import CatanDrillSignature
        lm = dspy.LM(model=model)
        dspy.configure(lm=lm)
        _dspy_module = dspy.ChainOfThought(CatanDrillSignature)
        _dspy_model = model
    return _dspy_module


def run_clustering_evaluation_agent(
    drill: Dict[str, Any],
    centroids: List[Dict[str, Any]],
    game_rules: str,
    model: str = "gpt-5.2",
    best_guidelines: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, Any], str, str]:
    """
    Run the clustering evaluation agent (from cluster_guideline_tree_meta.py).
    Uses DSPy directly to match the meta-clustering evaluation exactly.
    
    IMPORTANT: If best_guidelines is provided, use that (matching meta-clustering's
    best_per_drill). Otherwise, retrieve from centroids.
    
    Returns: (predicted_action_dict, reasoning, chosen_action_str)
    """
    observation = drill["observation"]
    viable_actions = drill["viable_actions"]
    drill_id = drill["drill_id"]
    
    # Use best guideline per drill if available (matching meta-clustering evaluation)
    if best_guidelines and str(drill_id) in best_guidelines:
        guideline = best_guidelines[str(drill_id)]
    else:
        # Fall back to retrieval
        guideline = retrieve_guideline_clustering(observation, centroids)
    
    # Use the extracted template with litellm directly (avoiding DSPy thread-safety issues)
    # This matches the frontend agent approach and avoids dspy.configure() in threads
    from dspy_ml.dspy_prompt_template import format_prompt
    prompt_dict = format_prompt(
        game_rules=game_rules,
        observation=observation,
        viable_actions=viable_actions,
        guideline=guideline
    )
    
    # Call LLM using litellm (matching DSPy's behavior)
    # DSPy uses temperature=None (model default), so we do too
    # Note: For GPT-5, litellm will handle temperature=None appropriately
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": prompt_dict["system"]},
            {"role": "user", "content": prompt_dict["user"]}
        ],
        temperature=None,  # Model default (matching DSPy)
    )
    
    response_text = response.choices[0].message.content
    
    # Parse response (same as frontend agent and meta-clustering script)
    reasoning = ""
    chosen_action_str = "null"
    
    reasoning_patterns = [
        r'\[\[ ## reasoning ## \]\]\s*(.*?)(?=\[\[ ## chosen_action ## \]\]|\[\[ ## completed ## \]\]|$)',
        r'\[\[ ## reasoning ## \]\]\s*(.*?)(?=\[\[ ## chosen_action ## \]\]|$)',
    ]
    for pattern in reasoning_patterns:
        reasoning_match = re.search(pattern, response_text, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            break
    
    chosen_action_patterns = [
        r'\[\[ ## chosen_action ## \]\]\s*(.*?)(?=\[\[ ## completed ## \]\]|$)',
        r'\[\[ ## chosen_action ## \]\]\s*(.*)',
    ]
    for pattern in chosen_action_patterns:
        chosen_action_match = re.search(pattern, response_text, re.DOTALL)
        if chosen_action_match:
            chosen_action_str = chosen_action_match.group(1).strip()
            break
    
    if chosen_action_str == "null" and "chosen_action" not in response_text.lower():
        json_match = re.search(r'\{[^{}]*"type"[^{}]*\}', response_text)
        if json_match:
            chosen_action_str = json_match.group(0)
    
    predicted = robust_parse(chosen_action_str)
    
    return predicted, reasoning, chosen_action_str


def run_frontend_agent(
    drill: Dict[str, Any],
    agent: GuidelineClusterAgent,
    centroids: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """
    Run the frontend GuidelineClusterAgent.
    BUT: Use the pre-computed observation and viable_actions from the dataset JSON
    instead of regenerating them from state, to match the clustering evaluation exactly.
    
    Returns: (predicted_action_dict, reasoning, error_message)
    """
    # Use the pre-computed observation and viable_actions from dataset JSON
    # (matching clustering evaluation exactly)
    observation = drill["observation"]  # From JSON, not regenerated
    viable_actions = drill["viable_actions"]  # From JSON, not regenerated
    game_rules = drill["game_rules"]  # From JSON
    
    # Retrieve guideline using the same method as clustering evaluation
    guideline = retrieve_guideline_clustering(observation, centroids)
    
    # Format prompt using extracted DSPy template (matching clustering evaluation)
    from dspy_ml.dspy_prompt_template import format_prompt
    prompt_dict = format_prompt(
        game_rules=game_rules,
        observation=observation,
        viable_actions=viable_actions,
        guideline=guideline
    )
    
    # Call LLM using litellm (matching frontend agent's approach)
    import litellm
    try:
        response = litellm.completion(
            model=agent.model,
            messages=[
                {"role": "system", "content": prompt_dict["system"]},
                {"role": "user", "content": prompt_dict["user"]}
            ],
            temperature=None,  # Model default
        )
        
        response_text = response.choices[0].message.content
        
        # Parse response (same as frontend agent)
        import re
        reasoning = ""
        chosen_action_str = "null"
        
        reasoning_patterns = [
            r'\[\[ ## reasoning ## \]\]\s*(.*?)(?=\[\[ ## chosen_action ## \]\]|\[\[ ## completed ## \]\]|$)',
            r'\[\[ ## reasoning ## \]\]\s*(.*?)(?=\[\[ ## chosen_action ## \]\]|$)',
        ]
        for pattern in reasoning_patterns:
            reasoning_match = re.search(pattern, response_text, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                break
        
        chosen_action_patterns = [
            r'\[\[ ## chosen_action ## \]\]\s*(.*?)(?=\[\[ ## completed ## \]\]|$)',
            r'\[\[ ## chosen_action ## \]\]\s*(.*)',
        ]
        for pattern in chosen_action_patterns:
            chosen_action_match = re.search(pattern, response_text, re.DOTALL)
            if chosen_action_match:
                chosen_action_str = chosen_action_match.group(1).strip()
                break
        
        if chosen_action_str == "null" and "chosen_action" not in response_text.lower():
            json_match = re.search(r'\{[^{}]*"type"[^{}]*\}', response_text)
            if json_match:
                chosen_action_str = json_match.group(0)
        
        predicted = robust_parse(chosen_action_str)
        
        # Convert to dict format for comparison
        if predicted:
            return predicted, reasoning or "", None
        else:
            return None, reasoning or "", "No action predicted"
            
    except Exception as e:
        return None, "", str(e)




def compare_drill(
    drill: Dict[str, Any],
    centroids: List[Dict[str, Any]],
    game_rules: str,
    frontend_agent: GuidelineClusterAgent,
    model: str = "gpt-5.2",
    best_guidelines: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Compare both agents on a single drill."""
    drill_id = drill["drill_id"]
    expected = drill["expected_action"]
    
    # Run both agents
    clustering_pred, clustering_reasoning, clustering_raw = run_clustering_evaluation_agent(
        drill, centroids, game_rules, model, best_guidelines
    )
    frontend_pred, frontend_reasoning, frontend_error = run_frontend_agent(drill, frontend_agent, centroids)
    
    # Check correctness - match meta-clustering evaluation logic exactly
    # Use state=None and normalize_setup_build separately (like meta-clustering script)
    def normalize_setup_build(canon):
        """Normalize build_road -> setup_place_road, build_settlement -> setup_place_settlement (always, regardless of phase)."""
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
    
    # Get correct_actions from drill (dataset has this field)
    correct_actions = drill.get("correct_actions", [expected] if expected else [])
    
    # Check clustering agent
    clustering_correct = False
    if clustering_pred:
        pred_canon = _canonical_action_dict(clustering_pred, state=None)
        for ca in correct_actions:
            ca_canon = _canonical_action_dict(ca, state=None)
            if normalize_setup_build(ca_canon) == normalize_setup_build(pred_canon):
                clustering_correct = True
                break
    
    # Check frontend agent
    frontend_correct = False
    if frontend_pred:
        pred_canon = _canonical_action_dict(frontend_pred, state=None)
        for ca in correct_actions:
            ca_canon = _canonical_action_dict(ca, state=None)
            if normalize_setup_build(ca_canon) == normalize_setup_build(pred_canon):
                frontend_correct = True
                break
    
    return {
        "drill_id": drill_id,
        "expected": expected,
        "clustering": {
            "predicted": clustering_pred,
            "reasoning": clustering_reasoning,
            "raw": clustering_raw,
            "correct": clustering_correct,
        },
        "frontend": {
            "predicted": frontend_pred,
            "reasoning": frontend_reasoning,
            "error": frontend_error,
            "correct": frontend_correct,
        },
        "match": clustering_correct == frontend_correct and (
            clustering_pred == frontend_pred if clustering_pred and frontend_pred else False
        ),
    }


def main():
    # Load dataset directly from JSON (matching meta-clustering script)
    dataset_path = Path(__file__).parent.parent / "data" / "drills_dataset.json"
    with open(dataset_path) as f:
        drills_json = json.load(f)
    drills_by_id_json = {d["drill_id"]: d for d in drills_json}
    
    # Also load via DrillDataset for state access
    dataset = DrillDataset()
    dataset.load_from_json(str(dataset_path))
    drills_by_id_state = {ex.drill_id: ex for ex in dataset.examples}
    
    # Load meta clusters
    meta_tree_path = Path(__file__).parent.parent / "data" / "guideline_tree_meta.json"
    centroids, drills_by_id = load_meta_clusters(meta_tree_path)
    
    # Load best guidelines per drill (matching meta-clustering evaluation)
    best_guidelines_path = Path(__file__).parent.parent / "data" / "best_guidelines_meta.json"
    best_guidelines = None
    if best_guidelines_path.exists():
        with open(best_guidelines_path) as f:
            best_guidelines = json.load(f)
        print(f"Loaded best guidelines for {len(best_guidelines)} drills", flush=True)
    else:
        print("Warning: best_guidelines_meta.json not found, will use retrieval", flush=True)
    
    # Get game rules (use from first drill in JSON, matching meta-clustering script)
    game_rules = drills_json[0]["game_rules"] if drills_json else ""
    
    # No need to configure DSPy - we're using the extracted template with litellm directly
    
    # Create frontend agent
    backend_dir = Path(__file__).parent.parent.parent
    frontend_agent = GuidelineClusterAgent(
        player_id="player_0",
        meta_tree_path=str(meta_tree_path.relative_to(backend_dir)),
        model="gpt-5.2",
    )
    
    # Convert dataset to dict format (matching meta-clustering script structure exactly)
    drills = []
    for d_json in drills_json:
        drill_id = d_json["drill_id"]
        # Get state from DrillDataset
        ex_state = drills_by_id_state.get(drill_id)
        drills.append({
            "drill_id": drill_id,
            "game_rules": d_json["game_rules"],  # Use from JSON
            "observation": d_json["observation"],  # Use from JSON
            "viable_actions": d_json["viable_actions"],  # Use from JSON
            "expected_action": d_json["expected_action"],  # Use from JSON
            "correct_actions": d_json["correct_actions"],  # Use from JSON
            "state": ex_state.state if ex_state else None,  # Get state from DrillDataset
        })
    
    # Compare all drills
    # Use ProcessPoolExecutor instead of ThreadPoolExecutor to match meta-clustering script
    # This allows each process to have its own DSPy configuration (avoiding thread-safety issues)
    print(f"Comparing {len(drills)} drills...", flush=True)
    results = []
    
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(8, len(drills))) as executor:
        futures = {
            executor.submit(compare_drill, drill, centroids, game_rules, frontend_agent, "gpt-5.2", best_guidelines): drill
            for drill in drills
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}", flush=True)
    
    # Sort by drill_id
    results.sort(key=lambda x: x["drill_id"])
    
    # Analyze
    clustering_correct = sum(1 for r in results if r["clustering"]["correct"])
    frontend_correct = sum(1 for r in results if r["frontend"]["correct"])
    matches = sum(1 for r in results if r["match"])
    
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS:", flush=True)
    print(f"  Clustering evaluation: {clustering_correct}/{len(results)} ({100*clustering_correct/len(results):.1f}%)", flush=True)
    print(f"  Frontend agent: {frontend_correct}/{len(results)} ({100*frontend_correct/len(results):.1f}%)", flush=True)
    print(f"  Exact matches: {matches}/{len(results)} ({100*matches/len(results):.1f}%)", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Show discrepancies
    discrepancies = [r for r in results if not r["match"]]
    print(f"Found {len(discrepancies)} discrepancies:\n", flush=True)
    
    for r in discrepancies[:10]:  # Show first 10
        print(f"Drill {r['drill_id']}:", flush=True)
        print(f"  Expected: {r['expected']}", flush=True)
        print(f"  Clustering: {r['clustering']['predicted']} (correct={r['clustering']['correct']})", flush=True)
        print(f"  Frontend: {r['frontend']['predicted']} (correct={r['frontend']['correct']})", flush=True)
        if r['frontend']['error']:
            print(f"  Frontend error: {r['frontend']['error']}", flush=True)
        if r['clustering']['raw']:
            print(f"  Clustering raw: {r['clustering']['raw'][:200]}...", flush=True)
        print(flush=True)
    
    # Save full results
    output_path = Path(__file__).parent.parent / "data" / "agent_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()

