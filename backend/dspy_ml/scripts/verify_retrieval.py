#!/usr/bin/env python3
"""
Verify that retrieval is routing each drill to the correct meta-cluster guideline.
For each drill, check:
1. Which meta-cluster it belongs to (from meta-clustering)
2. What guideline is retrieved for that drill
3. Whether it matches the best guideline for that meta-cluster
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import litellm
from dspy_ml.dataset import DrillDataset


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts using OpenAI embeddings."""
    embeddings = []
    for t in texts:
        resp = litellm.embedding(model=model, input=t)
        vec = resp["data"][0]["embedding"]
        embeddings.append(vec)
    return np.array(embeddings)


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
            "drill_ids": drill_ids,  # Store drill IDs for verification
            "guideline": cluster.get("best_guideline", ""),
            "embedding": emb,
        })
    
    return centroids, drills_by_id


def retrieve_guideline(observation: str, centroids: List[Dict[str, Any]]) -> Tuple[str, int]:
    """Retrieve guideline using clustering evaluation method. Returns (guideline, cluster_id)."""
    obs_emb = embed_texts([observation])[0]
    best = None
    best_sim = -1e9
    best_cluster_id = None
    for c in centroids:
        emb = c["embedding"]
        sim = float(np.dot(obs_emb, emb) / (np.linalg.norm(obs_emb) * np.linalg.norm(emb) + 1e-8))
        if sim > best_sim:
            best_sim = sim
            best = c
            best_cluster_id = c["cluster_id"]
    return best["guideline"] if best else "", best_cluster_id


def main():
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "data" / "drills_dataset.json"
    dataset = DrillDataset()
    dataset.load_from_json(str(dataset_path))
    
    # Load meta clusters
    meta_tree_path = Path(__file__).parent.parent / "data" / "guideline_tree_meta.json"
    centroids, drills_by_id = load_meta_clusters(meta_tree_path)
    
    # Build mapping: drill_id -> expected cluster_id
    drill_to_cluster = {}
    for c in centroids:
        for drill_id in c["drill_ids"]:
            drill_to_cluster[drill_id] = c["cluster_id"]
    
    # Verify retrieval for each drill
    correct_retrievals = 0
    total = 0
    mismatches = []
    
    for ex in dataset.examples:
        drill_id = ex.drill_id
        observation = ex.observation
        
        # Get expected cluster (the one this drill belongs to)
        expected_cluster_id = drill_to_cluster.get(drill_id)
        if expected_cluster_id is None:
            print(f"Warning: Drill {drill_id} not found in any meta-cluster", flush=True)
            continue
        
        # Get expected guideline (best guideline for this drill's cluster)
        expected_guideline = None
        for c in centroids:
            if c["cluster_id"] == expected_cluster_id:
                expected_guideline = c["guideline"]
                break
        
        if expected_guideline is None:
            print(f"Warning: No guideline found for cluster {expected_cluster_id}", flush=True)
            continue
        
        # Retrieve guideline
        retrieved_guideline, retrieved_cluster_id = retrieve_guideline(observation, centroids)
        
        # Check if correct
        is_correct = (retrieved_cluster_id == expected_cluster_id)
        if is_correct:
            correct_retrievals += 1
        else:
            mismatches.append({
                "drill_id": drill_id,
                "expected_cluster": expected_cluster_id,
                "retrieved_cluster": retrieved_cluster_id,
                "expected_guideline": expected_guideline[:100] + "..." if len(expected_guideline) > 100 else expected_guideline,
                "retrieved_guideline": retrieved_guideline[:100] + "..." if len(retrieved_guideline) > 100 else retrieved_guideline,
            })
        
        total += 1
    
    print(f"\n{'='*80}", flush=True)
    print(f"RETRIEVAL VERIFICATION RESULTS:", flush=True)
    print(f"  Correct retrievals: {correct_retrievals}/{total} ({100*correct_retrievals/total:.1f}%)", flush=True)
    print(f"  Mismatches: {len(mismatches)}/{total}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    if mismatches:
        print(f"First 10 mismatches:\n", flush=True)
        for m in mismatches[:10]:
            print(f"Drill {m['drill_id']}:", flush=True)
            print(f"  Expected cluster: {m['expected_cluster']}", flush=True)
            print(f"  Retrieved cluster: {m['retrieved_cluster']}", flush=True)
            print(f"  Expected guideline: {m['expected_guideline']}", flush=True)
            print(f"  Retrieved guideline: {m['retrieved_guideline']}", flush=True)
            print(flush=True)
    
    # Save full results
    output_path = Path(__file__).parent.parent / "data" / "retrieval_verification.json"
    with open(output_path, "w") as f:
        json.dump({
            "correct_retrievals": correct_retrievals,
            "total": total,
            "accuracy": correct_retrievals / total if total > 0 else 0.0,
            "mismatches": mismatches
        }, f, indent=2)
    print(f"Full results saved to {output_path}", flush=True)
    
    if correct_retrievals == total:
        print(f"\n✓ All {total} drills are correctly routed to their meta-clusters!", flush=True)
    else:
        print(f"\n✗ {len(mismatches)} drills are being routed to wrong clusters!", flush=True)


if __name__ == "__main__":
    main()

