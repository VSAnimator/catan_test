#!/usr/bin/env python3
"""
Create hybrid guideline system that uses meta-guidelines where performance is similar
to leaf-guidelines, and keeps leaf-guidelines where there's a significant drop.

Strategy:
1. Compare leaf vs meta performance per cluster
2. Where meta performs within threshold (e.g., <5% drop), use meta-guideline
3. Where meta performs significantly worse, keep leaf-guidelines
4. Output a JSON mapping drill_id -> guideline (either leaf or meta)
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.dataset import DrillDataset


def load_leaf_performance() -> Dict[int, Dict[str, Any]]:
    """Load leaf-level cluster performance."""
    leaf_eval_path = Path(__file__).parent.parent / "data" / "leaf_guidelines_evaluation.json"
    if not leaf_eval_path.exists():
        return {}
    
    with open(leaf_eval_path) as f:
        data = json.load(f)
    
    # Build drill_id -> leaf cluster performance mapping
    leaf_tree_path = Path(__file__).parent.parent / "data" / "guideline_tree.json"
    with open(leaf_tree_path) as f:
        leaf_tree = json.load(f)
    
    drill_to_leaf_cluster = {}
    for cluster in leaf_tree.get("clusters", []):
        cluster_id = cluster["cluster_id"]
        cluster_perf = data.get("per_cluster", {}).get(str(cluster_id), {})
        for drill_id in cluster.get("drill_ids", []):
            drill_to_leaf_cluster[drill_id] = {
                "cluster_id": cluster_id,
                "score": cluster_perf.get("score", 0.0),
                "size": cluster_perf.get("size", 0),
            }
    
    return drill_to_leaf_cluster


def load_meta_performance() -> Dict[int, Dict[str, Any]]:
    """Load meta-level cluster performance."""
    meta_eval_path = Path(__file__).parent.parent / "data" / "agent_comparison.json"
    if not meta_eval_path.exists():
        return {}
    
    with open(meta_eval_path) as f:
        meta_results = json.load(f)
    
    # Load meta tree
    meta_tree_path = Path(__file__).parent.parent / "data" / "guideline_tree_meta.json"
    with open(meta_tree_path) as f:
        meta_tree = json.load(f)
    
    # Build drill_id -> meta cluster mapping
    drill_to_meta_cluster = {}
    for cluster in meta_tree.get("meta_clusters", []):
        cluster_id = cluster["cluster_id"]
        for drill_id in cluster.get("drill_ids", []):
            drill_to_meta_cluster[drill_id] = cluster_id
    
    # Calculate per-cluster performance from meta results
    meta_cluster_perf = {}
    for r in meta_results:
        drill_id = r["drill_id"]
        cluster_id = drill_to_meta_cluster.get(drill_id)
        if cluster_id is None:
            continue
        if cluster_id not in meta_cluster_perf:
            meta_cluster_perf[cluster_id] = {"correct": 0, "total": 0}
        meta_cluster_perf[cluster_id]["total"] += 1
        if r.get("clustering", {}).get("correct", False):
            meta_cluster_perf[cluster_id]["correct"] += 1
    
    # Build drill_id -> meta performance mapping
    drill_to_meta_perf = {}
    for drill_id, cluster_id in drill_to_meta_cluster.items():
        perf = meta_cluster_perf.get(cluster_id, {"correct": 0, "total": 0})
        score = perf["correct"] / perf["total"] if perf["total"] > 0 else 0.0
        drill_to_meta_perf[drill_id] = {
            "cluster_id": cluster_id,
            "score": score,
            "size": perf["total"],
        }
    
    return drill_to_meta_perf


def create_hybrid_guidelines(
    performance_threshold: float = 0.05,
    min_meta_score: float = 0.80
) -> Dict[str, str]:
    """
    Create hybrid guidelines mapping drill_id -> guideline.
    
    Args:
        performance_threshold: Maximum acceptable drop from leaf to meta (default 5%)
        min_meta_score: Minimum meta score to consider using meta (default 80%)
    
    Returns:
        Dict mapping drill_id (str) -> guideline (str)
    """
    # Load best guidelines
    leaf_best_path = Path(__file__).parent.parent / "data" / "best_guidelines_leaf.json"
    meta_best_path = Path(__file__).parent.parent / "data" / "best_guidelines_meta.json"
    
    with open(leaf_best_path) as f:
        leaf_best = json.load(f)
    
    with open(meta_best_path) as f:
        meta_best = json.load(f)
    
    # Load performance data
    leaf_perf = load_leaf_performance()
    meta_perf = load_meta_performance()
    
    # Create hybrid mapping
    hybrid_guidelines = {}
    decisions = []
    
    for drill_id_str in leaf_best.keys():
        drill_id = int(drill_id_str)
        
        leaf_guideline = leaf_best[drill_id_str]
        meta_guideline = meta_best.get(drill_id_str, "")
        
        leaf_score = leaf_perf.get(drill_id, {}).get("score", 1.0)
        meta_score = meta_perf.get(drill_id, {}).get("score", 0.0)
        
        # Decision logic:
        # 1. If meta score is below minimum threshold, use leaf
        # 2. If meta score drop is within threshold, use meta
        # 3. Otherwise, use leaf
        
        if meta_score < min_meta_score:
            use_meta = False
            reason = f"meta_score {meta_score:.3f} < {min_meta_score}"
        elif (leaf_score - meta_score) <= performance_threshold:
            use_meta = True
            reason = f"drop {leaf_score - meta_score:.3f} <= {performance_threshold}"
        else:
            use_meta = False
            reason = f"drop {leaf_score - meta_score:.3f} > {performance_threshold}"
        
        if use_meta and meta_guideline:
            hybrid_guidelines[drill_id_str] = meta_guideline
            decision = "meta"
        else:
            hybrid_guidelines[drill_id_str] = leaf_guideline
            decision = "leaf"
        
        decisions.append({
            "drill_id": drill_id,
            "decision": decision,
            "leaf_score": leaf_score,
            "meta_score": meta_score,
            "drop": leaf_score - meta_score,
            "reason": reason,
        })
    
    return hybrid_guidelines, decisions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create hybrid leaf/meta guidelines")
    parser.add_argument("--threshold", type=float, default=0.05, help="Max acceptable drop (default 0.05 = 5%%)")
    parser.add_argument("--min-meta-score", type=float, default=0.80, help="Min meta score to consider (default 0.80)")
    parser.add_argument("--output", default=str(Path(__file__).parent.parent / "data" / "best_guidelines_hybrid.json"), help="Output path")
    parser.add_argument("--output-decisions", default="backend/dspy_ml/data/hybrid_guidelines_decisions.json", help="Decisions log path")
    args = parser.parse_args()
    
    print(f"Creating hybrid guidelines (threshold={args.threshold}, min_meta_score={args.min_meta_score})...", flush=True)
    
    hybrid_guidelines, decisions = create_hybrid_guidelines(
        performance_threshold=args.threshold,
        min_meta_score=args.min_meta_score
    )
    
    # Count decisions
    meta_count = sum(1 for d in decisions if d["decision"] == "meta")
    leaf_count = sum(1 for d in decisions if d["decision"] == "leaf")
    
    print(f"Decisions: {meta_count} meta, {leaf_count} leaf", flush=True)
    
    # Save hybrid guidelines
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hybrid_guidelines, f, indent=2)
    
    print(f"Saved hybrid guidelines to {output_path}", flush=True)
    
    # Save decisions log
    decisions_path = Path(args.output_decisions)
    with open(decisions_path, "w") as f:
        json.dump(decisions, f, indent=2)
    
    print(f"Saved decisions log to {decisions_path}", flush=True)


if __name__ == "__main__":
    main()

