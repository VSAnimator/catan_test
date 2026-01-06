#!/usr/bin/env python3
"""
Extract meta-guidelines from drills through clustering and LLM synthesis.

Pipeline:
1. Load difficulty analysis (hard vs easy drills)
2. Extract features for hard drills (embeddings + metadata)
3. Cluster hard drills to find common failure patterns
4. Synthesize meta-guidelines for each cluster using LLM
5. Save meta-guidelines for use in optimization
"""
import sys
import argparse
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.guideline_extraction import DrillFeatureExtractor, DrillClusterer
from dspy_ml.guideline_synthesis import GuidelineSynthesizer
from api.database import get_db_connection


def load_drills(drill_ids: list) -> list:
    """Load drill data from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    drills = []
    for drill_id in drill_ids:
        cursor.execute("""
            SELECT 
                d.id, 
                d.name,
                d.guideline_text,
                ds.expected_action_json,
                ds.state_json
            FROM drills d
            JOIN drill_steps ds ON d.id = ds.drill_id
            WHERE d.id = ? AND ds.idx = 0
        """, (drill_id,))
        
        row = cursor.fetchone()
        if row:
            drills.append({
                'drill_id': row['id'],
                'name': row['name'],
                'guideline_text': row['guideline_text'],
                'expected_action': json.loads(row['expected_action_json']),
                'state': json.loads(row['state_json'])
            })
    
    return drills


def main():
    parser = argparse.ArgumentParser(description="Extract meta-guidelines from drill clusters")
    parser.add_argument("--difficulty-analysis", required=True, help="Difficulty analysis JSON file")
    parser.add_argument("--output", required=True, help="Output meta-guidelines JSON file")
    parser.add_argument("--num-clusters", type=int, help="Number of clusters (None = auto-detect)")
    parser.add_argument("--synthesis-model", default="openai/gpt-5.2", help="Model for guideline synthesis")
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"], help="Reasoning effort for synthesis")
    parser.add_argument("--use-embeddings", action="store_true", default=True, help="Use OpenAI embeddings")
    parser.add_argument("--no-embeddings", dest="use_embeddings", action="store_false", help="Disable embeddings")
    
    args = parser.parse_args()
    
    # Load difficulty analysis
    print(f"Loading difficulty analysis from {args.difficulty_analysis}...", flush=True)
    with open(args.difficulty_analysis, 'r') as f:
        difficulty_data = json.load(f)
    
    # Filter to hard drills only
    hard_drill_ids = [
        d['drill_id'] for d in difficulty_data['drills']
        if d['predicted_difficulty'] == 'hard'
    ]
    
    print(f"Found {len(hard_drill_ids)} hard drills to cluster", flush=True)
    
    # Load drill data
    print("Loading drill data from database...", flush=True)
    drills = load_drills(hard_drill_ids)
    print(f"Loaded {len(drills)} drills", flush=True)
    
    # Extract features
    print("\nExtracting features...", flush=True)
    extractor = DrillFeatureExtractor(use_embeddings=args.use_embeddings)
    features_list = []
    for drill in drills:
        features = extractor.extract_features(drill)
        features_list.append(features)
        if len(features_list) % 10 == 0:
            print(f"  Extracted features for {len(features_list)}/{len(drills)} drills", flush=True)
    
    print(f"Extracted features for all {len(features_list)} drills", flush=True)
    
    # Cluster drills
    print("\nClustering drills...", flush=True)
    clusterer = DrillClusterer()
    labels, clustering_info = clusterer.cluster_drills(features_list, num_clusters=args.num_clusters)
    
    print(f"Clustering complete: {clustering_info}", flush=True)
    
    # Organize drills by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # Skip noise points
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(drills[i])
    
    print(f"\nOrganized into {len(clusters)} clusters", flush=True)
    for cluster_id, cluster_drills in sorted(clusters.items()):
        print(f"  Cluster {cluster_id}: {len(cluster_drills)} drills", flush=True)
    
    # Synthesize meta-guidelines for each cluster
    print("\nSynthesizing meta-guidelines...", flush=True)
    synthesizer = GuidelineSynthesizer(
        model=args.synthesis_model,
        reasoning_effort=args.reasoning_effort
    )
    
    meta_guidelines = []
    for cluster_id, cluster_drills in sorted(clusters.items()):
        print(f"\nProcessing cluster {cluster_id}...", flush=True)
        meta_guideline_info = synthesizer.synthesize_meta_guideline(cluster_drills, cluster_id)
        meta_guidelines.append(meta_guideline_info)
    
    # Handle noise points (unclustered drills)
    noise_drill_ids = [
        drills[i]['drill_id']
        for i, label in enumerate(labels)
        if label == -1
    ]
    
    # Save results
    output = {
        'clustering_method': clustering_info['method'],
        'num_clusters': len(clusters),
        'num_hard_drills': len(drills),
        'num_unclustered': len(noise_drill_ids),
        'clusters': meta_guidelines,
        'unclustered_drill_ids': noise_drill_ids
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Guideline extraction complete!", flush=True)
    print(f"Saved {len(meta_guidelines)} meta-guidelines to {args.output}", flush=True)
    
    # Print summary
    print("\nSummary:", flush=True)
    print(f"  Hard drills clustered: {len(drills) - len(noise_drill_ids)}", flush=True)
    print(f"  Unclustered (noise): {len(noise_drill_ids)}", flush=True)
    print(f"  Meta-guidelines synthesized: {len(meta_guidelines)}", flush=True)
    for mg in meta_guidelines:
        print(f"\n  Cluster {mg['cluster_id']}: {mg['num_drills']} drills, {mg['num_with_guidelines']} with guidelines", flush=True)
        print(f"    Failure pattern: {mg['failure_pattern']}", flush=True)
        print(f"    Actions: {', '.join(mg['action_types'])}", flush=True)
        print(f"    Meta-guideline ({len(mg['meta_guideline'])} chars):", flush=True)
        # Print first 200 chars
        preview = mg['meta_guideline'][:200]
        if len(mg['meta_guideline']) > 200:
            preview += "..."
        print(f"      {preview}", flush=True)


if __name__ == "__main__":
    main()

