#!/usr/bin/env python3
"""
Precompute embeddings for all drill observations and store them on disk.
This avoids making embedding API calls during agent initialization.
"""
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.dataset import DrillDataset
import litellm


def precompute_embeddings(dataset_path: str, output_path: str):
    """Precompute embeddings for all drill observations."""
    print(f"Loading dataset from {dataset_path}...", flush=True)
    dataset = DrillDataset()
    dataset.load_from_json(dataset_path)
    
    print(f"Found {len(dataset.examples)} drills", flush=True)
    
    # Collect unique observations
    unique_observations = {}
    for ex in dataset.examples:
        if ex.observation and ex.observation not in unique_observations:
            unique_observations[ex.observation] = ex.drill_id
    
    print(f"Found {len(unique_observations)} unique observations", flush=True)
    
    # Compute embeddings
    print("Computing embeddings...", flush=True)
    embeddings = {}
    for i, (obs, drill_id) in enumerate(unique_observations.items()):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(unique_observations)}", flush=True)
        
        resp = litellm.embedding(model="text-embedding-3-small", input=obs)
        emb = np.array(resp["data"][0]["embedding"])
        embeddings[obs] = emb.tolist()  # Convert to list for JSON serialization
    
    print(f"Computed {len(embeddings)} embeddings", flush=True)
    
    # Save to disk
    print(f"Saving to {output_path}...", flush=True)
    with open(output_path, "w") as f:
        json.dump(embeddings, f)
    
    print(f"✓ Saved {len(embeddings)} precomputed embeddings", flush=True)


if __name__ == "__main__":
    backend_dir = Path(__file__).parent.parent.parent
    dataset_path = backend_dir / "dspy_ml" / "data" / "drills_dataset.json"
    output_path = backend_dir / "dspy_ml" / "data" / "observation_embeddings.json"
    
    precompute_embeddings(str(dataset_path), str(output_path))

