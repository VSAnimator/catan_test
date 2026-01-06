"""
Guideline extraction system: feature extraction and clustering for hard drills.

Focuses on clustering drills WITH guidelines to discover failure patterns,
then assigns drills WITHOUT guidelines to the nearest clusters.
"""
import json
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import deserialize_game_state
from engine.serialization import state_to_text


@dataclass
class DrillFeatures:
    """Multi-modal features for a single drill."""
    drill_id: int
    has_guideline: bool
    
    # Text embeddings (if available)
    state_embedding: Optional[np.ndarray] = None
    guideline_embedding: Optional[np.ndarray] = None
    
    # Categorical features
    action_type: str = ""
    phase: str = ""
    
    # Numerical features
    turn_number: int = 0
    current_player_vp: int = 0
    num_resources: int = 0
    
    # Structural features
    action_payload_hash: str = ""
    name_tokens: List[str] = None


class DrillFeatureExtractor:
    """
    Extract multi-modal features from drills for clustering.
    
    Focuses on hard drills only (those with guidelines or predicted hard).
    """
    
    def __init__(self, use_embeddings: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            use_embeddings: Whether to use OpenAI embeddings (requires API key)
        """
        self.use_embeddings = use_embeddings
        
        if self.use_embeddings:
            try:
                import openai
                self.openai = openai
                print("OpenAI embeddings enabled", flush=True)
            except ImportError:
                print("Warning: openai not installed, embeddings disabled", flush=True)
                self.use_embeddings = False
    
    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array, or None if embedding fails
        """
        if not self.use_embeddings:
            return None
        
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            return embedding
        except Exception as e:
            print(f"Warning: Failed to get embedding: {e}", flush=True)
            return None
    
    def _hash_payload_structure(self, payload: Any) -> str:
        """
        Hash the structure of a payload (not exact values).
        
        This groups actions with similar payload structures together.
        """
        if payload is None:
            return "null"
        if not isinstance(payload, dict):
            return "non-dict"
        
        # Get sorted list of keys
        keys = tuple(sorted(payload.keys()))
        return hashlib.md5(str(keys).encode()).hexdigest()[:8]
    
    def _tokenize_name(self, name: str) -> List[str]:
        """Simple tokenization of drill name."""
        import re
        # Split on underscores and spaces
        tokens = re.split(r'[_\s]+', name.lower())
        return [t for t in tokens if t]
    
    def extract_features(self, drill: Dict[str, Any]) -> DrillFeatures:
        """
        Extract all features for a single drill.
        
        Args:
            drill: Drill dictionary with keys: drill_id, name, guideline_text,
                   expected_action, state
                   
        Returns:
            DrillFeatures object
        """
        drill_id = drill['drill_id']
        has_guideline = drill.get('guideline_text') is not None
        
        # Load state
        state_json = drill['state']
        try:
            state = deserialize_game_state(state_json)
            state_text = state_to_text(state, state.players[state.current_player_index].id)
        except Exception as e:
            print(f"Warning: Failed to deserialize state for drill {drill_id}: {e}", flush=True)
            state_text = json.dumps(state_json)[:500]  # Fallback to JSON
            state = None
        
        # Extract embeddings
        state_embedding = None
        guideline_embedding = None
        
        if self.use_embeddings:
            # State embedding
            state_embedding = self._embed_text(state_text)
            
            # Guideline embedding (only if guideline exists)
            if has_guideline and drill['guideline_text']:
                guideline_embedding = self._embed_text(drill['guideline_text'])
        
        # Extract categorical features
        expected_action = drill['expected_action']
        action_type = expected_action.get('type', 'unknown')
        phase = state_json.get('phase', 'unknown') if state is None else state.phase
        
        # Extract numerical features
        turn_number = state_json.get('turn_number', 0) if state is None else (state.turn_number if hasattr(state, 'turn_number') else 0)
        current_player_idx = state_json.get('current_player_index', 0) if state is None else state.current_player_index
        
        current_player_vp = 0
        num_resources = 0
        if state is not None and current_player_idx < len(state.players):
            current_player = state.players[current_player_idx]
            current_player_vp = current_player.victory_points
            num_resources = sum(current_player.resources.values())
        
        # Extract structural features
        action_payload_hash = self._hash_payload_structure(expected_action.get('payload'))
        name_tokens = self._tokenize_name(drill['name'])
        
        return DrillFeatures(
            drill_id=drill_id,
            has_guideline=has_guideline,
            state_embedding=state_embedding,
            guideline_embedding=guideline_embedding,
            action_type=action_type,
            phase=phase,
            turn_number=turn_number,
            current_player_vp=current_player_vp,
            num_resources=num_resources,
            action_payload_hash=action_payload_hash,
            name_tokens=name_tokens
        )


class DrillClusterer:
    """
    Cluster hard drills to identify common failure patterns.
    
    Strategy:
    1. Cluster drills WITH guidelines based on guideline similarity
    2. Assign drills WITHOUT guidelines to nearest guideline cluster
    """
    
    def __init__(self, min_cluster_size: int = 3):
        """
        Initialize clusterer.
        
        Args:
            min_cluster_size: Minimum drills per cluster (for HDBSCAN)
        """
        self.min_cluster_size = min_cluster_size
    
    def _create_feature_matrix(
        self,
        features_list: List[DrillFeatures],
        use_embeddings: bool = True
    ) -> np.ndarray:
        """
        Create feature matrix for clustering.
        
        Combines embeddings and categorical/numerical features.
        """
        n_drills = len(features_list)
        
        # Determine feature dimensions
        embedding_dim = 1536 if use_embeddings else 0  # text-embedding-3-small dimension
        
        # Categorical features (one-hot encoded)
        action_types = set(f.action_type for f in features_list)
        phases = set(f.phase for f in features_list)
        categorical_dim = len(action_types) + len(phases)
        
        # Numerical features
        numerical_dim = 4  # turn_number, vp, num_resources, has_guideline
        
        total_dim = embedding_dim * 2 + categorical_dim + numerical_dim  # 2x embeddings (state + guideline)
        
        # Build matrix
        feature_matrix = np.zeros((n_drills, total_dim))
        
        for i, features in enumerate(features_list):
            col = 0
            
            # State embedding
            if use_embeddings and features.state_embedding is not None:
                feature_matrix[i, col:col+embedding_dim] = features.state_embedding
            col += embedding_dim
            
            # Guideline embedding (zero if no guideline)
            if use_embeddings and features.guideline_embedding is not None:
                feature_matrix[i, col:col+embedding_dim] = features.guideline_embedding
            col += embedding_dim
            
            # One-hot encode action type
            action_idx = list(action_types).index(features.action_type)
            feature_matrix[i, col + action_idx] = 1.0
            col += len(action_types)
            
            # One-hot encode phase
            phase_idx = list(phases).index(features.phase)
            feature_matrix[i, col + phase_idx] = 1.0
            col += len(phases)
            
            # Numerical features (normalized)
            feature_matrix[i, col] = features.turn_number / 50.0  # Normalize turn
            feature_matrix[i, col + 1] = features.current_player_vp / 10.0  # Normalize VP
            feature_matrix[i, col + 2] = features.num_resources / 20.0  # Normalize resources
            feature_matrix[i, col + 3] = 1.0 if features.has_guideline else 0.0
        
        return feature_matrix
    
    def cluster_drills(
        self,
        features_list: List[DrillFeatures],
        num_clusters: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster drills using HDBSCAN or K-means.
        
        Args:
            features_list: List of DrillFeatures objects
            num_clusters: Number of clusters (None = auto-detect with HDBSCAN)
            
        Returns:
            (cluster_labels, clustering_info) tuple
        """
        # Create feature matrix
        X = self._create_feature_matrix(features_list, use_embeddings=True)
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        if num_clusters is None:
            # Use HDBSCAN for automatic cluster detection
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    metric='euclidean',
                    cluster_selection_epsilon=0.5
                )
                labels = clusterer.fit_predict(X_scaled)
                
                clustering_info = {
                    'method': 'hdbscan',
                    'num_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'num_noise': sum(1 for l in labels if l == -1)
                }
                
                print(f"HDBSCAN found {clustering_info['num_clusters']} clusters", flush=True)
                print(f"  Noise points: {clustering_info['num_noise']}", flush=True)
                
            except ImportError:
                print("Warning: hdbscan not installed, falling back to K-means with k=6", flush=True)
                num_clusters = 6
        
        if num_clusters is not None:
            # Use K-means with specified number of clusters
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            clustering_info = {
                'method': 'kmeans',
                'num_clusters': num_clusters,
                'inertia': kmeans.inertia_
            }
            
            print(f"K-means with k={num_clusters}", flush=True)
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for label in set(labels):
            if label != -1:  # Exclude noise
                cluster_sizes[int(label)] = sum(1 for l in labels if l == label)
        
        clustering_info['cluster_sizes'] = cluster_sizes
        
        return labels, clustering_info

