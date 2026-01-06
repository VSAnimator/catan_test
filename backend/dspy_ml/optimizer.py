"""
DSPy optimizer for drill performance using GEPA.
"""
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    raise ImportError("dspy-ai is not installed. Install it with: pip install dspy-ai")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .signature import CatanDrillSignature
from .dataset import DrillExample
from api.routes import _canonical_action_dict


def create_drill_metric(examples: List[DrillExample]) -> callable:
    """
    Create a metric function that evaluates drill performance.
    
    In DSPy 3.0+, metrics are callable functions that take (example, prediction, trace=None)
    and return a score.
    """
    # Create a mapping from example to correct actions for quick lookup
    example_to_correct_actions = {id(ex): ex.correct_actions for ex in examples}
    
    def drill_metric(example, prediction, trace=None) -> float:
        """
        Evaluate if the predicted action matches any correct action.
        
        Args:
            example: The drill example (DrillExample or dspy.Example)
            prediction: The predicted action (CatanDrillSignature or dict)
            trace: Optional trace (not used)
        
        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        try:
            # Get correct actions - handle both DrillExample and dspy.Example
            if isinstance(example, DrillExample):
                correct_actions = example.correct_actions
            else:
                # For dspy.Example, get the attached DrillExample
                drill_example = getattr(example, '_drill_example', None)
                if drill_example:
                    correct_actions = drill_example.correct_actions
                else:
                    # Fallback: try to get correct_actions directly
                    correct_actions = getattr(example, 'correct_actions', [])
            
            # Get prediction - handle both CatanDrillSignature and dict
            if hasattr(prediction, 'chosen_action'):
                chosen_action_str = prediction.chosen_action
            elif isinstance(prediction, dict):
                chosen_action_str = prediction.get('chosen_action', 'null')
            else:
                return 0.0
            
            # Parse chosen_action JSON string
            if not chosen_action_str or chosen_action_str.lower() == "null":
                return 0.0
            
            try:
                chosen_action = json.loads(chosen_action_str)
            except (json.JSONDecodeError, TypeError):
                return 0.0
            
            # Check if it matches any correct action
            canonical_predicted = _canonical_action_dict(chosen_action)
            for correct_action in correct_actions:
                if _canonical_action_dict(correct_action) == canonical_predicted:
                    return 1.0
            
            return 0.0
        except Exception as e:
            print(f"Error in drill_metric: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 0.0
    
    return drill_metric


class DrillOptimizer:
    """Optimizes DSPy modules using GEPA for drill performance."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        gepa_auto: str = "light"
    ):
        """
        Initialize the optimizer.
        
        Args:
            model_name: LLM model to use for optimization
            api_key: Optional API key (uses env vars if not provided)
            gepa_auto: GEPA auto mode ('light' or 'full')
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                "dspy-ai is not installed. Install it with: pip install dspy-ai"
            )
        
        self.model_name = model_name
        self.gepa_auto = gepa_auto
        
        # Set API key in environment if provided
        if api_key:
            import os
            # Determine provider from model name and set appropriate env var
            if "gpt" in model_name.lower() or "openai" in model_name.lower():
                os.environ["OPENAI_API_KEY"] = api_key
            elif "claude" in model_name.lower() or "anthropic" in model_name.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif "gemini" in model_name.lower() or "google" in model_name.lower():
                os.environ["GEMINI_API_KEY"] = api_key
            else:
                # Default to OpenAI
                os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize DSPy with the model
        lm = dspy.LM(model=model_name)
        dspy.configure(lm=lm)
        
        # Create the module
        self.module = dspy.ChainOfThought(CatanDrillSignature)
    
    def optimize(
        self,
        train_examples: List[DrillExample],
        val_examples: Optional[List[DrillExample]] = None
    ) -> Any:
        """
        Optimize the DSPy module using GEPA.
        
        Args:
            train_examples: Training examples
            val_examples: Optional validation examples
            
        Returns:
            Optimized DSPy module
        """
        # Convert examples to DSPy format
        dspy_examples = []
        for ex in train_examples:
            dspy_ex = dspy.Example(
                game_rules=ex.game_rules,
                observation=ex.observation,
                viable_actions=ex.viable_actions,
                reasoning="",  # Will be filled by metric
                chosen_action="null"
            ).with_inputs("game_rules", "observation", "viable_actions")
            # Attach the DrillExample so metric can access correct_actions
            dspy_ex._drill_example = ex
            dspy_examples.append(dspy_ex)
        
        # Prepare validation examples if provided
        val_dspy_examples = None
        if val_examples:
            val_dspy_examples = []
            for ex in val_examples:
                dspy_ex = dspy.Example(
                    game_rules=ex.game_rules,
                    observation=ex.observation,
                    viable_actions=ex.viable_actions,
                    reasoning="",
                    chosen_action="null"
                ).with_inputs("game_rules", "observation", "viable_actions")
                dspy_ex._drill_example = ex
                val_dspy_examples.append(dspy_ex)
        
        # Create metric function
        metric = create_drill_metric(train_examples)
        
        # Run GEPA optimization
        if not hasattr(dspy, 'GEPA'):
            raise ValueError("GEPA not available in this DSPy version.")
        
        # GEPA requires a reflection LM for proposing new instructions
        print(f"Setting up GEPA with reflection LM ({self.model_name})...", flush=True)
        # Adjust max_tokens based on model
        max_tokens = 16384 if "gpt-4o-mini" in self.model_name.lower() else 32000
        reflection_lm = dspy.LM(model=self.model_name, temperature=1.0, max_tokens=max_tokens)
        
        print("Compiling with GEPA (this may take a while)...", flush=True)
        optimizer = dspy.GEPA(
            metric=metric,
            auto=self.gepa_auto,
            reflection_lm=reflection_lm
        )
        
        compile_kwargs = {
            "student": self.module,
            "trainset": dspy_examples
        }
        if val_dspy_examples:
            compile_kwargs["valset"] = val_dspy_examples
        
        optimized_module = optimizer.compile(**compile_kwargs)
        print("GEPA optimization complete.", flush=True)
        
        return optimized_module
    
    def extract_optimized_prompt(self, module: Any) -> Optional[str]:
        """
        Extract optimized instructions/prompt from GEPA-optimized module.
        
        This is optional - GEPA optimizes instructions which may be embedded
        in the module's predictors.
        
        Args:
            module: Optimized DSPy module
            
        Returns:
            Optimized prompt string if extractable, None otherwise
        """
        try:
            # GEPA optimizes instructions, which may be in the module's predictors
            if hasattr(module, 'predictors'):
                predictors = module.predictors
                if predictors:
                    predictor = predictors[0]
                    # Check for optimized instructions
                    if hasattr(predictor, 'instructions'):
                        return predictor.instructions
                    # Check for signature instructions
                    if hasattr(predictor, 'signature') and hasattr(predictor.signature, 'instructions'):
                        return predictor.signature.instructions
        except Exception as e:
            print(f"Warning: Could not extract optimized prompt: {e}", flush=True)
        
        return None
    
    def save(
        self,
        module: Any,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save optimized module and metadata.
        
        Args:
            module: Optimized DSPy module
            filepath: Path to save module (will also save metadata as .json)
            metadata: Optional metadata to save
        """
        # Save module
        with open(filepath, 'wb') as f:
            pickle.dump(module, f)
        
        # Save metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "model_name": self.model_name,
            "optimization_method": "gepa",
            "gepa_auto": self.gepa_auto,
            "saved_at": datetime.now().isoformat(),
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, filepath: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load optimized module and metadata.
        
        Args:
            filepath: Path to module file
            
        Returns:
            (module, metadata) tuple
        """
        # Load module
        with open(filepath, 'rb') as f:
            module = pickle.load(f)
        
        # Load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        metadata = {}
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return module, metadata
    
    def list_models(self, directory: str = "data/optimized_modules") -> List[Dict[str, Any]]:
        """
        List available optimized modules with metadata.
        
        Args:
            directory: Directory containing optimized modules
            
        Returns:
            List of model info dicts
        """
        models = []
        dir_path = Path(__file__).parent.parent / directory
        
        if not dir_path.exists():
            return models
        
        for pkl_file in dir_path.glob("*.pkl"):
            metadata_path = pkl_file.with_suffix('_metadata.json')
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            models.append({
                "filepath": str(pkl_file),
                "name": pkl_file.stem,
                "metadata": metadata
            })
        
        return models

