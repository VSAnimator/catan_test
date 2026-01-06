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
    
    GEPA requires metrics with signature: (gold, pred, trace, pred_name, pred_trace)
    """
    # Create a mapping from example to correct actions for quick lookup
    example_to_correct_actions = {id(ex): ex.correct_actions for ex in examples}
    
    def drill_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
        """
        Evaluate if the predicted action matches any correct action.
        
        Args:
            gold: The gold/example (DrillExample or dspy.Example)
            pred: The predicted action (CatanDrillSignature or dict)
            trace: Optional trace (not used)
            pred_name: Predictor name (not used)
            pred_trace: Predictor trace (not used)
        
        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        try:
            # Get correct actions - handle both DrillExample and dspy.Example
            if isinstance(gold, DrillExample):
                correct_actions = gold.correct_actions
            else:
                # For dspy.Example, get the attached DrillExample
                drill_example = getattr(gold, '_drill_example', None)
                if drill_example:
                    correct_actions = drill_example.correct_actions
                else:
                    # Fallback: try to get correct_actions directly
                    correct_actions = getattr(gold, 'correct_actions', [])
            
            # Get prediction - handle both CatanDrillSignature and dict
            if hasattr(pred, 'chosen_action'):
                chosen_action_str = pred.chosen_action
            elif isinstance(pred, dict):
                chosen_action_str = pred.get('chosen_action', 'null')
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
        gepa_auto: str = "light",
        reflection_model: Optional[str] = None,
        reflection_reasoning_effort: Optional[str] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            model_name: LLM model to use for optimization
            api_key: Optional API key (uses env vars if not provided)
            gepa_auto: GEPA auto mode ('light' or 'full')
            reflection_model: Model to use for GEPA reflection (defaults to model_name)
            reflection_reasoning_effort: Reasoning effort for reflection model ('low', 'medium', 'high')
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                "dspy-ai is not installed. Install it with: pip install dspy-ai"
            )
        
        self.model_name = model_name
        self.gepa_auto = gepa_auto
        self.reflection_model = reflection_model or model_name
        self.reflection_reasoning_effort = reflection_reasoning_effort
        
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
        print(f"Setting up GEPA with reflection LM ({self.reflection_model})...", flush=True)
        
        # Adjust max_tokens based on model
        max_tokens = 16384 if "gpt-4o-mini" in self.reflection_model.lower() else 32000
        
        # Configure reflection LM with thinking/reasoning if specified
        reflection_lm_kwargs = {
            "model": self.reflection_model,
            "temperature": 1.0,
            "max_tokens": max_tokens
        }
        
        # Add reasoning effort if specified (for models that support extended thinking)
        if self.reflection_reasoning_effort:
            print(f"  Enabling extended thinking with reasoning_effort={self.reflection_reasoning_effort}", flush=True)
            reflection_lm_kwargs["reasoning_effort"] = self.reflection_reasoning_effort
        
        reflection_lm = dspy.LM(**reflection_lm_kwargs)
        
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
            # GEPA optimizes instructions - check various locations
            # For ChainOfThought, instructions are in predict.signature.instructions
            if hasattr(module, 'predict'):
                predictor = module.predict
                if hasattr(predictor, 'signature') and hasattr(predictor.signature, 'instructions'):
                    return predictor.signature.instructions
                if hasattr(predictor, 'instructions'):
                    return predictor.instructions
            
            # Try other predictor locations
            if hasattr(module, 'predictors'):
                predictors = module.predictors if not callable(module.predictors) else module.predictors()
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
        # Try DSPy's built-in save method first
        if hasattr(module, 'save'):
            try:
                module.save(filepath)
                print(f"Saved module using DSPy's built-in save method", flush=True)
            except Exception as e:
                print(f"Warning: DSPy save failed: {e}", flush=True)
                # Fall back to custom serialization
                self._save_fallback(module, filepath)
        else:
            # Use fallback method
            self._save_fallback(module, filepath)
        
        # Save metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "model_name": self.model_name,
            "optimization_method": "gepa",
            "gepa_auto": self.gepa_auto,
            "reflection_model": self.reflection_model,
            "reflection_reasoning_effort": self.reflection_reasoning_effort,
            "saved_at": datetime.now().isoformat(),
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_fallback(self, module: Any, filepath: str) -> None:
        """
        Fallback save method that extracts serializable state.
        
        This handles cases where the module contains non-picklable objects
        (like DSPy StringSignature instances).
        """
        try:
            # Extract optimized instructions
            optimized_instructions = self.extract_optimized_prompt(module)
            
            print(f"Extracted optimized instructions: {len(optimized_instructions) if optimized_instructions else 0} chars", flush=True)
            
            # Build serializable state dict
            state_dict = {
                'module_type': type(module).__name__,
                'optimized_instructions': optimized_instructions,
                'model_name': self.model_name,
                'reflection_model': self.reflection_model,
                'gepa_auto': self.gepa_auto,
            }
            
            # Try to extract signature information
            if hasattr(module, 'signature'):
                sig = module.signature
                state_dict['signature_type'] = type(sig).__name__
                
                # Extract field information
                if hasattr(sig, 'fields'):
                    state_dict['signature_fields'] = {
                        'input_fields': [f for f in sig.fields if sig.fields[f].json_schema_extra.get('__dspy_field_type') == 'input'],
                        'output_fields': [f for f in sig.fields if sig.fields[f].json_schema_extra.get('__dspy_field_type') == 'output']
                    }
            
            # Save as pickle
            with open(filepath, 'wb') as f:
                pickle.dump(state_dict, f)
            
            print(f"Saved module using fallback method (extracted instructions)", flush=True)
            
        except Exception as e:
            print(f"Error in fallback save: {e}", flush=True)
            raise
    
    def load(self, filepath: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load optimized module and metadata.
        
        Args:
            filepath: Path to module file
            
        Returns:
            (module, metadata) tuple
        """
        from .signature import CatanDrillSignature
        
        # Try DSPy's load first
        try:
            module = dspy.ChainOfThought(CatanDrillSignature)
            module.load(filepath)
            print("Loaded module using DSPy's built-in load method", flush=True)
        except:
            # Fall back to pickle load
            with open(filepath, 'rb') as f:
                loaded = pickle.load(f)
            
            # Check if this is a state dict (fallback format) or actual module
            if isinstance(loaded, dict):
                if 'optimized_instructions' in loaded:
                    # This is a fallback-saved module, reconstruct it
                    print("Loading from fallback format, reconstructing module...", flush=True)
                    module = self._reconstruct_module(loaded)
                else:
                    # This might be DSPy's dict format, try to reconstruct
                    print("Loading from DSPy dict format, reconstructing module...", flush=True)
                    module = self._reconstruct_from_dspy_dict(loaded)
            else:
                # This is a regular module
                module = loaded
        
        # Load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        metadata = {}
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return module, metadata
    
    def _reconstruct_from_dspy_dict(self, dspy_dict: Dict[str, Any]) -> Any:
        """
        Reconstruct module from DSPy's save format (which saves as dict).
        
        Args:
            dspy_dict: Dictionary from DSPy's save method
            
        Returns:
            Reconstructed module
        """
        from .signature import CatanDrillSignature
        
        # Create new module
        module = dspy.ChainOfThought(CatanDrillSignature)
        
        # Try to restore state from the dict
        # DSPy saves module.__dict__, so we can restore it
        if hasattr(module, '__dict__'):
            module.__dict__.update(dspy_dict)
        
        return module
    
    def _reconstruct_module(self, state_dict: Dict[str, Any]) -> Any:
        """
        Reconstruct a DSPy module from saved state dict.
        
        Args:
            state_dict: State dictionary from fallback save
            
        Returns:
            Reconstructed DSPy module
        """
        from .signature import CatanDrillSignature
        
        # Create a new module with the original signature
        module = dspy.ChainOfThought(CatanDrillSignature)
        
        # Apply the optimized instructions if available
        if state_dict.get('optimized_instructions'):
            # Set instructions on the module's predictor
            # ChainOfThought uses 'predict' attribute
            if hasattr(module, 'predict') and hasattr(module.predict, 'signature'):
                module.predict.signature.instructions = state_dict['optimized_instructions']
            elif hasattr(module, '__dict__'):
                # Try to find predictor in module's attributes
                for attr_name, attr_value in module.__dict__.items():
                    if hasattr(attr_value, 'signature'):
                        attr_value.signature.instructions = state_dict['optimized_instructions']
                        break
        
        return module
    
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

