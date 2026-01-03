"""
DSPy-based prompt optimizer for LLM agents.

This module uses DSPy to optimize the system prompt (rules/instructions) while keeping
the observation and action space formatting fixed.
"""
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    # Create a dummy dspy module for type hints when DSPy is not installed
    class DummyInputField:
        def __init__(self, *args, **kwargs):
            pass
    class DummyOutputField:
        def __init__(self, *args, **kwargs):
            pass
    class DummySignature:
        pass
    # Metric is just a callable function in DSPy 3.0+, not a class
    class DummyChainOfThought:
        def __init__(self, *args, **kwargs):
            pass
    class DummyBootstrapFewShot:
        def __init__(self, *args, **kwargs):
            pass
        def compile(self, *args, **kwargs):
            return DummyChainOfThought()
    class DummyMIPRO:
        def __init__(self, *args, **kwargs):
            pass
        def compile(self, *args, **kwargs):
            return DummyChainOfThought()
    class DummyLM:
        def __init__(self, *args, **kwargs):
            pass
    class DummyExample:
        def __init__(self, *args, **kwargs):
            pass
        def with_inputs(self, *args, **kwargs):
            return self
    class DummyDSPy:
        Signature = DummySignature
        InputField = DummyInputField
        OutputField = DummyOutputField
        ChainOfThought = DummyChainOfThought
        BootstrapFewShot = DummyBootstrapFewShot
        MIPRO = DummyMIPRO
        LM = DummyLM
        Example = DummyExample
        @staticmethod
        def configure(**kwargs):
            pass
    dspy = DummyDSPy()

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from engine import (
    GameState,
    Action,
    ActionPayload,
    deserialize_game_state,
    serialize_action,
    serialize_action_payload,
    state_to_text,
    legal_actions_to_text,
    legal_actions,
)
from agents.llm_agent import LLMAgent


@dataclass
class DrillExample:
    """A single drill step example for optimization."""
    state: GameState
    player_id: str
    legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    state_text: str
    legal_actions_text: str
    correct_actions: List[Dict[str, Any]]
    incorrect_actions: Optional[List[Dict[str, Any]]] = None
    include_higher_level_features: bool = False


class CatanActionSignature(dspy.Signature):
    """DSPy signature for Catan action selection."""
    state_text: str = dspy.InputField(desc="Current game state description")
    legal_actions_text: str = dspy.InputField(desc="Available legal actions")
    action_type: str = dspy.OutputField(desc="The action type to take")
    action_payload: str = dspy.OutputField(desc="JSON string of action payload, or null")


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
            prediction: The predicted action (CatanActionSignature or dict)
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
            
            # Get prediction - handle both CatanActionSignature and dict
            if hasattr(prediction, 'action_type'):
                action_type = prediction.action_type
                payload_str = prediction.action_payload
            elif isinstance(prediction, dict):
                action_type = prediction.get('action_type', '')
                payload_str = prediction.get('action_payload', 'null')
            else:
                return 0.0
            
            # Parse payload if not null
            payload_dict = None
            if payload_str and payload_str.lower() != "null" and payload_str:
                try:
                    payload_dict = json.loads(payload_str)
                except (json.JSONDecodeError, TypeError):
                    payload_dict = None
            
            # Build action dict
            predicted_action = {"type": action_type}
            if payload_dict:
                predicted_action["payload"] = payload_dict
            
            # Check if it matches any correct action
            from api.routes import _canonical_action_dict
            
            canonical_predicted = _canonical_action_dict(predicted_action)
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


class PromptOptimizer:
    """Optimizes system prompts using DSPy."""
    
    def __init__(
        self,
        base_system_prompt: str,
        model_name: str = "gpt-4o-mini",
        include_higher_level_features: bool = False,
        api_key: Optional[str] = None
    ):
        """
        Initialize the prompt optimizer.
        
        Args:
            base_system_prompt: The base system prompt to optimize
            model_name: LLM model to use for optimization
            include_higher_level_features: Whether to include higher-level features in prompts
            api_key: Optional API key (uses env vars if not provided)
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                "dspy-ai is not installed. Install it with: pip install dspy-ai"
            )
        
        self.base_system_prompt = base_system_prompt
        self.model_name = model_name
        self.include_higher_level_features = include_higher_level_features
        
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
        # Note: DSPy uses litellm under the hood, so we can use the same model names
        # LiteLLM will automatically pick up API keys from environment variables
        lm = dspy.LM(model=model_name)
        dspy.configure(lm=lm)
        
        # Create the module
        self.module = dspy.ChainOfThought(CatanActionSignature)
    
    def prepare_examples(
        self,
        drills: List[Dict[str, Any]]
    ) -> List[DrillExample]:
        """
        Prepare drill examples for optimization.
        
        Args:
            drills: List of drill data from database
            
        Returns:
            List of DrillExample objects
        """
        examples = []
        
        for drill in drills:
            drill_id = drill["id"]
            steps = drill.get("steps", [])
            
            for step in steps:
                # Load state
                state_json = step["state"]
                state = deserialize_game_state(state_json)
                player_id = step["player_id"]
                
                # Get legal actions
                la_list = legal_actions(state, player_id)
                
                # Get state and actions text
                state_text = state_to_text(
                    state,
                    player_id,
                    exclude_higher_level_features=not self.include_higher_level_features
                )
                actions_text = legal_actions_to_text(la_list, state=state, player_id=player_id)
                
                # Get correct/incorrect actions
                correct_actions = step.get("correct_actions")
                if not correct_actions:
                    # Fall back to expected_action for backward compatibility
                    expected_action = step.get("expected_action")
                    if expected_action:
                        correct_actions = [expected_action]
                
                if not correct_actions:
                    continue  # Skip steps without correct actions
                
                incorrect_actions = step.get("incorrect_actions")
                
                examples.append(DrillExample(
                    state=state,
                    player_id=player_id,
                    legal_actions_list=la_list,
                    state_text=state_text,
                    legal_actions_text=actions_text,
                    correct_actions=correct_actions,
                    incorrect_actions=incorrect_actions,
                    include_higher_level_features=self.include_higher_level_features
                ))
        
        return examples
    
    def optimize(
        self,
        train_examples: List[DrillExample],
        method: str = "bootstrap",
        num_iterations: int = 10,
        val_examples: Optional[List[DrillExample]] = None
    ) -> str:
        """
        Optimize the system prompt using DSPy.
        
        Args:
            train_examples: Training examples
            method: Optimization method ("bootstrap", "miprov2", "gepa", "copro")
            num_iterations: Number of optimization iterations
            val_examples: Optional validation examples. If None, uses train_examples for validation.
            
        Returns:
            Optimized system prompt
        """
        # Convert examples to DSPy format
        # We need to attach the DrillExample to each dspy.Example for the metric to access correct_actions
        dspy_examples = []
        for ex in train_examples:
            dspy_ex = dspy.Example(
                state_text=ex.state_text,
                legal_actions_text=ex.legal_actions_text,
                action_type="",  # Will be filled by metric
                action_payload="null"
            ).with_inputs("state_text", "legal_actions_text")
            # Attach the DrillExample so metric can access correct_actions
            dspy_ex._drill_example = ex
            dspy_examples.append(dspy_ex)
        
        # Prepare validation examples if provided
        val_dspy_examples = None
        if val_examples:
            val_dspy_examples = []
            for ex in val_examples:
                dspy_ex = dspy.Example(
                    state_text=ex.state_text,
                    legal_actions_text=ex.legal_actions_text,
                    action_type="",
                    action_payload="null"
                ).with_inputs("state_text", "legal_actions_text")
                dspy_ex._drill_example = ex
                val_dspy_examples.append(dspy_ex)
            print(f"Using {len(val_dspy_examples)} examples for validation", flush=True)
        else:
            print(f"Using training set for validation (no separate val set)", flush=True)
        
        # Create metric function
        metric = create_drill_metric(train_examples)
        
        # Run optimization
        print(f"Starting {method} optimization with {len(dspy_examples)} training examples...", flush=True)
        if method == "bootstrap":
            optimizer = dspy.BootstrapFewShot(metric=metric)
            print("Compiling with BootstrapFewShot...", flush=True)
            optimized_module = optimizer.compile(
                student=self.module,
                trainset=dspy_examples[:min(len(dspy_examples), 50)]  # Limit for efficiency
            )
            print("BootstrapFewShot optimization complete.", flush=True)
        elif method == "mipro" or method == "miprov2":
            # Use MIPROv2 (MIPRO doesn't exist in DSPy 3.0.4)
            if hasattr(dspy, 'MIPROv2'):
                print("Compiling with MIPROv2...", flush=True)
                optimizer = dspy.MIPROv2(metric=metric)
                optimized_module = optimizer.compile(
                    student=self.module,
                    trainset=dspy_examples[:min(len(dspy_examples), 50)]
                )
                print("MIPROv2 optimization complete.", flush=True)
            else:
                raise ValueError("MIPROv2 not available in this DSPy version. Use 'bootstrap' instead.")
        elif method == "gepa":
            # GEPA (Genetic Evolution of Prompts and Agents) - requires special metric format
            if hasattr(dspy, 'GEPA'):
                # GEPA expects a metric that can return ScoreWithFeedback
                # For now, we'll use a simple wrapper
                def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
                    # Convert to our metric format
                    score = metric(gold, pred, trace)
                    return score
                
                # GEPA requires a reflection LM for proposing new instructions
                print(f"Setting up GEPA with reflection LM ({self.model_name})...", flush=True)
                # Adjust max_tokens based on model - gpt-4o-mini supports max 16384
                max_tokens = 16384 if "gpt-4o-mini" in self.model_name.lower() else 32000
                reflection_lm = dspy.LM(model=self.model_name, temperature=1.0, max_tokens=max_tokens)
                
                print("Compiling with GEPA (this may take a while)...", flush=True)
                optimizer = dspy.GEPA(
                    metric=gepa_metric,
                    auto='light',  # Use light mode for faster testing
                    reflection_lm=reflection_lm
                )
                compile_kwargs = {
                    "student": self.module,
                    "trainset": dspy_examples[:min(len(dspy_examples), 50)]
                }
                # GEPA can use trainset for both if no valset provided
                if val_dspy_examples:
                    compile_kwargs["valset"] = val_dspy_examples[:min(len(val_dspy_examples), 50)]
                optimized_module = optimizer.compile(**compile_kwargs)
                print("GEPA optimization complete.", flush=True)
            else:
                raise ValueError("GEPA not available in this DSPy version. Use 'bootstrap' instead.")
        elif method == "copro":
            if hasattr(dspy, 'COPRO'):
                print("Compiling with COPRO...", flush=True)
                optimizer = dspy.COPRO(metric=metric)
                optimized_module = optimizer.compile(
                    student=self.module,
                    trainset=dspy_examples[:min(len(dspy_examples), 50)]
                )
                print("COPRO optimization complete.", flush=True)
            else:
                raise ValueError("COPRO not available in this DSPy version. Use 'bootstrap' instead.")
        else:
            raise ValueError(f"Unknown optimization method: {method}. Available: bootstrap, miprov2, gepa, copro")
        
        # Extract optimized prompt
        # DSPy optimizes by adding demonstrations (few-shot examples) to the predictor
        # The optimized module now contains these demonstrations internally
        # For BootstrapFewShot, the optimization adds demos that are used during inference
        # 
        # Note: DSPy doesn't directly expose the optimized prompt as a string
        # The demos are stored in the predictor and used at runtime
        # For now, we return the base prompt - the optimization is in the module itself
        # 
        # In practice, you would use the optimized_module directly for inference,
        # and it would automatically include the optimized demonstrations
        
        # Try to extract any instructions or demos if available
        try:
            # Check if there are predictors with demos
            if hasattr(optimized_module, 'predictors'):
                predictors = optimized_module.predictors
                if predictors:
                    # The first predictor typically contains the optimized demos
                    predictor = predictors[0]
                    # Check if it has demos
                    if hasattr(predictor, 'demos') and predictor.demos:
                        # Format demos as part of the prompt
                        demo_text = "\n\n## Optimized Examples:\n"
                        for demo in predictor.demos[:3]:  # Show first 3 demos
                            demo_text += f"\nExample: {demo}\n"
                        return f"{self.base_system_prompt}{demo_text}"
        except Exception:
            pass
        
        # Fallback: return base prompt (optimization is in the module, not the prompt string)
        # The optimized_module should be used directly for inference
        return self.base_system_prompt
    
    def evaluate(
        self,
        test_examples: List[DrillExample],
        system_prompt: str
    ) -> Dict[str, Any]:
        """
        Evaluate a system prompt on test examples.
        
        Args:
            test_examples: Test examples
            system_prompt: System prompt to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create a temporary agent with the system prompt
        # For evaluation, we'll use the LLMAgent with a custom prompt
        # This is a simplified evaluation - in practice, you'd want to run full agent evaluation
        
        correct = 0
        total = 0
        
        for ex in test_examples:
            total += 1
            # In a real implementation, we'd run the agent and check the result
            # For now, this is a placeholder
            # The actual evaluation should be done through the drill evaluation endpoint
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

