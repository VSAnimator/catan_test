"""
LLM-based validation for bug test results.

Uses LLM to check if test results match expected behavior.
"""
import os
import json
from typing import Dict, Any, Optional
from engine.serialization import state_to_text


def validate_with_llm(
    test_description: str,
    expected_behavior: str,
    undesired_behavior: str,
    state_before: Dict[str, Any],
    state_after: Optional[Dict[str, Any]],
    test_result: Dict[str, Any],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Validate test results using LLM.
    
    Args:
        test_description: Description of what is being tested
        expected_behavior: What should happen (desired behavior)
        undesired_behavior: What should NOT happen (bug behavior)
        state_before: Game state before the test action
        state_after: Game state after the test action (if action was taken)
        test_result: Test execution result dictionary
        model: LLM model to use
        api_key: Optional API key (uses env vars if not provided)
        
    Returns:
        Dictionary with validation results:
        - valid: bool - whether test passed validation
        - reasoning: str - LLM's reasoning
        - issues: List[str] - any issues found
    """
    try:
        import litellm
        
        # Set API key if provided
        if api_key:
            if model.startswith("claude") or "anthropic" in model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif model.startswith("gemini") or "google" in model.lower():
                os.environ["GEMINI_API_KEY"] = api_key
            elif model.startswith("gpt") or "openai" in model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
        
        # Build validation prompt
        state_before_text = state_to_text(state_before, player_id=None) if state_before else "N/A"
        state_after_text = state_to_text(state_after, player_id=None) if state_after else "N/A"
        
        prompt = f"""You are validating a bug regression test for a Catan game engine.

## Test Description:
{test_description}

## Expected Behavior (DESIRED):
{expected_behavior}

## Undesired Behavior (BUG - should NOT happen):
{undesired_behavior}

## Game State Before Test:
{state_before_text}

## Game State After Test:
{state_after_text}

## Test Execution Result:
{json.dumps(test_result, indent=2)}

## Your Task:
Analyze whether the test results match the expected behavior and do NOT exhibit the undesired behavior.

Respond in JSON format:
{{
  "valid": true/false,
  "reasoning": "Your detailed reasoning about whether the test passed or failed",
  "issues": ["list", "of", "any", "issues", "found"]
}}

Be thorough in your analysis. Check:
1. Does the final state match what was expected?
2. Are there any signs of the undesired behavior?
3. Did the test execute correctly?
4. Are there any inconsistencies or errors?
"""
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are a test validation expert for game engines. Analyze test results carefully and provide detailed feedback."},
            {"role": "user", "content": prompt}
        ]
        
        litellm.drop_params = True
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent validation
        )
        
        response_text = response.choices[0].message.content
        
        # Try to parse JSON from response
        try:
            # Extract JSON if wrapped in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            validation_result = json.loads(response_text)
            return validation_result
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return {
                "valid": False,
                "reasoning": f"Failed to parse LLM response as JSON. Response: {response_text}",
                "issues": ["LLM response format error"]
            }
    
    except ImportError:
        return {
            "valid": False,
            "reasoning": "litellm not installed. Cannot perform LLM validation.",
            "issues": ["Missing dependency: litellm"]
        }
    except Exception as e:
        return {
            "valid": False,
            "reasoning": f"LLM validation error: {str(e)}",
            "issues": [f"Validation error: {str(e)}"]
        }

