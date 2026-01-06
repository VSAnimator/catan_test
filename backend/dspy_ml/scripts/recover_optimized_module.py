#!/usr/bin/env python3
"""
Recover optimized DSPy module from GEPA optimization log.

When optimization completes but saving fails, this script extracts
the best instructions from the log and reconstructs the module.
"""
import sys
import re
import json
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.signature import CatanDrillSignature
from dspy_ml.optimizer import DrillOptimizer


def extract_best_instructions_from_log(log_path):
    """
    Parse GEPA optimization log to find the best performing instructions.
    
    Returns:
        (instructions_text, score, iteration) tuple
    """
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Find all iterations with their scores
    # Pattern: "Iteration X: Full valset score for new program: Y"
    iteration_pattern = r'Iteration (\d+): Full valset score for new program: ([\d.]+)'
    iterations = re.findall(iteration_pattern, log_content)
    
    # Find the best iteration
    best_iteration = None
    best_score = 0.0
    for iter_num, score in iterations:
        score_float = float(score)
        if score_float > best_score:
            best_score = score_float
            best_iteration = int(iter_num)
    
    if best_iteration is None:
        print("No iterations with improved scores found in log", flush=True)
        return None, None, None
    
    print(f"Best iteration: {best_iteration} with score: {best_score:.4f}", flush=True)
    
    # Extract the instructions proposed at that iteration
    # Pattern: "Iteration X: Proposed new text for predict: text\n<instructions>"
    # The instructions continue until the next log line starting with a timestamp
    
    # Find the proposed text for the best iteration
    pattern = f'Iteration {best_iteration}: Proposed new text for predict: text\n(.*?)\n\\d{{4}}/\\d{{2}}/\\d{{2}}'
    match = re.search(pattern, log_content, re.DOTALL)
    
    if match:
        instructions = match.group(1).strip()
        return instructions, best_score, best_iteration
    
    # If not found, the instructions might span multiple lines differently
    # Try to find it by looking for the iteration and reading until next timestamp
    lines = log_content.split('\n')
    in_instructions = False
    instructions_lines = []
    
    for i, line in enumerate(lines):
        if f'Iteration {best_iteration}: Proposed new text for predict: text' in line:
            in_instructions = True
            continue
        
        if in_instructions:
            # Stop at next timestamp line or next iteration
            if re.match(r'^\d{4}/\d{2}/\d{2}', line) or 'Iteration' in line:
                break
            instructions_lines.append(line)
    
    if instructions_lines:
        instructions = '\n'.join(instructions_lines).strip()
        return instructions, best_score, best_iteration
    
    print(f"Could not extract instructions for iteration {best_iteration}", flush=True)
    return None, best_score, best_iteration


def reconstruct_module(instructions_text):
    """
    Reconstruct a DSPy module with the given optimized instructions.
    
    Args:
        instructions_text: Optimized instructions extracted from log
        
    Returns:
        DSPy module with instructions applied
    """
    # Create base module
    module = dspy.ChainOfThought(CatanDrillSignature)
    
    # Apply instructions - ChainOfThought uses 'predict' attribute for the predictor
    if hasattr(module, 'predict'):
        predictor = module.predict
        if hasattr(predictor, 'signature'):
            predictor.signature.instructions = instructions_text
            print("Applied instructions to module signature", flush=True)
    elif hasattr(module, '__dict__'):
        # Try to find predictor in module's attributes
        for attr_name, attr_value in module.__dict__.items():
            if hasattr(attr_value, 'signature'):
                attr_value.signature.instructions = instructions_text
                print(f"Applied instructions to {attr_name}.signature", flush=True)
                break
    
    return module


def main():
    parser = argparse.ArgumentParser(description="Recover optimized module from log")
    parser.add_argument("--log", required=True, help="Path to optimization log file")
    parser.add_argument("--output", required=True, help="Output path for recovered module (.pkl)")
    parser.add_argument("--model", default="gpt-5.2", help="Model name for metadata")
    
    args = parser.parse_args()
    
    print(f"Parsing log file: {args.log}", flush=True)
    
    # Extract instructions from log
    instructions, score, iteration = extract_best_instructions_from_log(args.log)
    
    if instructions is None:
        print("Failed to extract instructions from log", flush=True)
        return 1
    
    print(f"\nExtracted instructions ({len(instructions)} chars):", flush=True)
    print("=" * 60, flush=True)
    print(instructions[:500], flush=True)
    if len(instructions) > 500:
        print(f"... ({len(instructions) - 500} more characters)", flush=True)
    print("=" * 60, flush=True)
    
    # Reconstruct module
    print("\nReconstructing module...", flush=True)
    
    # Initialize DSPy
    lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)
    
    module = reconstruct_module(instructions)
    
    # Verify instructions were applied
    if hasattr(module, 'predict') and hasattr(module.predict, 'signature'):
        if hasattr(module.predict.signature, 'instructions'):
            applied_instructions = module.predict.signature.instructions
            print(f"\nVerifying instructions were applied: {len(applied_instructions)} chars", flush=True)
            if applied_instructions != instructions:
                print("WARNING: Instructions don't match!", flush=True)
        else:
            print("WARNING: Module has no instructions attribute!", flush=True)
    
    # Save using the fixed optimizer
    print(f"\nSaving to {args.output}...", flush=True)
    optimizer = DrillOptimizer(model_name=args.model)
    
    metadata = {
        'recovered_from_log': True,
        'log_file': args.log,
        'best_iteration': iteration,
        'best_score': score,
        'instructions_length': len(instructions)
    }
    
    try:
        optimizer.save(module, args.output, metadata)
        print(f"Successfully saved recovered module to {args.output}", flush=True)
    except Exception as e:
        print(f"Error saving module: {e}", flush=True)
        
        # Try to save instructions as text file instead
        txt_path = args.output.replace('.pkl', '_instructions.txt')
        with open(txt_path, 'w') as f:
            f.write(instructions)
        print(f"Saved instructions as text to {txt_path}", flush=True)
        
        # Save metadata
        metadata_path = args.output.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}", flush=True)
        
        return 1
    
    print("\n✓ Recovery complete!", flush=True)
    print(f"Best iteration: {iteration}", flush=True)
    print(f"Best score: {score:.4f} ({score*58:.0f}/58 drills)", flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

