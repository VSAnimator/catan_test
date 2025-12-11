#!/usr/bin/env python3
"""
Script to manage bug test cases.

Usage:
    python -m bug_tests.manage_tests add --game-id GAME_ID --step-id STEP_ID --description "..." --expected "..." --undesired "..."
    python -m bug_tests.manage_tests list
    python -m bug_tests.manage_tests show TEST_ID
    python -m bug_tests.manage_tests remove TEST_ID
    python -m bug_tests.manage_tests run [TEST_ID]
"""
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bug_tests.test_registry import TestRegistry, BugTestCase


def add_test(args):
    """Add a new test case."""
    registry = TestRegistry()
    
    # Generate test ID if not provided
    test_id = args.test_id
    if not test_id:
        test_id = f"test_{args.game_id[:8]}_step{args.step_id}"
    
    # Parse test_action if provided
    test_action = None
    if args.test_action:
        test_action = json.loads(args.test_action)
    
    # Parse tags if provided
    tags = args.tags.split(",") if args.tags else []
    tags = [t.strip() for t in tags if t.strip()]
    
    test = BugTestCase(
        test_id=test_id,
        game_id=args.game_id,
        step_id=args.step_id,
        description=args.description,
        expected_behavior=args.expected,
        undesired_behavior=args.undesired,
        test_action=test_action,
        llm_validation_prompt=args.llm_prompt,
        tags=tags
    )
    
    registry.add_test(test)
    print(f"Added test: {test_id}")


def list_tests(args):
    """List all test cases."""
    registry = TestRegistry()
    tests = registry.list_tests()
    
    if args.tag:
        tests = [t for t in tests if args.tag in t.tags]
    
    if not tests:
        print("No tests found.")
        return
    
    print(f"Found {len(tests)} test(s):\n")
    for test in tests:
        print(f"  {test.test_id}")
        print(f"    Game: {test.game_id}, Step: {test.step_id}")
        print(f"    Description: {test.description[:80]}...")
        if test.tags:
            print(f"    Tags: {', '.join(test.tags)}")
        print()


def show_test(args):
    """Show details of a specific test."""
    registry = TestRegistry()
    test = registry.get_test(args.test_id)
    
    if not test:
        print(f"Test {args.test_id} not found.")
        return
    
    print(f"Test ID: {test.test_id}")
    print(f"Game ID: {test.game_id}")
    print(f"Step ID: {test.step_id}")
    print(f"\nDescription:")
    print(f"  {test.description}")
    print(f"\nExpected Behavior:")
    print(f"  {test.expected_behavior}")
    print(f"\nUndesired Behavior:")
    print(f"  {test.undesired_behavior}")
    if test.test_action:
        print(f"\nTest Action:")
        print(f"  {json.dumps(test.test_action, indent=2)}")
    if test.llm_validation_prompt:
        print(f"\nLLM Validation Prompt:")
        print(f"  {test.llm_validation_prompt}")
    if test.tags:
        print(f"\nTags: {', '.join(test.tags)}")


def remove_test(args):
    """Remove a test case."""
    registry = TestRegistry()
    registry.remove_test(args.test_id)
    print(f"Removed test: {args.test_id}")


def run_tests(args):
    """Run test(s)."""
    from bug_tests.test_runner import run_test, run_all_tests
    from bug_tests.llm_validator import validate_with_llm
    
    registry = TestRegistry()
    
    if args.test_id:
        # Run single test
        test = registry.get_test(args.test_id)
        if not test:
            print(f"Test {args.test_id} not found.")
            return
        
        print(f"Running test: {test.test_id}")
        result = run_test(test)
        
        print(f"\nTest Result:")
        print(f"  Success: {result['success']}")
        if result['error']:
            print(f"  Error: {result['error']}")
        for msg in result['messages']:
            print(f"  {msg}")
        
        # Run LLM validation if requested
        if args.validate and test.llm_validation_prompt:
            print(f"\nRunning LLM validation...")
            # Get state before
            from api.database import get_state_at_step
            state_before = get_state_at_step(test.game_id, test.step_id, use_state_before=True)
            
            validation = validate_with_llm(
                test.description,
                test.expected_behavior,
                test.undesired_behavior,
                state_before,
                result.get('state_after'),
                result,
                model=args.model
            )
            
            print(f"\nLLM Validation:")
            print(f"  Valid: {validation['valid']}")
            print(f"  Reasoning: {validation['reasoning']}")
            if validation.get('issues'):
                print(f"  Issues:")
                for issue in validation['issues']:
                    print(f"    - {issue}")
    else:
        # Run all tests
        results = run_all_tests(registry)
        
        print(f"\nTest Results Summary:")
        print(f"  Total: {results['total']}")
        print(f"  Passed: {results['passed']}")
        print(f"  Failed: {results['failed']}")
        
        if args.verbose:
            print(f"\nDetailed Results:")
            for test_id, result in results['test_results'].items():
                status = "PASS" if result['success'] else "FAIL"
                print(f"\n  {test_id}: {status}")
                if result['error']:
                    print(f"    Error: {result['error']}")


def main():
    parser = argparse.ArgumentParser(description="Manage bug regression tests")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Add test
    add_parser = subparsers.add_parser('add', help='Add a new test case')
    add_parser.add_argument('--test-id', help='Test ID (auto-generated if not provided)')
    add_parser.add_argument('--game-id', required=True, help='Game ID')
    add_parser.add_argument('--step-id', type=int, required=True, help='Step index (0-based)')
    add_parser.add_argument('--description', required=True, help='Test description')
    add_parser.add_argument('--expected', required=True, help='Expected behavior description')
    add_parser.add_argument('--undesired', required=True, help='Undesired behavior description')
    add_parser.add_argument('--test-action', help='Test action JSON (optional)')
    add_parser.add_argument('--llm-prompt', help='LLM validation prompt (optional)')
    add_parser.add_argument('--tags', help='Comma-separated tags (optional)')
    
    # List tests
    list_parser = subparsers.add_parser('list', help='List all test cases')
    list_parser.add_argument('--tag', help='Filter by tag')
    
    # Show test
    show_parser = subparsers.add_parser('show', help='Show test details')
    show_parser.add_argument('test_id', help='Test ID')
    
    # Remove test
    remove_parser = subparsers.add_parser('remove', help='Remove a test case')
    remove_parser.add_argument('test_id', help='Test ID')
    
    # Run tests
    run_parser = subparsers.add_parser('run', help='Run test(s)')
    run_parser.add_argument('test_id', nargs='?', help='Test ID (optional, runs all if not provided)')
    run_parser.add_argument('--validate', action='store_true', help='Run LLM validation')
    run_parser.add_argument('--model', default='gpt-4o-mini', help='LLM model for validation')
    run_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.command == 'add':
        add_test(args)
    elif args.command == 'list':
        list_tests(args)
    elif args.command == 'show':
        show_test(args)
    elif args.command == 'remove':
        remove_test(args)
    elif args.command == 'run':
        run_tests(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

