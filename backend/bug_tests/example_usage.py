#!/usr/bin/env python3
"""
Example usage of the bug testing system.

This demonstrates how to programmatically add and run tests.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bug_tests.test_registry import TestRegistry, BugTestCase
from bug_tests.test_runner import run_test, run_all_tests


def example_add_test():
    """Example: Adding a test case programmatically."""
    registry = TestRegistry()
    
    test = BugTestCase(
        test_id="example_test_1",
        game_id="example_game_123",
        step_id=42,
        description="Example test: Player should not be able to build settlement adjacent to another settlement",
        expected_behavior="Building a settlement adjacent to an existing settlement should raise ValueError",
        undesired_behavior="Settlement is built successfully, violating distance rule",
        test_action={
            "type": "build_settlement",
            "payload": {
                "type": "BuildSettlementPayload",
                "intersection_id": 123
            }
        },
        tags=["distance_rule", "bug", "example"]
    )
    
    registry.add_test(test)
    print(f"Added test: {test.test_id}")


def example_run_test():
    """Example: Running a test."""
    registry = TestRegistry()
    test = registry.get_test("example_test_1")
    
    if test:
        result = run_test(test)
        print(f"Test Result:")
        print(f"  Success: {result['success']}")
        if result['error']:
            print(f"  Error: {result['error']}")
        for msg in result['messages']:
            print(f"  {msg}")
    else:
        print("Test not found")


def example_list_tests():
    """Example: Listing all tests."""
    registry = TestRegistry()
    tests = registry.list_tests()
    
    print(f"Found {len(tests)} test(s):")
    for test in tests:
        print(f"  - {test.test_id}: {test.description[:60]}...")


if __name__ == "__main__":
    print("Bug Testing System - Example Usage\n")
    
    # Example 1: Add a test
    print("1. Adding a test...")
    example_add_test()
    print()
    
    # Example 2: List tests
    print("2. Listing tests...")
    example_list_tests()
    print()
    
    # Example 3: Run a test (will fail if game doesn't exist)
    print("3. Running a test...")
    print("   (Note: This will fail if the game doesn't exist in the database)")
    example_run_test()

