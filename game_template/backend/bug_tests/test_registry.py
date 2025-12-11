"""
Test registry for bug regression tests.

Stores test cases with:
- Game state restoration info (game_id, step_id)
- Test description
- Expected behavior (desired/undesired)
- LLM validation prompts
"""
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class BugTestCase:
    """A single bug test case."""
    test_id: str  # Unique identifier for this test
    game_id: str  # Game ID to restore from
    step_id: int  # Step index to restore state at
    description: str  # Human-readable description of the bug/test
    expected_behavior: str  # Description of what should happen (desired behavior)
    undesired_behavior: str  # Description of what should NOT happen (bug behavior)
    test_action: Optional[Dict[str, Any]] = None  # Optional action to take after restoring state
    llm_validation_prompt: Optional[str] = None  # Optional LLM prompt for validation
    tags: List[str] = None  # Optional tags for categorization
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class TestRegistry:
    """Registry for managing bug test cases."""
    
    def __init__(self, registry_file: Optional[Path] = None):
        """Initialize registry.
        
        Args:
            registry_file: Path to JSON file storing test cases. If None, uses default.
        """
        if registry_file is None:
            # Default to bug_tests directory
            registry_file = Path(__file__).parent / "test_registry.json"
        self.registry_file = registry_file
        self.tests: Dict[str, BugTestCase] = {}
        self.load()
    
    def load(self):
        """Load test cases from registry file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                for test_data in data.get('tests', []):
                    test = BugTestCase(**test_data)
                    self.tests[test.test_id] = test
        else:
            # Create empty registry file
            self.save()
    
    def save(self):
        """Save test cases to registry file."""
        data = {
            'tests': [asdict(test) for test in self.tests.values()]
        }
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_test(self, test: BugTestCase):
        """Add a test case to the registry."""
        self.tests[test.test_id] = test
        self.save()
    
    def get_test(self, test_id: str) -> Optional[BugTestCase]:
        """Get a test case by ID."""
        return self.tests.get(test_id)
    
    def list_tests(self) -> List[BugTestCase]:
        """List all test cases."""
        return list(self.tests.values())
    
    def remove_test(self, test_id: str):
        """Remove a test case from the registry."""
        if test_id in self.tests:
            del self.tests[test_id]
            self.save()
    
    def get_tests_by_tag(self, tag: str) -> List[BugTestCase]:
        """Get all tests with a specific tag."""
        return [test for test in self.tests.values() if tag in test.tags]

