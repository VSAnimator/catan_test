# Bug Regression Testing System

This directory contains an automated system for testing bugs discovered during gameplay. The system can restore game states from any step and validate expected behavior.

## Features

- **State Restoration**: Restore game state from any step in any game
- **Reproducible RNG**: Step-based seeding ensures RNG is reproducible when restoring from a specific step
- **Test Registry**: JSON-based registry of test cases
- **LLM Validation**: Optional LLM-based validation of test results
- **Easy Management**: Command-line tools for adding, listing, and running tests

## Directory Structure

- `test_registry.py`: Test case data structures and registry management
- `test_runner.py`: Test execution engine that restores states and runs tests
- `llm_validator.py`: LLM-based validation of test results
- `manage_tests.py`: Command-line tool for managing tests
- `test_registry.json`: JSON file storing all test cases (auto-generated)
- `test_*.py`: Standalone test scripts for specific scenarios (e.g., `test_llm_discard.py`)

**IMPORTANT**: All test files should be preserved and integrated into the test suite. Never delete test files - they are part of the regression testing database.

## Usage

### Adding a Test Case

When you discover a bug during gameplay:

1. Note the `game_id` and `step_id` where the bug occurs
2. Add a test case:

```bash
python -m bug_tests.manage_tests add \
  --game-id <game_id> \
  --step-id <step_id> \
  --description "Description of the bug/test" \
  --expected "What should happen (desired behavior)" \
  --undesired "What actually happens (bug behavior)" \
  --tags "bug,regression"
```

Optional parameters:
- `--test-id`: Custom test ID (auto-generated if not provided)
- `--test-action`: JSON string of action to execute after restoring state
- `--llm-prompt`: Custom LLM validation prompt

Example:
```bash
python -m bug_tests.manage_tests add \
  --game-id abc123 \
  --step-id 42 \
  --description "Player should not be able to build settlement adjacent to another settlement" \
  --expected "Building a settlement adjacent to an existing settlement should raise ValueError" \
  --undesired "Settlement is built successfully, violating distance rule" \
  --test-action '{"type": "build_settlement", "payload": {"intersection_id": 123}}' \
  --tags "distance_rule,bug"
```

### Listing Tests

```bash
# List all tests
python -m bug_tests.manage_tests list

# List tests with specific tag
python -m bug_tests.manage_tests list --tag bug
```

### Viewing Test Details

```bash
python -m bug_tests.manage_tests show <test_id>
```

### Running Tests

```bash
# Run all tests
python -m bug_tests.manage_tests run

# Run specific test
python -m bug_tests.manage_tests run <test_id>

# Run with LLM validation
python -m bug_tests.manage_tests run <test_id> --validate

# Verbose output
python -m bug_tests.manage_tests run --verbose
```

### Removing Tests

```bash
python -m bug_tests.manage_tests remove <test_id>
```

## How It Works

### State Restoration

When a test is run:
1. The system loads the game state from `state_before_json` at the specified step
2. A deterministic seed is computed from `game_id` + `step_id` using MD5 hash
3. The RNG is seeded with this value to ensure reproducible randomness

### Step-Based Seeding

The seed is computed as:
```python
seed_string = f"{game_id}:{step_id}"
seed_hash = hashlib.md5(seed_string.encode()).hexdigest()
seed = int(seed_hash[:8], 16)
```

This ensures that:
- Restoring the same game at the same step produces identical RNG sequences
- Different games or steps produce different seeds
- The seed is deterministic and reproducible

### Test Execution

1. **Restore State**: Load game state at specified step
2. **Set Seed**: Initialize RNG with step-based seed
3. **Execute Action** (if provided): Run the test action
4. **Validate**: Check results against expected/undesired behavior
5. **LLM Validation** (optional): Use LLM to analyze test results

### LLM Validation

When `--validate` is used, the LLM receives:
- Test description
- Expected behavior
- Undesired behavior
- Game state before and after
- Test execution results

The LLM analyzes whether:
- The test passed or failed
- The results match expected behavior
- Any undesired behavior occurred
- There are any issues or inconsistencies

## Test Registry Format

Tests are stored in `test_registry.json`:

```json
{
  "tests": [
    {
      "test_id": "test_abc123_step42",
      "game_id": "abc123",
      "step_id": 42,
      "description": "Test description",
      "expected_behavior": "What should happen",
      "undesired_behavior": "What should NOT happen",
      "test_action": {
        "type": "build_settlement",
        "payload": {"intersection_id": 123}
      },
      "llm_validation_prompt": null,
      "tags": ["bug", "regression"]
    }
  ]
}
```

## Integration with Game Engine

The system integrates with the existing game engine:
- Uses `get_state_at_step()` from `api.database` to restore states
- Uses `deserialize_game_state()` from `engine.serialization` to load states
- Uses `GameState.step()` to execute actions
- Uses global `random` module for RNG (seeded before each test)

## Best Practices

1. **Test Early**: Add tests as soon as bugs are discovered
2. **Clear Descriptions**: Write clear descriptions of expected vs undesired behavior
3. **Use Tags**: Tag tests for easy filtering (e.g., "bug", "regression", "distance_rule")
4. **Include Actions**: If testing a specific action, include it in `test_action`
5. **Run Regularly**: Run tests after code changes to catch regressions
6. **LLM Validation**: Use LLM validation for complex behavioral checks

## Example Workflow

1. **Discover Bug**: Notice a bug during gameplay at step 42 of game `abc123`
2. **Add Test**: Create a test case capturing the bug
3. **Fix Bug**: Implement the fix
4. **Run Test**: Verify the test now passes
5. **Keep Test**: Keep the test in the registry to prevent regression

## Test Scripts

In addition to registry-based tests, you can create standalone test scripts (e.g., `test_llm_discard.py`) that test specific scenarios. These scripts:

- Should be stored in the `bug_tests/` directory
- Can be run directly: `python -m bug_tests.test_llm_discard`
- Should follow a consistent structure for integration into automated test suites
- **NEVER DELETE TEST FILES** - they are part of the regression testing database

Example test script structure:
```python
def test_specific_scenario(game_id: str, ...) -> dict:
    """Test description."""
    result = {'success': False, 'error': None, 'messages': []}
    # ... test logic ...
    return result

if __name__ == "__main__":
    # Run test
    result = test_specific_scenario(...)
    sys.exit(0 if result['success'] else 1)
```

## Limitations

- Tests rely on the database containing the original game states
- RNG seeding works for the Python `random` module; other RNG sources may need separate handling
- LLM validation requires API access and may incur costs
- Test actions must be valid for the restored game state

