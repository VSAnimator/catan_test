# Parsing Error Analysis Report
## Game ID: 1ec82583-e656-4308-ab2e-62e5fd8f443f

### Summary
Analyzed 34 steps from the game for parsing errors where agent reasoning indicates one action but a different action was taken.

### Findings

**No clear parsing errors found** in this game based on the following checks:

1. **Explicit parsing error indicators**: Searched for terms like "failed to parse", "parse error", "fallback", etc. in reasoning fields - **0 found**

2. **Reasoning vs Action mismatches**: Analyzed reasoning text for explicit action statements (e.g., "choose build_settlement", "take propose_trade") and compared to actions taken - **0 mismatches found**

3. **Raw LLM response data**: Checked for raw_llm_response fields that could reveal parsing discrepancies - **0 steps have raw_llm_response data stored**

4. **Action JSON analysis**: Checked action_json fields for error indicators or discrepancies with reasoning - **none found**

### Observations

- Most steps (especially setup phase) have reasoning that matches the action taken
- The reasoning field often contains the same text as stored in action_json.reasoning
- No explicit fallback messages or parsing warnings in the stored data

### Limitations

1. **Missing raw LLM data**: Without raw_llm_response fields, we cannot detect cases where:
   - The LLM output one action but parsing selected a different one
   - The LLM's JSON was malformed and fell back to a default action
   - Field name mismatches (e.g., "road_id" vs "road_edge_id") caused fallbacks

2. **Agent fallback logic**: The codebase shows that agents have fallback logic (see `llm_agent.py` lines 422-446) that can silently use a different action when:
   - LLM requests a specific tile_id/road_id that doesn't match legal actions
   - The system falls back to the first available legal action
   - These cases print warnings but may not be stored in the database

### Recommendations

To better detect parsing errors in the future:

1. **Store raw_llm_response**: Ensure raw_llm_response is saved to the database for all LLM agent steps
2. **Log parsing warnings**: Store parsing warnings/fallbacks in the reasoning field or a separate error field
3. **Compare LLM intent**: Compare the action type in raw_llm_response JSON to the final parsed action
4. **Check payload mismatches**: Look for cases where the LLM specified a payload (tile_id, road_id, etc.) but a different one was used

### Code References

The agent code shows potential parsing fallback scenarios:
- `backend/agents/llm_agent.py:422-446` - Fallback when tile_id or road_id doesn't match legal actions
- `backend/agents/agent_runner.py:485-501` - PROPOSE_TRADE parsing error retry logic

