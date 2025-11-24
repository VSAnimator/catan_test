"""
LLM-based agent using ReAct pattern (Reasoning and Acting) with RAG.
Uses retrieval-augmented generation to learn from past games and user feedback.
"""
import os
import json
from typing import Tuple, Optional, List, Dict, Any
from engine import GameState, Action, ActionPayload
from engine.serialization import legal_actions, state_to_text, legal_actions_to_text
from .base_agent import BaseAgent
from .llm_retrieval import StateRetriever, RetrievedExample
from api.guidelines_db import get_guidelines, get_feedback


class LLMAgent(BaseAgent):
    """
    LLM agent using ReAct pattern with RAG.
    Supports multiple LLM providers via LiteLLM (OpenAI, Anthropic/Claude, Google, etc.).
    
    ReAct pattern:
    1. Observe: Understand current game state
    2. Think: Reason about what to do, using retrieved examples and guidelines
    3. Act: Choose an action
    
    Supported models (via LiteLLM):
    - OpenAI: gpt-4o-mini, gpt-4, gpt-3.5-turbo, etc.
    - Anthropic: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
    - Google: gemini/gemini-pro, gemini/gemini-1.5-pro
    - See https://docs.litellm.ai/docs/providers for full list
    """
    
    def __init__(
        self,
        player_id: str,
        api_key: Optional[str] = None,
        model: str = "gpt-5.1",
        temperature: float = 0.7,
        max_examples: int = 5,
        enable_retrieval: bool = True
    ):
        # Token usage tracking
        self.token_usage_history: List[Dict[str, int]] = []
        """
        Initialize the LLM agent.
        
        Args:
            player_id: ID of the player this agent controls
            api_key: API key for the LLM provider (defaults to env vars based on model)
                     - OpenAI: OPENAI_API_KEY
                     - Anthropic: ANTHROPIC_API_KEY
                     - Google: GEMINI_API_KEY
                     - etc. (see LiteLLM docs)
            model: Model to use (default: gpt-4o-mini)
                   Examples:
                   - OpenAI: "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"
                   - Anthropic: "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
                   - Google: "gemini/gemini-pro", "gemini/gemini-1.5-pro"
                   - See LiteLLM docs for full list
            temperature: Temperature for generation
            max_examples: Maximum number of examples to retrieve
            enable_retrieval: Whether to use RAG retrieval
        """
        super().__init__(player_id)
        self.model = model
        self.temperature = temperature
        self.enable_retrieval = enable_retrieval
        # Only create retriever if retrieval is enabled (for zero-shot, skip this)
        self.retriever = StateRetriever(max_examples=max_examples) if enable_retrieval else None
        
        # Set API key if provided, otherwise LiteLLM will use env vars
        if api_key:
            # Determine provider from model name and set appropriate env var
            if model.startswith("claude") or "anthropic" in model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif model.startswith("gemini") or "google" in model.lower():
                os.environ["GEMINI_API_KEY"] = api_key
            elif model.startswith("gpt") or "openai" in model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                # Default to OpenAI for unknown models
                os.environ["OPENAI_API_KEY"] = api_key
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM API using LiteLLM (supports OpenAI, Anthropic, Google, etc.).
        
        Args:
            messages: List of message dicts with "role" and "content"
            
        Returns:
            LLM response text
        """
        try:
            import litellm
            
            # GPT-5 models only support temperature=1
            # Set drop_params to handle unsupported parameters gracefully
            litellm.drop_params = True
            
            # Adjust temperature for GPT-5 models
            temperature = self.temperature
            if self.model.startswith("gpt-5"):
                temperature = 1.0
            
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0)
                }
                self.token_usage_history.append(usage)
            
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("litellm package required. Install with: pip install litellm")
        except Exception as e:
            raise RuntimeError(f"LLM API error: {str(e)}")
    
    def _retrieve_context(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> str:
        """
        Retrieve relevant context (examples and guidelines) for RAG.
        
        Args:
            state: Current game state
            legal_actions_list: List of legal actions
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Retrieve similar examples (only if retrieval is enabled)
        if self.enable_retrieval and self.retriever:
            examples = self.retriever.retrieve_similar_states(state, self.player_id)
            
            if examples:
                context_parts.append("## Similar Game Examples:")
                for i, example in enumerate(examples, 1):
                    outcome_str = f" (outcome: {example.outcome})" if example.outcome else ""
                    context_parts.append(
                        f"\n### Example {i} (similarity: {example.similarity_score:.2f}{outcome_str}):"
                    )
                    context_parts.append(f"Action taken: {example.action.get('type', 'unknown')}")
                    if example.action.get('payload'):
                        context_parts.append(f"Payload: {json.dumps(example.action['payload'], indent=2)}")
        
        # Retrieve guidelines (only if retrieval is enabled)
        if self.enable_retrieval:
            guidelines = get_guidelines(player_id=self.player_id, active_only=True)
            if guidelines:
                context_parts.append("\n## Guidelines:")
                for guideline in guidelines:
                    priority_str = " (HIGH PRIORITY)" if guideline['priority'] > 5 else ""
                    context_parts.append(f"- {guideline['guideline_text']}{priority_str}")
                    if guideline['context']:
                        context_parts.append(f"  (Context: {guideline['context']})")
            
            # Retrieve recent feedback
            feedback = get_feedback(player_id=self.player_id, limit=5)
            if feedback:
                context_parts.append("\n## Recent Feedback:")
                for fb in feedback:
                    context_parts.append(f"- [{fb['feedback_type']}] {fb['feedback_text']}")
        
        return "\n".join(context_parts) if context_parts else "No additional context available."
    
    def _format_state_and_actions(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> str:
        """
        Format the game state and legal actions for the LLM.
        
        Args:
            state: Current game state
            legal_actions_list: List of legal actions
            
        Returns:
            Formatted string
        """
        state_text = state_to_text(state, self.player_id)
        actions_text = legal_actions_to_text(legal_actions_list, state=state, player_id=self.player_id)
        
        return f"""## Current Game State:
{state_text}

## Available Legal Actions:
{actions_text}"""
    
    def choose_action(
        self,
        state: GameState,
        legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
    ) -> Tuple[Action, Optional[ActionPayload], Optional[str], Optional[str]]:
        """
        Choose an action using ReAct pattern with RAG.
        
        ReAct pattern:
        1. Observe: Format current state
        2. Think: Retrieve context, reason about best action
        3. Act: Parse LLM response and return action
        
        Args:
            state: Current game state
            legal_actions_list: List of legal actions
            
        Returns:
            Chosen action tuple
        """
        if not legal_actions_list:
            raise ValueError("No legal actions available")
        
        # Handle pending trade responses FIRST (must be done before other actions)
        if state.pending_trade_offer is not None:
            offer = state.pending_trade_offer
            current_player = state.players[state.current_player_index]
            
            # Check if this player is a target of the trade and needs to respond
            if current_player.id in offer['target_player_ids']:
                if current_player.id not in state.pending_trade_responses:
                    # Player needs to respond - prioritize this
                    accept_actions = [(a, p) for a, p in legal_actions_list if a == Action.ACCEPT_TRADE]
                    reject_actions = [(a, p) for a, p in legal_actions_list if a == Action.REJECT_TRADE]
                    
                    if accept_actions or reject_actions:
                        # If only one option, use it immediately
                        if accept_actions and not reject_actions:
                            return (accept_actions[0][0], accept_actions[0][1], "Accepting trade (only option available)")
                        elif reject_actions and not accept_actions:
                            return (reject_actions[0][0], reject_actions[0][1], "Rejecting trade (cannot afford)")
                        # If both available, let LLM decide but add urgent note to prompt
                        # Continue to LLM call below, but we'll add a note in the prompt
            
            # Check if this player is the proposer and needs to select a partner
            elif current_player.id == offer['proposer_id']:
                accepting_players = [pid for pid, accepted in state.pending_trade_responses.items() if accepted]
                if len(accepting_players) > 1:
                    # Multiple accepted - must select partner
                    select_actions = [(a, p) for a, p in legal_actions_list if a == Action.SELECT_TRADE_PARTNER]
                    if select_actions:
                        # Pick first accepting player (or could let LLM decide)
                        from engine import SelectTradePartnerPayload
                        return (Action.SELECT_TRADE_PARTNER, SelectTradePartnerPayload(selected_player_id=accepting_players[0]), 
                                f"Selecting trade partner: {accepting_players[0]}")
                elif len(accepting_players) == 1:
                    # Only one accepted - trade will execute automatically, just need to wait
                    # But we should still be able to continue, so let it fall through
                    pass
        
        # Step 1: Observe - Format current state
        state_and_actions = self._format_state_and_actions(state, legal_actions_list)
        
        # Step 2: Think - Retrieve context
        context = self._retrieve_context(state, legal_actions_list)
        
        # Step 3: Act - Build prompt and call LLM
        system_prompt = """You are an expert Catan player agent. Your goal is to win the game by reaching 10 victory points.

## CRITICAL CATAN RULES:

### Setup Phase:
- **Two unconnected settlements**: In setup, you place TWO settlements that do NOT need to be connected by roads. They are placed independently.
- **Setup order**: 
  - Round 1: Each player places one settlement + one road (road must connect to that settlement)
  - Round 2: Each player places one settlement + one road (road must connect to that settlement)
  - The two settlements from setup are NOT connected to each other
- **Distance rule**: Settlements and cities must be at least 2 edges apart (no adjacent intersections can both have buildings)

### Playing Phase:
- **One dev card per turn**: You can only play ONE development card per turn (except Victory Point cards, which are revealed at game end)
- **Road building**: Roads must connect to your existing roads or settlements/cities
- **Settlement placement**: Settlements must be at least 2 edges from any other settlement/city (distance rule)
- **City placement**: Cities can only be built by upgrading existing settlements

### Development Cards:
- **Knight**: Move robber and steal one resource from a player on that tile
- **Year of Plenty**: Take any 2 resources from the bank
- **Monopoly**: All players give you all resources of one type
- **Road Building**: Build 2 roads for free (must be legal placements)
- **Victory Point**: Worth 1 VP, revealed at game end

### Robber Rules:
- **Must move robber**: When a 7 is rolled or you play a Knight card, you must move the robber to a different tile
- **Steal after moving**: After moving the robber, you can steal one resource from a player who has buildings on that tile (if any)

### Trading:
- **Bank trades**: 4:1 default, 3:1 with matching port, 2:1 with specific resource port
- **Player trades**: Propose to one or more players, they accept/reject, proposer selects if multiple accept

### Victory:
- First player to reach 10+ victory points wins
- Victory points come from: Settlements (1 VP), Cities (2 VPs), Longest Road (2 VPs), Largest Army (2 VPs), VP cards (1 VP each)

---

You will receive:
1. The current game state
2. Available legal actions
3. Similar examples from past games
4. Guidelines and feedback

Use the ReAct pattern:
- **Observe**: Understand the current game state and available actions
- **Think**: Reason about the best action, considering:
  - Your current position (resources, VPs, buildings)
  - Opponent positions
  - Similar situations from past games
  - Guidelines and feedback
  - Strategic goals (winning conditions)
  - **CRITICAL**: Follow the Catan rules above!
- **Act**: Choose the best action from the legal actions

Respond in JSON format:
{
  "reasoning": "Your reasoning about what to do",
  "action_type": "The action type (e.g., 'build_settlement', 'trade_bank', etc.)",
  "action_payload": { ... } // Optional payload, matching the action type
}

Be strategic and consider:
- Building settlements/cities for VPs
- Building roads for expansion and longest road
- Buying development cards for flexibility
- Trading when beneficial
- Playing development cards at the right time
- Blocking opponents when advantageous"""
        
        # Add urgent note if trade response is needed
        trade_urgency_note = ""
        if state.pending_trade_offer is not None:
            current_player = state.players[state.current_player_index]
            offer = state.pending_trade_offer
            if (current_player.id in offer['target_player_ids'] and 
                current_player.id not in state.pending_trade_responses):
                trade_urgency_note = "\n\n⚠️ URGENT: You MUST respond to the pending trade offer. You can only choose ACCEPT_TRADE or REJECT_TRADE. No other actions are available until you respond.\n"
        
        user_prompt = f"""{state_and_actions}

{context}{trade_urgency_note}

Now reason about the best action and respond in JSON format as specified."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM
        response_text = self._call_llm(messages)
        
        # Store response for debugging
        self._last_llm_response = response_text
        
        # Parse response
        try:
            # Try to extract JSON from response (might have markdown code blocks)
            original_response = response_text
            response_text = response_text.strip()
            
            # Try multiple extraction methods
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                # Try to find JSON block
                parts = response_text.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Odd indices are code blocks
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        # Try to parse this part as JSON
                        try:
                            json.loads(part)
                            response_text = part
                            break
                        except:
                            continue
            
            # Try to find JSON object in text if not already extracted
            if not response_text.startswith("{"):
                # Look for first { and last }
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx+1]
            
            response_json = json.loads(response_text)
            
            action_type_str = response_json.get("action_type", "").lower()
            action_payload_dict = response_json.get("action_payload", {})
            reasoning = response_json.get("reasoning", None)  # Extract reasoning
            
            # Handle compound or invalid action names BEFORE normalization
            # If the LLM returns a compound action, try to map it to the appropriate single action
            if "move_robber" in action_type_str and "steal" in action_type_str:
                # Compound action: move_robber_and_steal
                # Check what's actually legal - if only steal is legal, use that
                has_steal = any(a == Action.STEAL_RESOURCE for a, _ in legal_actions_list)
                has_move = any(a == Action.MOVE_ROBBER for a, _ in legal_actions_list)
                if has_steal and not has_move:
                    action_type_str = "steal_resource"
                elif has_move:
                    action_type_str = "move_robber"
                else:
                    # Neither is legal, will be caught later
                    action_type_str = "steal_resource"  # Default fallback
            elif "pass" in action_type_str or "skip" in action_type_str:
                # LLM trying to "pass" or "skip" - map to appropriate action based on what's legal
                if any(a == Action.STEAL_RESOURCE for a, _ in legal_actions_list):
                    # Must steal before passing/skipping
                    action_type_str = "steal_resource"
                elif any(a == Action.END_TURN for a, _ in legal_actions_list):
                    action_type_str = "end_turn"
                else:
                    # Use first available action
                    if legal_actions_list:
                        # Map to the first legal action's type
                        first_action = legal_actions_list[0][0]
                        action_type_str = first_action.value
            elif "resolve" in action_type_str and "robber" in action_type_str:
                # LLM trying to "resolve_robber" - map to appropriate robber action
                if any(a == Action.STEAL_RESOURCE for a, _ in legal_actions_list):
                    action_type_str = "steal_resource"
                elif any(a == Action.MOVE_ROBBER for a, _ in legal_actions_list):
                    action_type_str = "move_robber"
            
            # Normalize action type string: replace spaces/hyphens with underscores, remove extra chars
            action_type_normalized = action_type_str.replace(" ", "_").replace("-", "_").strip()
            
            # Find matching action - try multiple matching strategies
            action_type_map = {
                "build_settlement": Action.BUILD_SETTLEMENT,
                "build_city": Action.BUILD_CITY,
                "build_road": Action.BUILD_ROAD,
                "buy_dev_card": Action.BUY_DEV_CARD,
                "buy_devcard": Action.BUY_DEV_CARD,  # Alternative spelling
                "play_dev_card": Action.PLAY_DEV_CARD,
                "play_devcard": Action.PLAY_DEV_CARD,  # Alternative spelling
                "play_development_card": Action.PLAY_DEV_CARD,  # Full name variant
                "play_knight": Action.PLAY_DEV_CARD,  # Common variant (knight is a dev card)
                "play_monopoly": Action.PLAY_DEV_CARD,  # Common variant
                "play_year_of_plenty": Action.PLAY_DEV_CARD,  # Common variant
                "play_road_building": Action.PLAY_DEV_CARD,  # Common variant
                "trade_bank": Action.TRADE_BANK,
                "propose_trade": Action.PROPOSE_TRADE,
                "accept_trade": Action.ACCEPT_TRADE,
                "reject_trade": Action.REJECT_TRADE,
                "select_trade_partner": Action.SELECT_TRADE_PARTNER,
                "move_robber": Action.MOVE_ROBBER,
                "robber_move": Action.MOVE_ROBBER,  # Alternative name
                "steal_resource": Action.STEAL_RESOURCE,
                "discard_resources": Action.DISCARD_RESOURCES,
                "end_turn": Action.END_TURN,
                "setup_place_settlement": Action.SETUP_PLACE_SETTLEMENT,
                "setup_place_road": Action.SETUP_PLACE_ROAD,
                "start_game": Action.START_GAME,
            }
            
            # Try exact match first (normalized)
            target_action = action_type_map.get(action_type_normalized)
            
            # Try original string too
            if not target_action:
                target_action = action_type_map.get(action_type_str)
            
            # Try matching against legal actions with fuzzy matching
            if not target_action:
                # Remove common words and normalize
                action_clean = action_type_normalized.replace("dev", "dev_card").replace("card", "dev_card")
                for action, _ in legal_actions_list:
                    action_value_lower = action.value.lower()
                    # Check if normalized string matches action value
                    if (action_type_normalized == action_value_lower or
                        action_clean == action_value_lower or
                        action_type_normalized in action_value_lower or
                        action_value_lower in action_type_normalized):
                        target_action = action
                        print(f"  Matched via fuzzy: '{action_type_str}' -> {action.value}", flush=True)
                        break
            
            if not target_action:
                # Try harder to find a match
                action_type_clean = action_type_str.replace("_", "").replace("-", "")
                for action, _ in legal_actions_list:
                    action_value_clean = action.value.replace("_", "").replace("-", "")
                    if action_type_clean in action_value_clean or action_value_clean in action_type_clean:
                        target_action = action
                        print(f"  Matched action via fuzzy matching: {action_type_str} -> {action.value}", flush=True)
                        break
                
                if not target_action:
                    print(f"Warning: Could not map action type '{action_type_str}' to any legal action", flush=True)
                    print(f"  Available actions: {[a.value for a, _ in legal_actions_list]}", flush=True)
                    raise ValueError(f"Could not map action type: {action_type_str}")
            
            # Find the matching legal action with payload
            matching_action = None
            exact_match = None
            preferred_match = None
            
            # For actions with tile_id (like MOVE_ROBBER), try to match the LLM's preference
            if target_action == Action.MOVE_ROBBER and action_payload_dict and "tile_id" in action_payload_dict:
                llm_tile_id = action_payload_dict["tile_id"]
                # Try to find exact match first
                for action, payload in legal_actions_list:
                    if action == target_action and payload and hasattr(payload, "tile_id"):
                        if payload.tile_id == llm_tile_id:
                            exact_match = (action, payload)
                            break
                        # Also store first match as fallback
                        if not preferred_match:
                            preferred_match = (action, payload)
                
                if exact_match:
                    matching_action = exact_match
                elif preferred_match:
                    # LLM specified a tile_id but it doesn't match any legal action
                    # This might be a parsing error - log it
                    print(f"Warning: LLM requested tile_id {llm_tile_id} but it's not in legal actions. Using first available.", flush=True)
                    matching_action = preferred_match
            
            # For other actions, use standard matching
            if not matching_action:
                for action, payload in legal_actions_list:
                    if action == target_action:
                        # If payload is provided, try to match it
                        if action_payload_dict and payload:
                            # Simple matching - check if keys match
                            payload_dict = self._payload_to_dict(payload)
                            if self._payloads_match(payload_dict, action_payload_dict):
                                matching_action = (action, payload)
                                break
                        elif not action_payload_dict and not payload:
                            # Both are None
                            matching_action = (action, payload)
                            break
                        elif not matching_action:
                            # Store first match as fallback
                            matching_action = (action, payload)
            
            if not matching_action:
                # Fallback: just pick the first matching action type
                for action, payload in legal_actions_list:
                    if action == target_action:
                        matching_action = (action, payload)
                        break
            
            if not matching_action:
                print(f"Warning: Could not find exact matching legal action for {action_type_str}", flush=True)
                print(f"  Target action enum: {target_action.value if target_action else 'None'}", flush=True)
                print(f"  Legal actions of this type: {[(a.value, type(p).__name__ if p else None) for a, p in legal_actions_list if a == target_action]}", flush=True)
                print(f"  All available legal actions: {[a.value for a, _ in legal_actions_list]}", flush=True)
                print(f"  LLM response JSON: {json.dumps(response_json, indent=2)[:500]}", flush=True)
                
                # Last resort: pick first matching action type
                for action, payload in legal_actions_list:
                    if action == target_action:
                        matching_action = (action, payload)
                        print(f"  Using first available action: {action.value}", flush=True)
                        break
                
                if not matching_action:
                    # The LLM returned an action that isn't legal. Try to find a similar legal action
                    # For example, if LLM said "move_robber" but only "steal_resource" is legal,
                    # that might mean the robber was already moved
                    if target_action == Action.MOVE_ROBBER:
                        # Check if STEAL_RESOURCE is available (robber already moved)
                        for action, payload in legal_actions_list:
                            if action == Action.STEAL_RESOURCE:
                                print(f"  LLM wanted MOVE_ROBBER but it's not legal. Using STEAL_RESOURCE instead.", flush=True)
                                matching_action = (action, payload)
                                break
                    elif target_action == Action.END_TURN:
                        # Check if STEAL_RESOURCE is available (must steal before ending turn)
                        for action, payload in legal_actions_list:
                            if action == Action.STEAL_RESOURCE:
                                print(f"  LLM wanted END_TURN but STEAL_RESOURCE is required first. Using STEAL_RESOURCE instead.", flush=True)
                                matching_action = (action, payload)
                                break
                    else:
                        # LLM returned an action that's not legal. Try to find the most appropriate legal action
                        # Common case: LLM wants to build/trade but must steal first
                        if any(a == Action.STEAL_RESOURCE for a, _ in legal_actions_list):
                            for action, payload in legal_actions_list:
                                if action == Action.STEAL_RESOURCE:
                                    print(f"  LLM wanted {target_action.value if target_action else action_type_str} but STEAL_RESOURCE is required first. Using STEAL_RESOURCE instead.", flush=True)
                                    matching_action = (action, payload)
                                    break
                        # If no special case matches, just use the first legal action
                        if not matching_action and legal_actions_list:
                            matching_action = legal_actions_list[0]
                            print(f"  LLM wanted {target_action.value if target_action else action_type_str} but it's not legal. Using first available action: {matching_action[0].value}", flush=True)
                    
                    if not matching_action:
                        raise ValueError(f"Could not find matching legal action for {action_type_str}. Available: {[a.value for a, _ in legal_actions_list]}")
            
            # Return action, payload, reasoning, and raw response
            action, payload = matching_action
            return (action, payload, reasoning, response_text)
            
        except json.JSONDecodeError as e:
            # Fallback: try to extract action from text using regex
            import re
            print(f"Warning: Failed to parse LLM response as JSON: {e}", flush=True)
            print(f"Response (first 500 chars): {response_text[:500]}", flush=True)
            
            # Try to extract action_type from text using regex
            action_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
            if not action_match:
                action_match = re.search(r'action_type["\']?\s*[:=]\s*["\']?([a-z_]+)', response_text, re.IGNORECASE)
            
            if action_match:
                action_type_str = action_match.group(1).lower()
                # Try to find matching action
                for action, payload in legal_actions_list:
                    if action_type_str in action.value.lower() or action.value.lower() in action_type_str:
                        print(f"  Extracted action from text: {action.value}", flush=True)
                        return (action, payload, f"Parsed from text (JSON parse failed): {e}", response_text)
            
            # Last resort: fallback to first legal action
            print(f"  Falling back to first legal action: {legal_actions_list[0][0].value}", flush=True)
            action, payload = legal_actions_list[0]
            return (action, payload, f"Failed to parse LLM response: {e}", response_text)
        except Exception as e:
            print(f"Warning: Error processing LLM response: {e}", flush=True)
            if 'response_text' in locals():
                print(f"Response (first 500 chars): {response_text[:500]}", flush=True)
            import traceback
            traceback.print_exc()
            # Fallback to first legal action
            action, payload = legal_actions_list[0]
            raw_response = response_text if 'response_text' in locals() else None
            return (action, payload, f"Error processing LLM response: {e}", raw_response)
    
    def _payload_to_dict(self, payload: ActionPayload) -> Dict[str, Any]:
        """Convert action payload to dictionary."""
        if payload is None:
            return {}
        return {
            k: v.value if hasattr(v, 'value') else v
            for k, v in payload.__dict__.items()
        }
    
    def _payloads_match(
        self,
        payload1: Dict[str, Any],
        payload2: Dict[str, Any]
    ) -> bool:
        """Check if two payloads match (simplified)."""
        # Check if key fields match
        for key in payload1:
            if key in payload2:
                val1 = payload1[key]
                val2 = payload2[key]
                # Handle ResourceType enums
                if hasattr(val1, 'value'):
                    val1 = val1.value
                if hasattr(val2, 'value'):
                    val2 = val2.value
                if val1 != val2:
                    return False
        return True

