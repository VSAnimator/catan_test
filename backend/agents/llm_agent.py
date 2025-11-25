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
        
        # Filter out propose_trade actions that were already taken this turn
        # to avoid repeated trade proposals
        filtered_actions = []
        current_player = state.players[state.current_player_index]
        player_actions_this_turn = [
            a for a in state.actions_taken_this_turn 
            if a["player_id"] == current_player.id and a["action"] == "propose_trade"
        ]
        
        for action, payload in legal_actions_list:
            if action == Action.PROPOSE_TRADE:
                # Check if this exact trade was already proposed this turn
                already_proposed = False
                if payload and hasattr(payload, 'give_resources') and hasattr(payload, 'receive_resources'):
                    # Normalize current payload to string keys (matching stored format)
                    current_give = {rt.value: count for rt, count in payload.give_resources.items()}
                    current_receive = {rt.value: count for rt, count in payload.receive_resources.items()}
                    
                    for prev_action in player_actions_this_turn:
                        prev_payload = prev_action.get("payload", {})
                        prev_give = prev_payload.get("give_resources", {})
                        prev_receive = prev_payload.get("receive_resources", {})
                        prev_targets = set(prev_payload.get("target_player_ids", []))
                        
                        if (prev_give == current_give and
                            prev_receive == current_receive and
                            prev_targets == set(payload.target_player_ids)):
                            already_proposed = True
                            break
                if not already_proposed:
                    filtered_actions.append((action, payload))
            else:
                filtered_actions.append((action, payload))
        
        # Use filtered actions
        legal_actions_list = filtered_actions if filtered_actions else legal_actions_list
        
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

### Building Costs (CRITICAL - Know These Exactly):
- **Settlement**: 1 wood, 1 brick, 1 sheep, 1 wheat (exactly one of each)
- **City**: 2 wheat, 3 ore (upgrades existing settlement)
- **Road**: 1 wood, 1 brick
- **Development Card**: 1 wheat, 1 sheep, 1 ore

**IMPORTANT - Resource Calculation:**
- When calculating if you can build after gaining resources, add your current resources + resources you'll gain, then check if you have enough of EACH resource type
- Example: If you have 1 wood and 2 wheat, and you gain 1 brick and 1 sheep, you will have: 1 wood (need 1) ✓, 1 brick (need 1) ✓, 2 wheat (need 1) ✓, 1 sheep (need 1) ✓ → You CAN build a settlement!
- Be careful: You only need ONE of each resource for a settlement, not multiple. Having 2 wheat doesn't mean you need 2 wood - you still only need 1 wood.
- **CRITICAL - Calculating Resources After Trades**: When evaluating a trade, calculate your resources AFTER the trade:
  - Start with your current resources
  - SUBTRACT what you GIVE in the trade
  - ADD what you RECEIVE in the trade
  - Check ALL resource types needed for what you want to build
  - **Example 1**: You have 1 brick, 2 wheat. You propose: give 1 brick + 1 wheat, receive 1 sheep.
    - After trade: 1 brick - 1 brick = 0 brick ❌, 2 wheat - 1 wheat = 1 wheat ✓, 0 sheep + 1 sheep = 1 sheep ✓, 0 wood = 0 wood ❌
    - For settlement (needs 1 wood, 1 brick, 1 sheep, 1 wheat): You're missing BOTH wood AND brick, not just wood!
  - **Example 2 (COMMON MISTAKE)**: You have 1 brick, 1 wheat. You propose: give 1 brick + 1 wheat, receive 1 wood + 1 sheep.
    - After trade: 1 brick - 1 brick = 0 brick ❌, 1 wheat - 1 wheat = 0 wheat ❌, 0 wood + 1 wood = 1 wood ✓, 0 sheep + 1 sheep = 1 sheep ✓
    - For settlement (needs 1 wood, 1 brick, 1 sheep, 1 wheat): You're missing BOTH brick AND wheat! You do NOT have "exactly the settlement set" - you're missing 2 resources!
    - **WRONG**: "After trade I'd have wood 1, brick 1, sheep 1, wheat 1" - NO! You gave away your brick and wheat, so you have 0 brick and 0 wheat!
  - **Always check ALL required resources, not just one!**
  - **Always SUBTRACT what you give - don't forget you're losing those resources!**

### Development Cards:
- **Knight**: Move robber and steal one resource from a player on that tile
- **Year of Plenty**: Take any 2 resources from the bank. **CRITICAL: After playing Year of Plenty, you immediately receive the resources and can use them in the SAME TURN to build settlements, cities, or roads!** For example, if you need wood+brick to build a settlement, you can play Year of Plenty to get wood+brick, then immediately build the settlement in the same turn. The legal actions you see are for your CURRENT resources - after using Year of Plenty, new build actions will become available.
- **Monopoly**: All players give you all resources of one type
- **Road Building**: When played, gives you 2 FREE roads that you can build immediately. These roads don't cost resources. You can build them even if you don't have wood/brick. **IMPORTANT: You must use ALL free roads before ending your turn - any unused free roads are lost when you end your turn.** Check the game state for "FREE ROADS AVAILABLE" to see how many you have remaining.
- **Victory Point**: Worth 1 VP, revealed at game end

### Robber Rules:
- **Must move robber**: When a 7 is rolled or you play a Knight card, you must move the robber to a different tile
- **Steal after moving**: After moving the robber, you can steal one resource from a player who has buildings on that tile (if any)

### Discarding Resources (When 7 is Rolled):
- **When to discard**: If a 7 is rolled and you have 8 or more resources, you MUST discard exactly half (rounded down) of your resources
- **How much to discard**: If you have N resources, discard N // 2 (half, rounded down)
  - Example: 8 resources → discard 4, 9 resources → discard 4, 10 resources → discard 5, 11 resources → discard 5
- **You choose which resources**: You decide which specific resources to discard (you must discard the exact amount)
- **CRITICAL - Discard Format**: When discard_resources appears in legal actions, you MUST provide the resources dict:
  - Format: { "action_type": "discard_resources", "action_payload": { "resources": {"wood": 2, "brick": 1, "sheep": 1} } }
  - The total of all resource amounts must equal exactly half your resources (rounded down)
  - Resource types: "wood", "brick", "sheep", "wheat", "ore"
  - You can only discard resources you actually have
  - Example: If you have 9 resources total (3 wood, 2 brick, 2 sheep, 1 wheat, 1 ore), you must discard 4:
    - Valid: {"wood": 2, "brick": 1, "sheep": 1} (total: 4)
    - Valid: {"wood": 1, "brick": 2, "wheat": 1} (total: 4)
    - Invalid: {"wood": 3} (only 3, need 4)
    - Invalid: {"wood": 5} (you only have 3 wood)

### Trading:
- **Bank trades**: 4:1 default, 3:1 with matching port, 2:1 with specific resource port
- **Player trades**: You can propose ANY trade to other players. You specify what resources you give and what resources you receive. You can trade with one or more players at once. They will accept/reject, and if multiple accept, you select which one to trade with.
- **IMPORTANT - Propose Trade Format**: When "propose_trade" appears in legal actions, you can propose ANY trade you want. You are not limited to pre-listed trades. You can propose any combination of resources you have. For example:
  - 1:1 trades (give 1 wood, receive 1 brick)
  - 2:1 trades (give 2 wood, receive 1 brick)
  - Multi-resource trades (give 1 wood + 1 brick, receive 1 sheep)
  - Any other combination you want
- **CRITICAL - No Repeated Trades**: Check the "Actions Taken This Turn" section in the game state. DO NOT propose the same trade (same give/receive resources to the same players) that you already proposed this turn. If a trade was rejected, try a different trade or different players instead.
- **CRITICAL - Proposing Trades (Think from Other Player's Perspective)**: When proposing a trade, you MUST think about whether the OTHER player would want to accept it:
  - **What does the other player NEED?** (resources they're missing for buildings they want to build)
  - **What does the other player have in ABUNDANCE?** (resources they produce easily - they won't want more of these)
  - **What does the other player have that's SCARCE?** (rare resources they need - they won't want to give these away)
  - **Example of BAD reasoning**: "Blake has good wheat income, so offering 1 wheat for 1 ore is attractive for Blake" - WRONG! If Blake has good wheat income, he doesn't need more wheat. If Blake only has 1 ore, he needs it for cities/dev cards and won't trade it away for wheat he can already produce.
  - **Example of GOOD reasoning**: "Blake has 3 wheat but no brick or sheep. He needs brick+sheep for settlements. I'll offer 1 brick + 1 sheep for 2 wheat - this helps him build settlements while giving me wheat I need for cities."
  - **Key principle**: Trade what you have in abundance for what the other player has in abundance, but they need what you're giving and you need what they're giving.
- **CRITICAL - Calculate Your Resources After Trade**: Before proposing a trade, ALWAYS calculate what resources you'll have AFTER the trade:
  - Start with current resources
  - SUBTRACT what you GIVE (you lose these!)
  - ADD what you RECEIVE
  - **Example**: You have 1 brick, 1 wheat. You propose: give 1 brick + 1 wheat, receive 1 wood + 1 sheep.
    - After: 0 brick, 0 wheat, 1 wood, 1 sheep
    - For settlement: You're missing brick AND wheat! You do NOT have all 4 resources!
  - **Don't forget: When you give resources, you LOSE them!**
- **CRITICAL - Accepting Trades**: When you accept a trade offer:
  - The trade executes immediately - you receive the resources right away
  - After the trade executes, the turn returns to the PROPOSER (not you)
  - You CANNOT build on the proposer's turn - you must wait for YOUR next turn
  - **However, you SHOULD accept trades if they give you resources you need for building on YOUR next turn!**
  - Example: If you have 1 wood and 2 wheat, and someone offers you 1 brick + 1 sheep for 1 ore, you should ACCEPT because:
    - You'll have: 1 wood, 1 brick, 1 sheep, 2 wheat (after giving 1 ore)
    - On YOUR next turn, you can build a settlement (1 wood, 1 brick, 1 sheep, 1 wheat)
    - Don't reject just because you can't build "this turn" (the proposer's turn) - you can build on YOUR turn!

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

**CRITICAL: Action Payload Format**
When you see actions like "On road edge 6" or "At intersection 44", use these EXACT field names:
- For build_road: { "road_edge_id": 6 }  (NOT "road_id")
- For build_settlement: { "intersection_id": 44 }  (NOT "intersection" or "intersectionId")
- For build_city: { "intersection_id": 44 }
- For move_robber: { "tile_id": 5 }
- For steal_resource: { "other_player_id": "player_1" }
- For trade_bank: { "give_resources": {"wood": 4}, "receive_resources": {"brick": 1} }
- For propose_trade: { "give_resources": {"wood": 1, "brick": 1}, "receive_resources": {"sheep": 1}, "target_player_ids": ["player_1", "player_2"] }
  - You can propose ANY trade - any combination of resources you have. Examples:
    - 1:1: {"give_resources": {"wood": 1}, "receive_resources": {"brick": 1}, "target_player_ids": ["player_1"]}
    - 2:1: {"give_resources": {"wood": 2}, "receive_resources": {"brick": 1}, "target_player_ids": ["player_1"]}
    - Multi-resource: {"give_resources": {"wood": 1, "ore": 1}, "receive_resources": {"sheep": 1}, "target_player_ids": ["player_1", "player_2"]}
  - Resource types: "wood", "brick", "sheep", "wheat", "ore"
  - You can trade with one or more players (list their IDs in target_player_ids)
- For play_dev_card: { "card_type": "knight" } (or "year_of_plenty", "monopoly", "road_building", "victory_point")
- For play_dev_card with year_of_plenty: { "card_type": "year_of_plenty", "year_of_plenty_resources": {"wood": 1, "brick": 1} }
- For play_dev_card with monopoly: { "card_type": "monopoly", "monopoly_resource_type": "wood" }
- For discard_resources: { "resources": {"wood": 2, "brick": 1} }
  - **REQUIRED**: You MUST provide the resources dict when discarding
  - The total must equal exactly half your resources (rounded down)
  - Example: If you have 9 resources, discard 4: {"wood": 2, "brick": 1, "sheep": 1}
- For select_trade_partner: { "selected_player_id": "player_1" }
- Actions without payload: end_turn, buy_dev_card, accept_trade, reject_trade (use null or omit action_payload)

Examples:
- To build road on edge 34: {"action_type": "build_road", "action_payload": {"road_edge_id": 34}}
- To build settlement at intersection 44: {"action_type": "build_settlement", "action_payload": {"intersection_id": 44}}
- To end turn: {"action_type": "end_turn", "action_payload": null}

Be strategic and consider:
- Building settlements/cities for VPs
- Building roads for expansion and longest road
- Buying development cards for flexibility
- Trading when beneficial (but NEVER repeat the same trade proposal in one turn)
- Playing development cards at the right time
- Blocking opponents when advantageous

**CRITICAL - Dynamic Legal Actions:**
- Legal actions shown are based on your CURRENT resources and board state
- Actions can CHANGE during your turn as you gain resources (from trades, Year of Plenty, Monopoly, etc.)
- If you don't see a build action now, you can still get resources and build in the SAME TURN!
- Example: Use Year of Plenty to get wood+brick+sheep+wheat → then build_settlement becomes available
- Example: Trade to get missing resources → then build actions become available
- Example: Build a road → then new settlement locations become available (connected to that road)
- The legal actions list updates after each action you take

**IMPORTANT**: Always check the "Actions Taken This Turn" section to see what you've already done. Do not repeat trade proposals you've already made this turn."""
        
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
            
            # Special handling: During setup phase, map "build_settlement" to "setup_place_settlement"
            # and "build_road" to "setup_place_road" if those are the legal actions
            # This is critical because LLMs often use "build_settlement" even during setup
            if state.phase == "setup":
                if action_type_str == "build_settlement":
                    # Check if setup_place_settlement is legal
                    if any(a == Action.SETUP_PLACE_SETTLEMENT for a, _ in legal_actions_list):
                        action_type_str = "setup_place_settlement"
                        print(f"  Mapped 'build_settlement' to 'setup_place_settlement' during setup phase", flush=True)
                elif action_type_str == "build_road":
                    # Check if setup_place_road is legal
                    if any(a == Action.SETUP_PLACE_ROAD for a, _ in legal_actions_list):
                        action_type_str = "setup_place_road"
                        print(f"  Mapped 'build_road' to 'setup_place_road' during setup phase", flush=True)
            
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
            
            # For BUILD_ROAD actions, handle both "road_id" and "road_edge_id" field names
            elif target_action == Action.BUILD_ROAD and action_payload_dict:
                # LLM might use "road_id" instead of "road_edge_id"
                llm_road_id = action_payload_dict.get("road_edge_id") or action_payload_dict.get("road_id")
                if llm_road_id is not None:
                    # Try to find exact match first
                    for action, payload in legal_actions_list:
                        if action == target_action and payload and hasattr(payload, "road_edge_id"):
                            if payload.road_edge_id == llm_road_id:
                                exact_match = (action, payload)
                                break
                            # Also store first match as fallback
                            if not preferred_match:
                                preferred_match = (action, payload)
                    
                    if exact_match:
                        matching_action = exact_match
                    elif preferred_match:
                        # LLM specified a road_id but it doesn't match any legal action
                        print(f"Warning: LLM requested road_id {llm_road_id} but it's not in legal actions. Using first available.", flush=True)
                        matching_action = preferred_match
            
            # For SETUP_PLACE_SETTLEMENT and BUILD_SETTLEMENT actions, match by intersection_id
            elif target_action in (Action.SETUP_PLACE_SETTLEMENT, Action.BUILD_SETTLEMENT) and action_payload_dict:
                # LLM might use "intersection" or "intersectionId" instead of "intersection_id"
                llm_intersection_id = (action_payload_dict.get("intersection_id") or 
                                      action_payload_dict.get("intersection") or
                                      action_payload_dict.get("intersectionId"))
                if llm_intersection_id is not None:
                    # Try to find exact match first
                    for action, payload in legal_actions_list:
                        if action == target_action and payload and hasattr(payload, "intersection_id"):
                            if payload.intersection_id == llm_intersection_id:
                                exact_match = (action, payload)
                                break
                            # Also store first match as fallback
                            if not preferred_match:
                                preferred_match = (action, payload)
                    
                    if exact_match:
                        matching_action = exact_match
                    elif preferred_match:
                        # LLM specified an intersection_id but it doesn't match any legal action
                        print(f"Warning: LLM requested intersection_id {llm_intersection_id} but it's not in legal actions. Using first available.", flush=True)
                        matching_action = preferred_match
            
            # For BUILD_CITY actions, match by intersection_id
            elif target_action == Action.BUILD_CITY and action_payload_dict:
                # LLM might use "intersection" or "intersectionId" instead of "intersection_id"
                llm_intersection_id = (action_payload_dict.get("intersection_id") or 
                                      action_payload_dict.get("intersection") or
                                      action_payload_dict.get("intersectionId"))
                if llm_intersection_id is not None:
                    # Try to find exact match first
                    for action, payload in legal_actions_list:
                        if action == target_action and payload and hasattr(payload, "intersection_id"):
                            if payload.intersection_id == llm_intersection_id:
                                exact_match = (action, payload)
                                break
                            # Also store first match as fallback
                            if not preferred_match:
                                preferred_match = (action, payload)
                    
                    if exact_match:
                        matching_action = exact_match
                    elif preferred_match:
                        # LLM specified an intersection_id but it doesn't match any legal action
                        print(f"Warning: LLM requested intersection_id {llm_intersection_id} but it's not in legal actions. Using first available.", flush=True)
                        matching_action = preferred_match
            
            # For PROPOSE_TRADE, construct payload from LLM response
            # This allows the agent to propose any trade, not just pre-enumerated ones
            if target_action == Action.PROPOSE_TRADE and action_payload_dict:
                if "give_resources" in action_payload_dict and "receive_resources" in action_payload_dict:
                    llm_give = action_payload_dict["give_resources"]
                    llm_receive = action_payload_dict["receive_resources"]
                    llm_target_players = action_payload_dict.get("target_player_ids", [])
                    
                    # Check if legal action has None payload (meaning we can construct any trade)
                    has_none_payload = any(action == target_action and payload is None for action, payload in legal_actions_list)
                    
                    if has_none_payload:
                        # Construct ProposeTradePayload from LLM's response
                        from engine import ResourceType, ProposeTradePayload
                        
                        # Convert string resource names to ResourceType enum
                        def convert_resource_dict(d):
                            """Convert resource dict with string keys to ResourceType keys."""
                            result = {}
                            for k, v in d.items():
                                if isinstance(k, str):
                                    # Find matching ResourceType
                                    found = False
                                    for rt in ResourceType:
                                        if rt.value == k.lower():
                                            result[rt] = v
                                            found = True
                                            break
                                    if not found:
                                        raise ValueError(f"Invalid resource type: {k}. Valid types: {[rt.value for rt in ResourceType]}")
                                else:
                                    result[k] = v
                            return result
                        
                        try:
                            give_resources = convert_resource_dict(llm_give)
                            receive_resources = convert_resource_dict(llm_receive)
                            
                            # Validate target_player_ids
                            if not llm_target_players:
                                raise ValueError("target_player_ids is required for PROPOSE_TRADE")
                            
                            # Validate that player has the resources they're giving
                            # (This validation will also happen in the engine, but we can catch it early)
                            
                            # Construct the payload
                            constructed_payload = ProposeTradePayload(
                                target_player_ids=llm_target_players,
                                give_resources=give_resources,
                                receive_resources=receive_resources
                            )
                            
                            # Use the constructed payload
                            matching_action = (target_action, constructed_payload)
                            print(f"  Constructed PROPOSE_TRADE payload: give={llm_give}, receive={llm_receive}, targets={llm_target_players}", flush=True)
                        except (ValueError, KeyError, TypeError) as e:
                            # Invalid trade format - raise error to trigger retry
                            error_msg = f"Invalid PROPOSE_TRADE format: {str(e)}. Required fields: give_resources (dict), receive_resources (dict), target_player_ids (list). Resource types must be: {[rt.value for rt in ResourceType]}"
                            print(f"Error: {error_msg}", flush=True)
                            raise ValueError(error_msg)
                    else:
                        # Old behavior: try to match against pre-enumerated trades
                        # Normalize LLM's resource dict (convert string keys to ResourceType if needed)
                        def normalize_llm_resource_dict(d):
                            """Normalize LLM's resource dict to match legal action format."""
                            result = {}
                            for k, v in d.items():
                                # Convert string resource names to ResourceType enum if needed
                                if isinstance(k, str):
                                    from engine import ResourceType
                                    try:
                                        # Try to find matching ResourceType
                                        for rt in ResourceType:
                                            if rt.value == k.lower():
                                                result[rt] = v
                                                break
                                        else:
                                            # Keep as string if no match
                                            result[k] = v
                                    except:
                                        result[k] = v
                                else:
                                    result[k] = v
                            return result
                        
                        llm_give_normalized = normalize_llm_resource_dict(llm_give)
                        llm_receive_normalized = normalize_llm_resource_dict(llm_receive)
                        
                        # Try to find exact match
                        for action, payload in legal_actions_list:
                            if action == target_action and payload:
                                payload_dict = self._payload_to_dict(payload)
                                legal_give = payload_dict.get("give_resources", {})
                                legal_receive = payload_dict.get("receive_resources", {})
                                # Check if give_resources and receive_resources match exactly
                                if legal_give == llm_give_normalized and legal_receive == llm_receive_normalized:
                                    matching_action = (action, payload)
                                    break
                        
                        if not matching_action:
                            print(f"Warning: LLM requested PROPOSE_TRADE with give_resources={llm_give}, receive_resources={llm_receive}, but no exact match found in legal actions", flush=True)
                            print(f"  Available trades: {[(self._payload_to_dict(p).get('give_resources'), self._payload_to_dict(p).get('receive_resources')) for a, p in legal_actions_list if a == target_action][:5]}", flush=True)
                            # Don't fallback - require exact match for PROPOSE_TRADE
                            # This prevents executing wrong trades
                else:
                    # Missing required fields - raise error to trigger retry
                    error_msg = "PROPOSE_TRADE requires 'give_resources' and 'receive_resources' in action_payload. Format: {\"action_type\": \"propose_trade\", \"action_payload\": {\"give_resources\": {\"wood\": 1}, \"receive_resources\": {\"sheep\": 1}, \"target_player_ids\": [\"player_1\"]}}"
                    print(f"Error: {error_msg}", flush=True)
                    raise ValueError(error_msg)
            
            # For DISCARD_RESOURCES, handle payload generation if LLM provides resources dict
            if target_action == Action.DISCARD_RESOURCES and action_payload_dict and "resources" in action_payload_dict:
                from engine import DiscardResourcesPayload, ResourceType
                
                # Convert LLM's resource dict (string keys) to ResourceType keys
                llm_resources = action_payload_dict["resources"]
                discard_dict = {}
                for k, v in llm_resources.items():
                    if isinstance(k, str):
                        # Find matching ResourceType
                        for rt in ResourceType:
                            if rt.value == k.lower():
                                discard_dict[rt] = v
                                break
                        else:
                            raise ValueError(f"Invalid resource type: {k}. Valid types: {[rt.value for rt in ResourceType]}")
                    else:
                        discard_dict[k] = v
                
                # Create payload from LLM's specification
                constructed_payload = DiscardResourcesPayload(resources=discard_dict)
                matching_action = (target_action, constructed_payload)
                print(f"  Constructed DISCARD_RESOURCES payload from LLM: {discard_dict}", flush=True)
            
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
                
                # Fallback only if still no match (and not PROPOSE_TRADE which requires exact match)
                if not matching_action and target_action != Action.PROPOSE_TRADE:
                    # Store first match as fallback
                    for action, payload in legal_actions_list:
                        if action == target_action:
                            matching_action = (action, payload)
                            break
            
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
            
            # Handle DISCARD_RESOURCES actions that have None payload
            # Generate a valid discard payload if needed
            if action == Action.DISCARD_RESOURCES and payload is None:
                from engine import DiscardResourcesPayload
                import random
                
                # Get the player
                player = next((p for p in state.players if p.id == self.player_id), None)
                if player:
                    total_resources = sum(player.resources.values())
                    discard_count = total_resources // 2
                    
                    # Create a list of all resources the player has
                    available_resources = []
                    for resource_type, amount in player.resources.items():
                        available_resources.extend([resource_type] * amount)
                    
                    # Randomly select resources to discard
                    if len(available_resources) >= discard_count:
                        resources_to_discard = random.sample(available_resources, discard_count)
                        
                        # Count resources by type
                        discard_dict = {}
                        for resource in resources_to_discard:
                            discard_dict[resource] = discard_dict.get(resource, 0) + 1
                        
                        # Create payload
                        payload = DiscardResourcesPayload(resources=discard_dict)
                        print(f"  Generated DISCARD_RESOURCES payload: {discard_dict}", flush=True)
                    else:
                        # Shouldn't happen, but raise error if it does
                        raise ValueError(f"Cannot discard {discard_count} resources when player only has {len(available_resources)}")
                else:
                    raise ValueError(f"Player {self.player_id} not found for DISCARD_RESOURCES")
            
            return (action, payload, reasoning, response_text)
            
        except json.JSONDecodeError as e:
            # Check if this is a PROPOSE_TRADE action - if so, don't fallback, raise error for retry
            import re
            action_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
            if action_match and action_match.group(1).lower() == "propose_trade":
                error_msg = f"Failed to parse PROPOSE_TRADE JSON response: {e}. Please ensure your response is valid JSON with the format: {{\"action_type\": \"propose_trade\", \"action_payload\": {{\"give_resources\": {{\"wood\": 1}}, \"receive_resources\": {{\"sheep\": 1}}, \"target_player_ids\": [\"player_1\"]}}}}"
                print(f"Error: {error_msg}", flush=True)
                raise ValueError(error_msg)
            
            # Fallback: try to extract action from text using regex
            print(f"Warning: Failed to parse LLM response as JSON: {e}", flush=True)
            print(f"Response (first 500 chars): {response_text[:500]}", flush=True)
            
            # Try to extract action_type from text using regex
            if not action_match:
                action_match = re.search(r'action_type["\']?\s*[:=]\s*["\']?([a-z_]+)', response_text, re.IGNORECASE)
            
            if action_match:
                action_type_str = action_match.group(1).lower()
                # If it's PROPOSE_TRADE, don't fallback
                if action_type_str == "propose_trade":
                    error_msg = f"Failed to parse PROPOSE_TRADE from text. Please provide valid JSON. Format: {{\"action_type\": \"propose_trade\", \"action_payload\": {{\"give_resources\": {{\"wood\": 1}}, \"receive_resources\": {{\"sheep\": 1}}, \"target_player_ids\": [\"player_1\"]}}}}"
                    print(f"Error: {error_msg}", flush=True)
                    raise ValueError(error_msg)
                
                # Try to find matching action
                for action, payload in legal_actions_list:
                    if action_type_str in action.value.lower() or action.value.lower() in action_type_str:
                        print(f"  Extracted action from text: {action.value}", flush=True)
                        return (action, payload, f"Parsed from text (JSON parse failed): {e}", response_text)
            
            # Last resort: fallback to first legal action (unless it's PROPOSE_TRADE)
            if any(a == Action.PROPOSE_TRADE for a, _ in legal_actions_list):
                error_msg = f"Failed to parse LLM response as JSON: {e}. PROPOSE_TRADE requires valid JSON format. Please retry with proper JSON."
                print(f"Error: {error_msg}", flush=True)
                raise ValueError(error_msg)
            
            print(f"  Falling back to first legal action: {legal_actions_list[0][0].value}", flush=True)
            action, payload = legal_actions_list[0]
            return (action, payload, f"Failed to parse LLM response: {e}", response_text)
        except ValueError as e:
            # Re-raise ValueError (these are our intentional errors for PROPOSE_TRADE)
            raise
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
        # For PROPOSE_TRADE, require exact match of give_resources and receive_resources
        if "give_resources" in payload1 and "give_resources" in payload2:
            # Normalize resource dicts for comparison
            def normalize_resource_dict(d):
                result = {}
                for k, v in d.items():
                    # Convert ResourceType enum to string if needed
                    if hasattr(k, 'value'):
                        k = k.value
                    result[k] = v
                return result
            
            give1 = normalize_resource_dict(payload1["give_resources"])
            give2 = normalize_resource_dict(payload2["give_resources"])
            if give1 != give2:
                return False
            
            receive1 = normalize_resource_dict(payload1.get("receive_resources", {}))
            receive2 = normalize_resource_dict(payload2.get("receive_resources", {}))
            if receive1 != receive2:
                return False
        
        # Check if key fields match for other fields
        for key in payload1:
            if key in payload2:
                val1 = payload1[key]
                val2 = payload2[key]
                # Handle ResourceType enums
                if hasattr(val1, 'value'):
                    val1 = val1.value
                if hasattr(val2, 'value'):
                    val2 = val2.value
                # Handle nested dicts (like for other action types)
                if isinstance(val1, dict) and isinstance(val2, dict):
                    if val1 != val2:
                        return False
                elif val1 != val2:
                    return False
        return True

