# LLM Agent Observation Space Documentation

This document describes what information the LLM agent receives when making decisions in Catan.

## Overview

The agent receives a text description of the game state and a list of legal actions. The observation is formatted using `state_to_text()` and `legal_actions_to_text()` functions.

## Observation Structure

### 1. Game Overview
- **Game ID**: Unique identifier for the game
- **Phase**: Current game phase (`setup` or `playing`)
- **Turn Number**: Current turn number (0-indexed)
- **Last Dice Roll**: Most recent dice roll (2-12, or null if no roll yet)
- **Current Player**: Name and ID of the player whose turn it is
  - In setup phase: Shows setup player and setup round (1 or 2)

### 2. Your Status (Agent's Player)
- **Victory Points**: Current VP count (need 10 to win)
- **Resources**: Count of each resource type
  - Wood, Brick, Wheat, Sheep, Ore
- **Buildings**: 
  - Number of settlements built
  - Number of cities built
- **Roads**: Number of roads built
- **Dev Cards**: 
  - Total count
  - List of card types (e.g., "knight", "victory_point", "year_of_plenty", etc.)
- **Special Status**:
  - Longest Road (if held)
  - Largest Army (if held)

### 3. Other Players
For each opponent, the agent sees:
- **Name**: Player name
- **Victory Points**: Their VP count
- **Total Resources**: Sum of all resources (exact counts not shown)
- **Buildings**: Number of settlements and cities

**Note**: The agent does NOT see:
- Opponents' exact resource counts by type
- Opponents' development cards
- Opponents' road counts
- Which intersections/roads opponents own (only counts)

### 4. Board Layout

#### Dice Probability Reference
- **Roll probabilities**: Table showing probability of each dice roll (2-12)
- **Best numbers**: Highlighted that 6 and 8 are most likely (13.9% each)
- **7 roll**: Noted that 7 triggers robber and produces no resources

#### Tiles
- **Tile details**: For each tile:
  - Tile ID
  - Resource type (wood, brick, wheat, sheep, ore, or desert)
  - Number token value
  - **Roll probability**: Percentage chance this number will be rolled
  - **Robber location**: Indicated if robber is on this tile

#### Intersections (Enhanced Spatial Information)

**Your Intersections:**
- Intersection ID and building type
- **Port information**: Port type if intersection has a port (3:1 or 2:1 specific resource)
- **Adjacent tiles**: List of tiles this intersection touches, with:
  - Tile ID
  - Resource type
  - Number token value
  - Roll probability percentage
- **Expected production**: Total production value per roll (sum of probabilities)
- **Resource breakdown**: Expected income per resource type per roll
- **Adjacent intersections**: List of connected intersections with ownership info

**Opponent Intersections:**
- Intersection ID, building type, and owner name
- Port information if applicable
- Adjacent tiles (abbreviated format)

**Available Intersections (for building):**
- Top 15 candidates sorted by production value
- Production value per roll
- Resource breakdown
- Port information if applicable
- Adjacent intersections list
- Only shows intersections that can be built (respects distance rule)

#### Road Network
- **Your roads**: List of all your roads with intersection connections
- **Opponent roads**: Total count

#### Robber
- Current robber location (tile ID and resource type)
- List of players whose production is blocked by the robber

### 5. Recent Actions (if available)
- Last 3 actions taken in the game
- Format: Action type with relevant details (e.g., "Build Settlement at intersection 5")

### 6. Legal Actions

The agent receives a formatted list of all legal actions it can take. Each action includes:

#### Action Types:
- **Setup Actions**:
  - `setup_place_settlement`: Place settlement at intersection (with intersection_id)
  - `setup_place_road`: Place road on edge (with road_edge_id)
  - `start_game`: Begin the game (after setup complete)

- **Building Actions**:
  - `build_settlement`: Build settlement (with intersection_id)
  - `build_city`: Upgrade settlement to city (with intersection_id)
  - `build_road`: Build road (with road_edge_id)

- **Development Cards**:
  - `buy_dev_card`: Purchase a development card
  - `play_dev_card`: Play a development card (with card_type and optional payload)
    - Knight: Move robber and steal
    - Year of Plenty: Choose 2 resources
    - Monopoly: Choose resource type to steal
    - Road Building: Build 2 free roads
    - Victory Point: Revealed at game end

- **Trading**:
  - `trade_bank`: Trade with bank (with give_resources, receive_resources, optional port_intersection_id)
  - `propose_trade`: Propose trade to other players (with give_resources, receive_resources, target_player_ids)
  - `accept_trade`: Accept a pending trade offer
  - `reject_trade`: Reject a pending trade offer
  - `select_trade_partner`: Choose which accepting player to trade with (with selected_player_id)

- **Robber Actions**:
  - `move_robber`: Move robber to a tile (with tile_id)
  - `steal_resource`: Steal a resource from a player (with other_player_id)

- **Other**:
  - `discard_resources`: Discard half resources when 7 is rolled (with resources dict)
  - `end_turn`: End your turn

#### Action Payloads:
Each action may include a payload with specific parameters:
- `intersection_id`: For settlement/city placement
- `road_edge_id`: For road building
- `tile_id`: For moving robber
- `card_type`: For playing dev cards
- `give_resources`: Dict of resources to give (e.g., `{"wood": 4}`)
- `receive_resources`: Dict of resources to receive
- `target_player_ids`: List of player IDs for trade proposals
- `selected_player_id`: Player ID for selecting trade partner
- `other_player_id`: Player ID for stealing
- `resources`: Dict for discarding resources
- `port_intersection_id`: Intersection ID for port trades

## Enhanced Information (Now Included)

The agent now receives:
1. **Dice probability table**: Roll probabilities for all numbers (2-12)
2. **Spatial relationships**: 
   - Which tiles each intersection touches (with resource types and number tokens)
   - Which intersections are adjacent to each other
3. **Port locations**: Which intersections have ports and their types (3:1 or 2:1 specific resource)
4. **Production analysis**: 
   - Expected production value per roll for each intersection
   - Resource breakdown showing expected income per resource type
5. **Opponent building locations**: Where opponents have built (intersection IDs)
6. **Road network structure**: Which roads connect which intersections
7. **Available intersection analysis**: Top candidates for building with production values sorted

## Still Missing Information

The agent does NOT receive:
1. **Opponent resource details**: Exact resource counts per type (only total)
2. **Opponent development cards**: What cards opponents have
3. **Tile positions**: Spatial coordinates (but adjacency is shown)
4. **Distance calculations**: How far settlements are from each other (but adjacency is shown)

## Current Limitations

The observation space is **text-based** and **high-level**. The agent must reason about:
- Which intersections are good for settlements (without seeing adjacency)
- Which resources are needed (without seeing production rates)
- Strategic positioning (without spatial information)
- Resource trading value (without market dynamics)

## Recommendations for Improvement

To improve agent performance, consider adding:
1. **Spatial information**: Tile adjacency, intersection connections
2. **Resource production analysis**: Expected income from each settlement
3. **Number token probabilities**: Dice roll probabilities (2-12)
4. **Port information**: Which ports exist and their types
5. **Opponent building locations**: Where opponents have built (for blocking)
6. **Road network visualization**: Which roads connect where
7. **Resource scarcity**: How many of each resource type exist on the board
8. **Distance metrics**: Distance between potential settlement locations

