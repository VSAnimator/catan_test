const API_BASE = 'http://localhost:8000/api'

export interface Player {
  id: string
  name: string
  color: string
  resources: Record<string, number>
  victory_points: number
  roads_built: number
  settlements_built: number
  cities_built: number
  dev_cards: string[]
  knights_played: number
  longest_road: boolean
  largest_army: boolean
}

export interface Tile {
  id: number
  resource_type: string | null
  number_token: number | null
  position: [number, number]
}

export interface Intersection {
  id: number
  position: [number, number]
  adjacent_tiles: number[]
  adjacent_intersections: number[]
  owner: string | null
  building_type: string | null
  port_type: string | null  // null = no port, "3:1" = generic port, or resource type for 2:1 port
}

export interface RoadEdge {
  id: number
  intersection1_id: number
  intersection2_id: number
  owner: string | null
}

export interface GameState {
  game_id: string
  players: Player[]
  current_player_index: number
  phase: string
  tiles: Tile[]
  intersections: Intersection[]
  road_edges: RoadEdge[]
  dice_roll: number | null
  turn_number: number
  setup_round: number
  setup_phase_player_index: number
  robber_tile_id: number | null
  waiting_for_robber_move: boolean
  waiting_for_robber_steal: boolean
  players_discarded?: string[]  // Players who have already discarded this turn (when 7 is rolled)
  robber_initial_tile_id?: number | null  // Robber position when 7 was rolled (to detect if it's been moved)
  roads_from_road_building?: Record<string, number>  // Player ID -> number of free roads remaining from road building card
  // Trade state
  pending_trade_offer?: {
    proposer_id: string
    target_player_ids: string[]
    give_resources: Record<string, number>
    receive_resources: Record<string, number>
  } | null
  pending_trade_responses?: Record<string, boolean>  // Player ID -> True if accepted, False if rejected
  pending_trade_current_responder_index?: number  // Index into target_player_ids for current responder
}

export interface LegalAction {
  type: string
  payload?: {
    type?: string
    road_edge_id?: number
    intersection_id?: number
    card_type?: string
    year_of_plenty_resources?: Record<string, number>  // For year_of_plenty: 2 resources to receive
    monopoly_resource_type?: string  // For monopoly: resource type to steal
    give_resources?: Record<string, number>  // Multi-resource support
    receive_resources?: Record<string, number>  // Multi-resource support
    port_intersection_id?: number | null
    other_player_id?: string
    tile_id?: number
    resources?: Record<string, number>
    selected_player_id?: string  // For SELECT_TRADE_PARTNER: which accepting player to trade with
    target_player_ids?: string[]  // For PROPOSE_TRADE: which players to propose to
    // Legacy single-resource fields (for backward compatibility during transition)
    give_resource?: string
    give_amount?: number
    receive_resource?: string
    receive_amount?: number
  }
}

export interface StepLog {
  state_before: GameState
  action: LegalAction
  state_after: GameState
  dice_roll: number | null
  timestamp: string
}

export interface ReplayResponse {
  game_id: string
  steps: StepLog[]
}

export interface CreateGameRequest {
  player_names: string[]
  rng_seed?: number
}

export interface CreateGameResponse {
  game_id: string
  initial_state: GameState
}

export interface ActRequest {
  player_id: string
  action: LegalAction
}

export interface ActResponse {
  new_state: GameState
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
  }
  return response.json()
}

export async function createGame(request: CreateGameRequest): Promise<CreateGameResponse> {
  const response = await fetch(`${API_BASE}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<CreateGameResponse>(response)
}

export async function getGameState(gameId: string): Promise<GameState> {
  const response = await fetch(`${API_BASE}/games/${gameId}`)
  return handleResponse<GameState>(response)
}

export async function getLegalActions(gameId: string, playerId: string): Promise<LegalAction[]> {
  const response = await fetch(`${API_BASE}/games/${gameId}/legal_actions?player_id=${playerId}`)
  const data = await handleResponse<{ legal_actions: LegalAction[] }>(response)
  return data.legal_actions || []
}

export async function postAction(gameId: string, playerId: string, action: LegalAction): Promise<GameState> {
  const request: ActRequest = {
    player_id: playerId,
    action: action
  }
  const response = await fetch(`${API_BASE}/games/${gameId}/act`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  const data = await handleResponse<ActResponse>(response)
  return data.new_state
}

export async function getReplay(gameId: string): Promise<ReplayResponse> {
  const response = await fetch(`${API_BASE}/games/${gameId}/replay`)
  return handleResponse<ReplayResponse>(response)
}

export async function restoreGameState(gameId: string, state: GameState): Promise<void> {
  const response = await fetch(`${API_BASE}/games/${gameId}/restore`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state)
  })
  await handleResponse<{ message: string; game_id: string }>(response)
}

export async function forkGame(gameId: string, state: GameState): Promise<CreateGameResponse> {
  const response = await fetch(`${API_BASE}/games/${gameId}/fork`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state)
  })
  return handleResponse<CreateGameResponse>(response)
}

export interface RunAgentsRequest {
  max_turns?: number
}

export interface RunAgentsResponse {
  game_id: string
  completed: boolean
  error?: string | null
  final_state: GameState
  turns_played: number
}

export async function runAgents(gameId: string, request: RunAgentsRequest = {}): Promise<RunAgentsResponse> {
  const response = await fetch(`${API_BASE}/games/${gameId}/run_agents`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<RunAgentsResponse>(response)
}

export interface WatchAgentsResponse {
  game_id: string
  game_continues: boolean
  error?: string | null
  new_state: GameState
  player_id?: string | null
}

export async function watchAgentsStep(gameId: string): Promise<WatchAgentsResponse> {
  const response = await fetch(`${API_BASE}/games/${gameId}/watch_agents_step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
  })
  return handleResponse<WatchAgentsResponse>(response)
}

export interface QueryEventsRequest {
  num_games?: number
  action_type?: string
  card_type?: string
  dice_roll?: number
  player_id?: string
  min_turn?: number
  max_turn?: number
  analyze?: string
  limit?: number
}

export interface GameEvent {
  game_id: string
  step_idx: number
  player_id: string
  action_type: string
  action_payload: any
  state_before: any
  state_after: any
  timestamp?: string
}

export interface QueryEventsResponse {
  events: GameEvent[]
  summary: {
    total_events: number
    unique_games: number
    action_types: Record<string, number>
    players: Record<string, number>
    turn_distribution: Record<number, number>
  }
  analysis?: any
}

export async function queryEvents(request: QueryEventsRequest): Promise<QueryEventsResponse> {
  const response = await fetch(`${API_BASE}/games/query_events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<QueryEventsResponse>(response)
}

