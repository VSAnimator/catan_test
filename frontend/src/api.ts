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
}

export interface LegalAction {
  type: string
  payload?: {
    type?: string
    road_edge_id?: number
    intersection_id?: number
    card_type?: string
    give_resources?: Record<string, number>  // Multi-resource support
    receive_resources?: Record<string, number>  // Multi-resource support
    port_intersection_id?: number | null
    other_player_id?: string
    tile_id?: number
    resources?: Record<string, number>
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

