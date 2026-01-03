// Adaptive backend port detection
// Priority: 1. Environment variable, 2. localStorage, 3. Auto-detect, 4. Default 8000

let cachedBackendPort: number | null = null

function getBackendPort(): number {
  // Check environment variable first
  const envPort = import.meta.env.VITE_API_PORT
  if (envPort) {
    const port = parseInt(envPort, 10)
    cachedBackendPort = port
    return port
  }
  
  // Use cached port if available
  if (cachedBackendPort !== null) {
    return cachedBackendPort
  }
  
  // Check localStorage for previously detected port
  const storedPort = localStorage.getItem('catan_backend_port')
  if (storedPort) {
    const port = parseInt(storedPort, 10)
    cachedBackendPort = port
    return port
  }
  
  // Default fallback
  return 8000
}

async function detectBackendPort(): Promise<number> {
  // Try common ports in order
  const portsToTry = [8000, 8001, 8002, 8003, 8004, 8005]
  
  for (const port of portsToTry) {
    try {
      const response = await fetch(`http://localhost:${port}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(500) // 500ms timeout
      })
      // Check if we get a successful response from the health endpoint
      if (response.ok) {
        cachedBackendPort = port
        localStorage.setItem('catan_backend_port', port.toString())
        return port
      }
    } catch (e) {
      // Port not available, try next
      continue
    }
  }
  
  // Fallback to default
  const defaultPort = 8000
  cachedBackendPort = defaultPort
  return defaultPort
}

// Getter function for API_BASE that always uses current port
function getApiBase(): string {
  return `http://localhost:${getBackendPort()}/api`
}

// Auto-detect backend port on module load (non-blocking)
detectBackendPort().then(port => {
  console.log(`Backend detected on port ${port}`)
}).catch(() => {
  console.log(`Using default backend port ${getBackendPort()}`)
})

// Export function to manually set backend port
export function setBackendPort(port: number) {
  cachedBackendPort = port
  localStorage.setItem('catan_backend_port', port.toString())
  console.log(`Backend port set to ${port}`)
}

// Export function to get current backend URL
export function getBackendUrl(): string {
  return getApiBase()
}

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
  setup_last_settlement_id?: number | null  // Intersection ID of the last settlement placed in setup
  setup_first_settlement_player_index?: number | null  // Index of player who placed first settlement
  robber_tile_id: number | null
  waiting_for_robber_move: boolean
  waiting_for_robber_steal: boolean
  players_discarded?: string[]  // Players who have already discarded this turn (when 7 is rolled)
  robber_initial_tile_id?: number | null  // Robber position when 7 was rolled (to detect if it's been moved)
  roads_from_road_building?: Record<string, number>  // Player ID -> number of free roads remaining from road building card
  dev_cards_bought_this_turn?: string[]  // Player IDs who bought dev cards this turn
  dev_cards_played_this_turn?: string[]  // Player IDs who played dev cards this turn
  // Trade state
  pending_trade_offer?: {
    proposer_id: string
    target_player_ids: string[]
    give_resources: Record<string, number>
    receive_resources: Record<string, number>
  } | null
  pending_trade_responses?: Record<string, boolean>  // Player ID -> True if accepted, False if rejected
  pending_trade_current_responder_index?: number  // Index into target_player_ids for current responder
  // Card counts
  resource_card_counts?: Record<string, number>  // Resource type -> count available
  dev_card_counts?: Record<string, number>  // Dev card type -> count available
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
  player_id?: string | null
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

// ---------------------------------------------------------------------------
// Drills API (curated "best action" datasets)
// ---------------------------------------------------------------------------

export interface DrillListItem {
  id: number
  created_at: string
  name: string | null
  guideline_text?: string | null
  source_game_id: string | null
  source_step_idx: number | null
  player_id: string
  num_steps: number
  metadata?: any
}

export interface DrillStepCreate {
  player_id: string
  state: GameState
  expected_action: LegalAction  // For backward compatibility
  correct_actions?: LegalAction[]
  incorrect_actions?: LegalAction[]
}

export interface CreateDrillRequest {
  name?: string | null
  guideline_text?: string | null
  source_game_id?: string | null
  source_step_idx?: number | null
  player_id: string
  steps: DrillStepCreate[]
  metadata?: any
}

export interface CreateDrillResponse {
  drill_id: number
  message: string
}

export async function listDrills(limit: number = 200): Promise<{ drills: DrillListItem[] }> {
  const response = await fetch(`${getApiBase()}/drills?limit=${limit}`)
  return handleResponse<{ drills: DrillListItem[] }>(response)
}

export async function createDrill(request: CreateDrillRequest): Promise<CreateDrillResponse> {
  const response = await fetch(`${getApiBase()}/drills`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<CreateDrillResponse>(response)
}

export interface EvaluateDrillRequest {
  agent_type: string
  include_guidelines?: boolean
  exclude_strategic_advice?: boolean
  exclude_higher_level_features?: boolean
}

export interface EvaluateDrillResultItem {
  idx: number
  player_id: string
  match: boolean
  expected_action: any
  actual_action: any
  error?: string
}

export interface EvaluateDrillResponse {
  drill_id: number
  agent_type: string
  passed: boolean
  results: EvaluateDrillResultItem[]
}

export async function evaluateDrill(drillId: number, request: EvaluateDrillRequest): Promise<EvaluateDrillResponse> {
  const response = await fetch(`${getApiBase()}/drills/${drillId}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<EvaluateDrillResponse>(response)
}

export interface EvaluateAllDrillsRequest {
  agent_type: string
  limit?: number
  include_step_results?: boolean
  include_guidelines?: boolean
  max_concurrency?: number
  drill_ids?: number[]
  exclude_strategic_advice?: boolean
  exclude_higher_level_features?: boolean
}

export interface EvaluateAllDrillsResponse {
  agent_type: string
  run_id?: string
  evaluated_at?: string
  include_guidelines?: boolean
  max_concurrency?: number
  results: Array<{
    drill_id: number
    name: string | null
    source_game_id?: string | null
    source_step_idx?: number | null
    player_id?: string
    num_steps?: number
    passed: boolean
    first_mismatch?: any
    step_results?: Array<{
      idx: number
      player_id: string
      match: boolean
      expected_action: any
      actual_action: any
      error?: string
    }>
    error?: string
  }>
}

export async function evaluateAllDrills(request: EvaluateAllDrillsRequest): Promise<EvaluateAllDrillsResponse> {
  const response = await fetch(`${getApiBase()}/drills/evaluate_all`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<EvaluateAllDrillsResponse>(response)
}

export interface GetDrillResponse {
  drill: {
    id: number
    created_at: string
    name: string | null
    guideline_text: string | null
    source_game_id: string | null
    source_step_idx: number | null
    player_id: string
    metadata?: any
  }
  steps: Array<{
    idx: number
    player_id: string
    state: GameState
    expected_action: any
    state_text?: string | null
    legal_actions_text?: string | null
  }>
}

export interface UpdateDrillRequest {
  name?: string | null
  guideline_text?: string | null
}

export async function updateDrill(drillId: number, request: UpdateDrillRequest): Promise<{ message: string }> {
  const response = await fetch(`${getApiBase()}/drills/${drillId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<{ message: string }>(response)
}

export async function getDrill(drillId: number): Promise<GetDrillResponse> {
  const response = await fetch(`${getApiBase()}/drills/${drillId}`)
  return handleResponse<GetDrillResponse>(response)
}

export interface CreateGameRequest {
  player_names: string[]
  rng_seed?: number
  exclude_strategic_advice?: boolean
  exclude_higher_level_features?: boolean
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
  const response = await fetch(`${getApiBase()}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<CreateGameResponse>(response)
}

export async function getGameState(gameId: string): Promise<GameState> {
  const response = await fetch(`${getApiBase()}/games/${gameId}`)
  return handleResponse<GameState>(response)
}

export async function getLegalActions(gameId: string, playerId: string): Promise<LegalAction[]> {
  const response = await fetch(`${getApiBase()}/games/${gameId}/legal_actions?player_id=${playerId}`)
  const data = await handleResponse<{ legal_actions: LegalAction[] }>(response)
  return data.legal_actions || []
}

export async function postAction(gameId: string, playerId: string, action: LegalAction): Promise<GameState> {
  const request: ActRequest = {
    player_id: playerId,
    action: action
  }
  const response = await fetch(`${getApiBase()}/games/${gameId}/act`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  const data = await handleResponse<ActResponse>(response)
  return data.new_state
}

export async function getReplay(gameId: string): Promise<ReplayResponse> {
  const response = await fetch(`${getApiBase()}/games/${gameId}/replay`)
  return handleResponse<ReplayResponse>(response)
}

export async function restoreGameState(gameId: string, state: GameState): Promise<void> {
  const response = await fetch(`${getApiBase()}/games/${gameId}/restore`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state)
  })
  await handleResponse<{ message: string; game_id: string }>(response)
}

export async function forkGame(gameId: string, state: GameState): Promise<CreateGameResponse> {
  const response = await fetch(`${getApiBase()}/games/${gameId}/fork`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state)
  })
  return handleResponse<CreateGameResponse>(response)
}

export interface RunAgentsRequest {
  max_turns?: number
  exclude_strategic_advice?: boolean
  exclude_higher_level_features?: boolean
}

export interface RunAgentsResponse {
  game_id: string
  completed: boolean
  error?: string | null
  final_state: GameState
  turns_played: number
}

export async function runAgents(gameId: string, request: RunAgentsRequest = {}): Promise<RunAgentsResponse> {
  const response = await fetch(`${getApiBase()}/games/${gameId}/run_agents`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<RunAgentsResponse>(response)
}

export interface WatchAgentsRequest {
  agent_mapping?: Record<string, string>  // player_id -> agent_type
  exclude_strategic_advice?: boolean
  exclude_higher_level_features?: boolean
}

export interface WatchAgentsResponse {
  game_id: string
  game_continues: boolean
  error?: string | null
  new_state: GameState
  player_id?: string | null
  reasoning?: string | null
}

export async function watchAgentsStep(
  gameId: string, 
  agentMapping?: Record<string, string>,
  excludeStrategicAdvice?: boolean,
  excludeHigherLevelFeatures?: boolean
): Promise<WatchAgentsResponse> {
  const response = await fetch(`${getApiBase()}/games/${gameId}/watch_agents_step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      agent_mapping: agentMapping || {},
      exclude_strategic_advice: excludeStrategicAdvice || false,
      exclude_higher_level_features: excludeHigherLevelFeatures || false
    })
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
  const response = await fetch(`${getApiBase()}/games/query_events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<QueryEventsResponse>(response)
}

// Guidelines and Feedback API

export interface Guideline {
  id: number
  player_id: string | null
  guideline_text: string
  context: string | null
  priority: number
  created_at: string
  updated_at: string
  active: boolean
}

export interface AddGuidelineRequest {
  guideline_text: string
  player_id?: string | null
  context?: string | null
  priority?: number
}

export interface Feedback {
  id: number
  game_id: string
  step_idx: number | null
  player_id: string | null
  action_taken: string | null
  feedback_text: string
  feedback_type: string
  created_at: string
}

export interface AddFeedbackRequest {
  feedback_text: string
  step_idx?: number | null
  player_id?: string | null
  action_taken?: string | null
  feedback_type?: string
}

export async function addGuideline(request: AddGuidelineRequest): Promise<{ guideline_id: number; message: string }> {
  const response = await fetch(`${getApiBase()}/guidelines`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<{ guideline_id: number; message: string }>(response)
}

export async function getGuidelines(playerId?: string | null, context?: string | null): Promise<{ guidelines: Guideline[] }> {
  const params = new URLSearchParams()
  if (playerId) params.append('player_id', playerId)
  if (context) params.append('context', context)
  const response = await fetch(`${getApiBase()}/guidelines?${params.toString()}`)
  return handleResponse<{ guidelines: Guideline[] }>(response)
}

export async function updateGuideline(guidelineId: number, request: Partial<AddGuidelineRequest & { active?: boolean }>): Promise<{ message: string }> {
  const response = await fetch(`${getApiBase()}/guidelines/${guidelineId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<{ message: string }>(response)
}

export async function deleteGuideline(guidelineId: number): Promise<{ message: string }> {
  const response = await fetch(`${getApiBase()}/guidelines/${guidelineId}`, {
    method: 'DELETE'
  })
  return handleResponse<{ message: string }>(response)
}

export async function addFeedback(gameId: string, request: AddFeedbackRequest): Promise<{ feedback_id: number; message: string }> {
  const response = await fetch(`${getApiBase()}/games/${gameId}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<{ feedback_id: number; message: string }>(response)
}

export async function getFeedback(gameId?: string, playerId?: string): Promise<{ feedback: Feedback[] }> {
  const params = new URLSearchParams()
  if (gameId) params.append('game_id', gameId)
  if (playerId) params.append('player_id', playerId)
  const response = await fetch(`${getApiBase()}/feedback?${params.toString()}`)
  return handleResponse<{ feedback: Feedback[] }>(response)
}

