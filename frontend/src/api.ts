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
  // In production, use environment variable or same origin (when behind nginx proxy)
  const apiUrl = import.meta.env.VITE_API_URL
  if (apiUrl) {
    return `${apiUrl}/api`
  }
  
  // In development, use localhost with port detection
  if (import.meta.env.DEV) {
    return `http://localhost:${getBackendPort()}/api`
  }
  
  // In production build without env var, assume same origin (nginx proxy)
  return '/api'
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
  agent_mapping?: Record<string, string>  // player_id -> agent_type
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
}

export interface WatchAgentsResponse {
  game_id: string
  game_continues: boolean
  error?: string | null
  new_state: GameState
  player_id?: string | null
  reasoning?: string | null
}

export async function watchAgentsStep(gameId: string, agentMapping?: Record<string, string>): Promise<WatchAgentsResponse> {
  const response = await fetch(`${getApiBase()}/games/${gameId}/watch_agents_step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agent_mapping: agentMapping || {} })
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

// Authentication API
export interface User {
  id: string
  username: string
  email?: string | null
  created_at: string
}

export interface Token {
  access_token: string
  token_type: string
  user: User
}

export interface RegisterRequest {
  username: string
  password: string
  email?: string | null
}

export interface LoginRequest {
  username: string
  password: string
}

export async function register(request: RegisterRequest): Promise<Token> {
  const response = await fetch(`${getApiBase()}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<Token>(response)
}

export async function login(request: LoginRequest): Promise<Token> {
  const response = await fetch(`${getApiBase()}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  return handleResponse<Token>(response)
}

export async function getCurrentUser(token: string): Promise<User> {
  const response = await fetch(`${getApiBase()}/auth/me`, {
    method: 'GET',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  })
  return handleResponse<User>(response)
}

export async function logout(token: string): Promise<{ message: string }> {
  const response = await fetch(`${getApiBase()}/auth/logout`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  })
  return handleResponse<{ message: string }>(response)
}

export async function refreshToken(token: string): Promise<Token> {
  const response = await fetch(`${getApiBase()}/auth/refresh`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  })
  return handleResponse<Token>(response)
}

// Room API
export interface RoomPlayer {
  user_id: string
  username: string
  player_id?: string | null
  joined_at: string
}

export interface Room {
  room_id: string
  host_user_id: string
  status: string
  max_players: number
  min_players: number
  players: RoomPlayer[]
  game_id?: string | null
  created_at: string
  is_private: boolean
  player_count: number
}

export interface CreateRoomRequest {
  max_players?: number
  min_players?: number
  is_private?: boolean
  password?: string | null
}

export interface JoinRoomRequest {
  password?: string | null
}

export async function createRoom(request: CreateRoomRequest, token: string): Promise<Room> {
  const response = await fetch(`${getApiBase()}/rooms`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(request)
  })
  return handleResponse<Room>(response)
}

export async function listRooms(): Promise<Room[]> {
  const response = await fetch(`${getApiBase()}/rooms`)
  return handleResponse<Room[]>(response)
}

export async function getRoom(roomId: string): Promise<Room> {
  const response = await fetch(`${getApiBase()}/rooms/${roomId}`)
  return handleResponse<Room>(response)
}

export async function joinRoom(roomId: string, request: JoinRoomRequest, token: string): Promise<Room> {
  const response = await fetch(`${getApiBase()}/rooms/${roomId}/join`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(request)
  })
  return handleResponse<Room>(response)
}

export async function leaveRoom(roomId: string, token: string): Promise<{ message: string }> {
  const response = await fetch(`${getApiBase()}/rooms/${roomId}/leave`, {
    method: 'POST',
    headers: { 
      'Authorization': `Bearer ${token}`
    }
  })
  return handleResponse<{ message: string }>(response)
}

export async function getMyRooms(token: string): Promise<Room[]> {
  const response = await fetch(`${getApiBase()}/rooms/user/my-rooms`, {
    method: 'GET',
    headers: { 
      'Authorization': `Bearer ${token}`
    }
  })
  return handleResponse<Room[]>(response)
}

export async function startGameFromRoom(roomId: string, token: string): Promise<{ game_id: string; room: Room; initial_state: GameState }> {
  const response = await fetch(`${getApiBase()}/rooms/${roomId}/start`, {
    method: 'POST',
    headers: { 
      'Authorization': `Bearer ${token}`
    }
  })
  return handleResponse<{ game_id: string; room: Room; initial_state: GameState }>(response)
}

// WebSocket client helper
export class GameWebSocket {
  private ws: WebSocket | null = null
  private gameId: string
  private token: string | null
  private onMessage: (data: any) => void
  private onError: (error: Error) => void
  private reconnectAttempts = 0
  private maxReconnectAttempts = 10
  private baseReconnectDelay = 1000  // Start with 1 second
  private maxReconnectDelay = 30000  // Max 30 seconds
  private reconnectTimeout: number | null = null
  private isManualDisconnect = false

  constructor(
    gameId: string,
    token: string | null,
    onMessage: (data: any) => void,
    onError: (error: Error) => void
  ) {
    this.gameId = gameId
    this.token = token
    this.onMessage = onMessage
    this.onError = onError
  }

  connect(): void {
    // Determine WebSocket URL
    let wsUrl: string
    const apiUrl = import.meta.env.VITE_API_URL
    
    if (apiUrl) {
      // Use environment variable
      const protocol = apiUrl.startsWith('https') ? 'wss:' : 'ws:'
      const host = apiUrl.replace(/^https?:\/\//, '').replace(/\/$/, '')
      wsUrl = `${protocol}//${host}/api/ws/game/${this.gameId}${this.token ? `?token=${encodeURIComponent(this.token)}` : ''}`
    } else if (import.meta.env.DEV) {
      // Development: use localhost with port detection
      const port = getBackendPort()
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      wsUrl = `${protocol}//localhost:${port}/api/ws/game/${this.gameId}${this.token ? `?token=${encodeURIComponent(this.token)}` : ''}`
    } else {
      // Production: use same origin (nginx proxy)
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      wsUrl = `${protocol}//${window.location.host}/api/ws/game/${this.gameId}${this.token ? `?token=${encodeURIComponent(this.token)}` : ''}`
    }
    
    console.log('Connecting to WebSocket:', wsUrl)
    
    try {
      this.ws = new WebSocket(wsUrl)
      
      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.reconnectAttempts = 0
        // Send ping to keep connection alive
        this.startHeartbeat()
      }
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'pong') {
            return // Ignore pong responses
          }
          this.onMessage(data)
        } catch (e) {
          console.error('Error parsing WebSocket message:', e)
        }
      }
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.onError(new Error('WebSocket connection error'))
      }
      
      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected', { code: event.code, reason: event.reason, wasClean: event.wasClean })
        
        // Don't reconnect if manually disconnected
        if (this.isManualDisconnect) {
          this.isManualDisconnect = false
          return
        }
        
        // Attempt to reconnect with exponential backoff
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++
          // Exponential backoff: baseDelay * 2^(attempt-1), capped at maxReconnectDelay
          const delay = Math.min(
            this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
            this.maxReconnectDelay
          )
          
          console.log(`Reconnecting in ${delay}ms... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
          
          this.reconnectTimeout = window.setTimeout(() => {
            console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`)
            this.connect()
          }, delay)
        } else {
          console.error('Max reconnection attempts reached')
          this.onError(new Error('WebSocket connection lost and reconnection failed after multiple attempts'))
        }
      }
    } catch (error) {
      console.error('Error creating WebSocket:', error)
      this.onError(new Error('Failed to create WebSocket connection'))
    }
  }

  private heartbeatInterval: number | null = null

  private startHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
    }
    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, 30000) // Send ping every 30 seconds
  }

  disconnect(): void {
    this.isManualDisconnect = true
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }
    
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
    
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    
    this.reconnectAttempts = 0
  }
  
  reconnect(): void {
    // Reset reconnection attempts and manually trigger reconnect
    this.reconnectAttempts = 0
    this.isManualDisconnect = false
    if (this.ws) {
      this.ws.close()
    } else {
      this.connect()
    }
  }
  
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }
  
  getReconnectAttempts(): number {
    return this.reconnectAttempts
  }

  send(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not open')
    }
  }
}

