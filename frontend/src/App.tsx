import React, { useState, useEffect, useRef } from 'react'
import './App.css'
import {
  createGame as apiCreateGame,
  getGameState,
  getLegalActions,
  postAction,
  getReplay,
  restoreGameState,
  forkGame,
  runAgents,
  watchAgentsStep,
  queryEvents,
  addFeedback,
  listDrills,
  createDrill,
  evaluateDrill,
  evaluateAllDrills,
  getDrill,
  updateDrill,
  type GameState,
  type LegalAction,
  type Player,
  type ReplayResponse,
  type QueryEventsResponse,
  type DrillListItem,
  type EvaluateAllDrillsResponse
} from './api'

type View = 'main' | 'game' | 'replay' | 'agent-watch' | 'event-query' | 'drills'

// Resource icons mapping
const RESOURCE_ICONS: Record<string, string> = {
  'wood': '🌲',
  'brick': '🧱',
  'wheat': '🌾',
  'sheep': '🐑',
  'ore': '⛏️'
}

// Get number of pips (probability dots) for a dice roll number
const getPipCount = (number: number): number => {
  // Dice probability: 2,12=1; 3,11=2; 4,10=3; 5,9=4; 6,8=5
  if (number === 2 || number === 12) return 1
  if (number === 3 || number === 11) return 2
  if (number === 4 || number === 10) return 3
  if (number === 5 || number === 9) return 4
  if (number === 6 || number === 8) return 5
  return 0
}

function App() {
  const [view, setView] = useState<View>('main')
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [playerId, setPlayerId] = useState<string>('')
  const [legalActions, setLegalActions] = useState<LegalAction[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [devMode, setDevMode] = useState(false)
  const refreshIntervalRef = useRef<number | null>(null)
  const discardPanelInitializedRef = useRef(false)  // Track if discard panel has been initialized to prevent clearing user input during auto-refresh
  const lastDiscardTurnRef = useRef<number | null>(null)  // Track turn number to detect new discard phases
  
  // For create/join game
  const [gameIdInput, setGameIdInput] = useState('')
  const [playerNameInput, setPlayerNameInput] = useState('')
  const [numPlayers, setNumPlayers] = useState(2)
  const [agentMapping, setAgentMapping] = useState<Record<string, string>>({})  // player_id -> agent_type
  const [createGameExcludeStrategicAdvice, setCreateGameExcludeStrategicAdvice] = useState<boolean>(false)
  const [createGameExcludeHigherLevelFeatures, setCreateGameExcludeHigherLevelFeatures] = useState<boolean>(false)

  // For replay view
  const [replayData, setReplayData] = useState<ReplayResponse | null>(null)
  const [replayStepIndex, setReplayStepIndex] = useState(0)
  const [replayGameId, setReplayGameId] = useState('')

  // Drill recording mode (started from replay viewer)
  const [drillRecording, setDrillRecording] = useState<null | {
    name: string
    guideline_text: string
    source_game_id: string
    source_step_idx: number
    drill_player_id: string
    forked_game_id: string
    steps: Array<{ player_id: string; state: GameState; expected_action: LegalAction }>
  }>(null)

  // Drills page state
  const [drillsList, setDrillsList] = useState<DrillListItem[]>([])
  const [drillsAgentType, setDrillsAgentType] = useState<string>('behavior_tree')
  const [drillsUseGuidelines, setDrillsUseGuidelines] = useState<boolean>(false)
  const [drillsExcludeStrategicAdvice, setDrillsExcludeStrategicAdvice] = useState<boolean>(false)
  const [drillsExcludeHigherLevelFeatures, setDrillsExcludeHigherLevelFeatures] = useState<boolean>(false)
  const [selectedDrillIds, setSelectedDrillIds] = useState<Set<number>>(new Set())
  const [drillsEval, setDrillsEval] = useState<EvaluateAllDrillsResponse | null>(null)
  const [drillsLoading, setDrillsLoading] = useState(false)
  // Batch evaluation metadata (single run that should back both row PASS/FAIL and details)
  const [drillsEvalRunMeta, setDrillsEvalRunMeta] = useState<{ run_id?: string; evaluated_at?: string } | null>(null)
  const [drillDetailsById, setDrillDetailsById] = useState<Record<number, {
    open: boolean
    loading: boolean
    error?: string | null
    original_action?: any
    original_player_id?: string | null
    expected_action?: any
    actual_action?: any
    match?: boolean | null
    agent_type?: string
    evaluated_at?: string
    drill?: any
    draft_guideline_text?: string
    saving_guideline?: boolean
  }>>({})
  
  // For agent-watching mode
  const [agentWatchGameId, setAgentWatchGameId] = useState('')
  const [isWatchingAgents, setIsWatchingAgents] = useState(false)
  const [watchInterval, setWatchInterval] = useState<number | null>(null)
  const [stepByStepMode, setStepByStepMode] = useState(false)  // Step-by-step mode (manual advance)
  const [lastReasoning, setLastReasoning] = useState<string | null>(null)  // Last agent reasoning
  const [agentWatchExcludeStrategicAdvice, setAgentWatchExcludeStrategicAdvice] = useState<boolean>(false)
  const [agentWatchExcludeHigherLevelFeatures, setAgentWatchExcludeHigherLevelFeatures] = useState<boolean>(false)
  
  // For trading UI
  const [showTradingPanel, setShowTradingPanel] = useState(false)
  const [giveResources, setGiveResources] = useState<Record<string, number>>({})
  const [receiveResources, setReceiveResources] = useState<Record<string, number>>({})
  const [selectedTradePlayers, setSelectedTradePlayers] = useState<Set<string>>(new Set())
  
  // For discard resources UI (when 7 is rolled)
  const [showDiscardPanel, setShowDiscardPanel] = useState(false)
  const [discardResources, setDiscardResources] = useState<Record<string, number>>({})
  const [playerWhoRolled7, setPlayerWhoRolled7] = useState<string | null>(null)
  const [isAutoDiscarding, setIsAutoDiscarding] = useState(false)

  // For event query view
  const [queryResults, setQueryResults] = useState<QueryEventsResponse | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryParams, setQueryParams] = useState({
    num_games: 100,
    action_type: '',
    card_type: '',
    dice_roll: '',
    player_id: '',
    min_turn: '',
    max_turn: '',
    analyze: '',
    limit: ''
  })

  // For feedback UI (LLM agent teaching)
  const [showFeedbackPanel, setShowFeedbackPanel] = useState(false)
  const [feedbackText, setFeedbackText] = useState('')
  const [feedbackType, setFeedbackType] = useState('general')
  const [lastAction, setLastAction] = useState<LegalAction | null>(null)
  const [lastStepIdx] = useState<number | null>(null)  // Used in feedback submission

  // Fetch legal actions when game state or player ID changes
  useEffect(() => {
    if (gameState && playerId && view === 'game') {
      fetchLegalActions()
    }
  }, [gameState, playerId, view])
  
  // Track when a 7 is rolled
  useEffect(() => {
    if (gameState && gameState.dice_roll === 7 && gameState.phase === 'playing') {
      const currentPlayer = gameState.players[gameState.current_player_index]
      if (currentPlayer && !playerWhoRolled7) {
        setPlayerWhoRolled7(currentPlayer.id)
      }
    } else if (gameState && gameState.dice_roll !== 7) {
      // Reset when 7 is no longer the roll
      setPlayerWhoRolled7(null)
      setIsAutoDiscarding(false)
    }
  }, [gameState?.dice_roll, gameState?.current_player_index, playerWhoRolled7])

  // Auto-open discard panel when 7 is rolled and player needs to discard
  // But only if we're still in the discard phase (not in robber phase)
  // And only if the player hasn't already discarded
  useEffect(() => {
    if (gameState && playerId && gameState.dice_roll === 7 && 
        !gameState.waiting_for_robber_move && !gameState.waiting_for_robber_steal) {
      const player = gameState.players.find(p => p.id === playerId)
      if (player) {
        const totalResources = Object.values(player.resources).reduce((a, b) => a + b, 0)
        const hasAlreadyDiscarded = gameState.players_discarded?.includes(playerId) || false
        
        // Reset ref if this is a new turn (different turn number)
        if (lastDiscardTurnRef.current !== gameState.turn_number) {
          discardPanelInitializedRef.current = false
          lastDiscardTurnRef.current = gameState.turn_number
        }
        
        if (totalResources >= 8 && !hasAlreadyDiscarded) {
          // Only open panel and clear selections if panel wasn't already initialized for this discard phase
          // This prevents clearing user input during auto-refresh
          if (!discardPanelInitializedRef.current) {
            setShowDiscardPanel(true)
            setDiscardResources({})
            discardPanelInitializedRef.current = true
          }
          // Panel stays open, preserve discardResources during auto-refresh
        } else {
          setShowDiscardPanel(false)
          discardPanelInitializedRef.current = false
          // Only clear discard resources if player has already discarded or no longer needs to
          if (hasAlreadyDiscarded || totalResources < 8) {
            setDiscardResources({})
          }
        }
      }
    } else {
      // Clear discard panel when not needed or when robber phase has started
      setShowDiscardPanel(false)
      discardPanelInitializedRef.current = false
      lastDiscardTurnRef.current = null
      // Only clear resources if we're closing the panel due to phase change
      if (gameState && (!gameState.dice_roll || gameState.dice_roll !== 7 || 
          gameState.waiting_for_robber_move || gameState.waiting_for_robber_steal)) {
        setDiscardResources({})
      }
    }
  }, [gameState, playerId])

  // Dev mode: Auto-switch to players who need to discard when 7 is rolled
  // But only if we're still in the discard phase (not in robber phase)
  useEffect(() => {
    if (drillRecording) return
    if (!devMode || !gameState || gameState.dice_roll !== 7) return
    
    // Don't auto-switch if we're already in the robber phase
    if (gameState.waiting_for_robber_move || gameState.waiting_for_robber_steal) {
      // Robber phase has started, switch back to player who rolled 7 if needed
      if (playerWhoRolled7 && playerId !== playerWhoRolled7 && isAutoDiscarding) {
        setPlayerId(playerWhoRolled7)
        setIsAutoDiscarding(false)
      }
      return
    }
    
    // Find all players who need to discard (and haven't discarded yet)
    const playersNeedingDiscard = gameState.players.filter(p => {
      const totalResources = Object.values(p.resources).reduce((a, b) => a + b, 0)
      const hasDiscarded = gameState.players_discarded?.includes(p.id) || false
      return totalResources >= 8 && !hasDiscarded
    })
    
    if (playersNeedingDiscard.length === 0) {
      // No one needs to discard, switch back to player who rolled 7
      if (playerWhoRolled7 && playerId !== playerWhoRolled7 && isAutoDiscarding) {
        setPlayerId(playerWhoRolled7)
        setIsAutoDiscarding(false)
      }
      return
    }
    
    // Check if current player needs to discard
    const currentPlayerNeedsDiscard = playersNeedingDiscard.some(p => p.id === playerId)
    
    if (!currentPlayerNeedsDiscard && !isAutoDiscarding) {
      // Switch to first player who needs to discard
      setIsAutoDiscarding(true)
      setPlayerId(playersNeedingDiscard[0].id)
    }
  }, [devMode, gameState?.dice_roll, gameState?.waiting_for_robber_move, gameState?.waiting_for_robber_steal, gameState?.players, playerId, playerWhoRolled7, isAutoDiscarding])

  // Auto-switch to current player in dev mode when turn changes
  // But don't interfere if we're auto-discarding
  useEffect(() => {
    if (drillRecording) return
    if (devMode && gameState && view === 'game' && !isAutoDiscarding) {
      // Don't auto-switch if a 7 was rolled and players need to discard
      if (gameState.dice_roll === 7) {
        const playersNeedingDiscard = gameState.players.filter(p => {
          const totalResources = Object.values(p.resources).reduce((a, b) => a + b, 0)
          return totalResources >= 8
        })
        if (playersNeedingDiscard.length > 0) {
          // Let the discard auto-switching handle it
          return
        }
      }
      
      let currentPlayer: Player | null = null
      if (gameState.phase === 'playing') {
        currentPlayer = gameState.players[gameState.current_player_index] || null
      } else {
        currentPlayer = gameState.players[gameState.setup_phase_player_index] || null
      }
      if (currentPlayer && currentPlayer.id !== playerId) {
        setPlayerId(currentPlayer.id)
      }
    }
  }, [gameState, devMode, view, playerId, isAutoDiscarding])

  // Auto-refresh polling
  useEffect(() => {
    if (autoRefresh && gameState && view === 'game') {
      refreshIntervalRef.current = setInterval(() => {
        refreshGameState()
      }, 3000) // Poll every 3 seconds
    } else {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
      }
    }
  }, [autoRefresh, gameState, view])

  // Cleanup watch interval on unmount or view change
  useEffect(() => {
    return () => {
      if (watchInterval) {
        clearInterval(watchInterval)
      }
    }
  }, [watchInterval])

  // Auto-advance agent turns (must be before early returns)
  useEffect(() => {
    // In drill recording mode, we freeze other players (including agents). The drill
    // is meant to capture only the selected player's sub-turn decisions.
    if (drillRecording) return
    if (!gameState || loading || view !== 'game') return
    
    const activePlayer = getCurrentPlayer()
    if (!activePlayer) return
    
    // Check if current player is an agent
    const currentPlayerIsAgent = activePlayer.id in agentMapping

    // Special case: after rolling a 7, *non-current* players may have a mandatory discard.
    // If those players are agent-controlled, advance the backend one step to let the agent discard,
    // even if it's currently the human player's turn.
    if (
      gameState.phase === 'playing' &&
      gameState.dice_roll === 7 &&
      !gameState.waiting_for_robber_move &&
      !gameState.waiting_for_robber_steal
    ) {
      const discarded = new Set<string>(gameState.players_discarded || [])
      const agentNeedsDiscard = gameState.players.some(p => {
        const total = Object.values(p.resources || {}).reduce((a, b) => a + b, 0)
        return total >= 8 && !discarded.has(p.id) && (p.id in agentMapping)
      })
      if (agentNeedsDiscard) {
        const advanceDiscard = async () => {
          try {
            setLoading(true)
            const result = await watchAgentsStep(
              gameState.game_id, 
              agentMapping,
              agentWatchExcludeStrategicAdvice,
              agentWatchExcludeHigherLevelFeatures
            )
            setGameState(result.new_state)
            if (result.reasoning) setLastReasoning(result.reasoning)
            if (result.error) setError(`Agent error: ${result.error}`)
          } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to advance agent discard')
          } finally {
            setLoading(false)
          }
        }
        const timeout = setTimeout(advanceDiscard, 250)
        return () => clearTimeout(timeout)
      }
    }
    
    // If there's a pending trade, only block auto-advancement if:
    // 1. The current player is the human player (they need to manually respond)
    // 2. The current player is the proposer waiting for responses (not their turn yet)
    // Otherwise, if it's an agent's turn to respond, allow auto-advancement
    if (gameState.pending_trade_offer) {
      const offer = gameState.pending_trade_offer
      const isProposer = offer.proposer_id === activePlayer.id
      const isTarget = offer.target_player_ids?.includes(activePlayer.id)
      
      // If current player is human and needs to respond, don't auto-advance
      if (activePlayer.id === playerId && isTarget) {
        return
      }
      
      // If current player is proposer waiting for responses, don't auto-advance yet
      // (unless multiple players accepted and they need to select)
      if (isProposer && !isTarget) {
        // Check if proposer needs to select a partner (multiple acceptances)
        const responses = gameState.pending_trade_responses || {}
        const acceptingPlayers = Object.keys(responses).filter(pid => responses[pid] === true)
        if (acceptingPlayers.length <= 1) {
          // Still waiting for responses, don't auto-advance
          return
        }
        // Multiple accepted - proposer needs to select, so allow auto-advance if agent
      }
      
      // If it's an agent's turn to respond to trade, allow auto-advancement
      if (!currentPlayerIsAgent && isTarget && activePlayer.id !== playerId) {
        // Not an agent but needs to respond - this shouldn't happen, but don't auto-advance
        return
      }
    }
    
    if (currentPlayerIsAgent && activePlayer.id !== playerId) {
      // It's an agent's turn (and not the human player)
      // Auto-advance by calling watch_agents_step (unless in step-by-step mode)
      if (stepByStepMode) {
        // In step-by-step mode, don't auto-advance
        return
      }
      
      const advanceAgentTurn = async () => {
        try {
          setLoading(true)
          const result = await watchAgentsStep(gameState.game_id, agentMapping)
          setGameState(result.new_state)
          
          // Store reasoning if available
          if (result.reasoning) {
            setLastReasoning(result.reasoning)
          }
          
          if (result.error) {
            setError(`Agent error: ${result.error}`)
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to advance agent turn')
        } finally {
          setLoading(false)
        }
      }
      
      // Small delay to avoid race conditions
      const timeout = setTimeout(advanceAgentTurn, 500)
      return () => clearTimeout(timeout)
    }
  }, [gameState?.current_player_index, gameState?.setup_phase_player_index, gameState?.phase, gameState?.game_id, gameState?.pending_trade_offer, gameState?.pending_trade_responses, gameState, agentMapping, playerId, loading, view, stepByStepMode])

  const refreshGameState = async () => {
    if (!gameState) return
    
    try {
      const newState = await getGameState(gameState.game_id)
      setGameState(newState)
      
      // Restore agent_mapping from metadata if available
      if ((newState as any)._metadata && (newState as any)._metadata.agent_mapping) {
        setAgentMapping((newState as any)._metadata.agent_mapping)
      }
    } catch (err) {
      // Silently fail for auto-refresh to avoid spam
      console.error('Failed to refresh game state:', err)
    }
  }

  const fetchLegalActions = async () => {
    if (!gameState || !playerId) return
    
    try {
      const actions = await getLegalActions(gameState.game_id, playerId)
      setLegalActions(actions)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setLegalActions([])
    }
  }

  const handleCreateGame = async () => {
    setLoading(true)
    setError(null)
    try {
      // Generate random names on backend, so just pass empty or same name
      const playerNames = Array.from({ length: numPlayers }, () => playerNameInput || '')
      
      const response = await apiCreateGame({ 
        player_names: playerNames,
        exclude_strategic_advice: createGameExcludeStrategicAdvice,
        exclude_higher_level_features: createGameExcludeHigherLevelFeatures
      })
      setGameState(response.initial_state)
      
      // Set player ID to first non-agent player, or first player if all are agents
      const firstNonAgentPlayer = response.initial_state.players.find(
        p => !(p.id in agentMapping)
      )
      setPlayerId(firstNonAgentPlayer?.id || response.initial_state.players[0]?.id || '')
      setView('game')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const handleLoadGame = async () => {
    if (!gameIdInput.trim()) {
      setError('Please enter a game ID')
      return
    }
    
    setLoading(true)
    setError(null)
    try {
      const data = await getGameState(gameIdInput)
      setGameState(data)
      if (data.players && data.players.length > 0) {
        const player = data.players.find((p: Player) => p.name === playerNameInput) || data.players[0]
        setPlayerId(player.id)
      }
      
      // Restore agent_mapping from metadata if available
      if ((data as any)._metadata && (data as any)._metadata.agent_mapping) {
        setAgentMapping((data as any)._metadata.agent_mapping)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const handleJoinGame = () => {
    if (!gameState || !playerId) {
      setError('Please load the game and select a player first')
      return
    }
    setView('game')
  }

  const handleExecuteAction = async (action: LegalAction): Promise<GameState | null> => {
    if (!gameState || !playerId) return
    
    setLoading(true)
    setError(null)
    try {
      // Ensure action has proper structure
      // The payload from backend includes a "type" field, but we need to pass it as-is
      // For PLAY_DEV_CARD actions, ensure the payload has the type field
      let payload = action.payload
      if (action.type === 'play_dev_card' && payload && !payload.type) {
        payload = {
          ...payload,
          type: 'PlayDevCardPayload'
        }
      }
      
      const actionToSend: LegalAction = {
        type: action.type,
        payload: payload || undefined
      }

      // If we are recording a drill, capture the decision point (state-before + chosen action)
      if (drillRecording && playerId === drillRecording.drill_player_id) {
        const stateSnapshot = JSON.parse(JSON.stringify(gameState)) as GameState
        setDrillRecording(prev => {
          if (!prev) return prev
          if (playerId !== prev.drill_player_id) return prev
          return {
            ...prev,
            steps: [
              ...prev.steps,
              { player_id: playerId, state: stateSnapshot, expected_action: actionToSend }
            ]
          }
        })
      }
      
      // Debug: log the action being sent
      console.log('Executing action:', actionToSend)
      
      const newState = await postAction(gameState.game_id, playerId, actionToSend)
      setGameState(newState)
      // Legal actions will be refetched via useEffect
      return newState
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('Action execution error:', err)
      return null
    } finally {
      setLoading(false)
    }
  }

  const handleIntersectionClick = (intersectionId: number) => {
    if (!gameState || !playerId || loading) return
    
    // Find legal action for this intersection
    const action = legalActions.find(a => {
      if (a.type === 'setup_place_settlement' || a.type === 'build_settlement') {
        return a.payload?.intersection_id === intersectionId
      }
      if (a.type === 'build_city') {
        return a.payload?.intersection_id === intersectionId
      }
      return false
    })
    
    if (action) {
      handleExecuteAction(action)
    }
  }

  const handleRoadClick = (roadId: number) => {
    if (!gameState || !playerId || loading) return
    
    // Find legal action for this road
    const action = legalActions.find(a => {
      if (a.type === 'setup_place_road' || a.type === 'build_road') {
        return a.payload?.road_edge_id === roadId
      }
      return false
    })
    
    if (action) {
      handleExecuteAction(action)
    }
  }

  const handleTileClick = (tileId: number) => {
    if (!gameState || !playerId || loading) return
    
    // Find legal action for moving robber to this tile
    const action = legalActions.find(a => {
      if (a.type === 'move_robber') {
        return a.payload?.tile_id === tileId
      }
      return false
    })
    
    if (action) {
      handleExecuteAction(action)
    }
  }

  const handleLoadReplay = async () => {
    if (!replayGameId.trim()) {
      setError('Please enter a game ID')
      return
    }
    
    setLoading(true)
    setError(null)
    try {
      const data = await getReplay(replayGameId)
      setReplayData(data)
      setReplayStepIndex(0)
      setView('replay')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const refreshDrills = async () => {
    setDrillsLoading(true)
    setError(null)
    try {
      console.log('[DEBUG] refreshDrills: Starting...')
      const data = await listDrills(200)
      console.log('[DEBUG] refreshDrills: Got data:', data)
      setDrillsList(data.drills || [])
      // Keep only selections that still exist
      setSelectedDrillIds(prev => {
        const allowed = new Set<number>((data.drills || []).map(d => d.id))
        const next = new Set<number>()
        for (const id of prev) if (allowed.has(id)) next.add(id)
        return next
      })
    } catch (err) {
      console.error('[ERROR] refreshDrills failed:', err)
      setError(err instanceof Error ? err.message : 'Failed to load drills')
    } finally {
      setDrillsLoading(false)
    }
  }

  // Auto-load drills when drills view is opened (only once)
  useEffect(() => {
    if (view === 'drills' && drillsList.length === 0 && !drillsLoading) {
      console.log('[DEBUG] Auto-loading drills on view change')
      refreshDrills()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]) // Only depend on view to avoid infinite loops

  const runAllDrillsEval = async () => {
    setDrillsLoading(true)
    setError(null)
    try {
      const result = await evaluateAllDrills({
        agent_type: drillsAgentType,
        limit: 200,
        include_step_results: true,
        include_guidelines: drillsUseGuidelines,
        exclude_strategic_advice: drillsExcludeStrategicAdvice,
        exclude_higher_level_features: drillsExcludeHigherLevelFeatures,
        drill_ids: selectedDrillIds.size > 0 ? Array.from(selectedDrillIds) : undefined
      })
      setDrillsEval(result)
      setDrillsEvalRunMeta({ run_id: result.run_id, evaluated_at: result.evaluated_at })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to evaluate drills')
    } finally {
      setDrillsLoading(false)
    }
  }

  const formatActionName = (action: LegalAction): string => {
    const type = action.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    
    if (action.payload) {
      const payload = action.payload
      if (payload.intersection_id !== undefined) {
        return `${type} at intersection ${payload.intersection_id}`
      }
      if (payload.road_edge_id !== undefined) {
        return `${type} on road edge ${payload.road_edge_id}`
      }
      if (payload.tile_id !== undefined) {
        return `${type} to tile ${payload.tile_id}`
      }
      if (payload.card_type) {
        let cardInfo = payload.card_type
        // Add details for year_of_plenty and monopoly
        if (payload.card_type === 'year_of_plenty' && 'year_of_plenty_resources' in payload && payload.year_of_plenty_resources) {
          const resourceList = Object.entries(payload.year_of_plenty_resources)
            .filter(([_, count]) => (count as number) > 0)
            .map(([resource, count]) => `${count} ${resource}`)
            .join(', ')
          cardInfo = `${payload.card_type}: ${resourceList}`
        } else if (payload.card_type === 'monopoly' && 'monopoly_resource_type' in payload && payload.monopoly_resource_type) {
          cardInfo = `${payload.card_type}: ${payload.monopoly_resource_type}`
        }
        return `${type} (${cardInfo})`
      }
      // New multi-resource trade format
      if (payload.give_resources && payload.receive_resources) {
        const giveStr = Object.entries(payload.give_resources)
          .filter(([_, count]) => count > 0)
          .map(([resource, count]) => `${count} ${resource}`)
          .join(', ')
        const receiveStr = Object.entries(payload.receive_resources)
          .filter(([_, count]) => count > 0)
          .map(([resource, count]) => `${count} ${resource}`)
          .join(', ')
        const portInfo = payload.port_intersection_id ? ` (port)` : ''
        return `${type}: Give ${giveStr}, receive ${receiveStr}${portInfo}`
      }
      // Legacy single-resource format (for backward compatibility)
      if (payload.give_resource && payload.receive_resource) {
        return `${type}: Give ${payload.give_amount} ${payload.give_resource}, receive ${payload.receive_amount} ${payload.receive_resource}`
      }
      if (payload.other_player_id && !payload.give_resources) {
        return `${type} from ${payload.other_player_id}`
      }
      if (payload.selected_player_id) {
        // For SELECT_TRADE_PARTNER, show player name if available
        const selectedPlayer = gameState?.players.find(p => p.id === payload.selected_player_id)
        const playerName = selectedPlayer?.name || payload.selected_player_id
        return `${type}: ${playerName}`
      }
      if (payload.resources) {
        const resourceList = Object.entries(payload.resources)
          .filter(([_, count]) => count > 0)
          .map(([resource, count]) => `${count} ${resource}`)
          .join(', ')
        return `${type}: ${resourceList}`
      }
    }
    
    return type
  }

  const getCurrentPlayer = (): Player | null => {
    if (!gameState) return null
    if (gameState.phase === 'playing') {
      return gameState.players[gameState.current_player_index] || null
    } else {
      return gameState.players[gameState.setup_phase_player_index] || null
    }
  }

  const getPlayerById = (id: string): Player | null => {
    if (!gameState) return null
    return gameState.players.find(p => p.id === id) || null
  }

  const renderBoard = (state: GameState | null, highlightPlayerId?: string, legalActionsList?: LegalAction[], isGameView: boolean = true) => {
    if (!state || state.tiles.length === 0) {
      return <div className="board-placeholder">No board data available</div>
    }

    // Use provided legal actions or empty array
    const legalActions = legalActionsList || []

    // Hex coordinate to pixel conversion (pointy-top hexagons)
    // Using axial coordinates (q, r) to pixel conversion
    const hexToPixel = (q: number, r: number) => {
      const hexSize = 65  // Size of hex (radius) - slightly larger for better visibility
      const sqrt3 = Math.sqrt(3)
      // Pointy-top hex conversion
      const x = hexSize * (sqrt3 * q + sqrt3 / 2 * r)
      const y = hexSize * (3.0 / 2.0 * r)
      return { x: x + 450, y: y + 390 }  // Center offset for 19-tile board
    }

    return (
      <div className="hex-board">
        {state.tiles.map(tile => {
          const { x, y } = hexToPixel(tile.position[0], tile.position[1])
          
          // Check if this tile has a legal action for moving robber (only in game view)
          const hasRobberMoveAction = isGameView && legalActions.some(a => {
            if (a.type === 'move_robber') {
              return a.payload?.tile_id === tile.id
            }
            return false
          })
          
          // Check if robber is on this tile
          const hasRobber = state.robber_tile_id === tile.id
          
          return (
          <div
            key={tile.id}
            className={`hex-tile ${hasRobberMoveAction ? 'clickable' : ''}`}
            style={{
              left: `${x}px`,
              top: `${y}px`
            }}
            onClick={() => hasRobberMoveAction && handleTileClick(tile.id)}
            title={`Tile ${tile.id}${tile.resource_type ? ` - ${tile.resource_type}` : ' - Desert'}${hasRobber ? ' (Robber here)' : ''}${hasRobberMoveAction ? ' - Click to move robber here' : ''}`}
          >
            <div className={`hex-resource ${tile.resource_type || 'desert'}`}>
              {tile.resource_type ? (
                <>
                  <div className="resource-name">{RESOURCE_ICONS[tile.resource_type] || ''} {tile.resource_type}</div>
                  {tile.number_token && (
                    <div className={`number-token ${tile.number_token === 6 || tile.number_token === 8 ? 'red-number' : ''}`} style={{ position: 'relative' }}>
                      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', marginTop: '-2px' }}>
                        {tile.number_token}
                      </div>
                      <div className="number-pips" style={{ 
                        position: 'absolute', 
                        bottom: '5px', 
                        left: '50%', 
                        transform: 'translateX(-50%)',
                        display: 'flex', 
                        gap: '1.5px', 
                        justifyContent: 'center',
                        alignItems: 'center'
                      }}>
                        {Array.from({ length: getPipCount(tile.number_token) }, (_, i) => (
                          <span key={i} style={{ 
                            width: '2.5px', 
                            height: '2.5px', 
                            borderRadius: '50%', 
                            backgroundColor: '#333',
                            display: 'inline-block'
                          }} />
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="resource-name">🏜️ Desert</div>
              )}
              {hasRobber && (
                <div className="robber-icon" title="Robber">👹</div>
              )}
              {hasRobberMoveAction && (
                <div className="tile-clickable-overlay" />
              )}
            </div>
          </div>
          )
        })}
        
        {/* Render intersections */}
        {state.intersections.map(intersection => {
          // Intersection positions are in hex coordinates (q, r)
          const { x, y } = hexToPixel(intersection.position[0], intersection.position[1])
          
          // Check if this intersection has a legal action (only in game view, not replay)
          const hasLegalAction = isGameView && legalActions.some(a => {
            if (a.type === 'setup_place_settlement' || a.type === 'build_settlement') {
              return a.payload?.intersection_id === intersection.id
            }
            if (a.type === 'build_city') {
              return a.payload?.intersection_id === intersection.id
            }
            return false
          })
          
          // Get player color if owned
          const ownerPlayer = intersection.owner ? state.players.find(p => p.id === intersection.owner) : null
          const ownerColor = ownerPlayer?.color || null
          
          return (
          <React.Fragment key={intersection.id}>
            <div
              className={`intersection ${intersection.owner === highlightPlayerId ? 'owned-by-me' : intersection.owner ? 'owned' : ''} ${hasLegalAction ? 'clickable' : ''}`}
              style={{
                left: `${x}px`,
                top: `${y}px`,
                ...(ownerColor && {
                  backgroundColor: ownerColor,
                  borderColor: ownerColor
                })
              }}
              onClick={() => handleIntersectionClick(intersection.id)}
              title={`Intersection ${intersection.id}${intersection.owner ? ` (${ownerPlayer?.name || intersection.owner})` : ''}${intersection.building_type ? ` - ${intersection.building_type}` : ''}${intersection.port_type ? ` - Port: ${intersection.port_type}` : ''}${hasLegalAction ? ' - Click to build' : ''}`}
            >
              {!intersection.building_type && '○'}
            </div>
            {intersection.building_type && (
              <div
                onClick={() => handleIntersectionClick(intersection.id)}
                style={{
                  position: 'absolute',
                  left: `${x}px`,
                  top: `${y}px`,
                  transform: 'translate(-50%, -50%)',
                  fontSize: intersection.building_type === 'city' ? '36px' : '30px',
                  lineHeight: '1',
                  zIndex: 21,
                  cursor: hasLegalAction ? 'pointer' : 'default',
                  filter: 'drop-shadow(2px 2px 4px rgba(0, 0, 0, 0.5))',
                  padding: intersection.building_type === 'city' ? '4px' : '3px',
                  borderRadius: '50%',
                  backgroundColor: ownerColor ? `${ownerColor}40` : 'rgba(255, 255, 255, 0.3)',
                  border: ownerColor ? `3px solid ${ownerColor}` : '3px solid #666',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: intersection.building_type === 'city' ? '44px' : '36px',
                  height: intersection.building_type === 'city' ? '44px' : '36px'
                }}
                title={`${intersection.building_type === 'city' ? 'City' : 'Settlement'} - ${ownerPlayer?.name || intersection.owner || 'Unknown'}${hasLegalAction ? ' - Click to build' : ''}`}
              >
                {intersection.building_type === 'city' ? '🏰' : '🏠'}
              </div>
            )}
          </React.Fragment>
          )
        })}
        
        {/* Render ports on coastal edges (between port pairs) */}
        {(() => {
          // First, identify all port pairs
          const portPairs: Array<{ inter1: typeof state.intersections[0], inter2: typeof state.intersections[0], portType: string }> = []
          const seenPairs = new Set<string>()
          
          for (const inter of state.intersections) {
            if (inter.port_type === null) continue
            
            const adjacentSameType = state.intersections.filter(
              adj => adj.id !== inter.id && 
                     inter.adjacent_intersections.includes(adj.id) && 
                     adj.port_type === inter.port_type
            )
            
            if (adjacentSameType.length === 0) continue
            
            const portPair = adjacentSameType.reduce((min, curr) => 
              curr.id < min.id ? curr : min, adjacentSameType[0])
            
            const pairKey = `${Math.min(inter.id, portPair.id)}-${Math.max(inter.id, portPair.id)}`
            if (seenPairs.has(pairKey)) continue
            if (inter.id > portPair.id) continue
            
            seenPairs.add(pairKey)
            portPairs.push({ inter1: inter, inter2: portPair, portType: inter.port_type || '' })
          }
          
          // Order ports by their position around the perimeter
          // Build perimeter order by traversing the coastline
          const perimeterIntersections = state.intersections.filter(i => i.adjacent_tiles.length < 3)
          const perimeterIds = new Set(perimeterIntersections.map(i => i.id))
          
          // Build coastline graph
          const coastlineGraph: Record<number, number[]> = {}
          for (const inter of perimeterIntersections) {
            coastlineGraph[inter.id] = []
            for (const adjId of inter.adjacent_intersections) {
              const adjInter = state.intersections.find(i => i.id === adjId)
              if (adjInter && adjInter.adjacent_tiles.length < 3) {
                coastlineGraph[inter.id].push(adjId)
              }
            }
          }
          
          // Remove unused variable warning
          void perimeterIds
          
          // Traverse perimeter to get ordered list
          const perimeterOrdered: number[] = []
          if (Object.keys(coastlineGraph).length > 0) {
            const start = Number(Object.keys(coastlineGraph)[0])
            const visited = new Set<number>()
            let current = start
            let prev: number | null = null
            
            while (visited.size < Object.keys(coastlineGraph).length) {
              visited.add(current)
              perimeterOrdered.push(current)
              
              let nextNode: number | null = null
              for (const neighbor of coastlineGraph[current] || []) {
                if (neighbor !== prev && !visited.has(neighbor)) {
                  nextNode = neighbor
                  break
                }
              }
              
              if (nextNode === null) {
                if (coastlineGraph[current]?.includes(start) && visited.size === Object.keys(coastlineGraph).length) {
                  break
                } else {
                  break
                }
              }
              
              prev = current
              current = nextNode
            }
          }
          
          // Build ordered list of perimeter edges
          const perimeterEdges: Array<[number, number]> = []
          for (let i = 0; i < perimeterOrdered.length; i++) {
            const node1 = perimeterOrdered[i]
            const node2 = perimeterOrdered[(i + 1) % perimeterOrdered.length]
            perimeterEdges.push([node1, node2])
          }
          
          // Find which edge each port pair is on, and order by perimeter position
          const portPairsWithOrder = portPairs.map(pair => {
            // Find which perimeter edge this pair is on
            for (let idx = 0; idx < perimeterEdges.length; idx++) {
              const [n1, n2] = perimeterEdges[idx]
              if ((n1 === pair.inter1.id && n2 === pair.inter2.id) ||
                  (n1 === pair.inter2.id && n2 === pair.inter1.id)) {
                return { ...pair, edgeIndex: idx }
              }
            }
            return { ...pair, edgeIndex: 999 } // Fallback for ports not on perimeter (shouldn't happen)
          })
          
          // Sort by perimeter position
          portPairsWithOrder.sort((a, b) => a.edgeIndex - b.edgeIndex)
          
          // Render ports in perimeter order, but still sort 3:1 ports first for z-index
          const sortedForRender = portPairsWithOrder.sort((a, b) => {
            // First by edge index (perimeter order)
            if (a.edgeIndex !== b.edgeIndex) return a.edgeIndex - b.edgeIndex
            // Then by port type (3:1 first for z-index)
            if (a.portType === '3:1' && b.portType !== '3:1') return -1
            if (a.portType !== '3:1' && b.portType === '3:1') return 1
            return 0
          })
          
          // Collect port data (lines and icons)
          const portLines: Array<{
            x1: number, y1: number, x2: number, y2: number,
            portX: number, portY: number, color: string
          }> = []
          
          const portIcons = sortedForRender.map(pair => {
            const { inter1: inter, inter2: portPair } = pair
            
            // Calculate midpoint between the two intersections
            const { x: x1, y: y1 } = hexToPixel(inter.position[0], inter.position[1])
            const { x: x2, y: y2 } = hexToPixel(portPair.position[0], portPair.position[1])
            const midX = (x1 + x2) / 2
            const midY = (y1 + y2) / 2
            
            // Calculate direction from board center to midpoint (outward direction)
            const boardCenterX = 600  // Center offset from hexToPixel (updated for larger map)
            const boardCenterY = 450
            const dx = midX - boardCenterX
            const dy = midY - boardCenterY
            const distance = Math.sqrt(dx * dx + dy * dy)
            
            // Offset port icon outward from the edge (away from board center)
            const offsetDistance = 55  // Pixels to offset from edge (increased for larger hexes)
            const offsetX = (dx / distance) * offsetDistance
            const offsetY = (dy / distance) * offsetDistance
            const portX = midX + offsetX
            const portY = midY + offsetY
            
            // Get port icon based on type
            let portIcon = '⚓' // Default 3:1 port
            let portColor = '#2196F3' // Blue for 3:1
            let portTitle = '3:1 Port (any resource)'
            
            if (inter.port_type !== '3:1') {
              // 2:1 specific resource port
              const resourceIcons: Record<string, { icon: string; color: string }> = {
                'sheep': { icon: '🐑', color: '#90EE90' },
                'ore': { icon: '⛏️', color: '#708090' },
                'wood': { icon: '🌲', color: '#8B4513' },
                'brick': { icon: '🧱', color: '#CD5C5C' },
                'wheat': { icon: '🌾', color: '#FFD700' }
              }
              const resourceInfo = resourceIcons[inter.port_type || ''] || { icon: '🏭', color: '#FF9800' }
              portIcon = resourceInfo.icon
              portColor = resourceInfo.color
              portTitle = `2:1 ${inter.port_type} Port`
            }
            
            // Store line data for SVG rendering
            portLines.push({
              x1, y1, x2, y2, portX, portY, color: portColor
            })
            
            return {
              portX, portY, portIcon, portColor, portTitle,
              portType: inter.port_type,
              key: `port-${inter.id}-${portPair.id}`
            }
          })
          
          return (
            <React.Fragment key="port-container">
              {/* Single SVG for all port connection lines */}
              <svg
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                  zIndex: 17
                }}
              >
                {portLines.map((line, idx) => (
                  <React.Fragment key={`port-line-${idx}`}>
                    <line
                      x1={line.x1}
                      y1={line.y1}
                      x2={line.portX}
                      y2={line.portY}
                      stroke="#888"
                      strokeWidth="2"
                      strokeDasharray="4,4"
                      opacity="0.5"
                    />
                    <line
                      x1={line.x2}
                      y1={line.y2}
                      x2={line.portX}
                      y2={line.portY}
                      stroke="#888"
                      strokeWidth="2"
                      strokeDasharray="4,4"
                      opacity="0.5"
                    />
                  </React.Fragment>
                ))}
              </svg>
              {/* Port icons */}
              {portIcons.map(icon => (
                <div
                  key={icon.key}
                  className="port-edge-indicator"
                  data-port-type={icon.portType}
                  style={{
                    left: `${icon.portX}px`,
                    top: `${icon.portY}px`,
                    color: icon.portColor
                  }}
                  title={icon.portTitle}
                >
                  {icon.portIcon}
                </div>
              ))}
            </React.Fragment>
          )
        })()}
        
        {/* Render roads */}
        {state.road_edges.map(road => {
          const inter1 = state.intersections.find(i => i.id === road.intersection1_id)
          const inter2 = state.intersections.find(i => i.id === road.intersection2_id)
          if (!inter1 || !inter2) return null
          
          const pos1 = hexToPixel(inter1.position[0], inter1.position[1])
          const pos2 = hexToPixel(inter2.position[0], inter2.position[1])
          const x1 = pos1.x
          const y1 = pos1.y
          const x2 = pos2.x
          const y2 = pos2.y
          
          // Calculate distance and angle
          const dx = x2 - x1
          const dy = y2 - y1
          const distance = Math.sqrt(dx * dx + dy * dy)
          const angle = Math.atan2(dy, dx) * 180 / Math.PI
          
          // Check if this road has a legal action (only in game view, not replay)
          const hasLegalAction = isGameView && legalActions.some(a => {
            if (a.type === 'setup_place_road' || a.type === 'build_road') {
              return a.payload?.road_edge_id === road.id
            }
            return false
          })
          
          // Get player color if owned
          const ownerPlayer = road.owner ? state.players.find(p => p.id === road.owner) : null
          const ownerColor = ownerPlayer?.color || null
          
          return (
            <div
              key={road.id}
              className={`road ${road.owner === highlightPlayerId ? 'owned-by-me' : road.owner ? 'owned' : ''} ${hasLegalAction ? 'clickable' : ''}`}
              style={{
                left: `${x1}px`,
                top: `${y1}px`,
                width: `${distance}px`,
                transform: `rotate(${angle}deg)`,
                transformOrigin: '0 50%',
                ...(ownerColor && {
                  backgroundColor: ownerColor
                })
              }}
              onClick={() => handleRoadClick(road.id)}
              title={`Road ${road.id}${road.owner ? ` (${ownerPlayer?.name || road.owner})` : ''}${hasLegalAction ? ' - Click to build' : ''}`}
            />
          )
        })}
      </div>
    )
  }

  if (view === 'main') {
    return (
      <div className="app">
        <header>
          <h1>Catan Game</h1>
        </header>
        <main>
          <div className="main-menu">
            <div className="menu-section">
              <h2>Create New Game</h2>
              <div className="form-group">
                <label>
                  Number of Players:
                  <input
                    type="number"
                    min="2"
                    max="4"
                    value={numPlayers}
                    onChange={(e) => setNumPlayers(parseInt(e.target.value) || 2)}
                  />
                </label>
              </div>
              <div className="form-group">
                <label>
                  Your Name (optional - players will get random names):
                  <input
                    type="text"
                    value={playerNameInput}
                    onChange={(e) => setPlayerNameInput(e.target.value)}
                    placeholder="Leave empty for random names"
                  />
                </label>
              </div>
              <div className="form-group">
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
                  Configure Agents (leave empty for human players):
                </label>
                {Array.from({ length: numPlayers }).map((_, idx) => {
                  const playerId = `player_${idx}`
                  const isAgent = playerId in agentMapping
                  return (
                    <div key={idx} style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <label style={{ minWidth: '80px' }}>
                        Player {idx + 1}:
                      </label>
                      <select
                        value={isAgent ? agentMapping[playerId] : ''}
                        onChange={(e) => {
                          const newMapping = { ...agentMapping }
                          if (e.target.value) {
                            newMapping[playerId] = e.target.value
                          } else {
                            delete newMapping[playerId]
                          }
                          setAgentMapping(newMapping)
                        }}
                        style={{ flex: 1, padding: '0.25rem' }}
                      >
                        <option value="">Human Player</option>
                        <option value="llm">LLM Agent (legacy: env LLM_MODEL)</option>
                        <option value="llm:gpt-4o">LLM Agent (gpt-4o)</option>
                        <option value="llm:gpt-4.1">LLM Agent (gpt-4.1)</option>
                        <option value="llm:gpt-5.2">LLM Agent (gpt-5.2)</option>
                        <option value="llm:gpt-5.2:thinking:medium">LLM Agent (gpt-5.2 thinking · medium)</option>
                        <option value="behavior_tree">Behavior Tree Agent</option>
                        <option value="balanced">Balanced Agent</option>
                        <option value="aggressive_builder">Aggressive Builder</option>
                        <option value="dev_card_focused">Dev Card Focused</option>
                        <option value="expansion">Expansion Agent</option>
                        <option value="defensive">Defensive Agent</option>
                        <option value="state_conditioned">State Conditioned</option>
                        <option value="random">Random Agent</option>
                      </select>
                    </div>
                  )
                })}
                <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.5rem' }}>
                  💡 Tip: Set 3 players as "Behavior Tree Agent" to play against 3 bots!
                </div>
              </div>
              <div className="form-group" style={{ fontSize: '0.9rem', marginTop: '1rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
                  LLM Agent Prompt Options (for LLM agents only):
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={createGameExcludeStrategicAdvice}
                    onChange={(e) => setCreateGameExcludeStrategicAdvice(e.target.checked)}
                  />
                  Exclude strategic advice from LLM prompt
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', marginTop: '0.5rem' }}>
                  <input
                    type="checkbox"
                    checked={createGameExcludeHigherLevelFeatures}
                    onChange={(e) => setCreateGameExcludeHigherLevelFeatures(e.target.checked)}
                  />
                  Exclude higher-level features (production analysis, etc.)
                </label>
                <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.5rem' }}>
                  These settings will be used when agents play in this game.
                </div>
              </div>
              <button onClick={handleCreateGame} disabled={loading}>
                {loading ? 'Creating...' : 'Create Game'}
              </button>
            </div>

            <div className="menu-divider">OR</div>

            <div className="menu-section">
              <h2>Join Existing Game</h2>
              <div className="form-group">
                <label>
                  Game ID:
                  <input
                    type="text"
                    value={gameIdInput}
                    onChange={(e) => setGameIdInput(e.target.value)}
                    placeholder="Enter game ID"
                  />
                </label>
              </div>
              <div className="form-group">
                <label>
                  Your Name:
                  <input
                    type="text"
                    value={playerNameInput}
                    onChange={(e) => setPlayerNameInput(e.target.value)}
                    placeholder="Your player name"
                  />
                </label>
              </div>
              <div className="button-group">
                <button 
                  onClick={handleLoadGame}
                  disabled={loading || !gameIdInput.trim()}
                  className="secondary"
                >
                  {loading ? 'Loading...' : 'Load Game'}
                </button>
              </div>
              {gameState && (
                <div className="form-group">
                  <label>
                    Select Your Player:
                    <select
                      value={playerId}
                      onChange={(e) => setPlayerId(e.target.value)}
                    >
                      <option value="">Select a player...</option>
                      {gameState.players.map(p => (
                        <option key={p.id} value={p.id}>{p.name} ({p.id})</option>
                      ))}
                    </select>
                  </label>
                </div>
              )}
              <button 
                onClick={handleJoinGame} 
                disabled={loading || !gameIdInput.trim() || !playerId || !gameState}
              >
                {loading ? 'Joining...' : 'Join Game'}
              </button>
            </div>

            <div className="menu-divider">OR</div>

            <div className="menu-section">
              <h2>Replay Game</h2>
              <div className="form-group">
                <label>
                  Game ID:
                  <input
                    type="text"
                    value={replayGameId}
                    onChange={(e) => setReplayGameId(e.target.value)}
                    placeholder="Enter game ID to replay"
                  />
                </label>
              </div>
              <button 
                onClick={handleLoadReplay}
                disabled={loading || !replayGameId.trim()}
                className="secondary"
              >
                {loading ? 'Loading...' : 'Load Replay'}
              </button>
            </div>

            <div className="menu-divider">OR</div>

            <div className="menu-section">
              <h2>🤖 Agent Testing</h2>
              <div className="form-group">
                <label>
                  Game ID:
                  <input
                    type="text"
                    value={agentWatchGameId}
                    onChange={(e) => setAgentWatchGameId(e.target.value)}
                    placeholder="Enter game ID for agent testing"
                  />
                </label>
              </div>
              <div className="form-group" style={{ fontSize: '0.9rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={agentWatchExcludeStrategicAdvice}
                    onChange={(e) => setAgentWatchExcludeStrategicAdvice(e.target.checked)}
                  />
                  Exclude strategic advice from LLM prompt
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', marginTop: '0.5rem' }}>
                  <input
                    type="checkbox"
                    checked={agentWatchExcludeHigherLevelFeatures}
                    onChange={(e) => setAgentWatchExcludeHigherLevelFeatures(e.target.checked)}
                  />
                  Exclude higher-level features (production analysis, etc.)
                </label>
              </div>
              <div className="button-group" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <button 
                  onClick={async () => {
                    if (!agentWatchGameId.trim()) {
                      setError('Please enter a game ID')
                      return
                    }
                    setLoading(true)
                    setError(null)
                    try {
                      const result = await runAgents(agentWatchGameId, { 
                        max_turns: 1000,
                        exclude_strategic_advice: agentWatchExcludeStrategicAdvice,
                        exclude_higher_level_features: agentWatchExcludeHigherLevelFeatures
                      })
                      if (result.error) {
                        alert(`Agents finished with error: ${result.error}\n\nGame ID: ${result.game_id}\nTurns played: ${result.turns_played}\nCompleted: ${result.completed}`)
                      } else {
                        alert(`Agents finished successfully!\n\nGame ID: ${result.game_id}\nTurns played: ${result.turns_played}\nCompleted: ${result.completed}`)
                      }
                      // Load the final state
                      const finalState = await getGameState(result.game_id)
                      setGameState(finalState)
                      setReplayGameId(result.game_id)
                      // Restore agent_mapping from metadata if available
                      if ((finalState as any)._metadata && (finalState as any)._metadata.agent_mapping) {
                        setAgentMapping((finalState as any)._metadata.agent_mapping)
                      }
                    } catch (err) {
                      setError(err instanceof Error ? err.message : 'Failed to run agents')
                    } finally {
                      setLoading(false)
                    }
                  }}
                  disabled={loading || !agentWatchGameId.trim()}
                  className="secondary"
                >
                  {loading ? 'Running...' : '▶ Run Agents Automatically'}
                </button>
                <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                  Runs game to completion (or max 1000 turns). Returns game ID for replay.
                </div>
                <button 
                  onClick={async () => {
                    if (!agentWatchGameId.trim()) {
                      setError('Please enter a game ID')
                      return
                    }
                    setLoading(true)
                    setError(null)
                    try {
                      const state = await getGameState(agentWatchGameId)
                      setGameState(state)
                      setView('agent-watch')
                      // Restore agent_mapping from metadata if available
                      if ((state as any)._metadata && (state as any)._metadata.agent_mapping) {
                        setAgentMapping((state as any)._metadata.agent_mapping)
                      }
                    } catch (err) {
                      setError(err instanceof Error ? err.message : 'Failed to load game')
                    } finally {
                      setLoading(false)
                    }
                  }}
                  disabled={loading || !agentWatchGameId.trim()}
                  className="secondary"
                >
                  {loading ? 'Loading...' : '👁️ Watch Agents Play'}
                </button>
                <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                  Watch agents play step-by-step with 1-second delay.
                </div>
              </div>
            </div>

            <div className="menu-section">
              <h2>🔍 Event Query & Analysis</h2>
              <div className="form-group">
                <button 
                  onClick={() => setView('event-query')}
                  className="action-button"
                  style={{ width: '100%' }}
                >
                  🔍 Query Game Events
                </button>
                <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                  Search and analyze specific events across multiple games (e.g., monopoly cards, 7-rolls).
                </div>
              </div>
            </div>

            <div className="menu-divider">OR</div>

            <div className="menu-section">
              <h2>🎯 Drills</h2>
              <div className="form-group">
                <button
                  onClick={async () => {
                    setView('drills')
                    await refreshDrills()
                  }}
                  className="action-button"
                  style={{ width: '100%' }}
                >
                  🎯 Open Drills
                </button>
                <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                  Run an agent against your curated “best action” drills and see pass/fail.
                </div>
              </div>
            </div>
          </div>
          {error && <div className="error">Error: {error}</div>}
        </main>
      </div>
    )
  }

  // Event Query View
  if (view === 'event-query') {
    const handleQuery = async () => {
      setQueryLoading(true)
      setError(null)
      try {
        const params: any = {
          num_games: queryParams.num_games
        }
        if (queryParams.action_type) params.action_type = queryParams.action_type
        if (queryParams.card_type) params.card_type = queryParams.card_type
        if (queryParams.dice_roll) params.dice_roll = parseInt(queryParams.dice_roll)
        if (queryParams.player_id) params.player_id = queryParams.player_id
        if (queryParams.min_turn) params.min_turn = parseInt(queryParams.min_turn)
        if (queryParams.max_turn) params.max_turn = parseInt(queryParams.max_turn)
        if (queryParams.analyze) params.analyze = queryParams.analyze
        if (queryParams.limit) params.limit = parseInt(queryParams.limit)

        const results = await queryEvents(params)
        setQueryResults(results)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to query events')
      } finally {
        setQueryLoading(false)
      }
    }

    return (
      <div className="app">
        <header>
          <h1>🔍 Event Query & Analysis</h1>
          <button onClick={() => setView('main')} className="back-button">
            Back to Menu
          </button>
        </header>
        <main>
          <div className="event-query-container" style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
            <div className="query-form" style={{ 
              backgroundColor: '#f5f5f5', 
              padding: '1.5rem', 
              borderRadius: '8px',
              marginBottom: '2rem'
            }}>
              <h2>Query Parameters</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                <div className="form-group">
                  <label>
                    Number of Games:
                    <input
                      type="number"
                      min="1"
                      value={queryParams.num_games}
                      onChange={(e) => setQueryParams({ ...queryParams, num_games: parseInt(e.target.value) || 100 })}
                    />
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    Action Type (e.g., PLAY_DEV_CARD, BUILD_CITY):
                    <input
                      type="text"
                      value={queryParams.action_type}
                      onChange={(e) => setQueryParams({ ...queryParams, action_type: e.target.value })}
                      placeholder="Optional"
                    />
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    Card Type (for PLAY_DEV_CARD, e.g., monopoly, knight):
                    <input
                      type="text"
                      value={queryParams.card_type}
                      onChange={(e) => setQueryParams({ ...queryParams, card_type: e.target.value })}
                      placeholder="Optional"
                    />
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    Dice Roll (e.g., 7):
                    <input
                      type="number"
                      min="2"
                      max="12"
                      value={queryParams.dice_roll}
                      onChange={(e) => setQueryParams({ ...queryParams, dice_roll: e.target.value })}
                      placeholder="Optional"
                    />
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    Player ID:
                    <input
                      type="text"
                      value={queryParams.player_id}
                      onChange={(e) => setQueryParams({ ...queryParams, player_id: e.target.value })}
                      placeholder="Optional"
                    />
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    Min Turn:
                    <input
                      type="number"
                      min="0"
                      value={queryParams.min_turn}
                      onChange={(e) => setQueryParams({ ...queryParams, min_turn: e.target.value })}
                      placeholder="Optional"
                    />
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    Max Turn:
                    <input
                      type="number"
                      min="0"
                      value={queryParams.max_turn}
                      onChange={(e) => setQueryParams({ ...queryParams, max_turn: e.target.value })}
                      placeholder="Optional"
                    />
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    Analysis Type:
                    <select
                      value={queryParams.analyze}
                      onChange={(e) => setQueryParams({ ...queryParams, analyze: e.target.value })}
                    >
                      <option value="">None</option>
                      <option value="monopoly">Monopoly Card</option>
                    </select>
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    Limit Results:
                    <input
                      type="number"
                      min="1"
                      value={queryParams.limit}
                      onChange={(e) => setQueryParams({ ...queryParams, limit: e.target.value })}
                      placeholder="Optional"
                    />
                  </label>
                </div>
              </div>
              <button
                onClick={handleQuery}
                disabled={queryLoading}
                className="action-button"
                style={{
                  marginTop: '1rem',
                  width: '100%',
                  padding: '0.75rem',
                  fontSize: '1rem',
                  fontWeight: 'bold'
                }}
              >
                {queryLoading ? 'Querying...' : '🔍 Run Query'}
              </button>
            </div>

            {error && <div className="error">Error: {error}</div>}

            {queryResults && (
              <div className="query-results">
                <h2>Results</h2>
                
                {/* Summary */}
                {queryResults.summary && (
                  <div style={{ 
                    backgroundColor: '#e3f2fd', 
                    padding: '1rem', 
                    borderRadius: '8px',
                    marginBottom: '1rem'
                  }}>
                    <h3>Summary</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                      <div><strong>Total Events:</strong> {queryResults.summary.total_events}</div>
                      <div><strong>Unique Games:</strong> {queryResults.summary.unique_games}</div>
                    </div>
                    {Object.keys(queryResults.summary.action_types).length > 0 && (
                      <div style={{ marginTop: '1rem' }}>
                        <strong>Action Types:</strong>
                        <ul>
                          {Object.entries(queryResults.summary.action_types).map(([action, count]) => (
                            <li key={action}>{action}: {count}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {/* Analysis */}
                {queryResults.analysis && (
                  <div style={{ 
                    backgroundColor: '#fff3cd', 
                    padding: '1rem', 
                    borderRadius: '8px',
                    marginBottom: '1rem'
                  }}>
                    <h3>Analysis</h3>
                    {queryResults.analysis.error ? (
                      <div style={{ color: '#d32f2f' }}>{queryResults.analysis.error}</div>
                    ) : (
                      <div>
                        <div><strong>Total Plays:</strong> {queryResults.analysis.total_plays || 0}</div>
                        <div><strong>Correct:</strong> {queryResults.analysis.correct || 0}</div>
                        <div><strong>Incorrect:</strong> {queryResults.analysis.incorrect || 0}</div>
                        {queryResults.analysis.issues && queryResults.analysis.issues.length > 0 && (
                          <div style={{ marginTop: '1rem' }}>
                            <strong>Issues ({queryResults.analysis.issues.length}):</strong>
                            <ul>
                              {queryResults.analysis.issues.slice(0, 10).map((issue: any, idx: number) => (
                                <li key={idx}>
                                  Game {issue.game_id?.substring(0, 8)}... Step {issue.step_idx}: {issue.issue}
                                </li>
                              ))}
                              {queryResults.analysis.issues.length > 10 && (
                                <li>... and {queryResults.analysis.issues.length - 10} more issues</li>
                              )}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Events List */}
                {queryResults.events && queryResults.events.length > 0 && (
                  <div>
                    <h3>Events ({queryResults.events.length})</h3>
                    <div style={{ maxHeight: '600px', overflowY: 'auto', border: '1px solid #ccc', borderRadius: '4px', padding: '1rem' }}>
                      {queryResults.events.map((event, idx) => (
                        <div 
                          key={idx} 
                          style={{ 
                            padding: '0.75rem', 
                            marginBottom: '0.5rem', 
                            backgroundColor: '#fff',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                            cursor: 'pointer'
                          }}
                          onClick={async () => {
                            setLoading(true)
                            setError(null)
                            try {
                              const replayData = await getReplay(event.game_id)
                              setReplayData(replayData)
                              setReplayStepIndex(event.step_idx)
                              setView('replay')
                            } catch (err) {
                              setError(err instanceof Error ? err.message : 'Failed to load replay')
                            } finally {
                              setLoading(false)
                            }
                          }}
                        >
                          <div><strong>Game:</strong> {event.game_id.substring(0, 12)}... | <strong>Step:</strong> {event.step_idx}</div>
                          <div><strong>Player:</strong> {event.player_id} | <strong>Action:</strong> {event.action_type}</div>
                          {event.action_payload && (
                            <div style={{ fontSize: '0.9rem', color: '#666' }}>
                              Payload: {JSON.stringify(event.action_payload)}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {queryResults.events && queryResults.events.length === 0 && (
                  <div style={{ padding: '2rem', textAlign: 'center', color: '#666' }}>
                    No events found matching the query criteria.
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    )
  }

  if (view === 'agent-watch') {
    const handleAdvanceStep = async () => {
      if (!gameState || loading) return
      try {
        setLoading(true)
        const result = await watchAgentsStep(
          gameState.game_id,
          undefined,
          agentWatchExcludeStrategicAdvice,
          agentWatchExcludeHigherLevelFeatures
        )
        setGameState(result.new_state)
        
        // Store reasoning if available
        if (result.reasoning) {
          setLastReasoning(result.reasoning)
        }
        
        if (!result.game_continues || result.error) {
          setIsWatchingAgents(false)
          if (watchInterval) {
            clearInterval(watchInterval)
            setWatchInterval(null)
          }
          if (result.error) {
            setError(`Agent watching stopped: ${result.error}`)
          } else {
            setError('Game finished or stopped')
          }
        }
      } catch (err) {
        setIsWatchingAgents(false)
        if (watchInterval) {
          clearInterval(watchInterval)
          setWatchInterval(null)
        }
        setError(err instanceof Error ? err.message : 'Failed to watch agents')
      } finally {
        setLoading(false)
      }
    }
    
    const handleStartWatching = () => {
      if (!gameState) return
      setIsWatchingAgents(true)
      
      if (stepByStepMode) {
        // Step-by-step mode: don't start interval, just enable manual advance
        return
      }
      
      const interval = setInterval(async () => {
        await handleAdvanceStep()
      }, 1000) // 1 second delay
      
      setWatchInterval(interval)
    }

    const handleStopWatching = () => {
      setIsWatchingAgents(false)
      if (watchInterval) {
        clearInterval(watchInterval)
        setWatchInterval(null)
      }
    }

    return (
      <div className="app">
        <header>
          <h1>🤖 Agent Watching Mode</h1>
          <div className="header-actions">
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '1rem' }}>
              <input
                type="checkbox"
                checked={stepByStepMode}
                onChange={(e) => {
                  setStepByStepMode(e.target.checked)
                  if (e.target.checked && isWatchingAgents) {
                    handleStopWatching()
                  }
                }}
              />
              <span>Step-by-Step Mode</span>
            </label>
            {stepByStepMode ? (
              <>
                {!isWatchingAgents ? (
                  <button 
                    onClick={handleStartWatching}
                    disabled={loading || !gameState}
                    className="action-button"
                    style={{ 
                      backgroundColor: '#4CAF50',
                      color: 'white'
                    }}
                  >
                    ▶ Start Watching
                  </button>
                ) : (
                  <button 
                    onClick={handleAdvanceStep}
                    disabled={loading || !gameState}
                    className="action-button"
                    style={{ 
                      backgroundColor: '#4CAF50',
                      color: 'white'
                    }}
                  >
                    {loading ? '⏳ Processing...' : '⏭️ Next Step'}
                  </button>
                )}
                <button 
                  onClick={handleStopWatching}
                  disabled={loading || !gameState || !isWatchingAgents}
                  className="action-button"
                  style={{ 
                    backgroundColor: '#d32f2f',
                    color: 'white'
                  }}
                >
                  ⏸ Stop
                </button>
              </>
            ) : (
              <button 
                onClick={isWatchingAgents ? handleStopWatching : handleStartWatching}
                disabled={loading || !gameState}
                className="action-button"
                style={{ 
                  backgroundColor: isWatchingAgents ? '#d32f2f' : '#4CAF50',
                  color: 'white'
                }}
              >
                {isWatchingAgents ? '⏸ Stop Watching' : '▶ Start Watching'}
              </button>
            )}
            <button onClick={() => {
              handleStopWatching()
              setView('main')
            }} className="back-button">
              Back to Menu
            </button>
          </div>
        </header>
        <main className="game-main">
          {error && <div className="error">Error: {error}</div>}
          
          {isWatchingAgents && (
            <div style={{ 
              padding: '1rem', 
              backgroundColor: '#e3f2fd', 
              border: '1px solid #2196F3',
              borderRadius: '4px',
              marginBottom: '1rem'
            }}>
              <strong>👁️ Watching agents play...</strong> {stepByStepMode ? 'Click "Next Step" to advance.' : 'Actions execute every 1 second.'}
            </div>
          )}
          
          {lastReasoning && (
            <div style={{ 
              padding: '1rem', 
              backgroundColor: '#fff3cd', 
              border: '1px solid #ffc107',
              borderRadius: '4px',
              marginBottom: '1rem'
            }}>
              <strong>🤔 Agent Reasoning:</strong>
              <div style={{ marginTop: '0.5rem', fontStyle: 'italic', color: '#666', whiteSpace: 'pre-wrap' }}>
                {lastReasoning}
              </div>
            </div>
          )}
          
          <div className="game-info-bar">
            <div>
              <strong>Game ID:</strong> {gameState?.game_id}
            </div>
            <div>
              <strong>Phase:</strong> {gameState?.phase}
            </div>
            <div>
              <strong>Turn:</strong> {gameState?.turn_number}
            </div>
            {gameState?.dice_roll && (
              <div>
                <strong>Last Dice Roll:</strong> {gameState.dice_roll}
              </div>
            )}
            {gameState && gameState.phase === 'playing' && (
              <div>
                <strong>Current Player:</strong> {gameState.players[gameState.current_player_index]?.name}
              </div>
            )}
          </div>

          <div className="game-layout">
            <div className="game-board-section">
              <h2>Board</h2>
              <div className="board-container">
                {renderBoard(gameState, undefined, [], false)}
              </div>
            </div>

            <div className="game-sidebar">
              {gameState && (
                <>
                  <div className="player-info">
                    <h2>Game State</h2>
                    <div className="info-section">
                      <div><strong>Phase:</strong> {gameState.phase}</div>
                      <div><strong>Turn:</strong> {gameState.turn_number}</div>
                      {gameState.dice_roll && (
                        <div><strong>Dice Roll:</strong> {gameState.dice_roll}</div>
                      )}
                    </div>
                  </div>

                  <div className="all-players">
                    <h2>Players</h2>
                    {gameState.players.map(player => (
                      <div 
                        key={player.id} 
                        className="player-card"
                        style={{
                          borderLeft: `4px solid ${player.color || '#ccc'}`
                        }}
                      >
                        <div><strong>{player.name}</strong> ({player.id})</div>
                        <div>VP: {player.victory_points}</div>
                        <div>Resources: {Object.values(player.resources).reduce((a, b) => a + b, 0)}</div>
                        <div>Settlements: {player.settlements_built}, Cities: {player.cities_built}</div>
                        <div>Roads: {player.roads_built}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </main>
      </div>
    )
  }

  if (view === 'drills') {
    return (
      <div className="app">
        <header>
          <h1>🎯 Drills</h1>
          <button onClick={() => setView('main')} className="back-button">
            Back to Menu
          </button>
        </header>
        <main className="game-main">
          {error && <div className="error">Error: {error}</div>}

          <div className="menu-section" style={{ maxWidth: '900px', margin: '0 auto' }}>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <button onClick={refreshDrills} disabled={drillsLoading} className="secondary">
                {drillsLoading ? 'Loading...' : 'Refresh Drills'}
              </button>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <strong>Agent:</strong>
                <select value={drillsAgentType} onChange={(e) => setDrillsAgentType(e.target.value)}>
                  <option value="llm">LLM Agent (legacy: env LLM_MODEL)</option>
                  <option value="llm:gpt-4o">LLM Agent (gpt-4o)</option>
                  <option value="llm:gpt-4.1">LLM Agent (gpt-4.1)</option>
                  <option value="llm:gpt-5.2">LLM Agent (gpt-5.2)</option>
                  <option value="llm:gpt-5.2:thinking:medium">LLM Agent (gpt-5.2 thinking · medium)</option>
                  <option value="behavior_tree">Behavior Tree Agent</option>
                  <option value="balanced">Balanced Agent</option>
                  <option value="aggressive_builder">Aggressive Builder</option>
                  <option value="dev_card_focused">Dev Card Focused</option>
                  <option value="expansion">Expansion Agent</option>
                  <option value="defensive">Defensive Agent</option>
                  <option value="state_conditioned">State Conditioned</option>
                  <option value="random">Random Agent</option>
                </select>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={drillsUseGuidelines}
                  onChange={(e) => setDrillsUseGuidelines(e.target.checked)}
                  disabled={!drillsAgentType.startsWith('llm')}
                />
                <span style={{ fontSize: '0.9rem' }}>
                  Use drill guidelines
                </span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={drillsExcludeStrategicAdvice}
                  onChange={(e) => setDrillsExcludeStrategicAdvice(e.target.checked)}
                  disabled={!drillsAgentType.startsWith('llm')}
                />
                <span style={{ fontSize: '0.9rem' }}>
                  Exclude strategic advice
                </span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={drillsExcludeHigherLevelFeatures}
                  onChange={(e) => setDrillsExcludeHigherLevelFeatures(e.target.checked)}
                  disabled={!drillsAgentType.startsWith('llm')}
                />
                <span style={{ fontSize: '0.9rem' }}>
                  Exclude higher-level features
                </span>
              </label>
              <button onClick={runAllDrillsEval} disabled={drillsLoading || drillsList.length === 0}>
                {drillsLoading ? 'Running...' : (selectedDrillIds.size > 0 ? 'Run selected' : 'Run all')}
              </button>
              <button
                className="secondary"
                disabled={drillsLoading || drillsList.length === 0}
                onClick={() => setSelectedDrillIds(new Set(drillsList.map(d => d.id)))}
              >
                Select all
              </button>
              <button
                className="secondary"
                disabled={drillsLoading || selectedDrillIds.size === 0}
                onClick={() => setSelectedDrillIds(new Set())}
              >
                Select none
              </button>
            </div>

            <div style={{ marginTop: '1rem', fontSize: '0.9rem', color: '#555' }}>
              Total drills: <strong>{drillsList.length}</strong>
              {' '}| Selected: <strong>{selectedDrillIds.size}</strong>
              {drillsEval && (
                <>
                  {' '}| Passed:{' '}
                  <strong>
                    {drillsEval.results.filter(r => r.passed).length}
                  </strong>
                  {' '} / {drillsEval.results.length}
                  {' '} (batch run: <strong>{drillsEval.agent_type}</strong>
                  {typeof drillsEval.include_guidelines === 'boolean' ? (
                    <> | guidelines: <strong>{drillsEval.include_guidelines ? 'on' : 'off'}</strong></>
                  ) : null}
                  {drillsEvalRunMeta?.evaluated_at ? <> @ {new Date(drillsEvalRunMeta.evaluated_at).toLocaleString()}</> : null}
                  {drillsEvalRunMeta?.run_id ? <> | run_id: {drillsEvalRunMeta.run_id}</> : null}
                  )
                </>
              )}
            </div>

            <div style={{ marginTop: '1rem' }}>
              {drillsList.length === 0 ? (
                <div style={{ color: '#666' }}>
                  No drills yet. Open a replay and use “Start Drill Mode” to record your best actions.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {drillsList.map(d => {
                    const evalRow = drillsEval?.results.find(r => r.drill_id === d.id)
                    const passed = evalRow ? evalRow.passed : null
                    const details = drillDetailsById[d.id]
                    const isSelected = selectedDrillIds.has(d.id)
                    return (
                      <div
                        key={d.id}
                        style={{
                          padding: '0.75rem',
                          border: '1px solid #ddd',
                          borderRadius: '6px',
                          background: passed == null ? '#fff' : passed ? '#e8f5e9' : '#ffebee'
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', flexWrap: 'wrap' }}>
                          <div>
                            <div style={{ fontWeight: 'bold' }}>
                              <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
                                <input
                                  type="checkbox"
                                  checked={isSelected}
                                  onChange={(e) => {
                                    setSelectedDrillIds(prev => {
                                      const next = new Set(prev)
                                      if (e.target.checked) next.add(d.id)
                                      else next.delete(d.id)
                                      return next
                                    })
                                  }}
                                />
                              </label>
                              #{d.id}{' '}
                              {d.name ? d.name : '(unnamed)'}
                            </div>
                            <div style={{ fontSize: '0.85rem', color: '#666' }}>
                              Player: {d.player_id} | Steps: {d.num_steps} | Source: {d.source_game_id} @ {d.source_step_idx}
                              {d.guideline_text ? <span> | Guideline: ✅</span> : <span> | Guideline: —</span>}
                            </div>
                          </div>
                          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                            <div style={{ fontWeight: 'bold' }}>
                              {passed == null ? '—' : passed ? 'PASS' : 'FAIL'}
                            </div>
                            {drillsEval && drillsEval.agent_type !== drillsAgentType && (
                              <div style={{ fontSize: '0.8rem', color: '#a15c00' }}>
                                (stale: showing {drillsEval.agent_type})
                              </div>
                            )}
                          </div>
                        </div>

                        {evalRow && !evalRow.passed && evalRow.first_mismatch && (
                          <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: '#b71c1c' }}>
                            First mismatch at step idx {evalRow.first_mismatch.idx}
                            {evalRow.first_mismatch.error && <>: {evalRow.first_mismatch.error}</>}
                          </div>
                        )}

                        <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                          <button
                            className="secondary"
                            onClick={async () => {
                              // Toggle open/closed
                              const isOpen = !!details?.open
                              setDrillDetailsById(prev => ({
                                ...prev,
                                [d.id]: {
                                  ...(prev[d.id] || { open: false, loading: false }),
                                  open: !isOpen,
                                  error: null
                                }
                              }))
                              if (isOpen) return

                              // If opening, use the cached batch evaluation result (same run)
                              // to avoid non-deterministic mismatches (especially for LLM).
                              const batchMatchesCurrent =
                                drillsEval &&
                                drillsEval.agent_type === drillsAgentType &&
                                (typeof drillsEval.include_guidelines !== 'boolean' || drillsEval.include_guidelines === drillsUseGuidelines)

                              if (!batchMatchesCurrent) {
                                setDrillDetailsById(prev => ({
                                  ...prev,
                                  [d.id]: {
                                    ...(prev[d.id] || { open: true }),
                                    open: true,
                                    loading: false,
                                    error: 'Run “Run All Drills” with the current agent + guideline toggle first (details are tied to that batch run).'
                                  }
                                }))
                                return
                              }

                              setDrillDetailsById(prev => ({
                                ...prev,
                                [d.id]: { ...(prev[d.id] || { open: true }), open: true, loading: true, error: null }
                              }))
                              try {
                                const drillResp = await getDrill(d.id)
                                const expectedAction = drillResp.steps?.[0]?.expected_action

                                let originalAction: any = null
                                let originalPlayerId: string | null = null
                                if (drillResp.drill.source_game_id != null && drillResp.drill.source_step_idx != null) {
                                  const replay = await getReplay(drillResp.drill.source_game_id)
                                  const srcIdx = drillResp.drill.source_step_idx
                                  const step = replay.steps?.[srcIdx]
                                  originalAction = step?.action ?? null
                                  originalPlayerId = (step as any)?.player_id ?? null
                                }

                                const batchRow = drillsEval.results.find(r => r.drill_id === d.id)
                                const step0 = batchRow?.step_results?.find(sr => sr.idx === 0)

                                setDrillDetailsById(prev => ({
                                  ...prev,
                                  [d.id]: {
                                    open: true,
                                    loading: false,
                                    error: null,
                                    agent_type: drillsAgentType,
                                    evaluated_at: drillsEvalRunMeta?.evaluated_at,
                                    drill: drillResp.drill,
                                    draft_guideline_text: drillResp.drill.guideline_text || '',
                                    original_action: originalAction,
                                    original_player_id: originalPlayerId,
                                    expected_action: expectedAction,
                                    actual_action: step0?.actual_action ?? null,
                                    match: step0?.match ?? null
                                  }
                                }))
                              } catch (err) {
                                setDrillDetailsById(prev => ({
                                  ...prev,
                                  [d.id]: {
                                    ...(prev[d.id] || { open: true }),
                                    open: true,
                                    loading: false,
                                    error: err instanceof Error ? err.message : 'Failed to load drill details'
                                  }
                                }))
                              }
                            }}
                            disabled={drillsLoading || details?.loading}
                          >
                            {details?.loading ? 'Loading…' : details?.open ? 'Hide details' : 'View details'}
                          </button>
                        </div>

                        {details?.open && (
                          <div style={{ marginTop: '0.75rem', background: '#fafafa', border: '1px solid #eee', borderRadius: '6px', padding: '0.75rem' }}>
                            {details.loading ? (
                              <div style={{ color: '#666' }}>Loading details…</div>
                            ) : details.error ? (
                              <div style={{ color: '#b71c1c' }}>{details.error}</div>
                            ) : (
                              <>
                                <div style={{ marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                                  <div>
                                    <strong>Evaluation (same batch run):</strong>{' '}
                                    {passed == null ? '—' : passed ? 'PASS' : 'FAIL'} | Match: <strong>{details.match ? 'true' : 'false'}</strong>{' '}
                                    (agent: {drillsEval?.agent_type}
                                    {typeof drillsEval?.include_guidelines === 'boolean' ? ` | guidelines: ${drillsEval.include_guidelines ? 'on' : 'off'}` : ''}
                                    {drillsEvalRunMeta?.evaluated_at ? ` @ ${new Date(drillsEvalRunMeta.evaluated_at).toLocaleString()}` : ''}
                                    {drillsEvalRunMeta?.run_id ? ` | run_id: ${drillsEvalRunMeta.run_id}` : ''}
                                    )
                                  </div>
                                </div>
                                <div style={{ marginBottom: '0.75rem' }}>
                                  <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>Drill guideline</div>
                                  <textarea
                                    value={details.draft_guideline_text ?? ''}
                                    onChange={(e) => {
                                      const v = e.target.value
                                      setDrillDetailsById(prev => ({
                                        ...prev,
                                        [d.id]: {
                                          ...(prev[d.id] || { open: true, loading: false }),
                                          draft_guideline_text: v
                                        }
                                      }))
                                    }}
                                    placeholder="(Optional) Guideline to show the LLM for this drill"
                                    style={{ width: '100%', padding: '0.5rem', minHeight: '70px', resize: 'vertical' }}
                                  />
                                  <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                                    <button
                                      className="secondary"
                                      disabled={details.saving_guideline}
                                      onClick={async () => {
                                        const newText = (details.draft_guideline_text ?? '').trim()
                                        setDrillDetailsById(prev => ({
                                          ...prev,
                                          [d.id]: { ...(prev[d.id] || { open: true }), saving_guideline: true, error: null }
                                        }))
                                        try {
                                          await updateDrill(d.id, { guideline_text: newText || null })
                                          // update drills list row
                                          setDrillsList(prev => prev.map(x => (x.id === d.id ? { ...x, guideline_text: newText || null } : x)))
                                          // update details cache
                                          setDrillDetailsById(prev => ({
                                            ...prev,
                                            [d.id]: {
                                              ...(prev[d.id] || { open: true }),
                                              saving_guideline: false,
                                              drill: { ...(prev[d.id]?.drill || {}), guideline_text: newText || null }
                                            }
                                          }))
                                        } catch (err) {
                                          setDrillDetailsById(prev => ({
                                            ...prev,
                                            [d.id]: {
                                              ...(prev[d.id] || { open: true }),
                                              saving_guideline: false,
                                              error: err instanceof Error ? err.message : 'Failed to update guideline'
                                            }
                                          }))
                                        }
                                      }}
                                    >
                                      {details.saving_guideline ? 'Saving…' : 'Save guideline'}
                                    </button>
                                    <div style={{ fontSize: '0.85rem', color: '#666' }}>
                                      Tip: toggle “Use drill guidelines” above + rerun to see if it changes LLM behavior.
                                    </div>
                                  </div>
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.75rem' }}>
                                  <div>
                                    <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>1) Original game action</div>
                                    {details.original_player_id && (
                                      <div style={{ fontSize: '0.85rem', color: '#666', marginBottom: '0.25rem' }}>
                                        player_id: {details.original_player_id}
                                      </div>
                                    )}
                                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(details.original_action, null, 2)}</pre>
                                  </div>
                                  <div>
                                    <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>2) Drill expected (your “correct”)</div>
                                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(details.expected_action, null, 2)}</pre>
                                  </div>
                                  <div>
                                    <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>3) Agent actual (evaluation)</div>
                                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(details.actual_action, null, 2)}</pre>
                                    {details.actual_action?.reasoning && (
                                      <div style={{ marginTop: '0.75rem' }}>
                                        <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>LLM reasoning (same run)</div>
                                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{details.actual_action.reasoning}</pre>
                                      </div>
                                    )}
                                    {details.actual_action?.raw_llm_response && (
                                      <div style={{ marginTop: '0.75rem' }}>
                                        <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>Raw LLM response (same run)</div>
                                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', maxHeight: '240px', overflow: 'auto' }}>{details.actual_action.raw_llm_response}</pre>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    )
  }

  if (view === 'replay') {
    const maxSteps = replayData ? replayData.steps.length : 0
    const currentStep = replayData && replayStepIndex >= 0 && replayStepIndex < maxSteps 
      ? replayData.steps[replayStepIndex] 
      : null
    // Use state_after from current step, or state_before from first step if at step 0, or null
    const displayState = currentStep?.state_after || 
                        (replayData && replayData.steps.length > 0 && replayStepIndex === 0 
                          ? replayData.steps[0]?.state_before 
                          : null)

    return (
      <div className="app">
        <header>
          <h1>Catan Game Replay</h1>
          <button onClick={() => setView('main')} className="back-button">
            Back to Menu
          </button>
        </header>
        <main className="game-main">
          {error && <div className="error">Error: {error}</div>}
          
          {replayData && (
            <>
              <div className="replay-controls">
                <div className="replay-info">
                  <strong>Game ID:</strong> {replayData.game_id} | 
                  <strong> Step:</strong> {replayStepIndex} / {Math.max(0, maxSteps - 1)}
                  {maxSteps === 0 && <span> (No steps recorded)</span>}
                </div>
                {maxSteps > 0 && (
                  <>
                    <div className="replay-scrubber">
                      <input
                        type="range"
                        min="0"
                        max={Math.max(0, maxSteps - 1)}
                        value={replayStepIndex}
                        onChange={(e) => setReplayStepIndex(parseInt(e.target.value))}
                        className="scrubber-slider"
                      />
                      <div className="scrubber-labels">
                        <span>0</span>
                        <span>{maxSteps - 1}</span>
                      </div>
                    </div>
                    <div className="replay-buttons">
                      <button onClick={() => setReplayStepIndex(Math.max(0, replayStepIndex - 1))} disabled={replayStepIndex === 0}>
                        Previous
                      </button>
                      <button onClick={() => setReplayStepIndex(Math.min(maxSteps - 1, replayStepIndex + 1))} disabled={replayStepIndex >= maxSteps - 1}>
                        Next
                      </button>
                    </div>
                  </>
                )}
              </div>

              {currentStep && (
                <div className="replay-step-info">
                  <h3>Step {replayStepIndex}</h3>
                  <div><strong>Action:</strong> {formatActionName(currentStep.action)}</div>
                  {currentStep.player_id && <div><strong>Player:</strong> {currentStep.player_id}</div>}
                  {currentStep.dice_roll && <div><strong>Dice Roll:</strong> {currentStep.dice_roll}</div>}
                  <div><strong>Timestamp:</strong> {currentStep.timestamp}</div>
                  <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <button
                      onClick={async () => {
                        if (!replayData || !currentStep?.state_before) return
                        if (!currentStep.player_id) {
                          alert('This replay step is missing player_id (update backend + reload).')
                          return
                        }
                        const defaultName = `Drill: ${replayData.game_id} @ step ${replayStepIndex} (${currentStep.player_id})`
                        const name = prompt('Name this drill:', defaultName) || defaultName
                        setLoading(true)
                        setError(null)
                        try {
                          const forkedGame = await forkGame(replayData.game_id, currentStep.state_before)
                          setGameState(forkedGame.initial_state)
                          setGameIdInput(forkedGame.game_id)

                          // Preserve agent mapping if present
                          if ((forkedGame.initial_state as any)._metadata && (forkedGame.initial_state as any)._metadata.agent_mapping) {
                            setAgentMapping((forkedGame.initial_state as any)._metadata.agent_mapping)
                          }

                          setPlayerId(currentStep.player_id)
                          // Freeze other players while recording a drill (prevents agents from auto-playing).
                          setStepByStepMode(true)
                          setDrillRecording({
                            name,
                            guideline_text: '',
                            source_game_id: replayData.game_id,
                            source_step_idx: replayStepIndex,
                            drill_player_id: currentStep.player_id,
                            forked_game_id: forkedGame.game_id,
                            steps: []
                          })
                          setView('game')
                          alert('Drill recording started. Take your actions, then use “Save Drill” in the game view.')
                        } catch (err) {
                          setError(err instanceof Error ? err.message : 'Failed to start drill')
                        } finally {
                          setLoading(false)
                        }
                      }}
                      disabled={loading || !currentStep?.state_before}
                      className="action-button"
                      style={{
                        backgroundColor: '#673ab7',
                        color: 'white',
                        padding: '0.75rem 1.5rem',
                        fontSize: '1rem',
                        fontWeight: 'bold'
                      }}
                    >
                      {loading ? 'Starting...' : '🎯 Start Drill Mode (fork + record actions)'}
                    </button>
                    <button
                      onClick={async () => {
                        if (!displayState) return
                        setLoading(true)
                        setError(null)
                        try {
                          // Fork the game to create a new copy
                          const forkedGame = await forkGame(replayData.game_id, displayState)
                          // Load the forked game state
                          setGameState(forkedGame.initial_state)
                          
                          // Restore agent_mapping from metadata if available (forked games preserve agent_mapping)
                          if ((forkedGame.initial_state as any)._metadata && (forkedGame.initial_state as any)._metadata.agent_mapping) {
                            setAgentMapping((forkedGame.initial_state as any)._metadata.agent_mapping)
                          }
                          
                          // Set player ID to current player
                          if (forkedGame.initial_state.phase === 'playing') {
                            const currentPlayer = forkedGame.initial_state.players[forkedGame.initial_state.current_player_index]
                            setPlayerId(currentPlayer.id)
                          } else {
                            const setupPlayer = forkedGame.initial_state.players[forkedGame.initial_state.setup_phase_player_index]
                            setPlayerId(setupPlayer.id)
                          }
                          // Switch to game view
                          setView('game')
                          // Show success message
                          alert(`Game forked! New game ID: ${forkedGame.game_id}\n\nOriginal game preserved. You can now play from this point.`)
                        } catch (err) {
                          setError(err instanceof Error ? err.message : 'Failed to fork game')
                        } finally {
                          setLoading(false)
                        }
                      }}
                      disabled={loading || !displayState}
                      className="action-button"
                      style={{ 
                        backgroundColor: '#4CAF50',
                        color: 'white',
                        padding: '0.75rem 1.5rem',
                        fontSize: '1rem',
                        fontWeight: 'bold'
                      }}
                    >
                      {loading ? 'Forking...' : '🔀 Fork from here (Recommended)'}
                    </button>
                    <div style={{ 
                      fontSize: '0.9rem', 
                      color: '#666',
                      fontStyle: 'italic'
                    }}>
                      Creates a new game copy. Original game is preserved.
                    </div>
                    <div style={{ 
                      marginTop: '0.5rem',
                      padding: '0.75rem',
                      backgroundColor: '#fff3cd',
                      border: '1px solid #ffc107',
                      borderRadius: '4px',
                      fontSize: '0.85rem'
                    }}>
                      <strong>⚠️ Advanced:</strong> You can also restore the original game (modifies it):
                    </div>
                    <button
                      onClick={async () => {
                        if (!displayState) return
                        if (!confirm('This will modify the original game. Continue?')) return
                        setLoading(true)
                        setError(null)
                        try {
                          // Restore the game state to this step's state
                          await restoreGameState(replayData.game_id, displayState)
                          // Load the restored game state
                          const restoredState = await getGameState(replayData.game_id)
                          setGameState(restoredState)
                          // Restore agent_mapping from metadata if available
                          if ((restoredState as any)._metadata && (restoredState as any)._metadata.agent_mapping) {
                            setAgentMapping((restoredState as any)._metadata.agent_mapping)
                          }
                          // Set player ID to current player
                          if (restoredState.phase === 'playing') {
                            const currentPlayer = restoredState.players[restoredState.current_player_index]
                            setPlayerId(currentPlayer.id)
                          } else {
                            const setupPlayer = restoredState.players[restoredState.setup_phase_player_index]
                            setPlayerId(setupPlayer.id)
                          }
                          // Switch to game view
                          setView('game')
                        } catch (err) {
                          setError(err instanceof Error ? err.message : 'Failed to restore game state')
                        } finally {
                          setLoading(false)
                        }
                      }}
                      disabled={loading || !displayState}
                      className="action-button"
                      style={{ 
                        backgroundColor: '#ff9800',
                        color: 'white',
                        padding: '0.5rem 1rem',
                        fontSize: '0.9rem'
                      }}
                    >
                      {loading ? 'Restoring...' : '▶ Restore original game (modifies it)'}
                    </button>
                  </div>
                </div>
              )}

              <div className="game-layout">
                <div className="game-board-section">
                  <h2>Board State</h2>
                  <div className="board-container">
                    {renderBoard(displayState, undefined, [], false)}
                  </div>
                </div>

                <div className="game-sidebar">
                  {displayState && (
                    <>
                      <div className="player-info">
                        <h2>Game State</h2>
                        <div className="info-section">
                          <div><strong>Phase:</strong> {displayState.phase}</div>
                          <div><strong>Turn:</strong> {displayState.turn_number}</div>
                          {displayState.dice_roll && (
                            <div><strong>Dice Roll:</strong> {displayState.dice_roll}</div>
                          )}
                        </div>
                      </div>

                      <div className="all-players">
                        <h2>Players</h2>
                        {displayState.players.map(player => (
                          <div 
                            key={player.id} 
                            className="player-card"
                            style={{
                              borderLeft: `4px solid ${player.color || '#ccc'}`
                            }}
                          >
                            <div><strong>{player.name}</strong> ({player.id})</div>
                            <div>VP: {player.victory_points}</div>
                            <div>Resources: {Object.values(player.resources).reduce((a, b) => a + b, 0)}</div>
                            <div>Settlements: {player.settlements_built}, Cities: {player.cities_built}</div>
                            <div>Roads: {player.roads_built}</div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    )
  }

  // Game view
  const currentPlayer = getPlayerById(playerId)
  const activePlayer = getCurrentPlayer()

  return (
    <div className="app">
      <header>
        <h1>Catan Game</h1>
        <div className="header-actions">
          <label className="dev-mode-toggle">
            <input
              type="checkbox"
              checked={devMode}
              onChange={(e) => setDevMode(e.target.checked)}
            />
            🛠️ Dev Mode
          </label>
          <button 
            onClick={refreshGameState} 
            disabled={loading || !gameState}
            className="refresh-button"
          >
            Refresh
          </button>
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button onClick={() => setView('main')} className="back-button">
            Back to Menu
          </button>
        </div>
      </header>
      <main className="game-main">
        {error && <div className="error">Error: {error}</div>}

        {/* Drill Recording Banner */}
        {drillRecording && (
          <div
            className="trading-panel"
            style={{ marginBottom: '1rem', backgroundColor: '#f3e5f5', borderColor: '#673ab7' }}
          >
            <h2 style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span>🎯 Drill Recording</span>
              <span style={{ fontSize: '0.9rem', color: '#555' }}>
                Steps recorded: <strong>{drillRecording.steps.length}</strong>
              </span>
            </h2>
            <div style={{ fontSize: '0.9rem', color: '#444', marginBottom: '0.5rem' }}>
              <div>
                <strong>Source:</strong> {drillRecording.source_game_id} @ step {drillRecording.source_step_idx} | <strong>Player:</strong> {drillRecording.drill_player_id}
              </div>
              <div>
                <strong>Forked game:</strong> {drillRecording.forked_game_id}
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <input
                type="text"
                value={drillRecording.name}
                onChange={(e) => {
                  const v = e.target.value
                  setDrillRecording(prev => (prev ? { ...prev, name: v } : prev))
                }}
                style={{ flex: 1, minWidth: '280px', padding: '0.5rem' }}
              />
              <textarea
                value={drillRecording.guideline_text}
                onChange={(e) => {
                  const v = e.target.value
                  setDrillRecording(prev => (prev ? { ...prev, guideline_text: v } : prev))
                }}
                placeholder="Optional guideline for this drill (shown to LLM when running with guidelines)"
                style={{ width: '100%', padding: '0.5rem', minHeight: '70px', resize: 'vertical' }}
              />
              <button
                onClick={async () => {
                  if (!drillRecording) return
                  if (drillRecording.steps.length === 0) {
                    alert('Record at least one action before saving.')
                    return
                  }
                  setLoading(true)
                  setError(null)
                  try {
                    const res = await createDrill({
                      name: drillRecording.name,
                      guideline_text: drillRecording.guideline_text || null,
                      source_game_id: drillRecording.source_game_id,
                      source_step_idx: drillRecording.source_step_idx,
                      player_id: drillRecording.drill_player_id,
                      steps: drillRecording.steps.map(s => ({
                        player_id: s.player_id,
                        state: s.state,
                        expected_action: s.expected_action
                      })),
                      metadata: { forked_game_id: drillRecording.forked_game_id }
                    })
                    setDrillRecording(null)
                    alert(`Saved drill #${res.drill_id}`)
                    setView('drills')
                    await refreshDrills()
                  } catch (err) {
                    setError(err instanceof Error ? err.message : 'Failed to save drill')
                  } finally {
                    setLoading(false)
                  }
                }}
                disabled={loading}
                className="action-button"
                style={{ backgroundColor: '#673ab7', color: 'white', fontWeight: 'bold' }}
              >
                {loading ? 'Saving...' : 'Save Drill'}
              </button>
              <button
                onClick={() => {
                  if (!confirm('Cancel drill recording? Recorded steps will be lost.')) return
                  setDrillRecording(null)
                }}
                disabled={loading}
                className="action-button"
                style={{ backgroundColor: '#9e9e9e', color: 'white' }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
        
        {/* Feedback Panel for LLM Agent Teaching */}
        {lastAction && (
          <div className="trading-panel" style={{ marginBottom: '1rem', backgroundColor: '#e3f2fd', borderColor: '#2196F3' }}>
            <h2>
              💬 Provide Feedback (Teach LLM Agent)
              <button
                onClick={() => {
                  setShowFeedbackPanel(!showFeedbackPanel)
                  if (!showFeedbackPanel) {
                    setFeedbackText('')
                    setFeedbackType('general')
                  }
                }}
                className="toggle-button"
                style={{ marginLeft: '0.5rem', fontSize: '0.8rem' }}
              >
                {showFeedbackPanel ? '▼' : '▶'}
              </button>
            </h2>
            {showFeedbackPanel && (
              <div className="trading-content">
                <p style={{ marginBottom: '0.5rem', fontSize: '0.9rem', color: '#666' }}>
                  Last action: <strong>{formatActionName(lastAction)}</strong>
                </p>
                <div style={{ marginBottom: '0.75rem' }}>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                    Feedback Type:
                  </label>
                  <select
                    value={feedbackType}
                    onChange={(e) => setFeedbackType(e.target.value)}
                    style={{ width: '100%', padding: '0.5rem', fontSize: '0.9rem' }}
                  >
                    <option value="general">General</option>
                    <option value="positive">Positive</option>
                    <option value="negative">Negative</option>
                    <option value="suggestion">Suggestion</option>
                  </select>
                </div>
                <div style={{ marginBottom: '0.75rem' }}>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                    Feedback:
                  </label>
                  <textarea
                    value={feedbackText}
                    onChange={(e) => setFeedbackText(e.target.value)}
                    placeholder="E.g., 'Good move! Building a settlement here gives you access to wheat and ore.' or 'Consider trading with the bank first to get the resources you need.'"
                    style={{ width: '100%', padding: '0.5rem', fontSize: '0.9rem', minHeight: '80px', resize: 'vertical' }}
                  />
                </div>
                <button
                  onClick={async () => {
                    if (!feedbackText.trim() || !gameState) return
                    setLoading(true)
                    try {
                      await addFeedback(gameState.game_id, {
                        feedback_text: feedbackText,
                        step_idx: lastStepIdx,
                        player_id: playerId,
                        action_taken: formatActionName(lastAction),
                        feedback_type: feedbackType
                      })
                      setFeedbackText('')
                      setLastAction(null)
                      setShowFeedbackPanel(false)
                      alert('Feedback submitted! The LLM agent will learn from this.')
                    } catch (err) {
                      setError(err instanceof Error ? err.message : 'Failed to submit feedback')
                    } finally {
                      setLoading(false)
                    }
                  }}
                  disabled={loading || !feedbackText.trim()}
                  className="action-button"
                  style={{ width: '100%', backgroundColor: '#4CAF50', color: 'white' }}
                >
                  {loading ? 'Submitting...' : 'Submit Feedback'}
                </button>
                <button
                  onClick={() => {
                    setFeedbackText('')
                    setLastAction(null)
                    setShowFeedbackPanel(false)
                  }}
                  className="action-button"
                  style={{ width: '100%', marginTop: '0.5rem', backgroundColor: '#999', color: 'white' }}
                >
                  Dismiss
                </button>
              </div>
            )}
          </div>
        )}
        
        {gameState?.pending_trade_offer && view === 'game' && (
          <div style={{ 
            padding: '1rem', 
            backgroundColor: '#e3f2fd', 
            border: '1px solid #2196F3',
            borderRadius: '4px',
            marginBottom: '1rem',
            marginTop: '1rem'
          }}>
            <strong>💼 Trade Pending:</strong>
            <div style={{ marginTop: '0.5rem', color: '#666' }}>
              {gameState.pending_trade_offer.proposer_id === playerId ? (
                <>You proposed a trade. Waiting for responses from other players...</>
              ) : (
                <>A trade has been proposed. {activePlayer?.id === playerId ? 'It\'s your turn to respond.' : 'Waiting for responses...'}</>
              )}
            </div>
          </div>
        )}
        
        {/* Step-by-step mode controls for main game view */}
        {view === 'game' && gameState && (
          <div style={{ 
            padding: '1rem', 
            backgroundColor: '#f5f5f5', 
            border: '1px solid #ddd',
            borderRadius: '4px',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '1rem',
            flexWrap: 'wrap'
          }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={stepByStepMode}
                onChange={(e) => setStepByStepMode(e.target.checked)}
              />
              <span>Step-by-Step Mode (read reasoning before advancing)</span>
            </label>
            
            {stepByStepMode && activePlayer && activePlayer.id !== playerId && (
              // Show button if:
              // 1. No pending trade, OR
              // 2. Pending trade but current player is an agent who needs to respond (target), OR
              // 3. Pending trade and current player is proposer who needs to select partner
              // Note: We show the button even if agentMapping is empty, as the backend may have agents configured
              (!gameState.pending_trade_offer || 
               (gameState.pending_trade_offer && (
                 gameState.pending_trade_offer.target_player_ids?.includes(activePlayer.id) ||
                 (gameState.pending_trade_offer.proposer_id === activePlayer.id && 
                  Object.keys(gameState.pending_trade_responses || {}).filter(pid => gameState.pending_trade_responses?.[pid] === true).length > 1)
               ))
              ) && (
              <button
                onClick={async () => {
                  if (!gameState || loading) return
                  try {
                    setLoading(true)
                    const result = await watchAgentsStep(
              gameState.game_id, 
              agentMapping,
              agentWatchExcludeStrategicAdvice,
              agentWatchExcludeHigherLevelFeatures
            )
                    setGameState(result.new_state)
                    
                    // Store reasoning if available
                    if (result.reasoning) {
                      setLastReasoning(result.reasoning)
                    }
                    
                    if (result.error) {
                      setError(`Agent error: ${result.error}`)
                    }
                  } catch (err) {
                    setError(err instanceof Error ? err.message : 'Failed to advance agent turn')
                  } finally {
                    setLoading(false)
                  }
                }}
                disabled={loading || !gameState}
                className="action-button"
                style={{ 
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  padding: '0.5rem 1rem'
                }}
              >
                {loading ? '⏳ Processing...' : '⏭️ Next Step'}
              </button>
            ))}
          </div>
        )}
        
        <div className="game-info-bar">
          <div>
            <strong>Game ID:</strong> {gameState?.game_id}
          </div>
          <div>
            <strong>Phase:</strong> {gameState?.phase}
          </div>
          <div>
            <strong>Turn:</strong> {gameState?.turn_number}
          </div>
          {gameState?.dice_roll && (
            <div>
              <strong>Last Dice Roll:</strong> {gameState.dice_roll}
            </div>
          )}
          {activePlayer && (
            <div>
              <strong>Current Player:</strong> {activePlayer.name}
            </div>
          )}
        </div>

        <div className="game-layout">
          <div className="game-board-section">
            <h2>Board</h2>
            <div className="board-container">
              {renderBoard(gameState, playerId, legalActions, true)}
            </div>
            {lastReasoning && view === 'game' && (
              <div className="reasoning-box">
                <strong>🤔 Agent Reasoning:</strong>
                <div className="reasoning-content">
                  {lastReasoning}
                </div>
              </div>
            )}
          </div>

          <div className="game-sidebar">
            <div className="game-sidebar-column">
            {devMode && gameState && activePlayer && (
              <div className="dev-mode-controls">
                <h2>🛠️ Dev Mode - Switch to Current Player</h2>
                {isAutoDiscarding && gameState.dice_roll === 7 && (
                  <div style={{ 
                    marginBottom: '1rem', 
                    padding: '0.75rem', 
                    backgroundColor: '#fff3cd', 
                    border: '1px solid #ffc107',
                    borderRadius: '4px',
                    color: '#856404'
                  }}>
                    🔄 Auto-discarding: Cycling through players who need to discard...
                  </div>
                )}
                <div className="form-group">
                  <label>
                    Play as:
                    <select
                      value={playerId}
                      onChange={(e) => {
                        setPlayerId(e.target.value)
                        setIsAutoDiscarding(false)  // Stop auto-discarding if manually switched
                      }}
                    >
                      <option key={activePlayer.id} value={activePlayer.id}>
                        {activePlayer.name} ({activePlayer.id}) [Current Turn]
                      </option>
                    </select>
                  </label>
                </div>
                <div className="dev-mode-warning">
                  ℹ️ Dev mode allows you to switch to the current player's turn. Turn validation is still enforced.
                  {gameState.dice_roll === 7 && (
                    <div style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>
                      When a 7 is rolled, dev mode automatically switches to each player who needs to discard.
                </div>
                  )}
                </div>
                {activePlayer.id !== playerId && !isAutoDiscarding && (
                  <button 
                    onClick={() => setPlayerId(activePlayer.id)}
                    className="action-button"
                    style={{ marginTop: '0.5rem' }}
                  >
                    Switch to {activePlayer.name}
                  </button>
                )}
              </div>
            )}

            {/* Combined Players Status */}
            <div className="all-players">
              <h2>Players</h2>
              {gameState?.players.map(player => {
                const isCurrentPlayer = player.id === playerId
                const isActiveTurn = activePlayer?.id === player.id
                const totalResources = Object.values(player.resources).reduce((a, b) => a + b, 0)
                const resourceEntries = Object.entries(player.resources)
                const resourcesDisplay = isCurrentPlayer 
                  ? resourceEntries.map(([resource, amount]) => `${RESOURCE_ICONS[resource] || ''} ${resource}: ${amount}`).join(', ')
                  : `${totalResources} total`
                
                return (
                  <div 
                    key={player.id} 
                    className={`player-card ${isCurrentPlayer ? 'current' : ''} ${isActiveTurn ? 'active-turn' : ''}`}
                    style={{
                      borderLeft: `6px solid ${player.color || '#ccc'}`,
                      padding: '0.5rem',
                      marginBottom: '0.5rem',
                      backgroundColor: isCurrentPlayer ? '#e3f2fd' : '#fff',
                      borderRadius: '4px',
                      border: `2px solid ${player.color || '#ddd'}`,
                      fontSize: '0.85rem'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.15rem' }}>
                      <div>
                        <strong style={{ color: player.color || '#333' }}>{player.name}</strong>
                        {isCurrentPlayer && <span style={{ color: '#1976d2', marginLeft: '0.5rem' }}>(You)</span>}
                        {isActiveTurn && <span className="turn-indicator"> [Current Turn]</span>}
                      </div>
                      <div><strong>VP:</strong> {player.victory_points}</div>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', fontSize: '0.8rem', alignItems: 'center' }}>
                      <div><strong>Resources:</strong> {resourcesDisplay}</div>
                      <div>
                        <strong>Dev Cards:</strong> {player.dev_cards?.length || 0}
                        {isCurrentPlayer && player.dev_cards && player.dev_cards.length > 0 && (
                          <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: '#666' }}>
                            ({player.dev_cards.join(', ')})
                          </span>
                        )}
                      </div>
                      <div>Buildings: {player.settlements_built}S/{player.cities_built}C</div>
                      <div>Roads: {player.roads_built}</div>
                      {player.longest_road && <div style={{ color: '#1976d2' }}>🏆</div>}
                      {player.largest_army && <div style={{ color: '#1976d2' }}>⚔️</div>}
                      {player.knights_played > 0 && <div style={{ color: '#1976d2' }}>⚔️{player.knights_played}</div>}
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="legal-actions">
              <h2>Legal Actions</h2>
              {loading ? (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading actions...</div>
              ) : legalActions.length === 0 ? (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>No legal actions available</div>
              ) : (
                <>
                  {/* Show trade offer details if accept/reject trade actions are present */}
                  {gameState?.pending_trade_offer && legalActions.some(a => a.type === 'accept_trade' || a.type === 'reject_trade') && (() => {
                    const offer = gameState.pending_trade_offer
                    const proposer = gameState.players.find(p => p.id === offer.proposer_id)
                    const proposerName = proposer?.name || offer.proposer_id
                    const giveResources = Object.entries(offer.give_resources || {})
                      .filter(([_, count]) => (count as number) > 0)
                      .map(([resource, count]) => `${count} ${resource}`)
                      .join(', ')
                    const receiveResources = Object.entries(offer.receive_resources || {})
                      .filter(([_, count]) => (count as number) > 0)
                      .map(([resource, count]) => `${count} ${resource}`)
                      .join(', ')
                    
                    return (
                      <div style={{
                        padding: '0.75rem',
                        backgroundColor: '#e3f2fd',
                        border: '1px solid #2196F3',
                        borderRadius: '4px',
                        marginBottom: '1rem',
                        fontSize: '0.9rem'
                      }}>
                        <strong style={{ display: 'block', marginBottom: '0.5rem', color: '#1976D2' }}>
                          💼 Trade Offer from {proposerName}
                        </strong>
                        <div style={{ marginBottom: '0.25rem' }}>
                          <strong>You give:</strong> {receiveResources || 'Nothing'}
                        </div>
                        <div>
                          <strong>You receive:</strong> {giveResources || 'Nothing'}
                        </div>
                      </div>
                    )
                  })()}
                  <div className="actions-list">
                    {legalActions
                      .filter(a => 
                        a.type !== 'trade_bank' && 
                        a.type !== 'trade_player' &&
                        a.type !== 'discard_resources'  // Hide discard from main list, show in discard panel
                      )
                      .map((action, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            console.log('Action button clicked:', action)
                            handleExecuteAction(action)
                          }}
                          disabled={loading || (activePlayer?.id !== playerId && action.type !== 'discard_resources')}
                          className="action-button"
                        >
                          {formatActionName(action)}
                        </button>
                      ))}
                  </div>
                </>
              )}
            </div>

            {/* Discard Resources Panel (when 7 is rolled) */}
            {gameState && gameState.dice_roll === 7 && currentPlayer && (() => {
              // Check if robber has been moved (discard phase is over)
              const robberHasBeenMoved = gameState.robber_initial_tile_id !== undefined && 
                                          gameState.robber_initial_tile_id !== null &&
                                          gameState.robber_tile_id !== gameState.robber_initial_tile_id
              
              if (gameState.waiting_for_robber_move || gameState.waiting_for_robber_steal || robberHasBeenMoved) {
                return null  // Discard phase is over
              }
              
              const totalResources = Object.values(currentPlayer.resources).reduce((a, b) => a + b, 0)
              const hasAlreadyDiscarded = gameState.players_discarded?.includes(playerId) || false
              const needsDiscard = totalResources >= 8 && !hasAlreadyDiscarded
              const discardCount = needsDiscard ? Math.floor(totalResources / 2) : 0
              const currentDiscardTotal = Object.values(discardResources).reduce((a, b) => a + b, 0)
              
              // Check if any other players still need to discard (and haven't discarded yet)
              const otherPlayersNeedDiscard = gameState.players.some(p => {
                const pResources = Object.values(p.resources).reduce((a, b) => a + b, 0)
                const pDiscarded = gameState.players_discarded?.includes(p.id) || false
                return p.id !== playerId && pResources >= 8 && !pDiscarded
              })
              
              if (!needsDiscard) {
                // Show status message if other players need to discard
                if (otherPlayersNeedDiscard) {
                  return (
                    <div className="trading-panel" style={{ backgroundColor: '#e3f2fd', borderColor: '#2196F3' }}>
                      <h2>⏳ Waiting for Other Players to Discard</h2>
                      <p style={{ margin: 0, color: '#666' }}>
                        A 7 was rolled. Other players with 8+ resources must discard half their resources before the robber can be moved.
                      </p>
                    </div>
                  )
                }
                return null
              }
              
              return (
                <div className="trading-panel" style={{ backgroundColor: '#fff3cd', borderColor: '#ffc107' }}>
                  <h2>
                    ⚠️ Discard Resources (7 Rolled)
                    <button
                      onClick={() => setShowDiscardPanel(!showDiscardPanel)}
                      className="toggle-button"
                      style={{ marginLeft: '0.5rem', fontSize: '0.8rem' }}
                    >
                      {showDiscardPanel ? '▼' : '▶'}
                    </button>
                  </h2>
                  {true && (  // Always show when needed - panel should be open
                    <div className="trading-content">
                      <div style={{ marginBottom: '1rem', padding: '0.75rem', backgroundColor: '#fff', borderRadius: '4px' }}>
                        <p style={{ margin: 0 }}>
                          <strong>You have {totalResources} resources.</strong> You must discard <strong>{discardCount} resources</strong> (half, rounded down).
                        </p>
                        <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', color: '#666' }}>
                          Currently selected: <strong>{currentDiscardTotal} / {discardCount}</strong>
                        </p>
                      </div>
                      
                      {(() => {
                        const resourceTypes = ['wood', 'brick', 'wheat', 'sheep', 'ore']
                        
                        return (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <div>
                              <h4 style={{ marginBottom: '0.5rem' }}>Resources to Discard</h4>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                {resourceTypes.map(resource => {
                                  const currentAmount = discardResources[resource] || 0
                                  const available = currentPlayer.resources[resource] || 0
                                  return (
                                    <div key={resource} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                      <span style={{ width: '80px' }}>{RESOURCE_ICONS[resource]} {resource}:</span>
                                      <button
                                        onClick={() => {
                                          if (currentAmount > 0) {
                                            setDiscardResources({ ...discardResources, [resource]: currentAmount - 1 })
                                          }
                                        }}
                                        disabled={currentAmount === 0}
                                        style={{ width: '30px', height: '30px' }}
                                      >
                                        -
                                      </button>
                                      <input
                                        type="number"
                                        min="0"
                                        max={available}
                                        value={currentAmount}
                                        onChange={(e) => {
                                          const val = Math.max(0, Math.min(available, parseInt(e.target.value) || 0))
                                          setDiscardResources({ ...discardResources, [resource]: val })
                                        }}
                                        style={{ width: '60px', textAlign: 'center' }}
                                      />
                                      <button
                                        onClick={() => {
                                          if (currentAmount < available && currentDiscardTotal < discardCount) {
                                            setDiscardResources({ ...discardResources, [resource]: currentAmount + 1 })
                                          }
                                        }}
                                        disabled={currentAmount >= available || currentDiscardTotal >= discardCount}
                                        style={{ width: '30px', height: '30px' }}
                                      >
                                        +
                                      </button>
                                      <span style={{ color: '#666', fontSize: '0.85rem' }}>
                                        (You have {available})
                                      </span>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                            
                            <div style={{ marginTop: '1rem' }}>
                              <button
                                onClick={async () => {
                                  if (currentDiscardTotal !== discardCount) {
                                    setError(`You must discard exactly ${discardCount} resources (currently ${currentDiscardTotal})`)
                                    return
                                  }
                                  
                                  // Validate player has enough of each resource
                                  for (const [resource, amount] of Object.entries(discardResources)) {
                                    if (amount > 0 && (currentPlayer.resources[resource] || 0) < amount) {
                                      setError(`You don't have enough ${resource} to discard`)
                                      return
                                    }
                                  }
                                  
                                  setLoading(true)
                                  setError(null)
                                  
                                  try {
                                    // Filter out zero amounts
                                    const filteredDiscardResources: Record<string, number> = {}
                                    for (const [resource, amount] of Object.entries(discardResources)) {
                                      if (amount > 0) {
                                        filteredDiscardResources[resource] = amount
                                      }
                                    }
                                    
                                    // Ensure we have resources to discard
                                    if (Object.keys(filteredDiscardResources).length === 0) {
                                      setError('No resources selected to discard')
                                      setLoading(false)
                                      return
                                    }
                                    
                                    const discardAction: LegalAction = {
                                      type: 'discard_resources',
                                      payload: {
                                        type: 'DiscardResourcesPayload',
                                        resources: filteredDiscardResources
                                      }
                                    }
                                    
                                    console.log('Executing discard action:', discardAction)
                                    // Drill recording: capture decision point
                                    if (drillRecording && playerId === drillRecording.drill_player_id) {
                                      const stateSnapshot = JSON.parse(JSON.stringify(gameState)) as GameState
                                      setDrillRecording(prev => {
                                        if (!prev) return prev
                                        if (playerId !== prev.drill_player_id) return prev
                                        return {
                                          ...prev,
                                          steps: [
                                            ...prev.steps,
                                            { player_id: playerId, state: stateSnapshot, expected_action: discardAction }
                                          ]
                                        }
                                      })
                                    }
                                    const newState = await postAction(gameState.game_id, playerId, discardAction)
                                    setGameState(newState)
                                    
                                    // Clear the discard form
                                    setDiscardResources({})
                                    setShowDiscardPanel(false)
                                    
                                    // Dev mode: Auto-switch to next player who needs to discard, or back to player who rolled 7
                                    if (devMode && isAutoDiscarding) {
                                      // Find all players who still need to discard (and haven't discarded yet)
                                      const playersStillNeedingDiscard = newState.players.filter(p => {
                                        const totalResources = Object.values(p.resources).reduce((a, b) => a + b, 0)
                                        const hasDiscarded = newState.players_discarded?.includes(p.id) || false
                                        return totalResources >= 8 && !hasDiscarded
                                      })
                                      
                                      if (playersStillNeedingDiscard.length > 0) {
                                        // Switch to next player who needs to discard
                                        const currentIndex = playersStillNeedingDiscard.findIndex(p => p.id === playerId)
                                        const nextIndex = (currentIndex + 1) % playersStillNeedingDiscard.length
                                        setPlayerId(playersStillNeedingDiscard[nextIndex].id)
                                      } else {
                                        // All discards done, switch back to player who rolled 7
                                        if (playerWhoRolled7) {
                                          setPlayerId(playerWhoRolled7)
                                          setIsAutoDiscarding(false)
                                        }
                                      }
                                    }
                                  } catch (err) {
                                    setError(err instanceof Error ? err.message : 'Failed to discard resources')
                                  } finally {
                                    setLoading(false)
                                  }
                                }}
                                disabled={loading || currentDiscardTotal !== discardCount}
                                className="action-button"
                                style={{
                                  width: '100%',
                                  padding: '0.75rem',
                                  fontSize: '1rem',
                                  fontWeight: 'bold',
                                  backgroundColor: currentDiscardTotal === discardCount ? '#4CAF50' : '#ccc',
                                  color: 'white'
                                }}
                              >
                                {loading ? 'Discarding...' : `Discard ${discardCount} Resources`}
                              </button>
                              {currentDiscardTotal !== discardCount && (
                                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: '#d32f2f' }}>
                                  Please select exactly {discardCount} resources to discard.
                                </div>
                              )}
                            </div>
                          </div>
                        )
                      })()}
                    </div>
                  )}
                </div>
              )
            })()}
            </div>

            <div className="game-sidebar-column">
            {/* Card Counts Display */}
            {gameState && (gameState.resource_card_counts || gameState.dev_card_counts) && (
              <div className="card-counts-panel" style={{ marginBottom: '0.75rem', padding: '0.75rem', backgroundColor: '#f5f5f5', borderRadius: '8px', fontSize: '0.85rem' }}>
                <h2 style={{ marginTop: 0, marginBottom: '0.5rem', fontSize: '1rem' }}>Available Cards</h2>
                
                {gameState.resource_card_counts && (
                  <div style={{ marginBottom: '0.75rem' }}>
                    <h3 style={{ marginTop: 0, marginBottom: '0.4rem', fontSize: '0.9rem' }}>Resource Cards</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(80px, 1fr))', gap: '0.4rem' }}>
                      {Object.entries(gameState.resource_card_counts).map(([resource, count]) => (
                        <div 
                          key={resource}
                          style={{
                            padding: '0.4rem',
                            backgroundColor: '#fff',
                            borderRadius: '4px',
                            border: '1px solid #ddd',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            fontSize: '0.8rem'
                          }}
                        >
                          <span style={{ textTransform: 'capitalize', fontWeight: '500' }}>{RESOURCE_ICONS[resource] || ''} {resource}:</span>
                          <span style={{ 
                            fontWeight: 'bold',
                            color: count === 0 ? '#d32f2f' : count < 5 ? '#f57c00' : '#2e7d32'
                          }}>
                            {count}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {gameState.dev_card_counts && (
                  <div>
                    <h3 style={{ marginTop: 0, marginBottom: '0.4rem', fontSize: '0.9rem' }}>Development Cards</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: '0.4rem' }}>
                      {Object.entries(gameState.dev_card_counts).map(([cardType, count]) => (
                        <div 
                          key={cardType}
                          style={{
                            padding: '0.4rem',
                            backgroundColor: '#fff',
                            borderRadius: '4px',
                            border: '1px solid #ddd',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            fontSize: '0.8rem'
                          }}
                        >
                          <span style={{ textTransform: 'capitalize', fontWeight: '500' }}>
                            {cardType.replace('_', ' ')}:
                          </span>
                          <span style={{ 
                            fontWeight: 'bold',
                            color: count === 0 ? '#d32f2f' : '#2e7d32'
                          }}>
                            {count}
                          </span>
                        </div>
                      ))}
                    </div>
                    <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', color: '#666' }}>
                      Total: {Object.values(gameState.dev_card_counts).reduce((a, b) => a + b, 0)} / 25
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Trading Panel */}
            {currentPlayer && activePlayer?.id === playerId && (
              <div className="trading-panel">
                <h2>
                  Trading
                  <button
                    onClick={() => setShowTradingPanel(!showTradingPanel)}
                    className="toggle-button"
                    style={{ marginLeft: '0.5rem', fontSize: '0.8rem' }}
                  >
                    {showTradingPanel ? '▼' : '▶'}
                  </button>
                </h2>
                {showTradingPanel && (
                  <div className="trading-content">
                    {/* Quick Trade Actions (from legal actions) */}
                    <div className="quick-trades">
                      <h3 style={{ marginTop: 0, marginBottom: '0.5rem', fontSize: '0.9rem' }}>Quick Trades</h3>
                      {legalActions
                        .filter(a => a.type === 'trade_bank' || a.type === 'trade_player')
                        .map((action, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleExecuteAction(action)}
                            disabled={loading}
                            className="action-button"
                            style={{ marginBottom: '0.5rem', fontSize: '0.9rem' }}
                          >
                            {formatActionName(action)}
                          </button>
                        ))}
                      {legalActions.filter(a => a.type === 'trade_bank' || a.type === 'trade_player').length === 0 && (
                        <div style={{ color: '#666', fontSize: '0.9rem' }}>No trade actions available</div>
                      )}
                    </div>
                    
                    {/* Custom Trade Builder */}
                    <div className="custom-trade" style={{ marginTop: '0.75rem', padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', backgroundColor: '#fff' }}>
                      <h3 style={{ marginTop: 0, marginBottom: '0.5rem', fontSize: '0.9rem' }}>Custom Trade</h3>
                      
                      {/* Resource Types */}
                      {(() => {
                        const resourceTypes = ['wood', 'brick', 'wheat', 'sheep', 'ore']
                        
                        return (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            {/* Give Resources */}
                            <div>
                              <h4 style={{ marginBottom: '0.4rem', fontSize: '0.85rem' }}>Resources to Give</h4>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                                {resourceTypes.map(resource => {
                                  const currentAmount = giveResources[resource] || 0
                                  const available = currentPlayer?.resources[resource] || 0
                                  return (
                                    <div key={resource} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'nowrap' }}>
                                      <span style={{ width: '75px', flexShrink: 0 }}>{RESOURCE_ICONS[resource]} {resource}:</span>
                                      <button
                                        onClick={() => {
                                          if (currentAmount > 0) {
                                            setGiveResources({ ...giveResources, [resource]: currentAmount - 1 })
                                          }
                                        }}
                                        disabled={currentAmount === 0}
                                        style={{ width: '28px', height: '28px', flexShrink: 0, fontSize: '0.8rem' }}
                                      >
                                        -
                                      </button>
                                      <input
                                        type="number"
                                        min="0"
                                        max={available}
                                        value={currentAmount}
                                        onChange={(e) => {
                                          const val = Math.max(0, Math.min(available, parseInt(e.target.value) || 0))
                                          setGiveResources({ ...giveResources, [resource]: val })
                                        }}
                                        style={{ width: '50px', textAlign: 'center', flexShrink: 0, fontSize: '0.8rem' }}
                                      />
                                      <button
                                        onClick={() => {
                                          if (currentAmount < available) {
                                            setGiveResources({ ...giveResources, [resource]: currentAmount + 1 })
                                          }
                                        }}
                                        disabled={currentAmount >= available}
                                        style={{ width: '28px', height: '28px', flexShrink: 0, fontSize: '0.8rem' }}
                                      >
                                        +
                                      </button>
                                      <span style={{ color: '#666', fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                                        (have {available})
                                      </span>
                      </div>
                                  )
                                })}
                    </div>
                  </div>
                            
                            {/* Receive Resources */}
                            <div>
                              <h4 style={{ marginBottom: '0.4rem', fontSize: '0.85rem' }}>Resources to Receive</h4>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                                {resourceTypes.map(resource => {
                                  const currentAmount = receiveResources[resource] || 0
                                  return (
                                    <div key={resource} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                      <span style={{ width: '80px' }}>{RESOURCE_ICONS[resource]} {resource}:</span>
                                      <button
                                        onClick={() => {
                                          if (currentAmount > 0) {
                                            setReceiveResources({ ...receiveResources, [resource]: currentAmount - 1 })
                                          }
                                        }}
                                        disabled={currentAmount === 0}
                                        style={{ width: '30px', height: '30px' }}
                                      >
                                        -
                                      </button>
                                      <input
                                        type="number"
                                        min="0"
                                        value={currentAmount}
                                        onChange={(e) => {
                                          const val = Math.max(0, parseInt(e.target.value) || 0)
                                          setReceiveResources({ ...receiveResources, [resource]: val })
                                        }}
                                        style={{ width: '60px', textAlign: 'center' }}
                                      />
                                      <button
                                        onClick={() => {
                                          setReceiveResources({ ...receiveResources, [resource]: currentAmount + 1 })
                                        }}
                                        style={{ width: '30px', height: '30px' }}
                                      >
                                        +
                                      </button>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                            
                            {/* Player Selection */}
                            {gameState && (
                              <div>
                                <h4 style={{ marginBottom: '0.4rem', fontSize: '0.85rem' }}>Trade With Players</h4>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                                  {gameState.players
                                    .filter(p => p.id !== playerId)
                                    .map(player => {
                                      const isSelected = selectedTradePlayers.has(player.id)
                                      const totalReceive = Object.values(receiveResources).reduce((a, b) => a + b, 0)
                                      const canAfford = totalReceive === 0 || Object.entries(receiveResources).every(([res, amt]) => 
                                        amt === 0 || (player.resources[res] || 0) >= amt
                                      )
                                      
                                      return (
                                        <label
                                          key={player.id}
                                          style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.5rem',
                                            padding: '0.5rem',
                                            backgroundColor: isSelected ? '#e3f2fd' : '#f5f5f5',
                                            borderRadius: '4px',
                                            cursor: canAfford ? 'pointer' : 'not-allowed',
                                            opacity: canAfford ? 1 : 0.6
                                          }}
                                        >
                                          <input
                                            type="checkbox"
                                            checked={isSelected}
                                            onChange={(e) => {
                                              const newSet = new Set(selectedTradePlayers)
                                              if (e.target.checked) {
                                                newSet.add(player.id)
                                              } else {
                                                newSet.delete(player.id)
                                              }
                                              setSelectedTradePlayers(newSet)
                                            }}
                                            disabled={!canAfford}
                                          />
                                          <span style={{ flex: 1 }}>
                                            <strong>{player.name}</strong>
                                            {!canAfford && totalReceive > 0 && (
                                              <span style={{ color: '#d32f2f', fontSize: '0.85rem', marginLeft: '0.5rem' }}>
                                                (Cannot afford this trade)
                                              </span>
                                            )}
                                          </span>
                                          <span style={{ fontSize: '0.85rem', color: '#666' }}>
                                            Resources: {Object.values(player.resources).reduce((a, b) => a + b, 0)}
                                          </span>
                                        </label>
                                      )
                                    })}
                                </div>
              </div>
            )}
                            
                            {/* Submit Button */}
                            <div style={{ marginTop: '1rem' }}>
                              <button
                                onClick={async () => {
                                  // Validate trade
                                  const totalGive = Object.values(giveResources).reduce((a, b) => a + b, 0)
                                  const totalReceive = Object.values(receiveResources).reduce((a, b) => a + b, 0)
                                  
                                  if (totalGive === 0) {
                                    setError('You must give at least one resource')
                                    return
                                  }
                                  
                                  if (totalReceive === 0) {
                                    setError('You must receive at least one resource')
                                    return
                                  }
                                  
                                  if (selectedTradePlayers.size === 0) {
                                    setError('You must select at least one player to trade with')
                                    return
                                  }
                                  
                                  // Check if current player has enough resources
                                  for (const [resource, amount] of Object.entries(giveResources)) {
                                    if (amount > 0 && (currentPlayer?.resources[resource] || 0) < amount) {
                                      setError(`You don't have enough ${resource} (have ${currentPlayer?.resources[resource] || 0}, need ${amount})`)
                                      return
                                    }
                                  }
                                  
                                  // Propose trade to selected players
                                  setLoading(true)
                                  setError(null)
                                  
                                  try {
                                    // Filter out zero amounts from resources
                                    const filteredGiveResources: Record<string, number> = {}
                                    const filteredReceiveResources: Record<string, number> = {}
                                    
                                    for (const [resource, amount] of Object.entries(giveResources)) {
                                      if (amount > 0) {
                                        filteredGiveResources[resource] = amount
                                      }
                                    }
                                    
                                    for (const [resource, amount] of Object.entries(receiveResources)) {
                                      if (amount > 0) {
                                        filteredReceiveResources[resource] = amount
                                      }
                                    }
                                    
                                    const tradeAction: LegalAction = {
                                      type: 'propose_trade',
                                      payload: {
                                        type: 'ProposeTradePayload',
                                        target_player_ids: Array.from(selectedTradePlayers),
                                        give_resources: filteredGiveResources,
                                        receive_resources: filteredReceiveResources
                                      }
                                    }
                                    
                                    // Drill recording: capture decision point
                                    if (drillRecording && playerId === drillRecording.drill_player_id) {
                                      const stateSnapshot = JSON.parse(JSON.stringify(gameState)) as GameState
                                      setDrillRecording(prev => {
                                        if (!prev) return prev
                                        if (playerId !== prev.drill_player_id) return prev
                                        return {
                                          ...prev,
                                          steps: [
                                            ...prev.steps,
                                            { player_id: playerId, state: stateSnapshot, expected_action: tradeAction }
                                          ]
                                        }
                                      })
                                    }
                                    const newState = await postAction(gameState!.game_id, playerId, tradeAction)
                                    setGameState(newState)
                                    
                                    // Refresh legal actions after proposing trade
                                    // (current player may have changed to target player)
                                    await fetchLegalActions()
                                    
                                    // Clear the trade form
                                    setGiveResources({})
                                    setReceiveResources({})
                                    setSelectedTradePlayers(new Set())
                                  } catch (err) {
                                    setError(err instanceof Error ? err.message : 'Failed to propose trade')
                                  } finally {
                                    setLoading(false)
                                  }
                                }}
                                disabled={loading}
                                className="action-button"
                                style={{
                                  width: '100%',
                                  padding: '0.75rem',
                                  fontSize: '1rem',
                                  fontWeight: 'bold',
                                  backgroundColor: '#4CAF50',
                                  color: 'white'
                                }}
                              >
                                {loading ? 'Proposing Trade...' : 'Propose Trade'}
                              </button>
                            </div>
                          </div>
                        )
                      })()}
                    </div>
                  </div>
                )}
              </div>
            )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
