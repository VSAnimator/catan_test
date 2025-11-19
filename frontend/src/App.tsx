import { useState, useEffect, useRef } from 'react'
import './App.css'
import {
  createGame as apiCreateGame,
  getGameState,
  getLegalActions,
  postAction,
  getReplay,
  type GameState,
  type LegalAction,
  type Player,
  type ReplayResponse,
  type StepLog
} from './api'

type View = 'main' | 'game' | 'replay'

function App() {
  const [view, setView] = useState<View>('main')
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [playerId, setPlayerId] = useState<string>('')
  const [legalActions, setLegalActions] = useState<LegalAction[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [devMode, setDevMode] = useState(false)
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null)
  
  // For create/join game
  const [gameIdInput, setGameIdInput] = useState('')
  const [playerNameInput, setPlayerNameInput] = useState('')
  const [numPlayers, setNumPlayers] = useState(2)

  // For replay view
  const [replayData, setReplayData] = useState<ReplayResponse | null>(null)
  const [replayStepIndex, setReplayStepIndex] = useState(0)
  const [replayGameId, setReplayGameId] = useState('')

  // Fetch legal actions when game state or player ID changes
  useEffect(() => {
    if (gameState && playerId && view === 'game') {
      fetchLegalActions()
    }
  }, [gameState, playerId, view])

  // Auto-switch to current player in dev mode when turn changes
  useEffect(() => {
    if (devMode && gameState && view === 'game') {
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
  }, [gameState, devMode, view, playerId])

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

  const refreshGameState = async () => {
    if (!gameState) return
    
    try {
      const newState = await getGameState(gameState.game_id)
      setGameState(newState)
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
      
      const response = await apiCreateGame({ player_names: playerNames })
      setGameState(response.initial_state)
      setPlayerId(response.initial_state.players[0]?.id || '')
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

  const handleExecuteAction = async (action: LegalAction) => {
    if (!gameState || !playerId) return
    
    setLoading(true)
    setError(null)
    try {
      // Ensure action has proper structure
      const actionToSend: LegalAction = {
        type: action.type,
        payload: action.payload || undefined
      }
      const newState = await postAction(gameState.game_id, playerId, actionToSend)
      setGameState(newState)
      // Legal actions will be refetched via useEffect
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
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
      if (payload.card_type) {
        return `${type} (${payload.card_type})`
      }
      if (payload.give_resource && payload.receive_resource) {
        return `${type}: Give ${payload.give_amount} ${payload.give_resource}, receive ${payload.receive_amount} ${payload.receive_resource}`
      }
      if (payload.other_player_id) {
        return `${type} with ${payload.other_player_id}`
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
      const hexSize = 45  // Size of hex (radius)
      const sqrt3 = Math.sqrt(3)
      // Pointy-top hex conversion
      const x = hexSize * (sqrt3 * q + sqrt3 / 2 * r)
      const y = hexSize * (3.0 / 2.0 * r)
      return { x: x + 400, y: y + 300 }  // Center offset for 19-tile board
    }

    return (
      <div className="hex-board">
        {state.tiles.map(tile => {
          const { x, y } = hexToPixel(tile.position[0], tile.position[1])
          
          return (
          <div
            key={tile.id}
            className="hex-tile"
            style={{
              left: `${x}px`,
              top: `${y}px`
            }}
          >
            <div className={`hex-resource ${tile.resource_type || 'desert'}`}>
              {tile.resource_type ? (
                <>
                  <div className="resource-name">{tile.resource_type}</div>
                  {tile.number_token && (
                    <div className="number-token">{tile.number_token}</div>
                  )}
                </>
              ) : (
                <div className="resource-name">Desert</div>
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
          <div
            key={intersection.id}
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
            title={`Intersection ${intersection.id}${intersection.owner ? ` (${ownerPlayer?.name || intersection.owner})` : ''}${intersection.building_type ? ` - ${intersection.building_type}` : ''}${hasLegalAction ? ' - Click to build' : ''}`}
          >
            {intersection.building_type === 'city' ? '🏰' : intersection.building_type === 'settlement' ? '🏠' : '○'}
          </div>
          )
        })}
        
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
          </div>
          {error && <div className="error">Error: {error}</div>}
        </main>
      </div>
    )
  }

  if (view === 'replay') {
    const maxSteps = replayData ? replayData.steps.length : 0
    const currentStep = replayData && replayStepIndex >= 0 && replayStepIndex < maxSteps 
      ? replayData.steps[replayStepIndex] 
      : null
    const displayState = currentStep?.state_after || null

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
                  {currentStep.dice_roll && <div><strong>Dice Roll:</strong> {currentStep.dice_roll}</div>}
                  <div><strong>Timestamp:</strong> {currentStep.timestamp}</div>
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
          </div>

          <div className="game-sidebar">
            {devMode && gameState && activePlayer && (
              <div className="dev-mode-controls">
                <h2>🛠️ Dev Mode - Switch to Current Player</h2>
                <div className="form-group">
                  <label>
                    Play as:
                    <select
                      value={playerId}
                      onChange={(e) => setPlayerId(e.target.value)}
                    >
                      <option key={activePlayer.id} value={activePlayer.id}>
                        {activePlayer.name} ({activePlayer.id}) [Current Turn]
                      </option>
                    </select>
                  </label>
                </div>
                <div className="dev-mode-warning">
                  ℹ️ Dev mode allows you to switch to the current player's turn. Turn validation is still enforced.
                </div>
                {activePlayer.id !== playerId && (
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

            {currentPlayer && (
              <div className="player-info" style={{ borderLeft: `4px solid ${currentPlayer.color || '#ccc'}` }}>
                <h2>Your Status ({currentPlayer.name})</h2>
                <div className="info-section">
                  <div><strong>Victory Points:</strong> {currentPlayer.victory_points}</div>
                  <div className="resources-list">
                    <strong>Resources:</strong>
                    <ul>
                      {Object.entries(currentPlayer.resources).map(([resource, amount]) => (
                        <li key={resource}>
                          {resource}: {amount}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <strong>Buildings:</strong> {currentPlayer.settlements_built} settlements, {currentPlayer.cities_built} cities
                  </div>
                  <div>
                    <strong>Roads:</strong> {currentPlayer.roads_built}
                  </div>
                  <div>
                    <strong>Dev Cards:</strong> {currentPlayer.dev_cards.length}
                    {currentPlayer.dev_cards.length > 0 && (
                      <ul>
                        {currentPlayer.dev_cards.map((card, idx) => (
                          <li key={idx}>{card}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                  {currentPlayer.longest_road && <div>🏆 Longest Road (+2 VP)</div>}
                  {currentPlayer.largest_army && <div>⚔️ Largest Army (+2 VP)</div>}
                  {currentPlayer.knights_played > 0 && <div>Knights Played: {currentPlayer.knights_played}</div>}
                </div>
              </div>
            )}

            <div className="legal-actions">
              <h2>Legal Actions</h2>
              {loading ? (
                <div>Loading actions...</div>
              ) : legalActions.length === 0 ? (
                <div>No legal actions available</div>
              ) : (
                <div className="actions-list">
                  {legalActions.map((action, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleExecuteAction(action)}
                      disabled={loading || activePlayer?.id !== playerId}
                      className="action-button"
                    >
                      {formatActionName(action)}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="all-players">
              <h2>All Players</h2>
              {gameState?.players.map(player => (
                <div 
                  key={player.id} 
                  className={`player-card ${player.id === playerId ? 'current' : ''} ${activePlayer?.id === player.id ? 'active-turn' : ''}`}
                  style={{
                    borderLeft: `4px solid ${player.color || '#ccc'}`
                  }}
                >
                  <div>
                    <strong>{player.name}</strong> ({player.id})
                    {activePlayer?.id === player.id && <span className="turn-indicator"> [Current Turn]</span>}
                  </div>
                  <div>VP: {player.victory_points}</div>
                  <div>Resources: {Object.values(player.resources).reduce((a, b) => a + b, 0)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
