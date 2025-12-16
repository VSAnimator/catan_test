# Testing Guide for Multiplayer Online System

This guide covers how to test the multiplayer online functionality, including user authentication, game rooms, and real-time gameplay.

## Prerequisites

1. **Backend dependencies installed:**
   ```bash
   cd backend && source .venv/bin/activate && uv pip install -r requirements.txt
   ```

2. **Frontend dependencies installed:**
   ```bash
   cd frontend && npm install
   ```

3. **Database initialized** (happens automatically on backend startup)

## Quick Start Testing

### 1. Start the Servers

**Terminal 1 - Backend:**
```bash
source ~/.zshrc && cd backend && source .venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend && npm run dev
```

Or use the Makefile:
```bash
make dev-backend  # Terminal 1
make dev-frontend # Terminal 2
```

### 2. Access Points

- **Frontend**: http://localhost:5173 (or port shown in terminal)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Manual Testing (UI)

### Test 1: User Registration & Login

1. Open the frontend in your browser
2. You should see a login/register page
3. **Register a new user:**
   - Enter username, password, and email
   - Click "Register"
   - Should see success message and redirect to main page
4. **Logout and login:**
   - Click logout
   - Enter username and password
   - Click "Login"
   - Should successfully authenticate

### Test 2: Create and Join Game Room

1. **As User 1:**
   - Create a room (if room system is enabled)
   - Note the room ID
2. **As User 2 (in different browser/incognito):**
   - Register/login as different user
   - Join the room using the room ID
3. **Start the game:**
   - User 1 clicks "Start Game"
   - Both users should see the game board

### Test 3: Multiplayer Gameplay

1. **Create a game with mixed players:**
   - User 1 creates a 4-player game
   - Set 2 players as LLM agents
   - User 1 and User 2 join as human players
2. **Play through setup:**
   - Both human players place settlements/roads
   - LLM agents should auto-play their turns
3. **Test real-time updates:**
   - User 1 makes a move
   - User 2 should see the update immediately (via WebSocket)

### Test 4: Database Persistence

1. **Create a user and game:**
   - Register a user
   - Create/join a game
2. **Restart the backend server:**
   - Stop the backend (Ctrl+C)
   - Start it again
3. **Verify persistence:**
   - Login with the same credentials (should work)
   - Load the game ID (should still exist)

## Automated Testing (Scripts)

### Test 1: Basic Multiplayer Flow

```bash
cd backend && source .venv/bin/activate && python scripts/test_multiplayer.py
```

This script:
- Registers 2 users
- Creates a room
- Both users join
- Starts a game
- Verifies game state

### Test 2: Mixed Players (Humans + LLM Agents)

```bash
cd backend && source .venv/bin/activate && python scripts/test_mixed_multiplayer_quick.py
```

This script:
- Registers 2 real players
- Creates a 4-player game with 2 LLM agents
- Plays through the entire setup phase
- Verifies agents auto-play correctly

### Test 3: User Authentication

```bash
cd backend && source .venv/bin/activate && python -c "
from api.database import init_db
from api.auth import create_user, get_user_by_username, authenticate_user
import sys

init_db()
print('Testing user authentication...')

# Create user
try:
    user = create_user('testuser', 'testpass123', 'test@example.com')
    print(f'✓ User created: {user.username}')
except Exception as e:
    print(f'✗ User creation failed: {e}')
    sys.exit(1)

# Retrieve user
retrieved = get_user_by_username('testuser')
if retrieved and retrieved.username == 'testuser':
    print('✓ User retrieval successful')
else:
    print('✗ User retrieval failed')
    sys.exit(1)

# Authenticate user
authenticated = authenticate_user('testuser', 'testpass123')
if authenticated:
    print('✓ User authentication successful')
else:
    print('✗ User authentication failed')
    sys.exit(1)

# Test wrong password
wrong_auth = authenticate_user('testuser', 'wrongpass')
if not wrong_auth:
    print('✓ Wrong password correctly rejected')
else:
    print('✗ Wrong password accepted (security issue!)')
    sys.exit(1)

print('\\nAll authentication tests passed!')
"
```

## API Testing (Using curl or httpie)

### Register a User

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123",
    "email": "test@example.com"
  }'
```

### Login

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'
```

Save the `access_token` from the response.

### Get Current User

```bash
curl -X GET http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Create a Room

```bash
curl -X POST http://localhost:8000/api/rooms \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "max_players": 4,
    "min_players": 2
  }'
```

### List Rooms

```bash
curl -X GET http://localhost:8000/api/rooms \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Testing WebSocket Connections

### Using Browser Console

1. Open browser console (F12)
2. Connect to WebSocket:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/game/YOUR_GAME_ID?token=YOUR_ACCESS_TOKEN');

ws.onopen = () => console.log('WebSocket connected');
ws.onmessage = (event) => console.log('Message:', JSON.parse(event.data));
ws.onerror = (error) => console.error('Error:', error);
ws.onclose = () => console.log('WebSocket closed');
```

3. Send a ping:
```javascript
ws.send(JSON.stringify({type: 'ping'}));
```

4. Request game state:
```javascript
ws.send(JSON.stringify({type: 'get_state'}));
```

## Testing Database Persistence

### Verify Users Table

```bash
cd backend && source .venv/bin/activate && python -c "
from api.database import get_db_connection
import json

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute('SELECT id, username, email, created_at FROM users')
rows = cursor.fetchall()
print(f'Found {len(rows)} users:')
for row in rows:
    print(f'  - {row[\"username\"]} (id: {row[\"id\"]}, email: {row[\"email\"]})')
"
```

### Verify Games Table

```bash
cd backend && source .venv/bin/activate && python -c "
from api.database import get_db_connection
import json

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute('SELECT id, created_at FROM games ORDER BY created_at DESC LIMIT 5')
rows = cursor.fetchall()
print(f'Found {len(rows)} recent games:')
for row in rows:
    print(f'  - Game ID: {row[\"id\"]}, Created: {row[\"created_at\"]}')
"
```

## Testing Environment Variables

### Check Current Configuration

```bash
cd backend && source .venv/bin/activate && python -c "
import os
from dotenv import load_dotenv
load_dotenv()

print('Environment Configuration:')
print(f'  SECRET_KEY: {\"Set\" if os.getenv(\"SECRET_KEY\") else \"Not set (using generated)\"}')
print(f'  CORS_ORIGINS: {os.getenv(\"CORS_ORIGINS\", \"Not set (using defaults)\")}')
print(f'  ACCESS_TOKEN_EXPIRE_MINUTES: {os.getenv(\"ACCESS_TOKEN_EXPIRE_MINUTES\", \"Not set (using default)\")}')
"
```

### Test with Custom CORS Origins

1. Edit `backend/.env`:
   ```
   CORS_ORIGINS=http://localhost:3000,https://example.com
   ```

2. Restart backend server

3. Verify in browser network tab that CORS headers match your settings

## Common Issues & Solutions

### Issue: "Users lost after restart"
- **Solution**: Verify users table exists and has data:
  ```bash
  cd backend && source .venv/bin/activate && python -c "from api.database import init_db; init_db(); from api.database import get_db_connection; conn = get_db_connection(); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM users'); print(f'Users in DB: {cursor.fetchone()[0]}')"
  ```

### Issue: "CORS errors in browser"
- **Solution**: Check `CORS_ORIGINS` in `.env` or verify frontend URL is in allowed origins

### Issue: "WebSocket connection fails"
- **Solution**: 
  1. Verify backend is running on port 8000
  2. Check WebSocket URL format: `ws://localhost:8000/api/ws/game/GAME_ID?token=TOKEN`
  3. Verify token is valid (not expired)

### Issue: "Cannot login after registration"
- **Solution**: 
  1. Check database has the user: `SELECT * FROM users WHERE username = 'yourusername'`
  2. Verify password hashing is working
  3. Check backend logs for errors

## Performance Testing

### Test Multiple Concurrent Users

```bash
# Run multiple test scripts in parallel
for i in {1..5}; do
  cd backend && source .venv/bin/activate && python scripts/test_multiplayer.py &
done
wait
```

### Monitor Database Size

```bash
cd backend && ls -lh catan.db
```

## Next Steps

After basic testing passes:
1. Test with 4+ players
2. Test agent auto-play during gameplay (not just setup)
3. Test game state persistence across server restarts
4. Test WebSocket reconnection handling
5. Load test with multiple concurrent games

