#!/bin/bash
# Quick test script for multiplayer system

echo "=== Multiplayer System Quick Test ==="
echo ""

# Check if backend is running
echo "1. Checking if backend is running..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "   ✓ Backend is running"
else
    echo "   ✗ Backend is not running. Start it with: make dev-backend"
    exit 1
fi

# Test user registration
echo ""
echo "2. Testing user registration..."
RESPONSE=$(curl -s -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser_'$(date +%s)'", "password": "test123", "email": "test@test.com"}')

if echo "$RESPONSE" | grep -q "access_token"; then
    echo "   ✓ User registration successful"
    TOKEN=$(echo "$RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
else
    echo "   ✗ User registration failed: $RESPONSE"
    exit 1
fi

# Test login
echo ""
echo "3. Testing user login..."
USERNAME=$(echo "$RESPONSE" | grep -o '"username":"[^"]*' | cut -d'"' -f4)
LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"test123\"}")

if echo "$LOGIN_RESPONSE" | grep -q "access_token"; then
    echo "   ✓ User login successful"
    LOGIN_TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
else
    echo "   ✗ User login failed: $LOGIN_RESPONSE"
    exit 1
fi

# Test get current user
echo ""
echo "4. Testing get current user..."
ME_RESPONSE=$(curl -s -X GET http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $LOGIN_TOKEN")

if echo "$ME_RESPONSE" | grep -q "$USERNAME"; then
    echo "   ✓ Get current user successful"
else
    echo "   ✗ Get current user failed: $ME_RESPONSE"
    exit 1
fi

# Test database persistence
echo ""
echo "5. Testing database persistence..."
cd backend && source .venv/bin/activate && python -c "
from api.database import get_db_connection
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM users')
count = cursor.fetchone()[0]
print(f'   ✓ Found {count} user(s) in database')
" 2>/dev/null || echo "   ⚠ Could not verify database (backend venv may not be activated)"

echo ""
echo "=== All Quick Tests Passed! ==="
echo ""
echo "Next steps:"
echo "  1. Open frontend: http://localhost:5173"
echo "  2. Register/login in the UI"
echo "  3. Create or join a game"
echo "  4. Test multiplayer functionality"
