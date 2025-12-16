#!/bin/bash
# Deployment script for Catan Agent multiplayer game

set -e

echo "🚀 Catan Agent Deployment Script"
echo "================================"

# Check if .env.production exists
if [ ! -f "backend/.env.production" ]; then
    echo "❌ Error: backend/.env.production not found"
    echo "📝 Please create it from backend/.env.production.example"
    echo ""
    echo "Required variables:"
    echo "  - SECRET_KEY (generate with: openssl rand -hex 32)"
    echo "  - CORS_ORIGINS (your domain, e.g., https://yourdomain.com)"
    echo "  - SENTRY_DSN (optional, for error tracking)"
    echo "  - ENABLE_CSRF (set to 'true' for production)"
    exit 1
fi

# Load environment variables
source backend/.env.production

# Check required variables
if [ -z "$SECRET_KEY" ]; then
    echo "❌ Error: SECRET_KEY not set in .env.production"
    exit 1
fi

if [ -z "$CORS_ORIGINS" ]; then
    echo "❌ Error: CORS_ORIGINS not set in .env.production"
    exit 1
fi

echo "✅ Environment variables loaded"

# Build frontend
echo ""
echo "📦 Building frontend..."
cd frontend
npm ci
npm run build
cd ..

# Build and start containers
echo ""
echo "🐳 Building Docker images..."
cd backend
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check health
echo ""
echo "🏥 Checking service health..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "⚠️  Frontend health check failed (may still be starting)"
fi

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "⚠️  Backend health check failed (may still be starting)"
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Services:"
echo "  - Frontend: http://localhost"
echo "  - Backend API: http://localhost:8000"
echo "  - Health Check: http://localhost/health"
echo "  - Metrics: http://localhost/metrics"
echo ""
echo "📝 Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "  - Update: ./deploy.sh"
echo ""

