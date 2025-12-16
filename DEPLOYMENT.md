# Deployment Guide

This guide covers deploying the Catan Agent multiplayer game to production.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)
- Domain name with SSL certificate
- Sentry account (for error tracking, optional)

## Environment Setup

### Required Environment Variables

Create a `.env` file in the `backend/` directory (or use `backend/env.example` as a template):

1. **Copy the example file:**
   ```bash
   cd backend
   cp env.example .env
   ```

2. **Generate a secure secret key:**
   ```bash
   openssl rand -hex 32
   ```

3. **Required Variables for Production:**

   **Application Configuration:**
   - `ENVIRONMENT=production` - **CRITICAL**: Must be set to `production` for HTTPS enforcement and security features
   - `SECRET_KEY=<generated-secret-key>` - Strong secret key for JWT tokens (generate with `openssl rand -hex 32`)
   - `CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com` - Comma-separated list of allowed frontend origins
   - `ACCESS_TOKEN_EXPIRE_MINUTES=10080` - JWT token expiration (default: 10080 = 7 days)
   - `ENABLE_CSRF=true` - **MUST be `true` in production** for CSRF protection

   **LLM API Keys (Required for LLM agents):**
   - `OPENAI_API_KEY=sk-proj-...` - **REQUIRED** if using OpenAI models (gpt-4o-mini, gpt-5.1, etc.)
   - `ANTHROPIC_API_KEY=sk-ant-...` - Optional: For Anthropic/Claude models
   - `GEMINI_API_KEY=...` - Optional: For Google Gemini models
   - `LLM_API_KEY=...` - Optional: Generic fallback API key
   - `LLM_MODEL=gpt-5.1` - Model to use (default: gpt-5.1). Options: gpt-5.1, gpt-4o-mini, gpt-4, claude-3-opus, etc.

   **Optional:**
   - `SENTRY_DSN=...` - Sentry DSN for error tracking (optional but recommended)
   - `BACKEND_PORT=8000` - Backend port (default: 8000)
   - `FRONTEND_PORT=80` - Frontend port (default: 80)

### Production Checklist

Before deploying to production, ensure:

- [ ] `ENVIRONMENT=production` is set (enables HTTPS enforcement)
- [ ] `SECRET_KEY` is a strong, randomly generated key
- [ ] `CORS_ORIGINS` includes only your production domain(s) (no wildcards)
- [ ] `ENABLE_CSRF=true` is set
- [ ] `OPENAI_API_KEY` (or other LLM API key) is set if using LLM agents
- [ ] `LLM_MODEL` is set to your preferred model (default: gpt-5.1)
- [ ] HTTPS is configured and working
- [ ] Database backups are configured
- [ ] `.env` file is **NOT** committed to git (it's in `.gitignore`)

## Docker Deployment

### Initial Setup

1. **Create `.env` file** (see Environment Setup above)

2. **Build and Run:**
   ```bash
   cd backend
   docker-compose build
   docker-compose up -d
   ```

3. **Verify containers are running:**
   ```bash
   docker-compose ps
   ```

4. **Check logs:**
   ```bash
   docker-compose logs -f backend
   ```

### Update

```bash
cd backend
docker-compose build
docker-compose up -d
```

### Important Notes for Docker

- The `.env` file in `backend/` directory is automatically loaded by docker-compose
- Environment variables in `docker-compose.yml` will override `.env` if both are set
- The backend container exposes port 8000 (configurable via `BACKEND_PORT`)
- The frontend container exposes port 80 (configurable via `FRONTEND_PORT`)
- Database file (`catan.db`) is mounted as a volume, so it persists across container restarts

## Manual Deployment

### Backend

1. Install dependencies:
   ```bash
   cd backend
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. Set environment variables (or use `.env` file):
   ```bash
   export ENVIRONMENT=production
   export SECRET_KEY=your-secret-key
   export CORS_ORIGINS=https://yourdomain.com
   export OPENAI_API_KEY=sk-proj-...
   export LLM_MODEL=gpt-5.1
   export ENABLE_CSRF=true
   # Optional:
   export SENTRY_DSN=your-sentry-dsn
   ```

3. Run with Gunicorn (recommended for production):
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
   ```

### Frontend

1. Build the frontend:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

2. Serve with a web server (nginx, Apache, etc.)

## Nginx Configuration

Example nginx configuration for reverse proxy:

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /api/ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Frontend
    location / {
        root /path/to/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
}
```

## Monitoring

### Health Check

The application exposes a health check endpoint:
```
GET /health
```

### Metrics

Prometheus metrics are available at:
```
GET /metrics
```

### Logging

Logs are output in JSON format (production) or console format (development).

## Security Checklist

- [ ] HTTPS enabled and enforced
- [ ] Secret key is strong and unique
- [ ] CORS origins are restricted
- [ ] CSRF protection enabled
- [ ] Rate limiting configured
- [ ] Account lockout enabled
- [ ] Password strength requirements enforced
- [ ] Security headers configured
- [ ] Database backups configured
- [ ] Error tracking (Sentry) configured

## Database Backups

SQLite database is stored at `backend/catan.db`. Set up regular backups:

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp backend/catan.db backups/catan_${DATE}.db
```

## Scaling Considerations

For high-traffic deployments:

1. **Database**: Migrate from SQLite to PostgreSQL
2. **Rate Limiting**: Use Redis instead of in-memory storage
3. **Session Storage**: Use Redis for CSRF tokens and sessions
4. **Load Balancing**: Use multiple backend instances behind a load balancer
5. **WebSocket**: Consider using a message broker (Redis Pub/Sub) for WebSocket scaling

## Troubleshooting

### WebSocket Connections Not Restoring

- Verify game state is persisted to database
- Check WebSocket reconnection logic in frontend
- Ensure WebSocket endpoint is accessible through proxy

### High Memory Usage

- Monitor database size and implement cleanup for old games
- Consider archiving completed games
- Use connection pooling for database

### Performance Issues

- Check Prometheus metrics at `/metrics`
- Review structured logs for slow queries
- Consider database indexing optimization

