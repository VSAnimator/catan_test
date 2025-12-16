# Deployment Guide

This guide covers deploying the Catan Agent multiplayer game to production.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)
- Domain name with SSL certificate
- Sentry account (for error tracking, optional)

## Environment Setup

1. Copy `.env.production.example` to `.env.production`:
   ```bash
   cp backend/.env.production.example backend/.env.production
   ```

2. Generate a secure secret key:
   ```bash
   openssl rand -hex 32
   ```

3. Update `.env.production` with your values:
   - `SECRET_KEY`: Generated secret key
   - `CORS_ORIGINS`: Your frontend domain(s)
   - `SENTRY_DSN`: Your Sentry DSN (optional)
   - `ENABLE_CSRF`: Set to `true` for production

## Docker Deployment

### Build and Run

```bash
cd backend
docker-compose up -d
```

### Update

```bash
docker-compose pull
docker-compose up -d
```

## Manual Deployment

### Backend

1. Install dependencies:
   ```bash
   cd backend
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export ENVIRONMENT=production
   export SECRET_KEY=your-secret-key
   export CORS_ORIGINS=https://yourdomain.com
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

