# Production Deployment Checklist

Quick reference for deploying to production. See `DEPLOYMENT.md` for detailed instructions.

## Before Deployment

### 1. Environment Variables (in `backend/.env`)

**CRITICAL - Must Set:**
- [ ] `ENVIRONMENT=production` - **REQUIRED** for HTTPS enforcement and security
- [ ] `SECRET_KEY=<strong-random-key>` - Generate with: `openssl rand -hex 32`
- [ ] `CORS_ORIGINS=https://yourdomain.com` - Your production frontend URL(s)
- [ ] `ENABLE_CSRF=true` - **MUST be true in production**
- [ ] `OPENAI_API_KEY=sk-proj-...` - Required if using LLM agents

**Recommended:**
- [ ] `LLM_MODEL=gpt-5.1` - Set to your preferred model
- [ ] `SENTRY_DSN=...` - For error tracking
- [ ] `ACCESS_TOKEN_EXPIRE_MINUTES=10080` - JWT expiration (default: 7 days)

**Optional:**
- [ ] `ANTHROPIC_API_KEY=...` - For Claude models
- [ ] `GEMINI_API_KEY=...` - For Gemini models
- [ ] `BACKEND_PORT=8000` - Backend port
- [ ] `FRONTEND_PORT=80` - Frontend port

### 2. Docker Configuration

- [ ] `.env` file exists in `backend/` directory
- [ ] All required environment variables are set
- [ ] `.env` is **NOT** committed to git (check `.gitignore`)

### 3. Security

- [ ] HTTPS is configured and working
- [ ] SSL certificate is valid
- [ ] CORS origins are restricted to production domain(s) only
- [ ] CSRF protection is enabled (`ENABLE_CSRF=true`)
- [ ] Secret key is strong and unique
- [ ] Database backups are configured

### 4. Infrastructure

- [ ] Domain name is configured
- [ ] DNS points to your server
- [ ] Firewall allows ports 80 (HTTP) and 443 (HTTPS)
- [ ] Reverse proxy (nginx) is configured (if using)
- [ ] WebSocket support is configured in nginx (if using)

## Deployment Steps

1. **Clone/Pull latest code:**
   ```bash
   git pull origin multiplayer_online
   ```

2. **Create/Update `.env` file:**
   ```bash
   cd backend
   cp env.example .env
   # Edit .env with your production values
   ```

3. **Build and start containers:**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

4. **Verify:**
   ```bash
   docker-compose ps
   docker-compose logs backend
   curl https://yourdomain.com/health
   ```

## Key Differences: Development vs Production

| Setting | Development | Production |
|---------|------------|------------|
| `ENVIRONMENT` | `development` | `production` |
| `ENABLE_CSRF` | `false` | `true` |
| `CORS_ORIGINS` | `http://localhost:80,http://localhost:5173` | `https://yourdomain.com` |
| HTTPS | Not required | **REQUIRED** |
| Security Headers | Relaxed | Strict |

## Common Issues

### CORS Errors
- Ensure `CORS_ORIGINS` includes your exact frontend URL (with protocol and port if not 80/443)
- Check that `ENVIRONMENT=production` is set correctly

### 401 Unauthorized
- Check that JWT tokens are being sent in Authorization header
- Verify `SECRET_KEY` matches between restarts

### LLM API Errors
- Verify `OPENAI_API_KEY` is set correctly
- Check API key has sufficient credits/permissions
- Ensure `LLM_MODEL` is a valid model name

### HTTPS Required Error
- Set `ENVIRONMENT=production`
- Ensure nginx/proxy sets `X-Forwarded-Proto: https` header
- Verify SSL certificate is valid

## Post-Deployment

- [ ] Test user registration
- [ ] Test user login
- [ ] Test game creation
- [ ] Test LLM agents (if enabled)
- [ ] Monitor logs for errors
- [ ] Set up monitoring/alerts
- [ ] Configure database backups

