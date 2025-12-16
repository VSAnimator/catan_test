# Deployment Checklist

Use this checklist to ensure everything is ready for deployment.

## Pre-Deployment

### Environment Configuration
- [ ] Created `backend/.env.production` from `.env.production.example`
- [ ] Generated strong `SECRET_KEY` (use `openssl rand -hex 32`)
- [ ] Set `CORS_ORIGINS` to your production domain(s)
- [ ] Set `ENVIRONMENT=production`
- [ ] Configured `ENABLE_CSRF=true` for production
- [ ] Added `SENTRY_DSN` (optional but recommended)

### Code Readiness
- [ ] All tests passing
- [ ] Frontend builds successfully (`npm run build`)
- [ ] Backend starts without errors
- [ ] Database migrations applied (if any)

### Security
- [ ] Strong SECRET_KEY generated
- [ ] CORS origins restricted to your domain
- [ ] CSRF protection enabled
- [ ] Password strength requirements active
- [ ] Account lockout enabled
- [ ] HTTPS configured (SSL certificate)

### Infrastructure
- [ ] Docker and Docker Compose installed
- [ ] Domain name registered (or IP address ready)
- [ ] SSL certificate obtained (Let's Encrypt recommended)
- [ ] Server/VPS provisioned (if self-hosting)
- [ ] Firewall configured (ports 80, 443, 8000 if needed)

## Deployment Steps

### 1. Initial Setup
- [ ] Clone repository on server
- [ ] Create `.env.production` file
- [ ] Set all required environment variables
- [ ] Test build locally first

### 2. Build & Deploy
- [ ] Run `./deploy.sh` or manual deployment
- [ ] Verify containers are running: `docker-compose ps`
- [ ] Check logs: `docker-compose logs`

### 3. Verification
- [ ] Health check passes: `curl http://localhost/health`
- [ ] Frontend loads: Open in browser
- [ ] Backend API responds: `curl http://localhost:8000/api/health`
- [ ] WebSocket connects: Test game connection
- [ ] User registration works
- [ ] User login works
- [ ] Game creation works
- [ ] Multiplayer connection works

### 4. Domain & SSL
- [ ] DNS records configured (A record pointing to server IP)
- [ ] SSL certificate installed (Let's Encrypt)
- [ ] HTTPS redirects working
- [ ] Security headers present (check with browser dev tools)

### 5. Monitoring
- [ ] Sentry error tracking configured (if using)
- [ ] Logs accessible and monitored
- [ ] Metrics endpoint accessible: `/metrics`
- [ ] Health checks automated (if using monitoring service)

## Post-Deployment

### Testing
- [ ] Test user registration
- [ ] Test user login
- [ ] Test game creation
- [ ] Test multiplayer game (2+ players)
- [ ] Test WebSocket reconnection
- [ ] Test game state persistence
- [ ] Test on mobile devices
- [ ] Test with different browsers

### Backup Setup
- [ ] Database backup script created
- [ ] Backup automation configured (cron job)
- [ ] Backup storage location secured
- [ ] Backup restoration tested

### Documentation
- [ ] Deployment process documented
- [ ] Environment variables documented
- [ ] Troubleshooting guide available
- [ ] Team members have access

## Maintenance

### Regular Tasks
- [ ] Monitor application logs
- [ ] Check Sentry for errors (if configured)
- [ ] Review Prometheus metrics
- [ ] Verify backups are running
- [ ] Update dependencies regularly
- [ ] Review security patches

### Scaling Considerations
- [ ] Monitor database size
- [ ] Monitor memory usage
- [ ] Monitor CPU usage
- [ ] Plan for database migration (SQLite → PostgreSQL)
- [ ] Plan for Redis integration (rate limiting, sessions)
- [ ] Plan for load balancing (if needed)

## Troubleshooting

### Common Issues

**Services won't start:**
- Check logs: `docker-compose logs`
- Verify environment variables
- Check port availability
- Verify Docker is running

**Frontend can't connect to backend:**
- Verify CORS_ORIGINS includes frontend domain
- Check nginx proxy configuration
- Verify backend is running
- Check network connectivity

**Database errors:**
- Verify database file permissions
- Check disk space
- Verify database file exists
- Check database integrity

**WebSocket issues:**
- Verify nginx WebSocket proxy config
- Check firewall rules
- Verify backend WebSocket endpoint
- Check browser console for errors

## Rollback Plan

If deployment fails:
1. Stop services: `docker-compose down`
2. Restore previous version from git
3. Restore database backup (if needed)
4. Redeploy previous version
5. Verify functionality

## Support Contacts

- Deployment issues: Check `DEPLOYMENT.md`
- Feature questions: Check `PRODUCTION_FEATURES.md`
- Quick start: Check `QUICK_DEPLOY.md`

