# Quick Deployment Guide

This is a simplified deployment guide for getting your Catan Agent game online quickly.

## Prerequisites

- Docker and Docker Compose installed
- A domain name (optional, can use IP address)
- SSL certificate (for HTTPS, can use Let's Encrypt)

## Step 1: Configure Environment

1. Create production environment file:
   ```bash
   cd backend
   cp .env.production.example .env.production
   ```

2. Generate a secret key:
   ```bash
   openssl rand -hex 32
   ```

3. Edit `backend/.env.production`:
   ```bash
   ENVIRONMENT=production
   SECRET_KEY=<paste-generated-key-here>
   CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
   ACCESS_TOKEN_EXPIRE_MINUTES=10080
   ENABLE_CSRF=true
   SENTRY_DSN=https://your-sentry-dsn-here  # Optional
   ```

## Step 2: Deploy

### Option A: Using the deployment script (recommended)

```bash
./deploy.sh
```

### Option B: Manual deployment

```bash
# Build frontend
cd frontend
npm ci
npm run build
cd ..

# Start services
cd backend
docker-compose up -d
```

## Step 3: Verify Deployment

1. Check health:
   ```bash
   curl http://localhost/health
   ```

2. Open in browser:
   - Frontend: http://localhost (or your domain)
   - Backend API: http://localhost:8000/api

## Step 4: Set Up Domain & SSL (Production)

### Using Nginx as Reverse Proxy

1. Install Nginx:
   ```bash
   sudo apt-get update
   sudo apt-get install nginx certbot python3-certbot-nginx
   ```

2. Get SSL certificate:
   ```bash
   sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
   ```

3. Configure Nginx (see DEPLOYMENT.md for full config):
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       return 301 https://$server_name$request_uri;
   }

   server {
       listen 443 ssl http2;
       server_name yourdomain.com;

       ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

       location / {
           proxy_pass http://localhost:80;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

## Platform-Specific Deployments

### Railway

1. Connect GitHub repository
2. Add environment variables in Railway dashboard
3. Railway auto-detects Docker and deploys

### Render

1. Create new Web Service
2. Connect GitHub repository
3. Set build command: `cd backend && docker-compose build`
4. Set start command: `cd backend && docker-compose up`
5. Add environment variables

### Fly.io

1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Launch: `fly launch`
4. Set secrets: `fly secrets set SECRET_KEY=... CORS_ORIGINS=...`

### DigitalOcean App Platform

1. Create new App
2. Connect GitHub repository
3. Add Docker Compose file
4. Set environment variables
5. Deploy

## Troubleshooting

### Services won't start

```bash
# Check logs
cd backend
docker-compose logs

# Check if ports are in use
netstat -tulpn | grep -E ':(80|8000)'
```

### Frontend can't connect to backend

- Verify CORS_ORIGINS includes your frontend domain
- Check nginx proxy configuration
- Verify backend is running: `curl http://localhost:8000/health`

### Database issues

```bash
# Backup database
cp backend/catan.db backend/catan.db.backup

# Check database
sqlite3 backend/catan.db ".tables"
```

## Maintenance

### Update the application

```bash
git pull
./deploy.sh
```

### View logs

```bash
cd backend
docker-compose logs -f
```

### Stop services

```bash
cd backend
docker-compose down
```

### Backup database

```bash
DATE=$(date +%Y%m%d_%H%M%S)
cp backend/catan.db backups/catan_${DATE}.db
```

## Next Steps

1. **Set up monitoring**: Configure Sentry for error tracking
2. **Set up backups**: Automate database backups
3. **Scale up**: Consider migrating to PostgreSQL for better performance
4. **Add Redis**: For rate limiting and session storage at scale
5. **Load balancing**: Add multiple backend instances for high traffic

## Support

- Check `DEPLOYMENT.md` for detailed deployment information
- Review `PRODUCTION_FEATURES.md` for feature documentation
- Check application logs for errors

