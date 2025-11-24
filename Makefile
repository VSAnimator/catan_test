.PHONY: help dev-backend dev-frontend dev install-backend install-frontend

help:
	@echo "Available targets:"
	@echo "  make dev              - Run both backend and frontend"
	@echo "  make dev-backend      - Run backend only"
	@echo "  make dev-frontend     - Run frontend only"
	@echo "  make install-backend  - Install Python dependencies"
	@echo "  make install-frontend - Install Node dependencies"

dev: dev-backend dev-frontend

dev-backend:
	@echo "Starting backend server..."
	source ~/.zshrc && cd backend && source .venv/bin/activate && uvicorn main:app --reload

dev-frontend:
	@echo "Starting frontend server..."
	cd frontend && npm run dev

install-backend:
	@echo "Installing Python dependencies..."
	cd backend && uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

install-frontend:
	@echo "Installing Node dependencies..."
	cd frontend && npm install

