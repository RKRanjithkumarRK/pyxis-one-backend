.PHONY: up down build migrate seed test test-backend test-frontend lint clean logs shell-backend shell-db

# ─── Docker Compose ──────────────────────────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build --parallel

logs:
	docker compose logs -f backend frontend

restart-backend:
	docker compose restart backend celery

# ─── Database ─────────────────────────────────────────────────────────────────
migrate:
	docker compose exec backend alembic upgrade head

migrate-local:
	cd backend && alembic upgrade head

migrate-new:
	cd backend && alembic revision --autogenerate -m "$(msg)"

# ─── Testing ──────────────────────────────────────────────────────────────────
test: test-backend test-frontend

test-backend:
	docker compose exec backend pytest -v --cov=app --cov-report=term-missing

test-backend-local:
	cd backend && pytest -v --cov=app --cov-report=term-missing

test-frontend:
	docker compose exec frontend npm test

# ─── Linting / type checking ──────────────────────────────────────────────────
lint:
	cd backend && ruff check app tests
	cd frontend && npm run type-check && npm run lint

# ─── Shells ───────────────────────────────────────────────────────────────────
shell-backend:
	docker compose exec backend bash

shell-db:
	docker compose exec postgres psql -U nexusai nexusai

# ─── Local dev (no Docker) ────────────────────────────────────────────────────
dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

# ─── Setup ────────────────────────────────────────────────────────────────────
setup:
	@[ -f .env ] || cp .env.example .env && echo "Created .env from example"
	@[ -f backend/.env ] || cp backend/.env.example backend/.env && echo "Created backend/.env"
	@[ -f frontend/.env.local ] || cp frontend/.env.example frontend/.env.local && echo "Created frontend/.env.local"

# ─── Clean ────────────────────────────────────────────────────────────────────
clean:
	docker compose down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf frontend/.next frontend/node_modules
