from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.usage import router as usage_router
from app.api.v1.agents import router as agents_router
from app.api.v1.research import router as research_router
from app.api.v1.canvas import router as canvas_router
from app.api.v1.memory import router as memory_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(conversations_router)
api_router.include_router(usage_router)
api_router.include_router(agents_router)
api_router.include_router(research_router)
api_router.include_router(canvas_router)
api_router.include_router(memory_router)
