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
from app.api.v1.projects import router as projects_router
from app.api.v1.knowledge_base import router as kb_router
from app.api.v1.image import router as image_router
from app.api.v1.voice import router as voice_router
from app.api.v1.sandbox import router as sandbox_router
from app.api.v1.workflows import router as workflows_router
from app.api.v1.computer_use import router as computer_use_router
from app.api.v1.sharing import router as sharing_router
from app.api.v1.search import router as search_router
from app.api.v1.export import router as export_router
from app.api.v1.settings import router as settings_router
from app.api.v1.billing import router as billing_router

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
api_router.include_router(projects_router)
api_router.include_router(kb_router)
api_router.include_router(image_router)
api_router.include_router(voice_router)
api_router.include_router(sandbox_router)
api_router.include_router(workflows_router)
api_router.include_router(computer_use_router)
api_router.include_router(sharing_router)
api_router.include_router(search_router)
api_router.include_router(export_router)
api_router.include_router(settings_router)
api_router.include_router(billing_router)
