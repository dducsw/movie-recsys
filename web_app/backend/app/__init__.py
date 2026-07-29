from .models.movie import MovieModel
from .models.user import UserModel, init_db_tables
from .services.recsys import RecsysService

from .controllers.chatbot import router as chatbot_router
from .controllers.movie import router as movie_router
from .controllers.recsys import router as recsys_router
from .controllers.events import router as events_router
from .controllers.auth import router as auth_router
from .controllers.onboarding import router as onboarding_router

__all__ = [
    "MovieModel",
    "UserModel",
    "init_db_tables",
    "RecsysService",

    # Routers
    "chatbot_router",
    "movie_router",
    "recsys_router",
    "events_router",
    "auth_router",
    "onboarding_router",
]
