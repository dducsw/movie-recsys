from .models.movie import MovieModel
from .services.recsys import RecsysService

from .controllers.chatbot import router as chatbot_router
from .controllers.movie import router as movie_router
from .controllers.recsys import router as recsys_router
from .controllers.events import router as events_router

___all__ = [
    "MovieModel",
    "RecsysService",

    # Router
    "chatbot_router",
    "movie_router",
    "recsys_router",
    "events_router",
]
