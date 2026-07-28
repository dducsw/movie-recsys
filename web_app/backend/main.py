import os
import uvicorn
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load API Key
from dotenv import load_dotenv
load_dotenv()

# Import routers from controllers
from app import movie_router, recsys_router, chatbot_router, events_router

app = FastAPI(
    title="MovieNex Recommendation System API (MVC)",
    description="Backend API for MovieNex Recsys Demo structured in MVC architecture",
    version="1.1.0"
)

# Enable CORS so the React Frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development ease
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(movie_router)
app.include_router(recsys_router)
app.include_router(chatbot_router)
app.include_router(events_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommender System API! Go to /docs for API documentation."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
