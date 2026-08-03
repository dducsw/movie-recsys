import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_recommendations_endpoint():
    response = client.get("/api/recommendations")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

def test_search_endpoint():
    response = client.get("/api/movies/search?query=Inception")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
