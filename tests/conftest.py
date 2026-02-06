"""
Pytest Configuration and Fixtures
"""
import pytest
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "api_key": os.getenv("GROK_API_KEY"),
        "model": "grok-beta"
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"id": "1", "text": "Sample document 1"},
        {"id": "2", "text": "Sample document 2"},
    ]
