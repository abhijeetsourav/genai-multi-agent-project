"""
Configuration Management
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # API Keys
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Grok Configuration
    GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
    GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")
    
    # Model settings
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    
    # Vector store
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    VECTOR_STORE_PATH = DATA_DIR / "embeddings"
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.GROK_API_KEY:
            raise ValueError("GROK_API_KEY not found in environment")
