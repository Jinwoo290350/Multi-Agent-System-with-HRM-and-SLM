import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings from environment variables"""
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Azure Configuration
    AZURE_SUBSCRIPTION_KEY: str = os.getenv("AZURE_SUBSCRIPTION_KEY", "")
    AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    
    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "./data/decision_agent.db")
    ENABLE_DATABASE_LOGGING: bool = os.getenv("ENABLE_DATABASE_LOGGING", "false").lower() == "true"
    
    # Web Scraping
    SCRAPING_TIMEOUT: int = int(os.getenv("SCRAPING_TIMEOUT", 10))
    MAX_CONCURRENT_SCRAPES: int = int(os.getenv("MAX_CONCURRENT_SCRAPES", 3))
    USER_AGENT: str = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; DecisionAgent/1.0)")
    
    # Search
    DEFAULT_SEARCH_ENGINE: str = os.getenv("DEFAULT_SEARCH_ENGINE", "duckduckgo")
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", 10))
    
    # Code Execution
    CODE_EXECUTION_TIMEOUT: int = int(os.getenv("CODE_EXECUTION_TIMEOUT", 5))
    ENABLE_CODE_EXECUTION: bool = os.getenv("ENABLE_CODE_EXECUTION", "true").lower() == "true"
    
    # System
    MAX_PROCESSING_HISTORY: int = int(os.getenv("MAX_PROCESSING_HISTORY", 1000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def cors_origins(self) -> List[str]:
        origins = os.getenv("CORS_ORIGINS", '["*"]')
        try:
            import json
            return json.loads(origins)
        except:
            return ["*"]

settings = Settings()