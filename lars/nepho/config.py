import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Default models
    DEFAULT_GPT_MODEL: str = os.getenv("DEFAULT_GPT_MODEL", "gpt-4-vision-preview")
    DEFAULT_OLLAMA_MODEL: str = os.getenv("DEFAULT_OLLAMA_MODEL", "llava")
    
    # Image processing settings
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    SUPPORTED_IMAGE_FORMATS: List[str] = os.getenv(
        "SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,gif,bmp,webp"
    ).split(",")
    
    # Parallel processing settings
    MAX_CONCURRENT_MODELS: int = int(os.getenv("MAX_CONCURRENT_MODELS", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))

    DEFAULT_ASK_SAGE_USER_URL = "https://api.asksage.anl.gov/user"
    DEFAULT_ASK_SAGE_SERVER_URL = "https://api.asksage.anl.gov/server"
    

config = Config()
