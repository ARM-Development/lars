import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any
from .base_model import BaseModel
from ..config import config

class OllamaModel(BaseModel):
    """Ollama model implementation for local models."""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        model_name = model_name or config.DEFAULT_OLLAMA_MODEL
        super().__init__(model_name)
        
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
    
    async def check_model_exists(self) -> bool:
        """Check if the model is available in Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        return self.model_name in models
                    return False
        except Exception:
            return False
    
    async def pull_model(self) -> bool:
        """Pull the model if it doesn't exist."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"name": self.model_name}
                async with session.post(
                    f"{self.base_url}/api/pull", 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                ) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Error pulling model {self.model_name}: {e}")
            return False
    
    async def chat(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Generate a response using Ollama model."""
        try:
            # Check if model exists, pull if not
            if not await self.check_model_exists():
                print(f"Model {self.model_name} not found. Attempting to pull...")
                if not await self.pull_model():
                    raise RuntimeError(f"Failed to pull model {self.model_name}")
            
            # Prepare the request payload
            if images:
                # For vision models, encode images as base64
                images_data = []
                for image_path in images:
                    if not self.validate_image(image_path):
                        raise ValueError(f"Invalid image: {image_path}")
                    
                    image_data = self.encode_image(image_path)
                    images_data.append(image_data)
                #if self.model_name == "llama4:scout":
                #    payload = {
                #        "model": self.model_name,
                #        "messages": [
                #            {"role": "user", "content": prompt, "images": images_data}
                #    ],
                #    "stream": False
                #}
                #else:    
                payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "images": images_data,
                        "stream": False
                }

                # Use generate endpoint for vision models
                url = self.api_url
            else:
                # For text-only models, use chat endpoint
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
                url = self.chat_url
            
            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                    
                    data = await response.json()

                    if images:
                        return data.get("response", "No response received")
                    else:
                        return data.get("message", {}).get("content", "No response received")
                        
        except Exception as e:
            raise RuntimeError(f"Error calling Ollama API: {e}")
    
    async def list_available_models(self) -> List[str]:
        """List all available models in Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
                    return []
        except Exception:
            return []
