import asyncio
from typing import List, Optional
from openai import AsyncOpenAI
from .base_model import BaseModel
from ..config import config

class GPTModel(BaseModel):
    """GPT model implementation using OpenAI API."""
    
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None, temperature: float = 0.7):
        model_name = model_name or config.DEFAULT_GPT_MODEL
        super().__init__(model_name)

        self.temperature = temperature
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = base_url or config.OPENAI_BASE_URL
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
    async def chat(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Generate a response using GPT model."""
        try:
            messages = []
            
            # Handle images if provided
            if images:
                content = [{"type": "text", "text": prompt}]
                
                for image_path in images:
                    if not self.validate_image(image_path):
                        raise ValueError(f"Invalid image: {image_path}")
                    
                    image_data = self.encode_image(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })
                
                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"Error calling GPT API: {e}")
    
    def supports_vision(self) -> bool:
        """Check if this model supports vision capabilities."""
        return "vision" in self.model_name.lower() or "gpt-4" in self.model_name.lower() or "gpt-5" in self.model_name.lower()
    
