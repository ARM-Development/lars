import asyncio
import json
from typing import List, Optional
from .base_model import BaseModel
from ..config import config

from asksageclient import AskSageClient


class AskSageModel(BaseModel):
    """Ask Sage model implementation using the asksageclient API."""

    def __init__(self, model_name: str, credentials_json: str):
        super().__init__(model_name)
        self.credentials = _load_credentials(credentials_json)
        self.api_key = self.credentials['credentials']['api_key']
        self.email = self.credentials['credentials']['Ask_sage_user_info']['username']
        self.client = AskSageClient(
            email=self.email,
            api_key=self.api_key,
            user_base_url=config.DEFAULT_ASK_SAGE_USER_URL,
            server_base_url=config.DEFAULT_ASK_SAGE_SERVER_URL
        )

    async def chat(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Generate a response using Ask Sage model."""
        try:
            loop = asyncio.get_event_loop()

            if images and self.supports_vision():
                for image_path in images:
                    if not self.validate_image(image_path):
                        raise ValueError(f"Invalid image: {image_path}")

                # query_with_file accepts a single path or a list
                file_arg = images[0] if len(images) == 1 else images
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.query_with_file(
                        message=prompt,
                        file=file_arg,
                        model=self.model_name
                    )
                )
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.query(
                        message=prompt,
                        model=self.model_name
                    )
                )

            return response.get("message", "No response received")

        except Exception as e:
            raise RuntimeError(f"Error calling Ask Sage API: {e}")

    def supports_vision(self) -> bool:
        """Check if this model supports vision capabilities."""
        vision_keywords = ["claude", "gpt-4o", "gpt-4-vision", "vision", "gpt-5"]
        return any(kw in self.model_name.lower() for kw in vision_keywords)

    async def list_available_models(self) -> List[str]:
        """List available Ask Sage models."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_models()
            )
            return response if isinstance(response, list) else []
        except Exception:
            return []


# Load credentials from file
def _load_credentials(filename):
    """Load API credentials from JSON file."""
    try:
        with open(filename) as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Credentials file '{filename}' not found.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in credentials file.")
