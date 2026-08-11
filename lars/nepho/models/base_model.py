import base64
import io
import os
import tempfile

from ..config import config
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from PIL import Image


class BaseModel(ABC):
    """Abstract base class for all chatbot models."""

    def __init__(self, model_name: str, downscale_factor: Optional[int] = None):
        if downscale_factor is not None and (not isinstance(downscale_factor, int) or downscale_factor < 1):
            raise ValueError("downscale_factor must be a positive integer")
        self.model_name = model_name
        self.downscale_factor = downscale_factor

    @abstractmethod
    async def chat(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Generate a response based on the prompt and optional images."""
        pass

    def _downscale_image(self, image_path: str) -> str:
        """Write a downscaled copy of the image to a temp file and return its path."""
        with Image.open(image_path) as img:
            img_format = img.format or "PNG"
            new_size = (
                max(1, img.width // self.downscale_factor),
                max(1, img.height // self.downscale_factor),
            )
            resized = img.resize(new_size, Image.LANCZOS)
            suffix = os.path.splitext(image_path)[1] or ".png"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                resized.save(tmp.name, format=img_format)
            finally:
                tmp.close()
            return tmp.name

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for API calls, downscaling first if configured."""
        try:
            temp_path = None
            if self.downscale_factor and self.downscale_factor > 1:
                temp_path = self._downscale_image(image_path)
            try:
                with open(temp_path or image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            finally:
                if temp_path:
                    os.remove(temp_path)
        except Exception as e:
            raise ValueError(f"Error encoding image {image_path}: {e}")
    
    def validate_image(self, image_path: str) -> bool:
        """Validate if image exists and is in supported format."""
        
        
        if not os.path.exists(image_path):
            return False
        
        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower().lstrip('.')
        if file_ext not in config.SUPPORTED_IMAGE_FORMATS:
            return False
        
        # Check file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > config.MAX_IMAGE_SIZE_MB:
            return False
        
        # Try to open with PIL to validate it's a valid image
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name})"
