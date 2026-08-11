import base64
import io

import pytest
from PIL import Image

from lars.nepho.models.gpt_model import GPTModel


def _make_test_image(path, size=(40, 20)):
    Image.new("RGB", size, color="red").save(path, format="PNG")


@pytest.fixture
def model():
    return GPTModel(model_name="gpt-4", api_key="test-key")


def test_downscale_factor_defaults_to_none(model):
    assert model.downscale_factor is None


def test_invalid_downscale_factor_raises():
    with pytest.raises(ValueError, match="downscale_factor must be a positive integer"):
        GPTModel(model_name="gpt-4", api_key="test-key", downscale_factor=0)


def test_encode_image_without_downscale_returns_original_bytes(model, tmp_path):
    image_path = tmp_path / "test.png"
    _make_test_image(image_path)

    encoded = model.encode_image(str(image_path))

    assert base64.b64decode(encoded) == image_path.read_bytes()


def test_encode_image_downscales_by_integer_factor(tmp_path):
    image_path = tmp_path / "test.png"
    _make_test_image(image_path, size=(40, 20))

    model = GPTModel(model_name="gpt-4", api_key="test-key", downscale_factor=4)
    encoded = model.encode_image(str(image_path))

    decoded_image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded_image.size == (10, 5)
