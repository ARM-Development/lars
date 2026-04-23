import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_openai():
    with patch("lars.nepho.models.gpt_model.AsyncOpenAI") as MockClient:
        fake_response = MagicMock()
        fake_response.choices[0].message.content = "test response"

        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=fake_response)
        yield instance


@pytest.mark.asyncio
async def test_chat_returns_content(mock_openai):
    from lars.nepho.models.gpt_model import GPTModel

    model = GPTModel(api_key="fake-key")
    result = await model.chat("What is rain?")
    assert result == "test response"


@pytest.mark.asyncio
async def test_chat_passes_temperature(mock_openai):
    from lars.nepho.models.gpt_model import GPTModel

    model = GPTModel(api_key="fake-key", temperature=0.2)
    await model.chat("What is rain?")
    _, kwargs = mock_openai.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_chat_raises_on_api_error(mock_openai):
    from lars.nepho.models.gpt_model import GPTModel

    mock_openai.chat.completions.create.side_effect = Exception("API down")
    model = GPTModel(api_key="fake-key")
    with pytest.raises(RuntimeError, match="Error calling GPT API"):
        await model.chat("What is rain?")
