"""The 'no codebook' baseline arm: class names reach the model, nothing else.

The sweep's other arms all hand the model a codebook -- per-class definitions plus
annotator guidelines. Without a definition-free control there is no way to separate
what the model knows about storm morphology from what the codebook told it. That
control is expressed by passing a category map whose descriptions are all empty,
so these tests pin down what such a map does to the prompt.
"""
import pytest
import pandas as pd

from lars.nepho.inference import label_radar_data

DEFINITIONS_HEADER = "Each category is defined as follows"

CATEGORIES_WITH_DESCRIPTIONS = {"Clear": "No echoes.", "Rain": "Widespread echoes."}
CATEGORIES_NAMES_ONLY = {"Clear": "", "Rain": ""}


class FakeModel:
    """Records every prompt it is handed and replies with a fixed label."""

    def __init__(self, response="Clear"):
        self.response = response
        self.calls = []

    async def chat(self, prompt, images=None):
        self.calls.append({"prompt": prompt, "images": images})
        return self.response


def make_df(n=1):
    return pd.DataFrame({
        "file_path": [f"/tmp/img{i}.png" for i in range(n)],
        "time": [f"2026-01-0{i + 1}" for i in range(n)],
        "label": ["Clear"] * n,
    })


async def prompt_for(categories, **kwargs):
    model = FakeModel()
    await label_radar_data(make_df(1), model, categories=categories,
                           verbose=False, **kwargs)
    return model.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_empty_descriptions_drop_the_definitions_block():
    prompt = await prompt_for(CATEGORIES_NAMES_ONLY)

    assert DEFINITIONS_HEADER not in prompt
    # The bare "Clear: ; Rain: ;" residue must not survive either.
    assert "Clear: ;" not in prompt
    assert "Rain: ;" not in prompt


@pytest.mark.asyncio
async def test_class_names_still_reach_the_model_without_descriptions():
    prompt = await prompt_for(CATEGORIES_NAMES_ONLY)

    assert "Clear" in prompt
    assert "Rain" in prompt


@pytest.mark.asyncio
async def test_descriptions_are_emitted_when_present():
    prompt = await prompt_for(CATEGORIES_WITH_DESCRIPTIONS)

    assert DEFINITIONS_HEADER in prompt
    assert "Clear: No echoes.;" in prompt
    assert "Rain: Widespread echoes.;" in prompt


@pytest.mark.asyncio
async def test_a_single_described_class_keeps_the_block():
    """The guard keys off 'any description', not 'all descriptions'."""
    prompt = await prompt_for({"Clear": "", "Rain": "Widespread echoes."})

    assert DEFINITIONS_HEADER in prompt
    assert "Rain: Widespread echoes.;" in prompt


@pytest.mark.asyncio
async def test_guidelines_are_independent_of_the_definitions_block():
    """The baseline passes guidelines=None; the grid passes real ones."""
    bare = await prompt_for(CATEGORIES_NAMES_ONLY, guidelines=None)
    guided = await prompt_for(CATEGORIES_NAMES_ONLY, guidelines=["Prefer Rain."])

    assert "annotator guidelines" not in bare
    assert "annotator guidelines" in guided
    assert "Prefer Rain." in guided
    assert DEFINITIONS_HEADER not in guided


@pytest.mark.asyncio
async def test_explicit_vmin_vmax_survive_without_a_codebook():
    """The baseline drops codebook_path but must still describe the real colorbar."""
    prompt = await prompt_for(CATEGORIES_NAMES_ONLY, vmin=-20, vmax=80)

    assert "-20 dBZ" in prompt
    assert "80 dBZ" in prompt


@pytest.mark.asyncio
async def test_labelling_still_works_with_names_only():
    model = FakeModel("Rain")
    df = await label_radar_data(make_df(2), model,
                                categories=CATEGORIES_NAMES_ONLY, verbose=False)

    assert list(df["llm_label"]) == ["Rain", "Rain"]


@pytest.mark.asyncio
async def test_parsing_ignores_descriptions():
    """Category matching reads only the keys, so empty values cannot break it."""
    model = FakeModel("Rain")
    df = await label_radar_data(make_df(1), model,
                                categories=CATEGORIES_NAMES_ONLY, verbose=False)
    assert df.loc[0, "llm_label"] == "Rain"

    model = FakeModel("nothing here")
    df = await label_radar_data(make_df(1), model,
                                categories=CATEGORIES_NAMES_ONLY, verbose=False)
    assert df.loc[0, "llm_label"] == "Unknown"
