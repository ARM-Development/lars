"""Where the prev-scan arm gets its 'previous label' from.

Historically ``use_previous_labels`` injected the *human* label of the
preceding scan. On the BNF CSAPR2 set that is close to leaking the answer:
consecutive scans are ~10 minutes apart and their labels rarely change, so
copying the previous hand label -- with no image at all -- reaches 0.80
macro-F1 against the two-human consensus, beating every model configuration
measured. ``previous_label_source="llm"`` feeds the model's own previous
prediction instead, which keeps ground truth out of the prompt and matches
what is actually available at inference time.
"""
import pytest
import pandas as pd

from lars.nepho.inference import label_radar_data

CATEGORIES = {"Clear": "No echoes.", "Rain": "Widespread echoes."}


class ScriptedModel:
    """Replies with a fixed sequence, recording each prompt it was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def chat(self, prompt, images=None):
        self.prompts.append(prompt)
        return self.responses[len(self.prompts) - 1]


def make_df(hand_labels):
    n = len(hand_labels)
    return pd.DataFrame({
        "file_path": [f"/tmp/img{i}.png" for i in range(n)],
        "time": [f"2026-01-{i + 1:02d}" for i in range(n)],
        "label": list(hand_labels),
    })


@pytest.mark.asyncio
async def test_hand_source_injects_the_human_label():
    # Hand labels say Rain; the model keeps answering Clear. Under "hand" the
    # prompt must quote the human label, not the model's own answer.
    model = ScriptedModel(["Clear", "Clear", "Clear"])
    await label_radar_data(make_df(["Rain", "Rain", "Rain"]), model,
                           categories=CATEGORIES, use_previous_labels=1,
                           previous_label_source="hand", verbose=False)

    assert "is Rain." in model.prompts[1]
    assert "is Clear." not in model.prompts[1]


@pytest.mark.asyncio
async def test_llm_source_injects_the_models_own_prediction():
    model = ScriptedModel(["Clear", "Clear", "Clear"])
    await label_radar_data(make_df(["Rain", "Rain", "Rain"]), model,
                           categories=CATEGORIES, use_previous_labels=1,
                           previous_label_source="llm", verbose=False)

    # The model said Clear for scene 0, so scene 1 must be told Clear -- and
    # must NOT be told the hand label Rain.
    assert "is Clear." in model.prompts[1]
    assert "is Rain." not in model.prompts[1]


@pytest.mark.asyncio
async def test_llm_source_never_leaks_ground_truth():
    """No hand label should appear in any prompt under the llm source."""
    model = ScriptedModel(["Clear"] * 4)
    df = make_df(["Rain", "Rain", "Rain", "Rain"])
    await label_radar_data(df, model, categories=CATEGORIES,
                           use_previous_labels=1,
                           previous_label_source="llm", verbose=False)

    for prompt in model.prompts:
        assert "is Rain." not in prompt


@pytest.mark.asyncio
async def test_llm_source_tracks_a_changing_prediction():
    """Each scene is told whatever the model actually said for the one before."""
    model = ScriptedModel(["Clear", "Rain", "Clear"])
    await label_radar_data(make_df(["Clear"] * 3), model,
                           categories=CATEGORIES, use_previous_labels=1,
                           previous_label_source="llm", verbose=False)

    assert "is Clear." in model.prompts[1]   # scene 0 -> Clear
    assert "is Rain." in model.prompts[2]    # scene 1 -> Rain


@pytest.mark.asyncio
async def test_first_scene_gets_no_previous_label_in_either_mode():
    for source in ("hand", "llm"):
        model = ScriptedModel(["Clear", "Clear"])
        await label_radar_data(make_df(["Rain", "Rain"]), model,
                               categories=CATEGORIES, use_previous_labels=1,
                               previous_label_source=source, verbose=False)
        assert "previous radar image" not in model.prompts[0], source


@pytest.mark.asyncio
async def test_default_source_is_hand_for_backward_compatibility():
    model = ScriptedModel(["Clear", "Clear"])
    await label_radar_data(make_df(["Rain", "Rain"]), model,
                           categories=CATEGORIES, use_previous_labels=1,
                           verbose=False)

    assert "is Rain." in model.prompts[1]


@pytest.mark.asyncio
async def test_llm_source_handles_an_unparseable_previous_reply():
    """A reply that parses to nothing must not inject an empty label."""
    model = ScriptedModel(["not a category at all", "Clear"])
    await label_radar_data(make_df(["Rain", "Rain"]), model,
                           categories=CATEGORIES, use_previous_labels=1,
                           previous_label_source="llm", verbose=False)

    # _parse_category falls back to "Unknown", which is a real string, so it is
    # passed through rather than producing "is ." -- the malformed case.
    assert "is ." not in model.prompts[1]


@pytest.mark.asyncio
async def test_multiple_previous_labels_use_the_chosen_source():
    model = ScriptedModel(["Clear", "Rain", "Clear"])
    await label_radar_data(make_df(["Rain"] * 3), model,
                           categories=CATEGORIES, use_previous_labels=2,
                           previous_label_source="llm", verbose=False)

    # Scene 2 looks back two steps: scene 1 -> Rain, scene 0 -> Clear.
    assert "is Rain." in model.prompts[2]
    assert "is Clear." in model.prompts[2]
