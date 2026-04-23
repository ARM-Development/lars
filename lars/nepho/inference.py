import asyncio
import os
import re

DEFAULT_CATEGORIES = {"No precipitation": "No echoes greater than 10 dBZ present. A circle of echoes near radar site may be present due to ground clutter.",
                      "Stratiform rain": "Widespread echoes between 0 and 35 dBZ, not present as a circular pattern around the radar site.",
                      "Scattered Convection": "Present as isolated to scattered cells with reflectivities between 35-65 dBZ",
                      "Linear convection": "Cells must be organized into a linear structure with reflectivities between 40-60 dBZ",
                      "Supercells": "Supercells contain the classic hook echo and bounded weak echo region signatures with reflectivities above 55 dBZ",
                      "Unknown": "If you cannot confidently classify the radar image into one of the above categories"}

def categories_from_codebook(codebook_path):
    """
    Parse label categories and descriptions from a LARS-format codebook markdown file.

    The function looks for a markdown table under a heading containing
    "Primary Classes" and extracts each ``| Label | Description |`` row.

    Parameters
    ----------
    codebook_path : str
        Path to the codebook markdown file.

    Returns
    -------
    dict
        Mapping of label name → description string, in the order they appear
        in the codebook.

    Raises
    ------
    ValueError
        If no primary-classes table is found in the file.
    """
    with open(codebook_path, "r") as f:
        text = f.read()

    # Find the section that contains the primary classes table.
    # We look for a heading with "Primary Classes" then capture everything
    # until the next heading of equal or higher level.
    section_match = re.search(
        r"(?:^|\n)#{1,6}[^\n]*Primary Classes[^\n]*\n(.*?)(?=\n#{1,6} |\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not section_match:
        raise ValueError(
            f"No 'Primary Classes' section found in codebook: {codebook_path}"
        )

    section = section_match.group(1)

    # Parse table rows: | cell | cell | — skip the separator row (---|---).
    categories = {}
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.fullmatch(r"[\|\s\-:]+", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        label, description = cells[0], cells[1]
        # Skip header row
        if label.lower() in ("label", "class", "category"):
            continue
        if label:
            categories[label] = description

    if not categories:
        raise ValueError(
            f"Primary Classes table found but contained no rows: {codebook_path}"
        )

    return categories


def guidelines_from_codebook(codebook_path):
    """
    Parse annotator guidelines from a LARS-format codebook markdown file.

    The function looks for a heading containing "Annotator Guidelines" and
    collects every bullet point (lines starting with ``-`` or ``*``) until
    the next heading, stripping markdown emphasis markers.

    Parameters
    ----------
    codebook_path : str
        Path to the codebook markdown file.

    Returns
    -------
    list of str
        Ordered list of guideline strings.

    Raises
    ------
    ValueError
        If no annotator-guidelines section is found in the file.
    """
    with open(codebook_path, "r") as f:
        text = f.read()

    section_match = re.search(
        r"(?:^|\n)#{1,6}[^\n]*Annotator Guidelines[^\n]*\n(.*?)(?=\n#{1,6} |\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not section_match:
        raise ValueError(
            f"No 'Annotator Guidelines' section found in codebook: {codebook_path}"
        )

    guidelines = []
    for line in section_match.group(1).splitlines():
        line = line.strip()
        if not line or not (line.startswith("-") or line.startswith("*")):
            continue
        # Strip the leading bullet character and clean markdown emphasis
        text_line = line.lstrip("-* ").strip()
        text_line = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text_line)
        if text_line:
            guidelines.append(text_line)

    if not guidelines:
        raise ValueError(
            f"Annotator Guidelines section found but contained no bullet points: {codebook_path}"
        )

    return guidelines


_DEFAULT_CODEBOOK = os.path.join(
    os.path.dirname(__file__), "..", "..", "CODEBOOK.md"
)
_default_codebook_path = os.path.normpath(_DEFAULT_CODEBOOK)
CODEBOOK_CATEGORIES = (
    categories_from_codebook(_default_codebook_path)
    if os.path.exists(_default_codebook_path) else None
)
CODEBOOK_GUIDELINES = (
    guidelines_from_codebook(_default_codebook_path)
    if os.path.exists(_default_codebook_path) else None
)

async def label_radar_data(radar_df, model, categories=None, guidelines=None,
                           site="Bankhead National Forest",
                           verbose=True, vmin=-20, vmax=60, model_output_dir=None):
    """
    Label radar data using a given model.

    Parameters
    ----------
    radar_df (pd.DataFrame): DataFrame containing radar data to be labeled.
    model: Model used for labeling the radar data.
    categories (dict, optional): Mapping of category name to description. Defaults to
        DEFAULT_CATEGORIES. Pass CODEBOOK_CATEGORIES to use the bundled codebook.
    guidelines (list of str, optional): Annotator guidelines appended to the prompt.
        Pass CODEBOOK_GUIDELINES to use the bundled codebook guidelines.
    site: str: Radar site identifier.
    model_output_dir: str: Directory to save model outputs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the labeled radar data.
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES
    prompt = "This is an image of weather radar base reflectivity data." \
                f" The radar site is the ARM Facility {site} site." \
             " Please classify the weather depicted into one of the following categories: " \
             f"{', '.join(categories) if categories else ', '.join(categories)}."
    prompt += "Each category is defined as follows: "
    for category, description in categories.items():
        prompt += f"{category}: {description}; "
    prompt += f"The reflectivity values range from {vmin} dBZ as indicated by the blue colors to {vmax} dBZ as indicated by the red colors."
    if guidelines:
        prompt += " When classifying, follow these annotator guidelines: "
        prompt += " ".join(guidelines)
    radar_df["llm_label"] = ""

    for fi in radar_df["file_path"].values:
        time = radar_df.loc[radar_df["file_path"] == fi, "time"].values[0]
        prompt_with_time = prompt + f"Please provide just the category label for the radar image taken at time {time}."      
        prompt_with_time = prompt_with_time + "Do not provide your reasoning for your selection, just the category."

        output_model = await model.chat(prompt_with_time, images=[fi])
        # Find the category label in the output
        output_model = output_model.strip()
        output = "Unknown"
        for category in categories.keys():
            output_lower = output_model.lower()
            last_line = output_lower.split("\n")[-1].strip().lower()
            if category.lower() in last_line:
                output = category
                break
        if verbose:
             print("Category assigned:", output)
             print("Model output:", output_model)
             print("Hand label:", radar_df.loc[radar_df["file_path"] == fi, "label"].values[0])
        if model_output_dir is not None:
            output_file = f"{model_output_dir}/{os.path.basename(fi).replace('.png', '_llm_output.txt')}"
            with open(output_file, "w") as f:
                f.write(output_model)
        if output[-1] == ".":
            output = output[:-1]
        radar_df.loc[radar_df["file_path"] == fi, "llm_label"] = output.strip()
        
        
    return radar_df