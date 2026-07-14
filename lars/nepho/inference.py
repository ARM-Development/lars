import asyncio
import os
import re

from ..preprocessing.labels import apply_criteria_to_labels

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


_CRITERION_PATTERN = re.compile(
    r"\b(percentage|number)\s+of\s+gates\s+"
    r"(?:with\s+reflectivity\s+)?"
    r"greater\s+than\s+(\d+(?:\.\d+)?)\s*dBZ\s+"
    r"must\s+not\s+exceed\s+(\d+(?:\.\d+)?)(?:\s*percent)?",
    re.IGNORECASE,
)

_RECLASSIFY_PATTERN = re.compile(
    r"\bIf\s+(?:it\s+(?:does\s+)?)?exceeds?\s+\d+(?:\.\d+)?(?:\s*percent)?\s*,?\s*"
    r"(?:then\s+)?classify\s+as\s+(?:a\s+|an\s+|the\s+)?([^.]+?)\s*\.",
    re.IGNORECASE,
)


def _format_threshold(threshold_str):
    val = float(threshold_str)
    return int(val) if val.is_integer() else val


def criteria_from_codebook(codebook_path):
    """
    Parse hard quantitative criteria from category descriptions in a
    LARS-format codebook markdown file.

    For every category description, this function looks for sentences of the
    form

        "the {percentage|number} of gates [with reflectivity] greater than
         X dBZ must not exceed Y[ percent]"

    and, optionally, an immediately following reclassification clause

        "If it does exceed Y[ percent], then classify as Z."

    Both percent-based (``pct_gates_<T>dbz``) and count-based
    (``n_gates_<T>dbz``) criteria are supported; the column name is derived
    from the matched phrasing.

    Parameters
    ----------
    codebook_path : str
        Path to the codebook markdown file.

    Returns
    -------
    dict
        Mapping of label name → list of criterion dicts. Each criterion has:

        * ``field`` (str): column name to compare against
          (e.g. ``"pct_gates_50dbz"`` or ``"n_gates_30dbz"``).
        * ``kind`` (str): ``"pct"`` or ``"count"``.
        * ``threshold_dbz`` (int or float): the reflectivity threshold.
        * ``max_value`` (float): the maximum allowed value; rows whose
          ``field`` is strictly greater than this violate the criterion.
        * ``reclassify_as`` (str or None): canonical label to assign when
          the criterion is violated, or ``None`` if the codebook does not
          specify a target.
    """
    categories = categories_from_codebook(codebook_path)
    canonical = {label.lower().strip(): label for label in categories}

    criteria = {}
    for label, description in categories.items():
        rules = []
        for m in _CRITERION_PATTERN.finditer(description):
            kind_word = m.group(1).lower()
            threshold = _format_threshold(m.group(2))
            max_value = float(m.group(3))
            kind = "pct" if kind_word.startswith("percent") else "count"
            field_prefix = "pct_gates" if kind == "pct" else "n_gates"
            field = f"{field_prefix}_{threshold}dbz"

            after = description[m.end():]
            rm = _RECLASSIFY_PATTERN.search(after)
            target = rm.group(1).strip() if rm else None
            reclassify_as = canonical.get(target.lower(), target) if target else None

            rules.append({
                "field": field,
                "kind": kind,
                "threshold_dbz": threshold,
                "max_value": max_value,
                "reclassify_as": reclassify_as,
            })
        if rules:
            criteria[label] = rules

    return criteria


DEFAULT_VMIN = -20
DEFAULT_VMAX = 60

_COLORMAP_LINE_RE = re.compile(r"^[^\n]*Color\s*scale[^\n]*$", re.IGNORECASE | re.MULTILINE)
_COLORMAP_NAME_RE = re.compile(r"([A-Za-z0-9_]+)\s+colormap", re.IGNORECASE)
_VMIN_RE = re.compile(r"vmin\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_VMAX_RE = re.compile(r"vmax\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _format_number(num_str):
    val = float(num_str)
    return int(val) if val.is_integer() else val


def colormap_from_codebook(codebook_path):
    """
    Parse the color-scale specification from a LARS-format codebook.

    The function looks for the ``Color scale`` line in the *Image Format*
    section (Section 2.2), e.g.::

        - **Color scale:** ChaseSpectral colormap with vmin=-10 and vmax=60

    and extracts the colormap name and its ``vmin`` / ``vmax`` bounds. Any
    field that is missing from the codebook falls back to its default:
    ``colormap=None``, ``vmin=DEFAULT_VMIN`` (-20), ``vmax=DEFAULT_VMAX`` (60).

    Parameters
    ----------
    codebook_path : str
        Path to the codebook markdown file.

    Returns
    -------
    dict
        Mapping with keys:

        * ``colormap`` (str or None): the colormap name, or ``None`` if the
          codebook does not name one.
        * ``vmin`` (int or float): lower bound of the color scale.
        * ``vmax`` (int or float): upper bound of the color scale.
    """
    with open(codebook_path, "r") as f:
        text = f.read()

    colormap = None
    vmin = DEFAULT_VMIN
    vmax = DEFAULT_VMAX

    line_match = _COLORMAP_LINE_RE.search(text)
    if line_match:
        line = line_match.group(0)
        name_match = _COLORMAP_NAME_RE.search(line)
        if name_match:
            colormap = name_match.group(1)
        vmin_match = _VMIN_RE.search(line)
        if vmin_match:
            vmin = _format_number(vmin_match.group(1))
        vmax_match = _VMAX_RE.search(line)
        if vmax_match:
            vmax = _format_number(vmax_match.group(1))

    return {"colormap": colormap, "vmin": vmin, "vmax": vmax}


COLOR_DBZ_RANGE = {
    "blue": (None, 10),
    "black": (None, 10),
    "green": (10, 30),
    "yellow": (30, 40),
    "red": (40, 50),
    "dark red": (40, 50),
    "pink": (50, None),
}

LOW_REFLECTIVITY_COLORS = {"blue", "black"}

DEFAULT_COLOR_PRESENCE_THRESHOLD = 0.1
DEFAULT_COLOR_ABSENCE_TOLERANCE = 0.5

_COLOR_NAMES_RE = "|".join(
    re.escape(c) for c in sorted(COLOR_DBZ_RANGE, key=len, reverse=True)
)
_COLOR_LIST_RE = (
    rf"((?:{_COLOR_NAMES_RE})"
    rf"(?:\s*(?:,\s*|\s+and\s+|\s+or\s+)(?:{_COLOR_NAMES_RE}))*)"
)
_COLOR_NAME_FINDER = re.compile(_COLOR_NAMES_RE, re.IGNORECASE)
_COLOR_ABSENCE_RE = re.compile(
    rf"\bno\s+{_COLOR_LIST_RE}\s*(?:colors?)?\b", re.IGNORECASE
)
_COLOR_EXCLUSIVE_RE = re.compile(
    rf"\bonly\s+have\s+{_COLOR_LIST_RE}\s*(?:colors?)?\b", re.IGNORECASE
)
_COLOR_PRESENCE_RE = re.compile(
    rf"\b(?:{_COLOR_LIST_RE}\s+(?:colors?\s+)?(?:are|must\s+be)\s+present"
    rf"|(?:must\s+have|have)\s+(?:regions?\s+of\s+)?{_COLOR_LIST_RE})\b",
    re.IGNORECASE,
)
_COLOR_DOMINANCE_RE = re.compile(
    rf"\b(?:over|more\s+than)\s+half\s+(?:of\s+the\s+image\s+)?must\s+be\s+{_COLOR_LIST_RE}\b",
    re.IGNORECASE,
)
_PERCENT_DOMINANCE_RE = re.compile(
    r"\b(?:over|more\s+than)\s+(\d+(?:\.\d+)?)\s*percent\b",
    re.IGNORECASE,
)


def _extract_colors(text):
    return [m.group(0).lower() for m in _COLOR_NAME_FINDER.finditer(text)]


def color_criteria_from_codebook(codebook_path,
                                 presence_threshold=DEFAULT_COLOR_PRESENCE_THRESHOLD,
                                 absence_tolerance=DEFAULT_COLOR_ABSENCE_TOLERANCE):
    """
    Parse color-based identification criteria from a LARS-format codebook.

    Each category description is split into sentences and matched against a
    small set of color phrasings:

    * **Exclusivity** — *"image will only have <colors>"* → all gates whose
      reflectivity exceeds the highest band of the listed colors must be
      ≤ ``absence_tolerance`` percent.
    * **Absence** — *"must have no <color>"* → ``pct_gates_<lo>dbz`` for that
      color must be ≤ ``absence_tolerance`` percent.
    * **Presence** — *"<colors> are present"* / *"must have [regions of]
      <colors>"* → ``pct_gates_<lo>dbz`` must be ≥ ``presence_threshold``
      percent for each listed color.
    * **Dominance** — *"Over half of the image must be <colors>"* → for
      low-reflectivity colors (blue/black) this maps to
      ``pct_gates_10dbz ≤ 50``.

    Color → dBZ band mapping is given by ``COLOR_DBZ_RANGE``, aligned with
    the default ``dbz_thresholds=(10, 20, 30, 40, 50)`` used by
    ``preprocess_radar_data``. Colors that fall in the low-reflectivity
    band (blue/black) cannot express absence or presence on their own and
    are skipped except for the exclusivity / dominance patterns.

    Parameters
    ----------
    codebook_path : str
    presence_threshold : float, optional
        Minimum ``pct_gates_*`` value to consider a color present.
    absence_tolerance : float, optional
        Maximum ``pct_gates_*`` value to consider a color absent.

    Returns
    -------
    dict
        Mapping of label → list of color rules. Each rule has:

        * ``kind`` (str): ``"min_pct_above"`` or ``"max_pct_above"``.
        * ``field`` (str): ``pct_gates_<T>dbz`` column name.
        * ``value`` (float): comparison threshold.
        * ``colors`` (list of str): colors involved.
        * ``phrase`` (str): the codebook sentence the rule was derived from.
    """
    categories = categories_from_codebook(codebook_path)
    out = {}
    for label, description in categories.items():
        rules = []
        for sentence in re.split(r"(?<=[.;])\s+", description):
            sentence = sentence.strip()
            if not sentence:
                continue

            m = _COLOR_EXCLUSIVE_RE.search(sentence)
            if m:
                colors = _extract_colors(m.group(1))
                allowed_max = max(
                    (COLOR_DBZ_RANGE[c][1] or 999)
                    for c in colors if c in COLOR_DBZ_RANGE
                )
                if allowed_max < 999:
                    rules.append({
                        "kind": "max_pct_above",
                        "field": f"pct_gates_{int(allowed_max)}dbz",
                        "value": absence_tolerance,
                        "colors": colors,
                        "phrase": sentence,
                    })
                continue

            m = _COLOR_ABSENCE_RE.search(sentence)
            if m:
                for c in _extract_colors(m.group(1)):
                    lo, _ = COLOR_DBZ_RANGE.get(c, (None, None))
                    if lo is None:
                        continue
                    rules.append({
                        "kind": "max_pct_above",
                        "field": f"pct_gates_{int(lo)}dbz",
                        "value": absence_tolerance,
                        "colors": [c],
                        "phrase": sentence,
                    })
                continue

            m = _COLOR_DOMINANCE_RE.search(sentence)
            if m:
                colors = _extract_colors(m.group(1))
                color_max_hi = max(
                    (COLOR_DBZ_RANGE[c][1] or 999)
                    for c in colors if c in COLOR_DBZ_RANGE
                )
                if color_max_hi < 999:
                    rules.append({
                        "kind": "max_pct_above",
                        "field": f"pct_gates_{int(color_max_hi)}dbz",
                        "value": 50.0,
                        "colors": colors,
                        "phrase": sentence,
                    })
                continue

            percent_m = _PERCENT_DOMINANCE_RE.search(sentence)
            if percent_m:
                colors = _extract_colors(sentence)
                color_max_hi = max(
                    (COLOR_DBZ_RANGE[c][1] or 999)
                    for c in colors if c in COLOR_DBZ_RANGE
                )
                if color_max_hi < 999:
                    rules.append({
                        "kind": "max_pct_above",
                        "field": f"pct_gates_{int(color_max_hi)}dbz",
                        "value": 50.0,
                        "colors": colors,
                        "phrase": sentence,
                    })
                continue

            m = _COLOR_PRESENCE_RE.search(sentence)
            if m:
                colors_text = next((g for g in m.groups() if g), "")
                for c in _extract_colors(colors_text):
                    lo, _ = COLOR_DBZ_RANGE.get(c, (None, None))
                    if lo is None:
                        continue
                    rules.append({
                        "kind": "min_pct_above",
                        "field": f"pct_gates_{int(lo)}dbz",
                        "value": presence_threshold,
                        "colors": [c],
                        "phrase": sentence,
                    })

        seen = set()
        unique = []
        for r in rules:
            key = (r["kind"], r["field"], r["value"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        if unique:
            out[label] = unique
    return out


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
CODEBOOK_CRITERIA = (
    criteria_from_codebook(_default_codebook_path)
    if os.path.exists(_default_codebook_path) else None
)
CODEBOOK_COLOR_CRITERIA = (
    color_criteria_from_codebook(_default_codebook_path)
    if os.path.exists(_default_codebook_path) else None
)
CODEBOOK_COLORMAP = (
    colormap_from_codebook(_default_codebook_path)
    if os.path.exists(_default_codebook_path) else None
)

async def label_radar_data(radar_df, model, categories=None, guidelines=None,
                           criteria=None, color_criteria=None,
                           mlflow_experiment=None, mlflow_run_name=None,
                           mlflow_tracking_uri=None, codebook_path=None,
                           site="Bankhead National Forest",
                           verbose=True, vmin=None, vmax=None, model_output_dir=None,
                           use_previous_labels=False):
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
    criteria (dict, optional): Hard quantitative criteria as returned by
        ``criteria_from_codebook``. When provided, any LLM label whose
        ``pct_gates_*`` / ``n_gates_*`` values violate the rules for that
        label is overridden in-place; the pre-override label and the rule
        that fired are recorded in ``llm_label_original`` and
        ``llm_label_criteria_violation``. Pass ``CODEBOOK_CRITERIA`` to
        enforce the bundled codebook.
    color_criteria (dict, optional): Color-based criteria as returned by
        ``color_criteria_from_codebook``. Used only for validation
        metric computation when ``mlflow_experiment`` is set; does not
        modify any labels. Pass ``CODEBOOK_COLOR_CRITERIA`` to evaluate
        against the bundled codebook.
    mlflow_experiment (str, optional): If provided, opens an MLflow run
        under this experiment and logs params, validation metrics
        (reflectivity-criteria + color-criteria violations and label
        agreement), the labelled CSV, the confusion matrix, and any raw
        model outputs in ``model_output_dir``. Requires the optional
        ``mlflow`` dependency.
    mlflow_run_name (str, optional): MLflow run name.
    mlflow_tracking_uri (str, optional): Forwarded to
        ``mlflow.set_tracking_uri``.
    codebook_path (str, optional): Path to the codebook used for this run;
        hashed and logged for traceability.
    site: str: Radar site identifier.
    vmin, vmax (float, optional): Bounds of the color scale described to the
        model in the prompt. When left as ``None`` and ``codebook_path`` is
        provided, they are read from the codebook's color-scale spec via
        ``colormap_from_codebook``; otherwise they fall back to
        ``DEFAULT_VMIN`` (-20) and ``DEFAULT_VMAX`` (60).
    model_output_dir: str: Directory to save model outputs.
    use_previous_labels: bool or int: If True, the function will use the previous *use_previous_labels* 
        labels as an additional input to the model for labeling. This can be useful if the model is being used to refine or validate existing labels.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the labeled radar data.
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES
    if (vmin is None or vmax is None) and codebook_path is not None:
        cmap = colormap_from_codebook(codebook_path)
        if vmin is None:
            vmin = cmap["vmin"]
        if vmax is None:
            vmax = cmap["vmax"]
    if vmin is None:
        vmin = DEFAULT_VMIN
    if vmax is None:
        vmax = DEFAULT_VMAX
    prompt = "This is an image of weather radar base reflectivity data." \
                f" The radar site is the ARM Facility {site} site." \
             " Please classify the weather depicted into one of the following categories: " \
             f"{', '.join(categories) if categories else ', '.join(categories)}."
    prompt += "Each category is defined as follows: "
    for category, description in categories.items():
        prompt += f"{category}: {description}; "
    prompt += f"The reflectivity values range from {vmin} dBZ as indicated by the blue colors to {vmax} dBZ as indicated by the red colors."
    for key in radar_df.columns:
        if key.startswith("pct_gates_") and key.endswith("dbz"):
            threshold = key[len("pct_gates_"):-len("dbz")]
            prompt += f" The percentage of gates with relfectivity above {threshold} dBZ is provided as {key} in the data."
        if key.startswith("n_gates_") and key.endswith("dbz"):
            threshold = key[len("n_gates_"):-len("dbz")]
            prompt += f" The number of gates with relfectivity above {threshold} dBZ is provided as {key} in the data."
    
    if guidelines:
        prompt += " When classifying, follow these annotator guidelines: "
        prompt += " ".join(guidelines)
    radar_df["llm_label"] = ""

    for fi in radar_df["file_path"].values:
        time = radar_df.loc[radar_df["file_path"] == fi, "time"].values[0]
        cur_index = radar_df.index[radar_df["file_path"] == fi][0]
        prompt_with_time = prompt + f"Please provide just the category label for the radar image taken at time {time}."      
        prompt_with_time = prompt_with_time + "Do not provide your reasoning for your selection, just the category."
        if use_previous_labels:
            for i in range(use_previous_labels):
                if cur_index - i - 1 >= 0:
                    prev_label = radar_df.loc[cur_index - i - 1, "label"]
                    prompt_with_time += f" The label for the previous radar image taken at time {radar_df.loc[cur_index - i - 1, 'time']} is {prev_label}."
                    
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

    if criteria:
        radar_df = apply_criteria_to_labels(radar_df, criteria,
                                            label_column="llm_label")

    if mlflow_experiment:
        from .tracking import log_run_to_mlflow
        log_run_to_mlflow(
            radar_df,
            experiment=mlflow_experiment,
            run_name=mlflow_run_name,
            tracking_uri=mlflow_tracking_uri,
            params={
                "model": getattr(model, "model_name", type(model).__name__),
                "site": site,
                "vmin": vmin,
                "vmax": vmax,
                "n_categories": len(categories),
                "criteria_enforced": criteria is not None,
            },
            criteria=criteria,
            color_criteria=color_criteria,
            codebook_path=codebook_path,
            model_output_dir=model_output_dir,
        )

    return radar_df