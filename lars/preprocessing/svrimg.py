"""Ingest for the SVRIMG severe convective mode benchmark.

SVRIMG (Haberlie, Ashley and Karpinski 2020) is a collection of hand-labelled
NEXRAD/GridRad column-maximum reflectivity images centred on SPC severe
weather reports. This module downloads those labels and images and emits them
in the same CSV schema as :func:`lars.preprocessing.preprocess_radar_data`, so
the existing inference and inter-annotator agreement tooling runs against
external ground truth without modification.

Two properties of the upstream format drive the implementation:

* The distributed PNGs are palette-mode images whose palette is a dummy linear
  ramp (index ``i`` maps to RGB ``(3i, 3i+1, 3i+2)``). The *palette indices*
  carry the data, which is why the files render as greyscale in an ordinary
  image viewer. :func:`read_svrimg_png` reads the indices, not the RGB.
* Those indices are reflectivity in dBZ, integer-truncated and floored at zero.
  The upstream GridRad ``REFC_MAX`` field is itself stored as ``uint8`` with
  ``units: dBZ``. There are no negative values, so the natural colour scale
  bounds are ``vmin=0``, ``vmax=80`` rather than the ``vmin=-20`` used for
  CSAPR2 imagery.

References
----------
Haberlie, A. M., W. S. Ashley, and M. Karpinski, 2020: Mean storms: Composites
of radar reflectivity images during two decades of severe thunderstorm events.
*International Journal of Climatology*.
"""

import os

import numpy as np
import pandas as pd
import requests
from PIL import Image

SVRIMG_BASE_URL = "https://nimbus.niu.edu/svrimg/data"

#: Canonical hand-classification table published upstream.
CLASSIFICATION_CSV = "sample_classifications_96-17.csv"

#: Grid is 136 x 136 at 3.75 km, i.e. roughly a 512 x 512 km box.
GRID_SIZE_PX = 136

#: Published train/validation/test split, keyed by inclusive year range.
SPLIT_YEARS = {
    "train": (1996, 2011),
    "val": (2012, 2013),
    "test": (2014, 2017),
}

#: The two convective mode classes retained by the binary collapse. The
#: upstream taxonomy also contains Tropical, Other, Noise and Missing.
BINARY_CLASSES = ("Cellular", "QLCS")

_DOWNLOAD_TIMEOUT = 120


def split_for_year(year):
    """
    Return the published SVRIMG split a given year belongs to.

    Parameters
    ----------
    year (int): Four-digit year.

    Returns
    -------
    str or None
        'train', 'val' or 'test', or None if the year falls outside the
        1996-2017 range covered by the dataset.
    """
    for split, (first, last) in SPLIT_YEARS.items():
        if first <= year <= last:
            return split
    return None


def _download(url, dest):
    """
    Download ``url`` to ``dest`` unless it already exists.

    Parameters
    ----------
    url (str): Source URL.
    dest (str): Destination path. Parent directories are created as needed.

    Returns
    -------
    str
        The destination path.
    """
    if os.path.exists(dest):
        return dest

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
    response.raise_for_status()

    # Write to a temporary name first so an interrupted download does not
    # leave a truncated file that later runs would treat as cached.
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
    os.replace(tmp, dest)

    return dest


def read_svrimg_png(path):
    """
    Read an SVRIMG PNG and return its reflectivity field in dBZ.

    SVRIMG PNGs are palette-mode images in which the palette is a meaningless
    linear ramp and the palette *indices* are the data. Converting such a file
    to RGB, as most image readers do by default, destroys the values. This
    function reads the indices directly and refuses any file that is not
    palette-mode, so a silently converted image fails loudly rather than
    producing plausible-looking nonsense.

    Parameters
    ----------
    path (str): Path to an SVRIMG .png file.

    Returns
    -------
    numpy.ndarray
        A (136, 136) uint8 array of reflectivity in dBZ. Zero means no echo.

    Raises
    ------
    ValueError
        If the image is not palette-mode.
    """
    with Image.open(path) as im:
        if im.mode != "P":
            raise ValueError(
                f"Expected a palette-mode ('P') SVRIMG PNG, got mode "
                f"'{im.mode}' for {path}. Reading this file as RGB would "
                f"discard the reflectivity values."
            )
        return np.array(im, dtype=np.uint8)


def reflectivity_stats(reflectivity, dbz_thresholds):
    """
    Compute reflectivity coverage statistics for a single SVRIMG scene.

    Mirrors the statistics that :func:`preprocess_radar_data` derives from
    CSAPR2 sweeps, so the resulting columns are directly comparable.

    Parameters
    ----------
    reflectivity (numpy.ndarray): Reflectivity in dBZ, zero meaning no echo.
    dbz_thresholds (sequence of float): Thresholds for which gate counts and
        coverage percentages are computed.

    Returns
    -------
    dict
        Mapping with keys 'ref_min', 'ref_max', and one
        'n_gates_<T>dbz'/'pct_gates_<T>dbz' pair per threshold. 'ref_min' and
        'ref_max' are computed over echoing gates only and are NaN for a scene
        with no echo at all.
    """
    reflectivity = np.asarray(reflectivity)
    echo = reflectivity[reflectivity > 0]

    stats = {
        "ref_min": float(echo.min()) if echo.size else float("nan"),
        "ref_max": float(echo.max()) if echo.size else float("nan"),
    }

    total_gates = reflectivity.size
    for threshold in dbz_thresholds:
        count = int(np.sum(reflectivity > threshold))
        stats[f"n_gates_{threshold}dbz"] = count
        stats[f"pct_gates_{threshold}dbz"] = round(count / total_gates * 100, 4)

    return stats


def fetch_svrimg_labels(data_dir, haz_type="tor", classes=BINARY_CLASSES,
                        base_url=SVRIMG_BASE_URL):
    """
    Download the SVRIMG hand classifications and join them to scene metadata.

    Parameters
    ----------
    data_dir (str): Directory in which downloaded tables are cached.
    haz_type (str): Hazard index to join against; 'tor', 'hail' or 'wind'.
        Default 'tor'.
    classes (sequence of str or None): Class names to retain. Defaults to
        ('Cellular', 'QLCS'), the binary convective mode collapse. Pass None
        to keep all six upstream classes.
    base_url (str): Base URL for the SVRIMG data server.

    Returns
    -------
    pd.DataFrame
        Indexed by 'unid', with columns 'label', 'radar_time', 'report_time',
        'split', and the upstream scene descriptors 'area', 'convection_area',
        'intense_area', 'max_intensity' and 'mean_intensity'.
    """
    class_csv = _download(
        f"{base_url}/{CLASSIFICATION_CSV}",
        os.path.join(data_dir, CLASSIFICATION_CSV),
    )
    index_name = f"96-17_{haz_type}_utc_svrimg_index.csv"
    index_csv = _download(
        f"{base_url}/{index_name}", os.path.join(data_dir, index_name)
    )

    labels = pd.read_csv(class_csv, index_col="UNID")
    labels = labels.rename(columns={"Class Name": "label"})
    labels.index.name = "unid"

    if classes is not None:
        labels = labels[labels["label"].isin(classes)]

    index = pd.read_csv(index_csv, index_col="unid")
    keep = [
        "radar_time", "report_time", "area", "convection_area",
        "intense_area", "max_intensity", "mean_intensity",
    ]
    joined = labels[["label"]].join(index[keep], how="inner")

    joined["radar_time"] = pd.to_datetime(
        joined["radar_time"], format="%m/%d/%Y %H:%M"
    )
    joined["report_time"] = pd.to_datetime(
        joined["report_time"], format="%m/%d/%Y %H:%M"
    )
    joined["split"] = joined["radar_time"].dt.year.map(split_for_year)

    return joined.sort_values("radar_time")


def svrimg_image_url(unid, haz_type="tor", base_url=SVRIMG_BASE_URL):
    """
    Build the upstream URL for a single SVRIMG scene.

    Parameters
    ----------
    unid (str): SVRIMG unique id, e.g. '199601180433z000000018'. The leading
        12 characters are YYYYMMDDHHMM.
    haz_type (str): 'tor', 'hail' or 'wind'. Default 'tor'.
    base_url (str): Base URL for the SVRIMG data server.

    Returns
    -------
    str
        Fully qualified URL of the scene's .png file.
    """
    year, month = unid[:4], unid[4:6]
    return f"{base_url}/{haz_type}/{year}/{month}/{unid}.png"


def preprocess_svrimg_data(output_path, data_dir, haz_type="tor",
                           classes=BINARY_CLASSES, splits=None,
                           dbz_thresholds=(10, 20, 30, 40, 50),
                           size_px=256, dpi=150, limit=None,
                           base_url=SVRIMG_BASE_URL, **kwargs):
    """
    Render SVRIMG scenes to labelled .png images for model evaluation.

    Downloads the SVRIMG classifications and the corresponding scenes, decodes
    each scene's reflectivity field, and re-renders it with a meteorological
    colormap so that a vision model sees imagery consistent with the codebook.
    Gates with no echo are masked rather than drawn as 0 dBZ.
    The returned DataFrame uses the same schema as
    :func:`preprocess_radar_data`, with 'unid' and 'split' added.

    Parameters
    ----------
    output_path (str): Directory in which rendered .png images are written.
    data_dir (str): Directory used to cache downloaded tables and source PNGs.
    haz_type (str): 'tor', 'hail' or 'wind'. Default 'tor'.
    classes (sequence of str or None): Class names to retain. Default is the
        binary ('Cellular', 'QLCS') collapse.
    splits (sequence of str or None): Restrict to these published splits, e.g.
        ('test',). Default None keeps all.
    dbz_thresholds (sequence of float): Reflectivity thresholds in dBZ for
        which gate counts and coverage percentages are computed.
    size_px (int): Width and height of the output PNG in pixels. Default 256.
    dpi (int): Dots per inch for the saved figure. Default 150.
    limit (int or None): Process at most this many scenes. Useful for smoke
        tests; default None processes everything.
    base_url (str): Base URL for the SVRIMG data server.

    **kwargs:

    Additional keyword arguments are passed to matplotlib's imshow. 'cmap',
    'vmin' and 'vmax' default to 'NWSRef', 0 and 80 respectively, matching
    CODEBOOK_SVRIMG.md.

    Returns
    -------
    label_df: pd.DataFrame
        DataFrame containing labels, paths and times, indexed by 'time'. Note
        that several reports can share one hourly radar volume, so the index
        is not unique; 'unid' and 'file_path' are the unique keys.
    """
    import matplotlib.pyplot as plt

    kwargs.setdefault("vmin", 0)
    kwargs.setdefault("vmax", 80)
    kwargs.setdefault("cmap", "NWSRef")
    if "ax" in kwargs:
        raise ValueError("Do not pass in an axis to this function.")

    labels = fetch_svrimg_labels(
        data_dir, haz_type=haz_type, classes=classes, base_url=base_url
    )
    if splits is not None:
        labels = labels[labels["split"].isin(splits)]
    if limit is not None:
        labels = labels.iloc[:limit]

    os.makedirs(output_path, exist_ok=True)

    dbz_thresholds = list(dbz_thresholds)
    count_cols = [f"n_gates_{t}dbz" for t in dbz_thresholds]
    pct_cols = [f"pct_gates_{t}dbz" for t in dbz_thresholds]
    columns = (["file_path", "time", "unid", "split", "label",
                "ref_min", "ref_max"] + count_cols + pct_cols)
    rows = []

    for unid, row in labels.iterrows():
        source = os.path.join(
            data_dir, haz_type, unid[:4], unid[4:6], f"{unid}.png"
        )
        try:
            _download(svrimg_image_url(unid, haz_type, base_url), source)
            reflectivity = read_svrimg_png(source)
        except (requests.HTTPError, ValueError) as err:
            print(f"Skipping {unid}: {err}")
            continue

        stats = reflectivity_stats(reflectivity, dbz_thresholds)

        # Zero means "no echo", not "0 dBZ". Mask it so those gates render as
        # blank background rather than as the bottom colour of the ramp, which
        # would fill the frame with apparent echo. This mirrors the
        # `where(field > min_ref)` masking in preprocess_radar_data.
        display = np.where(reflectivity > 0, reflectivity, np.nan)

        fig = plt.figure(figsize=(size_px / dpi, size_px / dpi))
        ax = plt.axes()
        # Row 0 of the stored array is the southern edge, so flip to draw
        # north-up. This matches the upstream svrimg draw_box_plot helper.
        ax.imshow(np.flipud(display), **kwargs)
        ax.axis("off")
        ax.set_title("")
        file_name = os.path.join(output_path, f"{unid}.png")
        fig.savefig(file_name, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        time_str = row["radar_time"].strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "file_path": file_name,
                "time": time_str,
                "unid": unid,
                "split": row["split"],
                "label": row["label"],
                **stats,
            }
        )

    out_df = pd.DataFrame(rows, columns=columns)
    out_df.set_index("time", inplace=True)

    return out_df.sort_index()
