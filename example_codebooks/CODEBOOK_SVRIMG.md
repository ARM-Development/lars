# Radar Image Labelling Codebook

A reference guide for annotators labelling radar imagery with LARS. Use this codebook to ensure consistent, reproducible labels across all annotators and sessions.

---

## 1. Overview

The purpose of this section is to label radar imagery by mode of convection

- **Radar type:** NEXRAD WSR-88D (GridRad three-dimensional merged mosaic)
- **Data source:** SVRIMG, scenes centred on SPC severe weather reports
- **Geographic scope:** Contiguous United States, 1996-2017
- **Labelling task:** convective mode classification

---

## 2. Data Description

### 2.1 Input Fields

| Field | Units | Description |
|--------------------------|-------|-------------------------------------|
| *reflectivity* | dBZ | Column-maximum intensity of returned radar signal |
| n_gates_30dBZ | percent | The percentage of gates greater than 30 dBZ |

### 2.2 Image Format

- **Spatial resolution:** 3.75 km by 3.75 km
- **Temporal resolution:** one scene per severe report, matched to the nearest hourly volume
- **Projection:** Lambert conformal conic, 136x136 grid covering approximately 512 km by 512 km
- **Color scale:** NWSRef colormap with vmin=0 and vmax=80

Source values are column-maximum reflectivity stored as integer dBZ and floored
at zero, so no negative reflectivity appears in these scenes. For reference, the
rendered colour bands are:

| Colour | Approximate reflectivity |
|--------|--------------------------|
| cyan to blue | 5-20 dBZ |
| green | 20-35 dBZ |
| yellow to orange | 35-50 dBZ |
| red | 50-65 dBZ |
| magenta to purple | 65-80 dBZ |

---

## 3. Label Classes

Each image or region-of-interest must be assigned exactly one primary class. 

### 3.1 Primary Classes

| Label | Description |
|------------------------|-----------------------------------------------------------------------------|
| Cellular | One or more discrete convective cores, each bounded on every side by weak or absent echo. Cores do not merge into a continuous band, and any echo bridging them is markedly weaker than the cores themselves. Most of the frame is free of echo. The percentage of gates greater than 30 dBZ must not exceed 6.5 percent. If it does exceed 6.5 percent, then classify as QLCS. |
| QLCS | A single continuous, quasi-linear band of strong echo at least roughly 100 km long, whose length is about three times its width or more. The band may be bowed or curved, and may trail a broad region of weaker stratiform echo behind it. Short breaks along the band are acceptable provided the overall structure still reads as one line rather than a field of separate cells. |
| Ambiguous / Uncertain | Cannot be classified with confidence. Use for transitional structures that are neither clearly one line nor clearly separate cells, and for scenes degraded by missing radar coverage. |

> **Note on enforcement.** Only the quantitative reflectivity thresholds in the
> descriptions above (e.g. "percentage of gates greater than X dBZ must not
> exceed Y percent") are validated automatically, and only when the
> corresponding `pct_gates_*` / `n_gates_*` columns are present in the data.
> Spatial and topological criteria — "bounded on every side by…", "continuous
> band", "length is about three times its width" — are judged by the annotator
> or model and are **not** checked programmatically.

> **Note on colour criteria.** This codebook deliberately describes classes in
> reflectivity and topology rather than colour. `COLOR_DBZ_RANGE` in
> `lars/nepho/inference.py` is calibrated against a ChaseSpectral-like ramp
> (green 10-30, yellow 30-40, red 40-50, pink 50+), which does not match the
> NWSRef bands tabulated in Section 2.2. Colour phrasings here would therefore
> be validated against the wrong reflectivity bands.

---

## 5. Labelling Procedure

1. Use :code:`lars.preprocessing.preprocess_svrimg_data` to download scenes and generate images and a .csv file
2. The csv file is populated with the published SVRIMG hand labels in the 'label' column, and a 'split' column giving the published train/validation/test assignment.
3. According to the criteria above, label all images in the 'file_path' column of the .csv file.

---

## 6. Annotator Guidelines

The bullets in this section are passed verbatim to automated labelling models,
so they must be self-contained for a single image with no external context.

- Every image is centred on a confirmed severe weather report, so an empty scene is never the right answer; if echo is present but its organisation is unclear, classify as Ambiguous / Uncertain.
- Judge the organisation of the whole image, not only the structures at its centre.
- If a line and separate cells are both present, classify according to whichever structure covers the larger area of the image.
- A broad region of weak echo on its own is not a class here; classify by the arrangement of the strong echo embedded within it.

### 6.1 Human Annotators Only

These apply to human annotators and the review process. They are intentionally
kept out of the bullet list above because an automated model labels each image
independently, with no temporal context and no access to the example gallery.

- Use the provided example gallery (Section 8) to calibrate your judgement.
- Inter-annotator agreement should be checked periodically; raise disagreements with the team lead.

---

## 7. Quality Control

| Check | Method |
|-------|--------|
| Completeness | All images have a primary label |
| Consistency | Random sample reviewed by second annotator |
| Agreement metric | Cohen's κ computed per annotator pair |
| Class balance | Report balanced accuracy and per-class recall; the published split carries a prior shift (train 54% Cellular, test 66% QLCS) |
| Outlier review | Labels deviating from model predictions flagged for review |

---

## 8. Example Gallery

Representative training scenes, chosen as the cases nearest each class median
for 30 dBZ coverage. Render them with
:code:`lars.preprocessing.preprocess_svrimg_data`; the source files are
palette-encoded and appear greyscale if opened directly.

| Class | Representative UNID | pct_gates_30dBZ | Notes |
|--------------------------|-----------------------------------|--------|--------------------------|
| Cellular | 200207250114z000000223 | 3.31 | Class median; discrete cores, mostly echo-free frame |
| Cellular | 200805282245z000000948 | 3.30 | Intense core (71 dBZ) yet still low total coverage |
| QLCS | 200210291000z000000352 | 14.33 | Class median; continuous band |
| QLCS | 201104042340z000292240 | 14.33 | Band with trailing stratiform region |

---

## 9. Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-09-14 | Robert Jackson | Initial release. Adapts CODEBOOK_NWSRef.md to the SVRIMG benchmark: binary Cellular/QLCS collapse (1,315 of 1,741 labelled scenes), vmin 0 rather than -20 because SVRIMG reflectivity is floored at zero, and colour criteria replaced by reflectivity/topology criteria. Thresholds calibrated on the 997-scene training split (see notebooks/reports/calibrate_svrimg_thresholds.py): pct_gates_30dBZ is the strongest single discriminator at AUC 0.935, Youden cut 6.48 percent (sensitivity 0.89, specificity 0.82), adopted as 6.5 percent. Also measured but not adopted, to avoid redundant correlated rules: pct_gates_20dBZ AUC 0.929, pct_gates_10dBZ AUC 0.903, pct_gates_40dBZ AUC 0.889. pct_gates_50dBZ does not discriminate (AUC 0.601) and pct_gates_60dBZ is weak and degenerate (AUC 0.684 at a 0.0 percent cut); neither is used. Caveat: these coverage fractions largely measure echo *extent* rather than linearity, so they are expected to misfire on large non-linear convective clusters. |

---

## 10. References

- Haberlie, A. M., W. S. Ashley, and M. Karpinski (2020). Mean storms: Composites of radar reflectivity images during two decades of severe thunderstorm events. *International Journal of Climatology*.
- Bowman, K. P., and C. R. Homeyer (2017). GridRad - Three-Dimensional Gridded NEXRAD WSR-88D Radar Data. Research Data Archive at NCAR.
- American Meteorological Society Glossary: https://glossary.ametsoc.org
