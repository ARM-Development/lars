import os
import pytest
import pandas as pd


def _write_csv(path, contents):
    with open(path, "w") as f:
        f.write(contents)
    return str(path)


def test_combine_labels_matches_on_file_path(tmp_path):
    from lars.preprocessing.labels import combine_labels

    human_csv = _write_csv(
        tmp_path / "human.csv",
        "file_path,label\n"
        "/a/img1.png,No Precipitation\n"
        "/a/img2.png,Stratiform Precipitation\n",
    )
    llm_csv = _write_csv(
        tmp_path / "llm.csv",
        "file_path,label\n"
        "/b/img1.png,No Precipitation\n"
        "/b/img2.png,Isolated Convection\n",
    )

    combined = combine_labels(
        [human_csv, llm_csv], ["human_alice", "llm_l4scout"]
    )

    assert list(combined.columns) == ["file_path", "label", "source"]
    assert len(combined) == 4
    assert set(combined["source"]) == {"human_alice", "llm_l4scout"}
    assert set(combined["file_path"]) == {"img1.png", "img2.png"}


def test_combine_labels_matches_on_time(tmp_path):
    from lars.preprocessing.labels import combine_labels

    human_csv = _write_csv(
        tmp_path / "human.csv",
        "time,label\n3/4/25 0:00,No Precipitation\n3/4/25 0:12,Isolated Convection\n",
    )
    llm_csv = _write_csv(
        tmp_path / "llm.csv",
        "time,label\n3/4/25 0:00,No Precipitation\n3/4/25 0:12,Stratiform Precipitation\n",
    )

    combined = combine_labels(
        [human_csv, llm_csv], ["human_bob", "llm_l4scout"], match_on="time"
    )

    assert list(combined.columns) == ["time", "label", "source"]
    assert len(combined) == 4
    assert set(combined["time"]) == {"3/4/25 0:00", "3/4/25 0:12"}


def test_combine_labels_custom_label_column(tmp_path):
    from lars.preprocessing.labels import combine_labels

    csv1 = _write_csv(
        tmp_path / "a.csv",
        "file_path,llm_label\n/a/img1.png,No Precipitation\n",
    )
    csv2 = _write_csv(
        tmp_path / "b.csv",
        "file_path,llm_label\n/b/img1.png,Isolated Convection\n",
    )

    combined = combine_labels(
        [csv1, csv2], ["run1", "run2"], label_column="llm_label"
    )

    assert list(combined.columns) == ["file_path", "llm_label", "source"]
    assert combined["llm_label"].tolist() == [
        "No Precipitation", "Isolated Convection"
    ]


def test_combine_labels_mismatched_lengths_raises(tmp_path):
    from lars.preprocessing.labels import combine_labels

    csv1 = _write_csv(tmp_path / "a.csv", "file_path,label\n/a/img1.png,x\n")

    with pytest.raises(ValueError):
        combine_labels([csv1], ["only_one", "too_many"])


def test_combine_labels_invalid_match_on_raises(tmp_path):
    from lars.preprocessing.labels import combine_labels

    csv1 = _write_csv(tmp_path / "a.csv", "file_path,label\n/a/img1.png,x\n")

    with pytest.raises(ValueError):
        combine_labels([csv1], ["source1"], match_on="bogus")


def test_standardize_labels_maps_ambiguous_and_unknown_variants():
    from lars.preprocessing.labels import standardize_labels

    df = pd.DataFrame({
        "label": ["Ambiguous", "UNKNOWN", "unknown", " Unknown ", "ambiguous"],
    })

    result = standardize_labels(df)

    assert result["label"].tolist() == ["Ambiguous / Uncertain"] * 5


def test_standardize_labels_maps_bare_stratiform():
    from lars.preprocessing.labels import standardize_labels

    df = pd.DataFrame({"label": ["Stratiform", "stratiform", "STRATIFORM"]})

    result = standardize_labels(df)

    assert result["label"].tolist() == ["Stratiform Precipitation"] * 3


def test_standardize_labels_leaves_canonical_and_other_labels_unchanged():
    from lars.preprocessing.labels import standardize_labels

    df = pd.DataFrame({
        "label": [
            "No Precipitation",
            "Stratiform Precipitation",
            "Isolated Convection",
            "Mesoscale Convective System",
        ],
    })

    result = standardize_labels(df)

    assert result["label"].tolist() == df["label"].tolist()


def test_standardize_labels_preserves_missing_values():
    from lars.preprocessing.labels import standardize_labels

    df = pd.DataFrame({"label": ["Ambiguous", None]})

    result = standardize_labels(df)

    assert result["label"].iloc[0] == "Ambiguous / Uncertain"
    assert pd.isna(result["label"].iloc[1])


def test_standardize_labels_custom_label_column():
    from lars.preprocessing.labels import standardize_labels

    df = pd.DataFrame({"llm_label": ["UNKNOWN", "Stratiform"]})

    result = standardize_labels(df, label_column="llm_label")

    assert result["llm_label"].tolist() == [
        "Ambiguous / Uncertain", "Stratiform Precipitation"
    ]


def test_standardize_labels_does_not_mutate_input():
    from lars.preprocessing.labels import standardize_labels

    df = pd.DataFrame({"label": ["UNKNOWN"]})

    standardize_labels(df)

    assert df["label"].tolist() == ["UNKNOWN"]
