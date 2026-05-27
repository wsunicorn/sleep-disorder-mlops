import pandas as pd

from feature_engineering.cap_features import DISEASE_FILES, FEATURE_NAMES
from training.train import load_training_data, summarize_training_frame


def _rows(labels, *, verified=None):
    rows = []
    for index, label in enumerate(labels):
        row = {name: float(index + offset) for offset, name in enumerate(FEATURE_NAMES)}
        row["disease"] = label
        if verified is not None:
            row["training_approved"] = verified
        rows.append(row)
    return rows


def test_unverified_extra_data_is_skipped(tmp_path):
    labels = list(DISEASE_FILES)
    base_path = tmp_path / "base.parquet"
    extra_path = tmp_path / "extra.parquet"
    pd.DataFrame(_rows(labels)).to_parquet(base_path, index=False)
    pd.DataFrame(_rows(labels)).to_parquet(extra_path, index=False)

    x, _y, _encoder, df = load_training_data(
        str(base_path),
        class_limits={},
        extra_data=[str(extra_path)],
        extra_data_policy="verified_only",
    )

    assert len(x) == len(labels)
    assert summarize_training_frame(df)["source_counts"] == {"base": len(labels)}


def test_verified_extra_data_is_appended(tmp_path):
    labels = list(DISEASE_FILES)
    base_path = tmp_path / "base.parquet"
    extra_path = tmp_path / "extra.parquet"
    pd.DataFrame(_rows(labels)).to_parquet(base_path, index=False)
    pd.DataFrame(_rows(labels, verified=True)).to_parquet(extra_path, index=False)

    x, _y, _encoder, df = load_training_data(
        str(base_path),
        class_limits={},
        extra_data=[str(extra_path)],
        extra_data_policy="verified_only",
    )

    assert len(x) == len(labels) * 2
    assert summarize_training_frame(df)["source_counts"] == {
        "base": len(labels),
        "extra_verified": len(labels),
    }
