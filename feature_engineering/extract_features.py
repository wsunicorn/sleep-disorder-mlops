"""Extract notebook-standard CAP Sleep features from preprocessed EEG epochs.

The project model is trained from notebooks/kaggle_cap_training.ipynb, so this
script now emits the same 24-feature schema used by that notebook and by the
serving API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

try:
    from feature_engineering.cap_features import (
        FEATURE_NAMES,
        WINDOW_SEC,
        bandpower,
        disease_label_from_subject,
        extract_feature_matrix,
        extract_features,
    )
except ModuleNotFoundError:  # Allows running this file directly from its folder.
    from cap_features import (  # type: ignore
        FEATURE_NAMES,
        WINDOW_SEC,
        bandpower,
        disease_label_from_subject,
        extract_feature_matrix,
        extract_features,
    )


def get_label_from_filename(subject_id: str) -> str:
    """Return the 7-class disorder label inferred from a CAP subject id."""
    return disease_label_from_subject(subject_id)


def _select_channel(epoch: np.ndarray) -> np.ndarray:
    """Use the first EEG channel to match the single-channel Kaggle dataset."""
    arr = np.asarray(epoch)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] > 0:
        return arr[0]
    raise ValueError(f"Unsupported epoch shape: {arr.shape}")


def process_npz_file(npz_path: Path) -> pd.DataFrame:
    """Convert one preprocessed .npz file into notebook-standard features."""
    data = np.load(npz_path, allow_pickle=True)
    epochs = data["epochs"]
    valid_mask = data["valid_mask"]
    sfreq = float(data["sfreq"])
    subject_id = str(data["subject_id"])
    label = get_label_from_filename(subject_id)
    window_samples = int(WINDOW_SEC * sfreq)

    rows: list[dict[str, object]] = []
    feature_row_index = 0
    for epoch_index, (epoch, is_valid) in enumerate(zip(epochs, valid_mask)):
        if not is_valid:
            continue

        try:
            signal = _select_channel(epoch)
            matrix = extract_feature_matrix(
                signal,
                sfreq=sfreq,
                window_samples=window_samples,
            )
            for subwindow_index, values in enumerate(matrix):
                row = dict(zip(FEATURE_NAMES, values.astype(float).tolist()))
                row["epoch_index"] = feature_row_index
                row["source_epoch_index"] = int(epoch_index)
                row["subwindow_index"] = int(subwindow_index)
                row["subject_id"] = subject_id
                row["disease"] = label
                rows.append(row)
                feature_row_index += 1
        except Exception as exc:
            logger.warning(f"Failed epoch {epoch_index} of {subject_id}: {exc}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CAP Sleep 24-feature extractor")
    parser.add_argument("--input-dir", required=True, help="Directory with *_epochs.npz files")
    parser.add_argument("--output-dir", required=True, help="Directory for features.parquet")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*_epochs.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_epochs.npz files found in {input_dir}")

    all_dfs = []
    for npz_file in tqdm(npz_files, desc="Extracting CAP features"):
        df = process_npz_file(npz_file)
        if not df.empty:
            all_dfs.append(df)
        logger.info(f"{npz_file.stem}: {len(df)} feature windows")

    if not all_dfs:
        raise RuntimeError("No features extracted.")

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = output_dir / "features.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(combined)} windows to {output_path}")
    if "disease" in combined:
        logger.info(f"Label distribution:\n{combined['disease'].value_counts()}")


if __name__ == "__main__":
    main()
