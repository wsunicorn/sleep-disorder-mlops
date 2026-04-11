"""
Feature Engineering — Tiền xử lý tín hiệu EEG từ .edf files.
- Bandpass filter 0.5–40 Hz
- Notch filter 50 Hz
- Cắt epoch 30 giây
- Lưu ra data/processed/ dưới dạng numpy .npz
"""

import os
import argparse
import numpy as np
import mne
from pathlib import Path
from loguru import logger
from tqdm import tqdm

EPOCH_DURATION_SEC = 30
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
NOTCH_FREQ = 50.0
ARTIFACT_THRESHOLD = 150e-6  # 150 µV

# Channels ưu tiên (EEG) — chọn subset nếu có
EEG_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2",
                 "Fp1", "Fp2", "EEG F3-A2", "EEG C3-A2", "EEG O1-A2"]


def preprocess_edf(edf_path: str, output_dir: str):
    """
    Xử lý một file .edf, lưu epochs ra thư mục output.
    """
    edf_path = Path(edf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = edf_path.stem  # e.g. "n1"
    output_file = output_dir / f"{subject_id}_epochs.npz"

    if output_file.exists():
        logger.info(f"Already processed: {subject_id}. Skipping.")
        return

    logger.info(f"Processing {edf_path.name}...")

    # Load raw EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Chọn chỉ EEG channels (nếu có)
    available = raw.ch_names
    eeg_picks = [ch for ch in EEG_CHANNELS if ch in available]
    if not eeg_picks:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads").tolist()
        logger.warning(f"Using all EEG channels: {len(eeg_picks)} channels")

    if eeg_picks:
        raw.pick(eeg_picks)

    # Notch filter — chỉ áp dụng nếu Nyquist > notch frequency
    nyquist = raw.info["sfreq"] / 2.0
    if nyquist > NOTCH_FREQ:
        raw.notch_filter(NOTCH_FREQ, verbose=False)
    else:
        logger.info(f"Skipping notch filter: Nyquist ({nyquist} Hz) ≤ {NOTCH_FREQ} Hz")

    # Bandpass filter — h_freq không được vượt quá Nyquist
    h_freq = min(BANDPASS_HIGH, nyquist - 0.5)
    raw.filter(
        l_freq=BANDPASS_LOW,
        h_freq=h_freq,
        method="fir",
        fir_window="hamming",
        verbose=False,
    )

    # Cắt thành epochs cố định
    sfreq = raw.info["sfreq"]
    n_samples_epoch = int(EPOCH_DURATION_SEC * sfreq)
    data, _ = raw[:, :]  # (n_channels, n_times)

    n_epochs = data.shape[1] // n_samples_epoch
    epochs_list = []
    valid_mask = []

    for i in range(n_epochs):
        start = i * n_samples_epoch
        epoch = data[:, start: start + n_samples_epoch]

        # Artifact rejection: loại epoch có biên độ quá cao
        if np.max(np.abs(epoch)) > ARTIFACT_THRESHOLD:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
        epochs_list.append(epoch.astype(np.float32))

    epochs_array = np.stack(epochs_list)  # (n_epochs, n_channels, n_samples)
    valid_mask = np.array(valid_mask)

    np.savez_compressed(
        output_file,
        epochs=epochs_array,
        valid_mask=valid_mask,
        sfreq=sfreq,
        channel_names=np.array(raw.ch_names),
        subject_id=subject_id,
    )

    n_valid = valid_mask.sum()
    logger.info(
        f"Saved {n_epochs} epochs ({n_valid} valid, "
        f"{n_epochs - n_valid} rejected) → {output_file}"
    )


def main():
    parser = argparse.ArgumentParser(description="EDF Preprocessor")
    parser.add_argument("--input-dir", required=True, help="Dir with .edf files")
    parser.add_argument("--output-dir", required=True, help="Output dir for .npz files")
    args = parser.parse_args()

    edf_files = list(Path(args.input_dir).glob("*.edf"))
    if not edf_files:
        logger.error(f"No .edf files found in {args.input_dir}")
        return

    logger.info(f"Found {len(edf_files)} .edf files.")
    for edf_file in tqdm(edf_files, desc="Preprocessing"):
        try:
            preprocess_edf(str(edf_file), args.output_dir)
        except Exception as e:
            logger.error(f"Failed {edf_file.name}: {e}")


if __name__ == "__main__":
    main()
