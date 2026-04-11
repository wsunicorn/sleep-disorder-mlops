"""
Feature Engineering — Trích xuất đặc trưng từ epochs EEG.
Features: Band power, Spectral entropy, HRV, Nonlinear features.
Output: Parquet file với mỗi hàng là một epoch.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy
from loguru import logger
from tqdm import tqdm

# Frequency bands (Hz)
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """Tính power trong một dải tần từ PSD."""
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if idx.sum() == 0:
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))


def spectral_entropy(psd: np.ndarray) -> float:
    """Entropy của phổ tần số (đo độ phức tạp tín hiệu)."""
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(scipy_entropy(psd_norm + 1e-12))


def peak_frequency(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Tần số có power lớn nhất."""
    return float(freqs[np.argmax(psd)])


def mean_frequency(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Tần số trung bình có trọng số."""
    total = psd.sum() + 1e-12
    return float(np.sum(freqs * psd) / total)


def extract_epoch_features(epoch: np.ndarray, sfreq: float) -> dict:
    """
    Trích xuất đặc trưng từ một epoch.

    Args:
        epoch: numpy array shape (n_channels, n_samples)
        sfreq: sampling frequency in Hz

    Returns:
        dict với tất cả features
    """
    features = {}
    n_channels, n_samples = epoch.shape

    # Tính PSD cho từng channel dùng Welch method
    all_band_powers = {band: [] for band in FREQ_BANDS}
    all_spec_entropy = []
    all_peak_freq = []
    all_mean_freq = []

    for ch_idx in range(n_channels):
        ch_data = epoch[ch_idx, :]
        nperseg = min(256, n_samples)
        freqs, psd = scipy_signal.welch(ch_data, fs=sfreq, nperseg=nperseg)

        # Band powers
        total_power = bandpower(psd, freqs, 0.5, 40.0) + 1e-12
        for band_name, (fmin, fmax) in FREQ_BANDS.items():
            bp = bandpower(psd, freqs, fmin, fmax)
            all_band_powers[band_name].append(bp)
            # Relative power
            features[f"ch{ch_idx}_{band_name}_rel"] = bp / total_power

        all_spec_entropy.append(spectral_entropy(psd))
        all_peak_freq.append(peak_frequency(psd, freqs))
        all_mean_freq.append(mean_frequency(psd, freqs))

    # Giá trị trung bình qua các channels
    for band_name in FREQ_BANDS:
        features[f"{band_name}_power_mean"] = float(np.mean(all_band_powers[band_name]))
        features[f"{band_name}_power_std"] = float(np.std(all_band_powers[band_name]))

    features["spectral_entropy_mean"] = float(np.mean(all_spec_entropy))
    features["peak_frequency_mean"] = float(np.mean(all_peak_freq))
    features["mean_frequency_mean"] = float(np.mean(all_mean_freq))

    # Thống kê thời gian
    flat = epoch.flatten()
    features["amplitude_mean"] = float(np.mean(np.abs(flat)))
    features["amplitude_std"] = float(np.std(flat))
    features["rms"] = float(np.sqrt(np.mean(flat ** 2)))

    # Tỉ số delta/beta (marker buồn ngủ)
    delta_mean = features["delta_power_mean"]
    beta_mean = features["beta_power_mean"] + 1e-12
    features["delta_beta_ratio"] = delta_mean / beta_mean

    # Theta/alpha ratio
    theta_mean = features["theta_power_mean"]
    alpha_mean = features["alpha_power_mean"] + 1e-12
    features["theta_alpha_ratio"] = theta_mean / alpha_mean

    return features


def get_label_from_filename(subject_id: str) -> str:
    """Lấy nhãn bệnh lý từ tên file."""
    prefixes = {
        "n": "healthy",
        "nfle": "nfle",
        "rbd": "rbd",
        "plm": "plm",
        "ins": "insomnia",
        "narco": "narcolepsy",
        "sdb": "sdb",
        "brux": "bruxism",
    }
    for prefix, label in sorted(prefixes.items(), key=lambda x: -len(x[0])):
        if subject_id.startswith(prefix):
            return label
    return "unknown"


def process_npz_file(npz_path: Path) -> pd.DataFrame:
    """Xử lý một file .npz → DataFrame features."""
    data = np.load(npz_path, allow_pickle=True)
    epochs = data["epochs"]          # (n_epochs, n_channels, n_samples)
    valid_mask = data["valid_mask"]  # (n_epochs,)
    sfreq = float(data["sfreq"])
    subject_id = str(data["subject_id"])
    label = get_label_from_filename(subject_id)

    rows = []
    for i, (epoch, is_valid) in enumerate(zip(epochs, valid_mask)):
        if not is_valid:
            continue
        try:
            feat = extract_epoch_features(epoch, sfreq)
            feat["epoch_index"] = i
            feat["subject_id"] = subject_id
            feat["label"] = label
            rows.append(feat)
        except Exception as e:
            logger.warning(f"Failed epoch {i} of {subject_id}: {e}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="EEG Feature Extractor")
    parser.add_argument("--input-dir", required=True, help="Dir with .npz files")
    parser.add_argument("--output-dir", required=True, help="Output dir for parquet")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*_epochs.npz"))
    if not npz_files:
        logger.error(f"No .npz files found in {input_dir}")
        return

    logger.info(f"Found {len(npz_files)} epoch files.")
    all_dfs = []

    for npz_file in tqdm(npz_files, desc="Extracting features"):
        try:
            df = process_npz_file(npz_file)
            all_dfs.append(df)
            logger.info(f"{npz_file.stem}: {len(df)} valid epochs")
        except Exception as e:
            logger.error(f"Failed {npz_file.name}: {e}")

    if not all_dfs:
        logger.error("No features extracted.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = output_dir / "features.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(combined)} epochs → {output_path}")
    logger.info(f"Label distribution:\n{combined['label'].value_counts()}")


if __name__ == "__main__":
    main()
