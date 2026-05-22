"""CAP Sleep feature schema shared by training, API inference, and demos.

This module mirrors the feature extraction logic used in
notebooks/kaggle_cap_training.ipynb:
- 1-D EEG windows
- 512 Hz sampling rate
- 1024 samples per window
- 24 handcrafted spectral/statistical/Hjorth features
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew

SFREQ = 512
WINDOW_SAMPLES = 1024
WINDOW_SEC = WINDOW_SAMPLES / SFREQ

FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

FEATURE_NAMES = [
    "delta_power",
    "delta_rel",
    "theta_power",
    "theta_rel",
    "alpha_power",
    "alpha_rel",
    "beta_power",
    "beta_rel",
    "gamma_power",
    "gamma_rel",
    "spectral_entropy",
    "peak_frequency",
    "mean_frequency",
    "amplitude_mean",
    "amplitude_std",
    "rms",
    "delta_beta_ratio",
    "theta_alpha_ratio",
    "skewness",
    "kurtosis",
    "zero_crossing_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
]

DISEASE_FILES = {
    "healthy": "bal_healthy.csv",
    "nfle": "bal_nfle.csv",
    "insomnia": "bal_ins.csv",
    "narcolepsy": "bal_narco.csv",
    "plm": "bal_plm.csv",
    "rbd": "bal_rbd.csv",
    "sdb": "bal_sdb.csv",
}

NOTEBOOK_MAX_PER_CLASS = {
    "healthy": None,
    "nfle": 20_000,
    "insomnia": None,
    "narcolepsy": None,
    "plm": None,
    "rbd": 20_000,
    "sdb": None,
}

RELATIVE_FEATURE_INDICES = [1, 3, 5, 7, 9]
CLIP_LOW = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.nan,
        np.nan,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


def bandpower(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """Integrate power spectral density over one frequency band."""
    idx = (freqs >= lo) & (freqs <= hi)
    if not idx.any():
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))


def extract_features(window: np.ndarray, sfreq: float = SFREQ) -> dict[str, float]:
    """Extract the notebook-standard 24 features from one 1-D EEG window."""
    w = np.asarray(window, dtype=np.float64).reshape(-1)
    if w.size < 2:
        raise ValueError("A feature window must contain at least 2 samples.")

    freqs, psd = scipy_signal.welch(w, fs=sfreq, nperseg=min(256, len(w)))
    total = bandpower(psd, freqs, 0.5, 40.0) + 1e-12

    features: dict[str, float] = {}
    for band, (lo, hi) in FREQ_BANDS.items():
        power = bandpower(psd, freqs, lo, hi)
        features[f"{band}_power"] = power
        features[f"{band}_rel"] = power / total

    psd_norm = psd / (psd.sum() + 1e-12)
    features["spectral_entropy"] = float(scipy_entropy(psd_norm + 1e-12))
    features["peak_frequency"] = float(freqs[np.argmax(psd)])
    features["mean_frequency"] = float(np.sum(freqs * psd) / (psd.sum() + 1e-12))

    features["amplitude_mean"] = float(np.mean(np.abs(w)))
    features["amplitude_std"] = float(np.std(w))
    features["rms"] = float(np.sqrt(np.mean(w**2)))

    features["delta_beta_ratio"] = features["delta_power"] / (
        features["beta_power"] + 1e-12
    )
    features["theta_alpha_ratio"] = features["theta_power"] / (
        features["alpha_power"] + 1e-12
    )

    features["skewness"] = float(skew(w))
    features["kurtosis"] = float(scipy_kurtosis(w))
    features["zero_crossing_rate"] = float(np.mean(np.diff(np.sign(w)) != 0))

    d1 = np.diff(w)
    d2 = np.diff(d1)
    var0 = np.var(w) + 1e-12
    var1 = np.var(d1) + 1e-12
    var2 = np.var(d2) + 1e-12
    features["hjorth_activity"] = float(var0)
    features["hjorth_mobility"] = float(np.sqrt(var1 / var0))
    features["hjorth_complexity"] = float(np.sqrt(var2 / var1) / np.sqrt(var1 / var0))

    return {name: float(features[name]) for name in FEATURE_NAMES}


def extract_feature_vector(window: np.ndarray, sfreq: float = SFREQ) -> list[float]:
    """Return features in the exact order expected by the exported model."""
    features = extract_features(window, sfreq=sfreq)
    return [features[name] for name in FEATURE_NAMES]


def iter_windows(
    signal: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
) -> list[np.ndarray]:
    """Split a 1-D signal into full non-overlapping windows."""
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    n_windows = arr.size // window_samples
    return [
        arr[index * window_samples : (index + 1) * window_samples]
        for index in range(n_windows)
    ]


def extract_feature_matrix(
    signal: np.ndarray,
    sfreq: float = SFREQ,
    window_samples: int | None = None,
) -> np.ndarray:
    """Extract a 2-D matrix of notebook-standard features from a 1-D signal."""
    samples = window_samples or int(WINDOW_SEC * sfreq)
    rows = [extract_feature_vector(window, sfreq=sfreq) for window in iter_windows(signal, samples)]
    return np.asarray(rows, dtype=np.float32)


def disease_label_from_subject(subject_id: str) -> str:
    """Infer the 7-class disorder label from CAP-style subject/file names."""
    normalized = subject_id.lower()
    prefixes = {
        "narco": "narcolepsy",
        "nfle": "nfle",
        "ins": "insomnia",
        "plm": "plm",
        "rbd": "rbd",
        "sdb": "sdb",
        "n": "healthy",
    }
    for prefix, label in sorted(prefixes.items(), key=lambda item: -len(item[0])):
        if normalized.startswith(prefix):
            return label
    return "unknown"


def load_balanced_cap_dataset(
    data_dir: str | Path,
    max_per_class: Mapping[str, int | None] | None = None,
    *,
    synthetic_if_missing: bool = False,
    synthetic_per_class: int = 300,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load Balanced CAP CSV files and extract notebook-standard features."""
    data_path = Path(data_dir)
    limits = dict(NOTEBOOK_MAX_PER_CLASS)
    if max_per_class:
        limits.update(max_per_class)

    all_dfs: list[pd.DataFrame] = []
    found_any = any((data_path / filename).exists() for filename in DISEASE_FILES.values())
    if not found_any:
        if not synthetic_if_missing:
            raise FileNotFoundError(
                f"No Balanced CAP CSV files found in {data_path}. "
                "Expected files such as bal_healthy.csv and bal_ins.csv."
            )
        rng = np.random.default_rng(random_seed)
        for index, label in enumerate(DISEASE_FILES):
            scale = 1.0 + index * 0.25
            rows = [
                extract_features(rng.normal(0.0, scale, WINDOW_SAMPLES)) | {"disease": label}
                for _ in range(synthetic_per_class)
            ]
            all_dfs.append(pd.DataFrame(rows))
        return pd.concat(all_dfs, ignore_index=True)

    for label, filename in DISEASE_FILES.items():
        csv_path = data_path / filename
        if not csv_path.exists():
            continue

        nrows = limits.get(label)
        raw_df = pd.read_csv(csv_path, nrows=nrows, header=0)
        windows = raw_df.iloc[:, :WINDOW_SAMPLES].to_numpy(dtype=np.float64)
        rows = [extract_features(windows[row_index]) | {"disease": label} for row_index in range(len(windows))]
        all_dfs.append(pd.DataFrame(rows))

    if not all_dfs:
        raise FileNotFoundError(f"No usable Balanced CAP CSV files found in {data_path}.")
    return pd.concat(all_dfs, ignore_index=True)


def load_feature_stats(path: str | Path) -> dict[str, dict[str, list[float]]]:
    """Load per-class feature mean/std used by the multi-patient demo."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample_feature_vector(
    stats: Mapping[str, Mapping[str, list[float]]],
    disease: str,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Sample one plausible 24-feature row from class mean/std statistics."""
    if disease not in stats:
        raise KeyError(f"No feature statistics for disease '{disease}'.")
    generator = rng or np.random.default_rng()
    mean = np.asarray(stats[disease]["mean"], dtype=np.float64)
    std = np.asarray(stats[disease]["std"], dtype=np.float64)
    sampled = generator.normal(mean, std)

    for index, low in enumerate(CLIP_LOW):
        if not np.isnan(low):
            sampled[index] = max(sampled[index], low)

    rel_sum = sampled[RELATIVE_FEATURE_INDICES].sum()
    if rel_sum > 0:
        sampled[RELATIVE_FEATURE_INDICES] /= rel_sum

    return sampled.astype(float).tolist()
