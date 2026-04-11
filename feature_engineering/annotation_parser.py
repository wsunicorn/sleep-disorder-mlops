"""
Feature Engineering — Parse annotation files (.txt) từ CAP Sleep DB.
Trả về DataFrame với cột: time_sec, sleep_stage, event, duration_sec, location.

Format thực tế của CAP Sleep DB .txt:
  Sleep Stage  TAB  Position  TAB  HH:MM:SS  TAB  Event  TAB  Duration[s]  TAB  Location
  W            TAB  Unknown   TAB  22:35:17  TAB  SLEEP-S0  TAB  30  TAB  EOG
"""

import re
import pandas as pd
from pathlib import Path
from loguru import logger

# Event → integer stage mapping
EVENT_TO_STAGE = {
    "SLEEP-S0": 0,   # Wake
    "SLEEP-N":  0,   # Wake / unscored (treat as wake)
    "SLEEP-S1": 1,
    "SLEEP-S2": 2,
    "SLEEP-S3": 3,
    "SLEEP-S4": 4,
    "SLEEP-REM": 5,
    "SLEEP-MT":  6,  # Body movement
    "SLEEP-UNSCORED": -1,
}

# Column header line identifier
_HEADER_MARKER = "Sleep Stage"


def parse_txt_annotation(txt_path: str) -> pd.DataFrame:
    """
    Parse file annotation .txt từ CAP Sleep DB (REMlogic export format).

    Returns:
        DataFrame với cột: time_sec, sleep_stage (int), event, duration_sec, location
    """
    records = []
    txt_path = Path(txt_path)

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Tìm dòng header "Sleep Stage\tPosition\tTime..."
    data_start = None
    for i, line in enumerate(lines):
        if _HEADER_MARKER in line and "Time" in line:
            data_start = i + 1   # dòng tiếp theo là data
            break

    if data_start is None:
        logger.warning(f"Could not find header in {txt_path.name}. Trying fallback.")
        # Fallback: tìm dòng đầu tiên có timestamp HH:MM:SS
        for i, line in enumerate(lines):
            if re.search(r"\b\d{2}:\d{2}:\d{2}\b", line):
                data_start = i
                break

    if data_start is None:
        logger.error(f"No data found in {txt_path.name}")
        return pd.DataFrame()

    for line in lines[data_start:]:
        line = line.rstrip("\n")
        if not line.strip():
            continue

        parts = re.split(r"\t", line)
        # Cần ít nhất: stage, position, time, event, duration
        if len(parts) < 5:
            continue

        try:
            # parts[2] = HH:MM:SS
            time_str = parts[2].strip()
            if not re.match(r"\d{2}:\d{2}:\d{2}", time_str):
                continue

            h, m, s = map(int, time_str.split(":"))
            time_sec = h * 3600 + m * 60 + s

            event        = parts[3].strip()
            duration_sec = float(parts[4].strip()) if parts[4].strip() else 30.0
            location     = parts[5].strip() if len(parts) > 5 else ""

            # Chỉ lấy epoch-level sleep stage events
            stage_int = EVENT_TO_STAGE.get(event.upper(), None)

            records.append({
                "time_sec":    time_sec,
                "sleep_stage": stage_int,
                "event":       event,
                "duration_sec": duration_sec,
                "location":    location,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    logger.info(
        f"Parsed {len(df)} annotations from {txt_path.name} "
        f"({df['sleep_stage'].notna().sum()} with stage labels)"
    )
    return df


def get_epoch_labels(annotation_df: pd.DataFrame, epoch_duration: float = 30.0) -> dict:
    """
    Trả về dict: {epoch_index: sleep_stage} dựa trên annotation.
    """
    labels = {}
    stage_rows = annotation_df[annotation_df["sleep_stage"].notna()].copy()

    for _, row in stage_rows.iterrows():
        n_epochs = max(1, int(row["duration_sec"] / epoch_duration))
        start_epoch = int(row["time_sec"] / epoch_duration)
        for i in range(n_epochs):
            labels[start_epoch + i] = int(row["sleep_stage"])

    return labels
