"""
Feature Engineering — Gắn nhãn sleep stage cho từng epoch.
Join features.parquet với annotation .txt → dataset có label thực tế.
Task: Sleep Stage Classification (Wake / S1 / S2 / S3 / S4 / REM)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from annotation_parser import parse_txt_annotation, EVENT_TO_STAGE

STAGE_NAMES = {0: "Wake", 1: "S1", 2: "S2", 3: "S3", 4: "S4", 5: "REM", 6: "Movement"}
EPOCH_DURATION = 30.0


def build_labeled_dataset(features_dir: str, raw_dir: str, output_dir: str):
    features_path = Path(features_dir) / "features.parquet"
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading features: {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Features shape: {df.shape}")

    labeled_rows = []

    for subject_id in df["subject_id"].unique():
        txt_path = raw_dir / f"{subject_id}.txt"
        if not txt_path.exists():
            logger.warning(f"No annotation for {subject_id}, skipping stage labeling.")
            continue

        ann = parse_txt_annotation(str(txt_path))
        # Chỉ lấy sleep-stage events
        stage_ann = ann[ann["sleep_stage"].notna() & (ann["sleep_stage"] >= 0)].copy()

        # Tính epoch_index từ time_sec của annotation
        # EDF recording bắt đầu từ giây 0, annotation time_sec là giờ trong ngày
        # Cần tính offset: lấy time_sec annotation đầu tiên làm t0
        if stage_ann.empty:
            continue

        t0 = stage_ann["time_sec"].iloc[0]
        stage_ann["epoch_from_start"] = ((stage_ann["time_sec"] - t0) / EPOCH_DURATION).round().astype(int)

        # Tạo dict {epoch_index: sleep_stage}
        epoch_label_map = {}
        for _, row in stage_ann.iterrows():
            n_epochs = max(1, int(row["duration_sec"] / EPOCH_DURATION))
            start_e = int(row["epoch_from_start"])
            for i in range(n_epochs):
                epoch_label_map[start_e + i] = int(row["sleep_stage"])

        # Gán nhãn vào feature rows
        subj_df = df[df["subject_id"] == subject_id].copy()
        subj_df["sleep_stage"] = subj_df["epoch_index"].map(epoch_label_map)

        n_labeled = subj_df["sleep_stage"].notna().sum()
        logger.info(f"{subject_id}: {len(subj_df)} epochs, {n_labeled} labeled with sleep stage")

        subj_df = subj_df.dropna(subset=["sleep_stage"])
        subj_df["sleep_stage"] = subj_df["sleep_stage"].astype(int)
        subj_df["stage_name"] = subj_df["sleep_stage"].map(STAGE_NAMES)
        labeled_rows.append(subj_df)

    if not labeled_rows:
        logger.error("No labeled data produced.")
        return

    result = pd.concat(labeled_rows, ignore_index=True)
    out_path = output_dir / "dataset_labeled.parquet"
    result.to_parquet(out_path, index=False)

    logger.info(f"\nSaved {len(result)} labeled epochs → {out_path}")
    logger.info("Sleep stage distribution:")
    dist = result.groupby(["sleep_stage", "stage_name"]).size()
    logger.info(f"\n{dist.to_string()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build labeled dataset")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--raw-dir",      default="data/raw")
    parser.add_argument("--output-dir",   default="data/features")
    args = parser.parse_args()

    build_labeled_dataset(args.features_dir, args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
