"""
IoT Simulation — MQTT Simulator
Đọc từng epoch 30 giây từ file .edf, publish lên Mosquitto MQTT broker.
Topic: sleep/sensor/<patient_id>
"""

import json
import time
import argparse
import os
import numpy as np
import mne
import paho.mqtt.client as mqtt
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "sleep/sensor")
EPOCH_DURATION_SEC = 30
PUBLISH_DELAY_SEC = 0.1  # Tốc độ publish (0.1s = nhanh cho demo)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"Connected to MQTT broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
    else:
        logger.error(f"MQTT connection failed with code {rc}")


def read_edf_epochs(edf_path: str):
    """
    Đọc file .edf, áp dụng bandpass filter, cắt thành epoch 30s.
    Trả về generator của (epoch_index, epoch_data dict).
    """
    logger.info(f"Loading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Bandpass filter 0.5–40 Hz
    raw.filter(l_freq=0.5, h_freq=40.0, method="fir", verbose=False)

    sfreq = raw.info["sfreq"]
    n_samples_per_epoch = int(EPOCH_DURATION_SEC * sfreq)
    n_epochs = int(raw.n_times // n_samples_per_epoch)
    channel_names = raw.ch_names

    logger.info(
        f"Sampling rate: {sfreq} Hz | "
        f"Channels: {len(channel_names)} | "
        f"Total epochs: {n_epochs}"
    )

    for i in range(n_epochs):
        start = i * n_samples_per_epoch
        stop = start + n_samples_per_epoch
        # Shape: (n_channels, n_samples)
        epoch_data, _ = raw[:, start:stop]

        yield i, {
            "epoch_index": i,
            "start_sample": start,
            "sfreq": sfreq,
            "channels": channel_names,
            # Gửi dạng list of list (JSON-serializable)
            "data": epoch_data.tolist(),
            "timestamp": datetime.utcnow().isoformat(),
        }


def publish_epochs(edf_path: str, patient_id: str, delay: float = PUBLISH_DELAY_SEC):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    client.loop_start()

    topic = f"{MQTT_TOPIC_PREFIX}/{patient_id}"
    logger.info(f"Publishing to topic: {topic}")

    for epoch_idx, epoch_payload in read_edf_epochs(edf_path):
        # Thêm patient metadata vào payload
        epoch_payload["patient_id"] = patient_id
        message = json.dumps(epoch_payload)

        result = client.publish(topic, message, qos=1)
        result.wait_for_publish(timeout=5)

        logger.info(
            f"Published epoch {epoch_idx:04d} | "
            f"patient={patient_id} | "
            f"size={len(message)} bytes"
        )
        time.sleep(delay)

    client.loop_stop()
    client.disconnect()
    logger.info("Simulation complete.")


def main():
    parser = argparse.ArgumentParser(description="EDF → MQTT Simulator")
    parser.add_argument("--edf", required=True, help="Path to .edf file")
    parser.add_argument("--patient-id", required=True, help="Patient ID (e.g. n1)")
    parser.add_argument(
        "--delay",
        type=float,
        default=PUBLISH_DELAY_SEC,
        help="Delay between epoch publishes (seconds)",
    )
    args = parser.parse_args()

    publish_epochs(args.edf, args.patient_id, args.delay)


if __name__ == "__main__":
    main()
