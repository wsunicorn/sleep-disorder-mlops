"""
IoT Simulation — MQTT Subscriber
Nhận epoch từ MQTT broker, lưu raw data lên S3, ghi metadata vào PostgreSQL.
"""

import json
import os
import io
import numpy as np
import boto3
import psycopg2
import psycopg2.extras
import paho.mqtt.client as mqtt
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "sleep/sensor")
S3_BUCKET = os.getenv("S3_BUCKET", "sleep-mlops-data")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
DATABASE_URL = os.getenv("DATABASE_URL")

# AWS clients
s3_client = boto3.client("s3", region_name=AWS_REGION)


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def ensure_table_exists():
    """Tạo bảng epoch_metadata nếu chưa có."""
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS epoch_metadata (
                    id SERIAL PRIMARY KEY,
                    patient_id VARCHAR(50) NOT NULL,
                    epoch_index INTEGER NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    sfreq FLOAT NOT NULL,
                    n_channels INTEGER NOT NULL,
                    s3_key TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_epoch_patient
                    ON epoch_metadata (patient_id, epoch_index);
            """)
    conn.close()
    logger.info("Database table ready.")


def save_epoch_to_s3(patient_id: str, epoch_index: int, epoch_data: list) -> str:
    """
    Lưu epoch data dưới dạng numpy .npy lên S3.
    Trả về S3 key.
    """
    arr = np.array(epoch_data, dtype=np.float32)
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)

    s3_key = f"raw-epochs/{patient_id}/epoch_{epoch_index:05d}.npy"
    s3_client.upload_fileobj(
        buffer,
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )
    return s3_key


def save_metadata_to_db(metadata: dict):
    """Ghi thông tin epoch vào PostgreSQL."""
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO epoch_metadata
                    (patient_id, epoch_index, timestamp, sfreq, n_channels, s3_key)
                VALUES (%(patient_id)s, %(epoch_index)s, %(timestamp)s,
                        %(sfreq)s, %(n_channels)s, %(s3_key)s)
                """,
                metadata,
            )
    conn.close()


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        patient_id = payload["patient_id"]
        epoch_index = payload["epoch_index"]
        epoch_data = payload["data"]
        sfreq = payload["sfreq"]
        timestamp = payload["timestamp"]
        n_channels = len(payload["channels"])

        # Lưu data lên S3
        s3_key = save_epoch_to_s3(patient_id, epoch_index, epoch_data)

        # Ghi metadata vào RDS
        save_metadata_to_db({
            "patient_id": patient_id,
            "epoch_index": epoch_index,
            "timestamp": timestamp,
            "sfreq": sfreq,
            "n_channels": n_channels,
            "s3_key": s3_key,
        })

        logger.info(
            f"Saved epoch {epoch_index:04d} | patient={patient_id} | s3={s3_key}"
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        topic = f"{MQTT_TOPIC_PREFIX}/#"
        client.subscribe(topic, qos=1)
        logger.info(f"Subscribed to {topic}")
    else:
        logger.error(f"MQTT connection failed, code={rc}")


def main():
    ensure_table_exists()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    logger.info("Subscriber running. Waiting for messages...")
    client.loop_forever()


if __name__ == "__main__":
    main()
