"""
upload_models_to_seaweedfs.py
------------------------------
Utility script to upload trained ML recommendation models to SeaweedFS (S3-compatible object storage).
Uploads artifacts from evaluation/ml_pipeline/models/ to SeaweedFS buckets (recsys-data & mlflow).
"""

import os
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SeaweedFSUploader")

SEAWEEDFS_FILER = os.getenv("SEAWEEDFS_FILER_URL", "http://localhost:8888")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "evaluation", "ml_pipeline", "models")


def upload_model_file(local_path: str, bucket: str = "recsys-data", prefix: str = "models"):
    """Upload single model file to SeaweedFS via HTTP REST API."""
    if not os.path.exists(local_path):
        logger.warning(f"File not found: {local_path}")
        return False

    filename = os.path.basename(local_path)
    target_url = f"{SEAWEEDFS_FILER}/{bucket}/{prefix}/{filename}"
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)

    logger.info(f"Uploading '{filename}' ({file_size_mb:.2f} MB) to {target_url}...")

    try:
        with open(local_path, "rb") as f:
            res = requests.put(target_url, data=f, timeout=60.0)

        if res.status_code in (200, 201):
            logger.info(f"Successfully uploaded '{filename}' to SeaweedFS bucket '{bucket}'.")
            return True
        else:
            logger.error(f"Failed to upload '{filename}': Status {res.status_code} - {res.text}")
            return False
    except Exception as e:
        logger.error(f"Error uploading '{filename}' to SeaweedFS: {e}")
        return False


def upload_all_models():
    """Upload all model artifacts to SeaweedFS object storage."""
    logger.info(f"Starting model upload to SeaweedFS Filer at {SEAWEEDFS_FILER}...")
    if not os.path.exists(MODELS_DIR):
        logger.error(f"Models directory not found at {MODELS_DIR}")
        return

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".pkl", ".json", ".pt", ".onnx"))]
    if not model_files:
        logger.warning("No model files found to upload.")
        return

    success_count = 0
    for filename in model_files:
        path = os.path.join(MODELS_DIR, filename)
        if upload_model_file(path, bucket="recsys-data", prefix="models"):
            upload_model_file(path, bucket="mlflow", prefix="models")
            success_count += 1

    logger.info(f"SeaweedFS Upload Completed: {success_count}/{len(model_files)} models uploaded successfully.")


if __name__ == "__main__":
    upload_all_models()
